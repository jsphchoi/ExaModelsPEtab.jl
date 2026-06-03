#######################################################
# STATE AS OF: 05/30/26
#######################################################

# (*) Main function for creating ExaModels collocation equations (*)
function _create_collocation(
    c::ExaCore,
    PEmodel::PEtabModel,
    PEprob::PEtabODEProblem,
    PEinfo::PEInfo
)
    ########################################################################
    # Unpack problem info
    ########################################################################
    (; N, K, Np, Nc, Nz, Ncv, h, taus, pscale) = PEinfo
    z = c.z
    p = c.p
    if Ncv >= 1
        cv = c.cv
    end

    ########################################################################
    # Lagrange collocation equations
    ########################################################################
    # Get ODE RHS functions (and whether they depend on time t)
    fs, has_t = _get_rhs_funcs(PEmodel, PEprob)

    # Create constraint: hᵢf(...) = (...)
    if has_t
        # t_ij = start of interval i + tau_k * h_i  (collocation time)
        h_cum    = cumsum(h) .- h  # start time of each interval
        itr_coll = [(i,k,cidx,h[i],h_cum[i] + taus[k+1]*h[i]) for i in 1:N, k in 1:K, cidx in 1:Nc]
        c_coll   = [
            ExaModels.@add_con(c,
                -hi*f(
                    ntuple(v -> z[v,i,k,cidx], Nz)...,         # state vars
                    ntuple(m -> _p_phys(p,m,pscale), Np)...,   # physical params (10^θ)
                    ntuple(m -> cv[m,cidx], Ncv)...,           # condition-dep. vars
                    t_ij                                        # time at collocation point
                )
                for (i,k,cidx,hi,t_ij) in itr_coll
            )
            for f in fs
        ]
    else
        itr_coll = [(i,k,cidx,h[i]) for i in 1:N, k in 1:K, cidx in 1:Nc]
        c_coll   = [
            ExaModels.@add_con(c,
                -hi*f(
                    ntuple(v -> z[v,i,k,cidx], Nz)...,         # state vars
                    ntuple(m -> _p_phys(p,m,pscale), Np)...,   # physical params (10^θ)
                    ntuple(m -> cv[m,cidx], Ncv)...            # condition-dep. vars
                )
                for (i,k,cidx,hi) in itr_coll
            )
            for f in fs
        ]
    end

    # Constraint augmentation: (...) = ∑dlⱼdτ(τₖ)*zᵢⱼ
    DLDTAU  = [_eval_dldtau(j,k,taus) for j in 0:K, k in 1:K]
    itr_coll! = [(i,j,k,cidx,DLDTAU[j+1,k]) for i in 1:N, j in 0:K, k in 1:K, cidx in 1:Nc]
    for v in eachindex(c_coll)
        ExaModels.@add_con!(c,
            c_coll[v],
            (i,k,cidx) => z[v,i,j,cidx]*DLDTAU
            for (i,j,k,cidx,DLDTAU) in itr_coll!
        )
    end

    ########################################################################
    # Auxiliary variable constraints for condition-dependent variables, cv
    ########################################################################
    # Unpack DataFrame: row = experimental condition, col = condition-dependent variable
    conditions_df = PEmodel.petab_tables[:conditions]
    cv_cols = _get_cv_colnames(PEmodel) # cv column names, aligned 1:Ncv (no positional offset)
    cond_rows = _get_cond_rows(PEmodel) # cidx => conditions-table row (cidx ≠ row when pre-eq-only rows exist)
    dict_pstr_pidx = _get_dict_pstr_pidx(PEprob) # string of unknown parameter, p => index of decision variable, p

    # Create iterators
    itr_cv_fix  = Tuple{Int, Int, Float64}[]
    itr_cv_p    = Tuple{Int, Int, Int}[]
    for cidx in 1:Nc
        for cvidx in 1:Ncv
            val = conditions_df[cond_rows[cidx], cv_cols[cvidx]] # by conditionId-aligned row
            if val isa Number
                # If the value is a numeric value...
                push!(itr_cv_fix, (cvidx, cidx, Float64(val)))
            elseif val isa String || val isa Symbol
                # If the value is an unknown parameter, p...
                str_val = String(val)
                parsed_val = tryparse(Float64, str_val)
                if parsed_val !== nothing
                    push!(itr_cv_fix, (cvidx, cidx, parsed_val))
                else
                    if haskey(dict_pstr_pidx, str_val)
                        pidx = dict_pstr_pidx[str_val]
                        push!(itr_cv_p, (cvidx, cidx, pidx))
                    else
                        error("Condition variable '$str_val' not found in unknown parameter list.")
                    end
                end
            end
        end
    end

    # Create auxiliary variable constraints
    if !isempty(itr_cv_fix)
        # Condition-dependent variable 'cvidx' at condition 'cidx' is a fixed value
        ExaModels.@add_con(c,
            cv[cvidx,cidx] - val
            for (cvidx, cidx, val) in itr_cv_fix
        )
    end
    if !isempty(itr_cv_p)
        # Condition-dependent variable 'cvidx' at condition 'cidx' equals the PHYSICAL
        # value of unknown parameter p[pidx]. pidx is a per-entry data index here (not a
        # literal), so the scale can't be branched inside the kernel — partition by scale
        # and emit one fixed-form constraint per scale group (see _p_phys docstring).
        for sc in (:log10, :log, :lin)
            grp = [t for t in itr_cv_p if pscale[t[3]] === sc]
            isempty(grp) && continue
            if sc === :log10
                ExaModels.@add_con(c, cv[cvidx,cidx] - exp(log(10.0)*p[pidx]) for (cvidx,cidx,pidx) in grp)
            elseif sc === :log
                ExaModels.@add_con(c, cv[cvidx,cidx] - exp(p[pidx])           for (cvidx,cidx,pidx) in grp)
            else
                ExaModels.@add_con(c, cv[cvidx,cidx] - p[pidx]                for (cvidx,cidx,pidx) in grp)
            end
        end
    end

    return c
end
