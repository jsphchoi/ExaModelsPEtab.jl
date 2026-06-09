
# (*) Main function for creating ExaModels collocation equations (*)
function _create_collocation(
        c::ExaCore,
        PEmodel::PEtabModel,
        PEprob::PEtabODEProblem,
        PEinfo::PEInfo
    )
    c = _create_lagrange(c, PEmodel, PEprob, PEinfo)
    c = _create_cv_constraints(c, PEmodel, PEprob, PEinfo)
    return c
end

# Create lagrange collocation equations
function _create_lagrange(c::ExaCore, PEmodel::PEtabModel, PEprob::PEtabODEProblem, PEinfo::PEInfo)
    ########################################################################
    # Unpack problem info
    ########################################################################
    (; N, K, Np, Nc, Nz, Ncv, h, taus, pscale, gate_syms, gate_vals, t_nodes) = PEinfo
    Ng = length(gate_syms)
    z = c.z
    p = c.p
    if Ncv >= 1
        cv = c.cv
    end

    ########################################################################
    # Lagrange collocation equations
    ########################################################################
    # ODE RHS functions; t is always the final arg (autonomous RHS just ignores it — see
    # _get_rhs_funcs), and the live piecewise(time) gates are trailing args before it. Their
    # per-(interval,condition) values gate_vals[:,i,cidx] are carried in the iterator tuple as an
    # NTuple field `gv` (exactly like h[i]/t_ij), then splatted into f via ntuple(g->gv[g],Ng)...
    # (ExaModels supports getindex on a tuple data field but NOT splatting it directly). Ng==0 makes
    # `gv` an empty tuple and the splat a no-op, so non-event models are unchanged.
    fs = _get_rhs_funcs(PEmodel, PEprob, gate_syms)

    # Create constraint: hᵢf(...) = (...). t_ij = start of interval i + tau_k * h_i (collocation time).
    # t_nodes[i] = the EXACT start time of interval i (original mesh node, no cumsum(h) round-off drift).
    itr_coll = [(i,k,cidx,h[i],t_nodes[i] + taus[k+1]*h[i], ntuple(g->gate_vals[g,i,cidx],Ng))
                for i in 1:N, k in 1:K, cidx in 1:Nc]
    c_coll   = [
        ExaModels.@add_con(c,
            -hi*f(
                ntuple(v -> z[v,i,k,cidx], Nz)...,         # state vars
                ntuple(m -> _p_phys(p,m,pscale), Np)...,   # physical params (10^θ)
                ntuple(m -> cv[m,cidx], Ncv)...,           # condition-dep. vars
                ntuple(g -> gv[g], Ng)...,                 # piecewise(time) gate values
                t_ij                                       # time at collocation point
            )
            for (i,k,cidx,hi,t_ij,gv) in itr_coll
        )
        for f in fs
    ]

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

    return c
end

# Auxiliary variable constraints binding each cv[cvidx,cidx] to its per-condition value
# (a numeric literal or the PHYSICAL value of an unknown parameter p). Shared by the
# collocation (time-course) path and the steady-state path. No-op when Ncv == 0.
function _create_cv_constraints(c::ExaCore, PEmodel::PEtabModel, PEprob::PEtabODEProblem, PEinfo::PEInfo)
    ########################################################################
    # Unpack problem info
    ########################################################################
    (; Np, Ncv, Nc, pscale) = PEinfo
    Ncv >= 1 || return c
    p  = c.p
    cv = c.cv

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
