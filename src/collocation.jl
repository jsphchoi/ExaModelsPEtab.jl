# (*) Main function for creating ExaModels collocation equations (*)
function _create_collocation(
        c::ExaCore,
        PEmodel::PEtabModel,
        PEprob::PEtabODEProblem,
        PEinfo::PEInfo;
        adaptive_mesh::Bool = false
    )
    c = _create_lagrange(c, PEmodel, PEprob, PEinfo; adaptive_mesh = adaptive_mesh)
    c = _create_cv_constraints(c, PEmodel, PEprob, PEinfo)
    return c
end

# Create lagrange collocation equations
# Set adaptive_mesh = true to make h[i] and t[i,j] into ExaModels parameters instead of
# constants (for AMREE implementation later on). Prolongs compile time
function _create_lagrange(
        c::ExaCore,
        PEmodel::PEtabModel,
        PEprob::PEtabODEProblem,
        PEinfo::PEInfo;
        adaptive_mesh::Bool = false
    )
    # Unpack problem info
    (; N, K, Np, Nc, Nz, Ncv, h, taus, pscale, gate_vals, t_nodes) = PEinfo

    gate_syms = _get_gate_syms(PEprob)
    Ng = length(gate_syms)

    fs = _get_rhs_funcs(PEmodel, PEprob) # obtain ODE rhs equations

    if adaptive_mesh
        #############################################################################################
        # AMREE PATH: h / t_ij / gates as ExaModels parameters, indexed symbolically
        #############################################################################################

        # Create mesh parameters, unpack variables
        c, h_par = ExaModels.add_par(c, h; name = Val(:h_mesh))
        t_init   = [t_nodes[i] + taus[k+1]*h[i] for i in 1:N, k in 1:K]
        c, t_par = ExaModels.add_par(c, t_init; name = Val(:t_mesh))
        g_par = nothing
        if Ng >= 1
            c, g_par = ExaModels.add_par(c, gate_vals; name = Val(:g_mesh))
        end
        z = c.z; p = c.p
        if Ncv >= 1
            cv = c.cv
        end

        # Create collocation equations: -hi*f(...) = (...)
        itr_coll = [(i,k,cidx) for i in 1:N, k in 1:K, cidx in 1:Nc]   # integer indices only
        c_coll   = [
            ExaModels.@add_con(c,
                -h_par[i]*f(
                    ntuple(v -> z[v,i,k,cidx], Nz)...,         # state vars
                    ntuple(m -> _p_phys(p,m,pscale), Np)...,   # physical params (10^θ)
                    ntuple(m -> cv[m,cidx], Ncv)...,           # condition-dep. vars
                    ntuple(g -> g_par[g,i,cidx], Ng)...,       # piecewise(time) gate values (θ)
                    t_par[i,k]                                 # time at collocation point (θ)
                )
                for (i,k,cidx) in itr_coll
            )
            for f in fs
        ]
    else
        ####################################################################
        # CONSTANT MESH (NO AMREE)
        ####################################################################
        # Unpack variables
        z = c.z; p = c.p
        if Ncv >= 1
            cv = c.cv
        end

        # Create collocation equations: -hi*f(...) = (...)
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
    end

    # Constraint augmentation: (...) = ∑dlⱼdτ(τₖ)*zᵢⱼ  (mesh-independent; shared by both paths)
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

# Create auxiliary variable constraints for cv[cvidx,cidx] = {numeric value, p}
function _create_cv_constraints(
        c::ExaCore,
        PEmodel::PEtabModel,
        PEprob::PEtabODEProblem,
        PEinfo::PEInfo
    )
    # Unpack problem info
    (; Ncv, pscale) = PEinfo
    Ncv >= 1 || return c
    p  = c.p
    cv = c.cv

    # Unpack DataFrame: row = experimental condition, col = condition-dependent variable
    conditions_df = PEmodel.petab_tables[:conditions]
    cv_cols = _get_cv_names(PEmodel) # cv column names, aligned 1:Ncv (no positional offset)
    # cv columns = simulation conditions (1:Nc) + distinct pre-equilibration conditions; bind ALL
    # of them so the extra pre-eq columns the steady-state residual reads are constrained too.
    cv_rows = _get_cv_cid_rows(PEmodel, PEprob) # cv column => conditions-table row
    Ncc = length(cv_rows)
    dict_pstr_pidx = _get_dict_pstr_pidx(PEprob) # string of unknown parameter, p => index of decision variable, p

    # Create iterators
    itr_cv_fix  = Tuple{Int, Int, Float64}[]
    itr_cv_p    = Tuple{Int, Int, Int}[]
    for cidx in 1:Ncc
        for cvidx in 1:Ncv
            val = conditions_df[cv_rows[cidx], cv_cols[cvidx]] # by conditionId-aligned row
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
        # cv[cvidx,cidx] = linearized value of p[pidx]
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
