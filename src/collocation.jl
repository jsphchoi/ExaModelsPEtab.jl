
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

    ########################################################################
    # Mesh geometry as MUTABLE PARAMETERS (enables rebuild-free r-refinement)
    ########################################################################
    # The interval widths h[i], the collocation times t_ij, and the piecewise(time) gate
    # values all depend on node PLACEMENT, not on the variable/constraint STRUCTURE. Lifting
    # them from baked iterator literals to ExaModels parameters (c.θ) lets an outer moving-mesh
    # loop relocate nodes and re-solve via set_value! WITHOUT rebuilding the model. taus/DLDTAU/
    # L1 depend only on K (invariant under node relocation), so they stay baked.
    #   h_mesh[i]        interval widths             (length N)
    #   t_mesh[i,k]      collocation time t_ij       (N×K), = t_nodes[i] + taus[k+1]*h[i]
    #   g_mesh[g,i,cidx] piecewise(time) gate value  (Ng×N×Nc); absent when Ng==0
    # Named (Val(:…)) so the outer loop can recover the handle as model.h_mesh / .t_mesh / .g_mesh.
    c, h_par = ExaModels.add_par(c, h; name = Val(:h_mesh))
    t_init   = [t_nodes[i] + taus[k+1]*h[i] for i in 1:N, k in 1:K]  # exact mesh, no cumsum drift
    c, t_par = ExaModels.add_par(c, t_init; name = Val(:t_mesh))
    g_par = nothing
    if Ng >= 1
        c, g_par = ExaModels.add_par(c, gate_vals; name = Val(:g_mesh))
    end

    # Capture variable handles AFTER add_par (add_par returns a fresh ExaCore; handles are
    # layout-stable, but read from the final core for clarity).
    z = c.z
    p = c.p
    if Ncv >= 1
        cv = c.cv
    end

    ########################################################################
    # Lagrange collocation equations
    ########################################################################
    # ODE RHS functions; t is always the final arg (autonomous RHS just ignores it — see
    # _get_rhs_funcs), and the live piecewise(time) gates are trailing args before it. The
    # iterator now carries only INTEGER indices (i,k,cidx); h[i], t_ij and the gate values all
    # flow through the parameter buffer c.θ, indexed symbolically (h_par[i], t_par[i,k],
    # g_par[g,i,cidx]). Ng==0 makes the gate splat a no-op, so non-event models are unchanged.
    fs = _get_rhs_funcs(PEmodel, PEprob, gate_syms)

    # Create constraint: hᵢf(...) = (...). t_ij = start of interval i + tau_k * h_i (collocation time).
    itr_coll = [(i,k,cidx) for i in 1:N, k in 1:K, cidx in 1:Nc]
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
    (; Np, Ncv, pscale) = PEinfo
    Ncv >= 1 || return c
    p  = c.p
    cv = c.cv

    ########################################################################
    # Auxiliary variable constraints for condition-dependent variables, cv
    ########################################################################

    # Unpack DataFrame: row = experimental condition, col = condition-dependent variable
    conditions_df = PEmodel.petab_tables[:conditions]
    cv_cols = _get_cv_colnames(PEmodel) # cv column names, aligned 1:Ncv (no positional offset)
    # cv columns = simulation conditions (1:Nc) + distinct pre-equilibration conditions; bind ALL
    # of them so the extra pre-eq columns the steady-state residual reads are constrained too.
    cv_rows = _get_cv_cond_rows(PEmodel, PEprob) # cv column => conditions-table row
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
