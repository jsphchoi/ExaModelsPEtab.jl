# Collocation, continuity, initial-condition, and condition-value constraints

# The ONE scale-partition helper: emits lhs(row) - physical(p[row[pat]]) per scale group.
# The scale must partition the iterator, never branch in-kernel (0*exp(big) = NaN).
function _add_scaled_cons(c::ExaCore, p, rows, pscale::Vector{Symbol}, lhs; pat::Int)
    for sc in (:log10, :log, :lin)
        grp = [r for r in rows if pscale[r[pat]] === sc]
        isempty(grp) && continue
        g = sc === :log10 ? (r -> lhs(r) - exp(log(10.0) * p[r[pat]])) :
            sc === :log   ? (r -> lhs(r) - exp(p[r[pat]]))             :
                            (r -> lhs(r) - p[r[pat]])
        c, _ = ExaModels.add_con(c, Base.Generator(g, grp))
    end
    return c
end

# (*) Main function for creating ExaModels collocation equations (*)
function _create_collocation(c::ExaCore, spec::PEtabSpec, mesh::PEtabMesh)
    c = _create_lagrange(c, spec, mesh)
    c = _create_cv_constraints(c, spec)
    return c
end

# Create lagrange collocation equations
function _create_lagrange(c::ExaCore, spec::PEtabSpec, mesh::PEtabMesh)
    (; Np, Nc, Nz, Ncv, Nu, pscale, simcv) = spec
    N, K = c.N, c.K
    u_vals = mesh.u_vals

    # Unpack variables (the core carries h[i], t[i,k], and the dlⱼdτ(τₖ) weights)
    z = c.z; p = c.p
    if Ncv >= 1
        cv = c.cv
    end

    # Create collocation equations: ∑dlⱼdτ(τₖ)*zᵢⱼ = hi*f(...), one call per rhs equation
    for (vidx, f) in enumerate(spec.rhs_fns)
        itr_coll = [(vidx, cidx, simcv[cidx], ntuple(g -> u_vals[g,i,cidx], Nu), i, k)
                    for cidx in 1:Nc, i in 1:N, k in 1:K]
        EMC.@add_con_collocation(c, z[vidx,cidx],
            f(
                ntuple(v -> z[v,cidx,i,k], Nz)...,         # state vars
                ntuple(m -> _p_phys(p,m,pscale), Np)...,   # physical params (10^θ)
                ntuple(m -> cv[m,sc], Ncv)...,             # condition-dep. vars
                ntuple(g -> uv[g], Nu)...,                 # piecewise(time) control values
                t                                          # time at collocation point
            )
            for (vidx, cidx, sc, uv, i, k) in itr_coll
        )
    end
    return c
end

# Create auxiliary variable constraints for cv[cvidx,col] = {numeric value, p}
function _create_cv_constraints(c::ExaCore, spec::PEtabSpec)
    (; Ncv, Ncc, pscale, cv_cells) = spec
    Ncv >= 1 || return c
    p  = c.p
    cv = c.cv

    # cv columns = every distinct referenced conditionId; bind ALL of them so the extra
    # pre-eq columns the steady-state residual reads are constrained too.
    itr_cv_fix = Tuple{Int, Int, Float64}[]
    itr_cv_p   = Tuple{Int, Int, Int}[]
    for col in 1:Ncc, cvidx in 1:Ncv
        cell = cv_cells[cvidx, col]
        cell isa Float64  && push!(itr_cv_fix, (cvidx, col, cell))
        cell isa ParamRef && push!(itr_cv_p, (cvidx, col, cell.pidx))
    end

    if !isempty(itr_cv_fix)
        # Condition-dependent variable 'cvidx' at column 'col' is a fixed value
        ExaModels.@add_con(c,
            cv[cvidx,col] - val
            for (cvidx, col, val) in itr_cv_fix
        )
    end
    if !isempty(itr_cv_p)
        # cv[cvidx,col] = linearized value of p[pidx]
        c = _add_scaled_cons(c, p, itr_cv_p, pscale, r -> cv[r[1], r[2]]; pat = 3)
    end
    return c
end

# (*) Main function for creating collocation continuity constraints (*)
function _create_continuity(c::ExaCore, spec::PEtabSpec, mesh::PEtabMesh)
    c = _create_interval_continuity(c)
    c = _create_initial_conditions(c, spec, mesh)
    return c
end

# Create cross-interval continuity constraints
function _create_interval_continuity(c::ExaCore)
    z = c.z
    # Create interval continuity equations: ∑ⱼlⱼ(1)*zᵢⱼ = zᵢ₊₁,₀ over every collocated slot
    EMC.@add_con_continuity(c, z)
    return c
end

# Create initial condition continuity constraints
function _create_initial_conditions(c::ExaCore, spec::PEtabSpec, mesh::PEtabMesh)
    (; Nz, Nc, Np, Ncv, pscale, simcv) = spec
    z = c.z
    p = c.p
    if Ncv >= 1
        cv = c.cv
    end

    if _has_preequilibration(spec)
        ###############################################################
        # Initial condition equations: steady-state pre-equilibration
        ###############################################################
        zss = c.zss
        overridden = Set(_overridden_states(spec))

        # Constraint 1: non-overridden simulation initial conditions carry the
        # pre-equilibration steady state. zss is indexed by condition pair cidx.
        itr_ss1 = [(v, cidx) for cidx in 1:Nc for v in 1:Nz if !(v in overridden)]
        ExaModels.@add_con(c,
            # z[:,cidx,1,0] = zss[:,cidx]
            z[v,cidx,1,0] - zss[v,cidx]
            for (v, cidx) in itr_ss1
        )

        # Overridden states take their conditions-table species column instead
        itr_ss_cv = [(v, cidx, spec.z0[v].cvidx, simcv[cidx])
                     for cidx in 1:Nc for v in 1:Nz if v in overridden]
        if !isempty(itr_ss_cv)
            ExaModels.@add_con(c,
                z[v,cidx,1,0] - cv[cvidx,col]
                for (v, cidx, cvidx, col) in itr_ss_cv
            )
        end

        # Constraint 2: steady-state residual f(zss[:,cidx]) = 0 evaluated under the
        # PRE-EQUILIBRATION condition's inputs cv[:, precv[cidx]]
        c = _create_residual_ss(c, spec, mesh.u_vals_ss, cidx -> spec.precv[cidx])
    else
        ###############################################################
        # Initial condition equations
        ###############################################################
        # x0 classes precomputed in the spec: fixed value, p, cv, or f(p,cv)
        itr_z0_fix  = Tuple{Int, Int, Float64}[]
        itr_z0_p    = Tuple{Int, Int, Int}[]
        itr_z0_cv   = Tuple{Int, Int, Int, Int}[]
        itr_z0_func = Tuple{Int, Any}[]
        for v in 1:Nz
            class = spec.z0[v]
            if class isa Float64
                # if z0 is a numeric value...
                append!(itr_z0_fix, ((v, cidx, class) for cidx in 1:Nc))
            elseif class isa ParamRef
                # if z0 is an unknown parameter p...
                append!(itr_z0_p, ((v, cidx, class.pidx) for cidx in 1:Nc))
            elseif class isa CvRef
                # if z0 is a condition-dependent variable cv...
                append!(itr_z0_cv, ((v, cidx, class.cvidx, simcv[cidx]) for cidx in 1:Nc))
            else
                # if z0 is some arbitrary function of [p,cv]...
                push!(itr_z0_func, (v, class.fn))
            end
        end

        # Create constraints
        if !isempty(itr_z0_fix)
            # Initial condition is a fixed numeric value
            ExaModels.@add_con(c,
                z[v,cidx,1,0] - val
                for (v, cidx, val) in itr_z0_fix
            )
        end
        if !isempty(itr_z0_p)
            # Initial condition is the linearized value of unknown parameter, p
            c = _add_scaled_cons(c, p, itr_z0_p, pscale, r -> z[r[1], r[2], 1, 0]; pat = 3)
        end
        if !isempty(itr_z0_cv)
            # Initial condition is a condition-dependent variable, cv
            ExaModels.@add_con(c,
                z[v,cidx,1,0] - cv[cvidx,col]
                for (v, cidx, cvidx, col) in itr_z0_cv
            )
        end
        if !isempty(itr_z0_func)
            # Initial condition is some arbitrary function, f(p,cv)
            for (v, z0_func) in itr_z0_func
                itr = [(v, cidx, simcv[cidx]) for cidx in 1:Nc]
                ExaModels.@add_con(c,
                    z[v,cidx,1,0] - z0_func(
                        ntuple(m -> _p_phys(p,m,pscale), Np)...,
                        ntuple(m -> cv[m,col], Ncv)...
                    )
                    for (v, cidx, col) in itr
                )
            end
        end
    end

    return c
end
