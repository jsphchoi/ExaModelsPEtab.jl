# !!! for steady-state models := measurement time = Inf
# Instead of collocation equations for the ODE RHS, set f(zss...) = 0

# A model is (pure) steady-state iff every measurement time is inf. Error any mixed cases
function _is_steady_state(spec::PEtabSpec)::Bool
    any_inf = any(isinf, spec.meas_time)
    any_inf || return false
    all(isinf, spec.meas_time) || error(
        "ExaModelsPEtab: unsupported mix of finite-time and steady-state (time=Inf) measurements."
    )
    return true
end

# Steady-state initial guesses: one forward solve per condition to t = _T_SS
function _steady_states(spec::PEtabSpec, modelsys::PEtabModelSys, theta0::Vector{Float64})
    solver = _default_solver(spec.Nz, spec.Np)
    zss0 = Matrix{Float64}(undef, spec.Nz, spec.Nc)
    for cidx in 1:spec.Nc
        zss0[:, cidx] = _equilibrate(spec, modelsys, theta0, spec.simcv[cidx], solver)
    end
    return zss0
end

# Control values at t = 0 per condition (no mesh on the steady-state path)
function _ss_u_vals(spec::PEtabSpec)
    u_vals_ss = Matrix{Float64}(undef, spec.Nu, spec.Nc)
    for cidx in 1:spec.Nc, g in 1:spec.Nu
        u_vals_ss[g, cidx] = _control_value(spec.controls[g, spec.simcv[cidx]], 0.0)
    end
    return u_vals_ss
end

# Create decision variables for the steady-state state variables
# Creation order fixes the x layout: p, y, sigma, cv, zss
function _create_variables_ss(c::ExaCore, spec::PEtabSpec, theta0::Vector{Float64}, zss0)
    c = _create_p(c, spec, theta0)
    c = _create_y(c, spec)
    c = _create_sigma(c, spec)
    if spec.Ncv >= 1
        c = _create_cv(c, spec, theta0)
    end
    c = _create_zss(c, spec; init = zss0)
    return c
end

# Creates all of the constraints for pure steady-state models
function _create_constraints_ss(c::ExaCore, spec::PEtabSpec, theta0::Vector{Float64},
                                u_vals_ss::Matrix{Float64})
    c = _create_cv_constraints(c, spec)
    W, b, keep_rows = _conservation_ss(c, spec, theta0, u_vals_ss)
    c = _create_residual_ss(c, spec, u_vals_ss, cidx -> spec.simcv[cidx]; keep_rows)
    c = _add_conservation_constraints(c, spec, W, b)
    return c
end

# Creates steady-state ODE RHS equations (used for x0SSpre and pure steady-state only models)
function _create_residual_ss(c::ExaCore, spec::PEtabSpec, u_vals_ss::Matrix{Float64},
                             cvcol_of; keep_rows = nothing)
    (; Nz, Nc, Np, Ncv, Nu, pscale) = spec
    p   = c.p
    zss = c.zss
    if Ncv >= 1
        cv = c.cv
    end

    fs = spec.rhs_fns
    keep_rows !== nothing && (fs = fs[keep_rows]) # get rid of the empty rows (redunant eqn)

    # Create steady-state ODE rhs residual equation f(zss...) = 0
    itr_ss = [(cidx, cvcol_of(cidx), ntuple(g -> u_vals_ss[g,cidx], Nu)) for cidx in 1:Nc]
    for f in fs
        ExaModels.@add_con(c,
            f(
                ntuple(v -> zss[v,cidx], Nz)...,
                ntuple(m -> _p_phys(p,m,pscale), Np)...,
                ntuple(m -> cv[m,cvcol], Ncv)...,
                ntuple(g -> uv[g], Nu)...,
                0.0
            )
            for (cidx, cvcol, uv) in itr_ss
        )
    end
    return c
end

# Steady-state node context: the single class zss[·,cidx]
function _ss_ctx(c::ExaCore, spec::PEtabSpec, zss0)
    (; Nz) = spec
    zss = c.zss
    class = NodeClass(
        n -> true,
        r -> ntuple(v -> zss[v, r[4]], Nz),
        r -> zss[r[2], r[3]],
    )
    return NodeCtx(
        midx -> (spec.meas_cidx[midx],),
        n -> ntuple(v -> zss0[v, n[1]], Nz),
        n -> spec.simcv[n[1]],
        NodeClass[class],
    )
end

# Detects conservation laws to remove additional constraint to make f(zss) = 0 sqaure wrt zss
# left null space of jacobian RHS is the conserved species
function _conservation_ss(c::ExaCore, spec::PEtabSpec, theta0::Vector{Float64},
                          u_vals_ss::Matrix{Float64})
    (; Nz, Nc, Np, Ncv, pscale) = spec
    zss0 = reshape(_var_starts(c, c.zss), Nz, Nc)
    cv0  = Ncv >= 1 ? reshape(_var_starts(c, c.cv), Ncv, :) : zeros(Float64, 0, Nc)
    p0   = [_p_phys_val(theta0, m, pscale) for m in 1:Np]

    # Derive initial guess for the ss variables
    uss = size(u_vals_ss, 2) >= 1 ? u_vals_ss[:, 1] : Float64[]
    fs  = spec.rhs_fns
    cvc = Ncv >= 1 ? cv0[:, 1] : Float64[]
    Fz  = z -> Float64[f(z..., p0..., cvc..., uss..., 0.0) for f in fs]

    # Finite difference approx of jacobian at initial states (initial guess satisfies conservation law)
    z1 = zss0[:, 1]
    F0 = Fz(z1)
    J  = zeros(Float64, Nz, Nz)
    for j in 1:Nz
        hj = 1.0e-7 * (abs(z1[j]) + 1.0)
        zp = copy(z1); zp[j] += hj
        J[:, j] = (Fz(zp) .- F0) ./ hj
    end

    # Calculate left null space of J
    Fsvd = LinearAlgebra.svd(J)
    smax = isempty(Fsvd.S) ? 0.0 : Fsvd.S[1]
    tol  = 1.0e-7 * max(smax, 1.0)
    null_idx = findall(s -> s < tol, Fsvd.S)
    r = length(null_idx)
    r == 0 && return (zeros(Float64, 0, Nz), zeros(Float64, 0, Nc), collect(1:Nz))

    W = Matrix(transpose(Fsvd.U[:, null_idx]))   # r×Nz (rows are left-null / conservation vectors)

    # Drop redundant equations from conservation law
    drop_rows = sort(LinearAlgebra.qr(W, LinearAlgebra.ColumnNorm()).p[1:r])
    keep_rows = setdiff(1:Nz, drop_rows)

    # Make sure that our finite diff on J did not depend on initial theta guess
    u0s, ic_theta_dep = _initial_conditions_ss(spec, theta0)
    ic_theta_dep && error(
        "ExaModelsPEtab: unsupported parameter-dependent conserved total in steady-state conservation law."
    )
    b = zeros(Float64, r, Nc)
    for cidx in 1:Nc, k in 1:r
        b[k, cidx] = LinearAlgebra.dot(view(W, k, :), view(u0s, :, cidx))
    end
    return (W, b, keep_rows)
end

# Get the initial conditions for the steady-state model
function _initial_conditions_ss(spec::PEtabSpec, theta0::Vector{Float64})
    (; Nz, Nc) = spec
    # Baseline u0 per condition at nominal theta
    u0s = zeros(Float64, Nz, Nc)
    for cidx in 1:Nc
        u0s[:, cidx] = _initial_state(spec, theta0, spec.simcv[cidx])
    end
    # Check for theta dependence of initial condition
    theta2 = theta0 .+ 0.1
    ic_theta_dep = false
    for cidx in 1:Nc
        u0_2 = _initial_state(spec, theta2, spec.simcv[cidx])
        if maximum(abs, u0_2 .- view(u0s, :, cidx)) > 1.0e-8 * (1.0 + maximum(abs, view(u0s, :, cidx)))
            ic_theta_dep = true
            break
        end
    end
    return u0s, ic_theta_dep
end

# Creates the conservation law constraints
function _add_conservation_constraints(c::ExaCore, spec::PEtabSpec, W::AbstractMatrix, b::AbstractMatrix)
    r = size(W, 1)
    r == 0 && return c
    (; Nz, Nc) = spec
    zss = c.zss

    # Ax=b
    itr_base = [(k, cidx, b[k, cidx]) for cidx in 1:Nc for k in 1:r]
    con = ExaModels.@add_con(c,
        -bval
        for (k, cidx, bval) in itr_base
    )
    itr_aug = [
        (pos, W[k, v], v, cidx)
        for (pos, (k, cidx, bval)) in enumerate(itr_base) for v in 1:Nz if W[k, v] != 0.0
    ]
    ExaModels.@add_con!(c, con,
        pos => Wkv*zss[v, cidx]
        for (pos, Wkv, v, cidx) in itr_aug
    )
    return c
end
