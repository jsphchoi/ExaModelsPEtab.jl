# Nominal ODE solves for initial guesses

const _ODE_ABSTOL = 1.0e-8
const _ODE_RELTOL = 1.0e-8
const _T_SS = 1.0e8   # steady-state-by-simulation horizon

# PEtab.jl's small-model default, with FBDF above it (OrdinaryDiffEq v7 no longer
# re-exports PEtab's QNDF/KenCarp4 picks)
_default_solver(Nz::Int, Np::Int) =
    (Nz <= 15 && Np <= 20) ? ODE.Rodas5P() : ODE.FBDF()

_has_preequilibration(spec::PEtabSpec) = any(>(0), spec.precv)

# States whose initial value a conditions-table species column overrides
_overridden_states(spec::PEtabSpec) =
    [v for v in 1:spec.Nz if string(spec.z_syms_bare[v]) in spec.cv_names]

# Numeric name => value map for one cv column at estimation-scale theta
function _condition_values(spec::PEtabSpec, theta::Vector{Float64}, col::Int)
    vals = Dict{String, Float64}()
    for (k, v) in spec.fixed_vals
        Symbolics.value(v) isa Number && (vals[string(k)] = Float64(Symbolics.value(v)))
    end
    for m in 1:spec.Np
        vals[spec.pnames[m]] = _p_phys_val(theta, m, spec.pscale)
    end
    for (cvidx, name) in enumerate(spec.cv_names)
        vals[name] = _cell_value(spec, theta, cvidx, col)
    end
    # Resolve the few fixed entries that stayed symbolic (assignment chains over p, cv)
    subs = Dict{Symbolics.Num, Float64}(
        Symbolics.variable(Symbol(k)) => v for (k, v) in vals
    )
    for (k, v) in spec.fixed_vals
        Symbolics.value(v) isa Number && continue
        r = Symbolics.value(Symbolics.substitute(v, subs))
        r isa Number ||
            error("ExaModelsPEtab: cannot resolve fixed parameter '$k' to a numeric value.")
        vals[string(k)] = Float64(r)
    end
    return vals
end

# Numeric value of one condition cell at estimation-scale theta
_cell_value(spec::PEtabSpec, theta, cvidx::Int, col::Int) =
    (cell = spec.cv_cells[cvidx, col];
     cell isa ParamRef ? _p_phys_val(theta, cell.pidx, spec.pscale) : cell)

# Numeric initial state for one cv column, from the z0 classification
function _initial_state(spec::PEtabSpec, theta::Vector{Float64}, col::Int)
    u0 = Vector{Float64}(undef, spec.Nz)
    for v in 1:spec.Nz
        class = spec.z0[v]
        u0[v] = class isa Float64  ? class :
                class isa ParamRef ? _p_phys_val(theta, class.pidx, spec.pscale) :
                class isa CvRef    ? _cell_value(spec, theta, class.cvidx, col) :
                Float64(class.fn(
                    (_p_phys_val(theta, m, spec.pscale) for m in 1:spec.Np)...,
                    (_cell_value(spec, theta, cvidx, col) for cvidx in 1:spec.Ncv)...,
                ))
    end
    return u0
end

# ODEProblem for one cv column, numeric u0 and parameters, no MTK initialization
function _make_odeproblem(spec::PEtabSpec, modelsys::PEtabModelSys, theta, col::Int,
                          tspan; u0_override = nothing)
    vals = _condition_values(spec, theta, col)
    u0 = u0_override === nothing ? _initial_state(spec, theta, col) : u0_override
    u0map = Dict{Any, Any}(state => u0[v] for (v, state) in enumerate(spec.z_syms))
    pmap = Dict{Any, Any}()
    for pp in MTK.parameters(modelsys.sys)
        name = string(pp)
        if occursin("__parameter_ifelse", name)
            pmap[pp] = 0.0   # set by the callbacks' initialize at solve start
        elseif haskey(vals, name)
            pmap[pp] = vals[name]
        else
            error("ExaModelsPEtab: no value for model parameter '$name'.")
        end
    end
    return ODE.ODEProblem(modelsys.sys, merge(u0map, pmap), tspan;
                          build_initializeprob = false)
end

function _assert_solved(sol, what::String)
    ODE.SciMLBase.successful_retcode(sol) ||
        error("ExaModelsPEtab: the nominal $what solve failed (retcode $(sol.retcode)).")
    return nothing
end

# Dense nominal solve of one condition pair, pre-equilibrating first when required.
# Returns `(sol, zss)` with `zss = nothing` without pre-eq. Errors if the solve fails.
function _condition_solution(spec::PEtabSpec, modelsys::PEtabModelSys, theta::Vector{Float64},
                             cidx::Int, tend::Float64, solver; tstops)
    col = spec.simcv[cidx]
    zss = nothing
    if spec.precv[cidx] > 0
        # Pre-equilibrate, then carry the steady state into every non-overridden state
        zss = _equilibrate(spec, modelsys, theta, spec.precv[cidx], solver)
        overridden = _overridden_states(spec)
        u0 = _initial_state(spec, theta, col)
        for v in 1:spec.Nz
            v in overridden || (u0[v] = zss[v])
        end
        oprob = _make_odeproblem(spec, modelsys, theta, col, (0.0, tend); u0_override = u0)
    else
        oprob = _make_odeproblem(spec, modelsys, theta, col, (0.0, tend))
    end
    sol = ODE.solve(oprob, solver; callback = modelsys.callbacks, tstops = tstops,
                    abstol = _ODE_ABSTOL, reltol = _ODE_RELTOL)
    _assert_solved(sol, "condition '$(spec.conds[cidx].sim)'")
    return sol, zss
end

# Terminal state of the pre-equilibration solve to t = _T_SS
function _equilibrate(spec::PEtabSpec, modelsys::PEtabModelSys, theta, col::Int, solver)
    oprob = _make_odeproblem(spec, modelsys, theta, col, (0.0, _T_SS))
    sol = ODE.solve(oprob, solver; callback = modelsys.callbacks,
                    abstol = _ODE_ABSTOL, reltol = _ODE_RELTOL, save_everystep = false)
    _assert_solved(sol, "pre-equilibration")
    return Vector{Float64}(sol.u[end])
end

"""
    _solve_conditions(spec, modelsys, theta, mesh, taus)

One nominal solve per condition pair. Returns `(z_init, zss_init)`:
`z_init[v, cidx, i, k+1]` interpolated at every collocation point, and
`zss_init[v, cidx]` the pre-equilibration steady states (`nothing` without pre-eq).
"""
function _solve_conditions(spec::PEtabSpec, modelsys::PEtabModelSys, theta::Vector{Float64},
                           mesh::PEtabMesh, taus::AbstractVector)
    (; Nz, Nc) = spec
    N, K = mesh.N, length(taus)
    solver = _default_solver(Nz, spec.Np)
    z_init = Array{Float64, 4}(undef, Nz, Nc, N, K + 1)
    zss_init = _has_preequilibration(spec) ? Matrix{Float64}(undef, Nz, Nc) : nothing

    for cidx in 1:Nc
        tend = mesh.nodes[cidx, end]
        sol, zss = _condition_solution(spec, modelsys, theta, cidx, tend, solver;
                                       tstops = mesh.nodes[cidx, :])
        zss === nothing || (zss_init[:, cidx] = zss)
        for i in 1:N
            h = mesh.nodes[cidx, i + 1] - mesh.nodes[cidx, i]
            z_init[:, cidx, i, 1] = sol(mesh.nodes[cidx, i])
            for k in 1:K
                z_init[:, cidx, i, k + 1] = sol(mesh.nodes[cidx, i] + h * taus[k])
            end
        end
    end
    return z_init, zss_init
end
