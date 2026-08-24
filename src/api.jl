# examodel_petab entry point and PEtabExaModel wrapper

# Condition axis = mesh axis by default; :shared is the fallback with one global mesh
const _VARIANT = :percond

"""
    PEtabMeta

The maps a consumer needs to read a solution, carried on the model as `model.petab`.

- `pnames`, `pscale` : decision-parameter names and estimation scales, aligned to `p[1:Np]`
- `conds` : the (preeq, sim) condition pairs, aligned to the condition axis
- `meas_cidx`, `meas_iidx` : per measurement, its condition and mesh interval (0 = the t0
  node, empty on the steady-state path)
- `tables` : the raw parsed PEtab tables
"""
struct PEtabMeta
    pnames::Vector{Symbol}
    pscale::Vector{Symbol}
    conds::Vector{NamedTuple{(:preeq, :sim), Tuple{String, String}}}
    meas_cidx::Vector{Int}
    meas_iidx::Vector{Int}
    tables::PEtabTables
end

# Carry the PEtab labelling onto the model, readable as model.petab
function _attach_petab_meta(c::ExaCore, spec::PEtabSpec, tables::PEtabTables,
                            meas_iidx::Vector{Int})
    meta = PEtabMeta(Symbol.(spec.pnames), copy(spec.pscale), copy(spec.conds),
                     copy(spec.meas_cidx), meas_iidx, tables)
    return ExaCore(c; refs = (; getfield(c, :refs)..., petab = meta))
end

"""
    examodel_petab(filename::String; backend = nothing, K = 4, subdivide = 4, p0 = nothing, kwargs...)

Builds an `ExaModels.ExaModel` from a PEtab YAML file at `filename`, with the PEtab maps
attached as `model.petab` (a [`PEtabMeta`](@ref)).

kwargs:
- `backend` : array backend, `nothing` (default) for CPU, `CUDA.CUDABackend()` for GPU
- `K` : degree of the collocating polynomial
- `subdivide` : equal parts each required mesh interval is split into
- `p0` : start values for the estimated parameters (estimation scale, parameters-table
  order), defaulting to the nominal values
- kwargs passed on to `CollocationExaCore`

# Example
```julia
using ExaModelsPEtab, MadNLP

# Build the ExaModel from a PEtab problem YAML file
model = examodel_petab("path/to/petab.yaml")   # CPU
res = madnlp(model)

# Build the ExaModel using CUDA backend and solve with MadNLPGPU
using CUDA, MadNLPGPU
model = examodel_petab(
  "path/to/petab.yaml";
  backend = CUDA.CUDABackend()
)
res = madnlp(model; tol = 1e-6)
```
"""
function examodel_petab(
        filename::String;
        backend = nothing,
        K::Int = 4,
        subdivide::Int = 4,
        p0 = nothing,
        kwargs...
    )
    # Radau collocation on a fixed mesh is the only supported configuration
    for kw in (:adaptive, :roots, :basis)
        haskey(kwargs, kw) && error("ExaModelsPEtab: the '$kw' option is unsupported. " *
                                    "Radau collocation on a fixed mesh is the only configuration.")
    end

    tables = PEtabTables(filename)
    modelsys = _load_model(tables)
    spec = _compile_spec(tables, modelsys)
    theta0 = _resolve_theta0(spec, p0)

    if _is_steady_state(spec)
        _has_preequilibration(spec) && error(
            "ExaModelsPEtab: unsupported pre-equilibration with steady-state (time=Inf) measurements."
        )
        core = ExaModels.ExaCore(; backend = backend)
        return _build_ss(core, tables, spec, modelsys, theta0)
    else
        mesh = _build_mesh(spec; subdivide = subdivide, variant = _VARIANT)
        core = EMC.CollocationExaCore(_core_nodes(mesh, spec.Nc), K; backend = backend, kwargs...)
        return _build(core, tables, spec, modelsys, theta0, mesh)
    end
end

# Builds the PEtabExaModel
function _build(c, tables::PEtabTables, spec::PEtabSpec, modelsys::PEtabModelSys,
                theta0::Vector{Float64}, mesh::PEtabMesh)
    # Nominal solves: dense state guesses at every collocation point
    z_init, zss_init = _solve_conditions(spec, modelsys, theta0, mesh, c.weights.taus)

    # Create decision variables {p,z,y,sigma,cv,zss}
    c = _create_variables(c, spec, mesh, theta0, z_init, zss_init)

    # Create collocation constraints
    c = _create_collocation(c, spec, mesh)

    # Create cross-interval continuity, initial condition, auxiliary variable {cv,} constraints
    c = _create_continuity(c, spec, mesh)

    # Create objective function (Gaussian negative log-likelihood) and auxiliary variable {y,sigma} constraints
    z0arr = reshape(_var_starts(c, c.z), spec.Nz, spec.Nc, mesh.N, c.K + 1)
    c, y0, sigma0 = _create_objective(c, spec, tables, _collocation_ctx(c, spec, mesh, z0arr))

    # Every estimated parameter must be read somewhere
    _assert_no_flat_p(spec)

    # Carry the PEtab labelling onto the model
    c = _attach_petab_meta(c, spec, tables, copy(mesh.meas_iidx))

    # Create ExaModel
    model = ExaModels.ExaModel(c)

    # Provide good initial guess for objective function auxiliary variables {y,sigma}
    ExaModels.set_start!(model, c.y, y0)
    ExaModels.set_start!(model, c.sigma, sigma0)

    return model
end

# Builds the PEtabExaModel for steady-state models (no collocation mesh)
function _build_ss(c::ExaCore, tables::PEtabTables, spec::PEtabSpec,
                   modelsys::PEtabModelSys, theta0::Vector{Float64})
    # Steady-state guesses and control values at t = 0
    zss0 = _steady_states(spec, modelsys, theta0)
    u_vals_ss = _ss_u_vals(spec)

    # Create decision variables {p,y,sigma,cv,zss}
    c = _create_variables_ss(c, spec, theta0, zss0)

    # Create steady-state ODE RHS constraints, f(zss...) = 0
    c = _create_constraints_ss(c, spec, theta0, u_vals_ss)

    # Create objective function (Gaussian negative log-likelihood) and auxiliary variable {y,sigma} constraints
    # (Evaluated at zss)
    c, y0, sigma0 = _create_objective(c, spec, tables, _ss_ctx(c, spec, zss0))

    # Every estimated parameter must be read somewhere
    _assert_no_flat_p(spec)

    # Carry the PEtab labelling onto the model
    c = _attach_petab_meta(c, spec, tables, Int[])

    # Create ExaModel
    model = ExaModels.ExaModel(c)

    # Provide good initial guess for objective function auxiliary variables {y,sigma}
    ExaModels.set_start!(model, c.y, y0)
    ExaModels.set_start!(model, c.sigma, sigma0)

    return model
end
