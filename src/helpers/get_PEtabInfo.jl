# gets PEtabInfo by simulating the ODE system at the nominal guess theta0
# 1. obtain good initial guesses for the discretized state variables
# 2. determine mesh node placements
# 3. create PEtabInfo
function _get_PEtabInfo(filename)
    # Parse petab .yaml file
    petab = _parse_yaml(filename)

    # Determine model size
    Nz = _get_Nz(petab.model)
    Ntheta_per_cond = _get_Ntheta_per_cond(petab)
    model_size = :large
    if Nz <= 15 && Ntheta_per_cond <= 20
        model_size = :small
    elseif Nz <= 50 && Ntheta_per_cond <= 70
        model_size = :medium
    end

    # Simulate at nominal theta0
    sols, sols_ss = _initial_solve(petab, model_size)

    # Determine mesh nodes and K
    nodes, K = _determine_mesh(petab, sols, model_size)

    # Initial guesses {z0, zss0}
    z0 = _get_z0(sols, nodes, K)
    zss0 = _get_zss0(sols_ss)

    # PEtabInfo: PEtabTables, 
    return PEtabInfo(petab..., nodes, K, _get_theta0(petab), z0, zss0)
end

# Parse the PEtab problem files
function _parse_yaml(filename)
    files = _read_yaml(filename)
    parameters = _get_parameters(_read_tsv(files.parameters))
    observables = _get_observables(_read_tsv(files.observables))
    conditions_table = _read_tsv(files.conditions)
    measurements_table = _read_tsv(files.measurements)

    # Condition axes: simulation conditions cidx = 1:Nc, pre-equilibration conditions ssidx = 1:Nss
    sim_column = measurements_table.simulationConditionId
    preeq_column = _get_column(measurements_table, :preequilibrationConditionId, "")
    sim_ids = unique(sim_column)
    preeq_ids = filter(!isempty, unique(preeq_column))
    preeq_idxs = [
        _get_preeq_idx(sim_id, sim_column, preeq_column, preeq_ids) 
        for sim_id in sim_ids
    ]

    # Steady-state models: every simulation condition is its own pre-equilibration condition
    if all(time -> isinf(parse(Float64, time)), measurements_table.time)
        isempty(preeq_ids) || 
            throw(ArgumentError("pre-equilibration with time = inf measurements is not supported"))
        preeq_ids = sim_ids
        preeq_idxs = collect(eachindex(sim_ids))
    end

    return (
        model            = _get_model(files.sbml),
        parameters       = parameters,
        conditions       = [_get_condition(conditions_table, id, parameters) for id in sim_ids],
        preeq_conditions = [_get_condition(conditions_table, id, parameters) for id in preeq_ids],
        preeq_idxs       = preeq_idxs,
        observables      = observables,
        events           = _get_events(files.sbml),
        measurements     = _get_measurements(measurements_table, sim_ids, observables, parameters),
    )
end

# Return PEtabModel from SBMLImporter model
function _get_model(sbml_file)
    reactionsys, callbacks = SBMLImporter.load_SBML(sbml_file)
    sys = MTK.mtkcompile(SBMLImporter.Catalyst.ode_model(reactionsys))
    return PEtabModel(
        sys,
        SBMLImporter.get_u0_map(reactionsys),
        SBMLImporter.get_parameter_map(reactionsys),
        callbacks
    )
end

# Return PEtabParameter from SBMLImporter parameters table
function _get_parameters(table)
    prior_types = _get_column(table, :objectivePriorType, "")
    prior_parameters = _get_column(table, :objectivePriorParameters, "")
    return [
        PEtabParameter(
            table.parameterId[i],
            table.estimate[i] == "1",
            parse(Float64, table.nominalValue[i]),
            _float(table.lowerBound[i]),
            _float(table.upperBound[i]),
            Symbol(table.parameterScale[i]),
            isempty(prior_types[i]) ? :none : Symbol(prior_types[i]),
            Float64[parse(Float64, part) for part in split(prior_parameters[i], ';'; keepempty = false)],
        )
        for i in eachindex(table.parameterId)
    ]
end

# Return PEtabObservable from SBMLImporter observables table
function _get_observables(table)
    transforms = _get_column(table, :observableTransformation, "lin")
    distributions = _get_column(table, :noiseDistribution, "normal")
    all(distribution -> distribution in ("", "normal"), distributions) ||
        throw(ArgumentError("only the normal noiseDistribution is supported"))
    return [
        PEtabObservable(
            table.observableId[i],
            table.observableFormula[i],
            table.noiseFormula[i],
            isempty(transforms[i]) ? :lin : Symbol(transforms[i]),
        )
        for i in eachindex(table.observableId)
    ]
end

# Return PEtabCondition from SBMLImporter conditions table
function _get_condition(table, condition_id, parameters)
    i = _get_index(condition_id, table.conditionId)
    target_ids = String[]
    target_values = Union{Float64, Int}[]
    for key in keys(table)
        key in (:conditionId, :conditionName) && continue
        isempty(table[key][i]) && continue
        push!(target_ids, String(key))
        push!(target_values, _resolve_cell(table[key][i], parameters))
    end
    return PEtabCondition(
        condition_id,
        target_ids,
        target_values
    )
end

# Return ssidx of the pre-eqbm condition
function _get_preeq_idx(sim_id, sim_column, preeq_column, preeq_ids)
    ids = unique(preeq_column[sim_column .== sim_id])
    length(ids) == 1 ||
        throw(ArgumentError("simulation condition '$sim_id' has several pre-equilibration conditions"))
    return isempty(ids[1]) ? 0 : _get_index(ids[1], preeq_ids)
end

# SBML events
function _get_events(sbml_file)
    events = SBMLImporter.parse_SBML(
        sbml_file, 
        false; 
        model_as_string = false, 
        inline_assignment_rules = false
    ).events
    return [
        PEtabEvent(
            [strip(split(formula, '='; limit = 2)[1]) for formula in event.formulas],
            [strip(split(formula, '='; limit = 2)[2]) for formula in event.formulas],
            _get_event_time(event.trigger, name),
        )
        for (name, event) in events
    ]
end

# Event time of a `t <op> number` trigger
function _get_event_time(trigger, name)
    sides = strip.(split(trigger, r"≥|≤|>=|<=|==|>|<"))
    time = length(sides) == 2 && "t" in sides ? tryparse(Float64, sides[1] == "t" ? sides[2] : sides[1]) : nothing
    isnothing(time) && throw(ArgumentError("event '$name': only fixed-time events are supported, got '$trigger'"))
    return time
end

# Measurements-table rows
function _get_measurements(table, sim_ids, observables, parameters)
    observable_ids = [observable.observable_id for observable in observables]
    observable_parameters = _get_column(table, :observableParameters, "")
    noise_parameters = _get_column(table, :noiseParameters, "")
    return [
        PEtabMeasurement(
            parse(Float64, table.time[m]),
            parse(Float64, table.measurement[m]),
            _resolve_cells(observable_parameters[m], parameters),
            _resolve_cells(noise_parameters[m], parameters),
            _get_index(table.simulationConditionId[m], sim_ids),
            _get_index(table.observableId[m], observable_ids),
        )
        for m in eachindex(table.time)
    ]
end

_is_steadystate(petab::NamedTuple) = all(measurement -> isinf(measurement.time), petab.measurements)

# theta0: nominal values of the estimated parameters on their scale
_get_theta0(petab) = [
    parameter.scale == :log10 ? log10(parameter.value) : parameter.scale == :log ? log(parameter.value) : parameter.value
    for parameter in petab.parameters if parameter.estimate
]

# Number of ODE states
_get_Nz(model) = length(MTK.unknowns(model.sys))

# PEtab.jl's n_xdynamic_sys
function _get_Ntheta_per_cond(petab)
    sys_ids = string.(MTK.parameters(petab.model.sys))
    n_sys = count(parameter -> parameter.estimate && parameter.parameter_id in sys_ids, petab.parameters)
    n_condition = maximum(count(cell -> cell isa Int, condition.target_values) for condition in [petab.conditions; petab.preeq_conditions]; init = 0)
    return n_sys + n_condition
end

# Simulate every condition at theta0
function _initial_solve(petab, model_size)
    solver = _get_odesolver(model_size)
    theta0 = _get_theta0(petab)
    sols_ss = _get_sols_ss(petab, theta0, solver)
    _is_steadystate(petab) && return nothing, sols_ss
    t_stops = _get_t_stops(petab)
    sols = [
        _solve(
            _get_odeproblem(petab, theta0, cidx, sols_ss, t_stops), 
            solver; 
            callback = petab.model.callbacks, 
            tstops = t_stops[cidx]
        )
        for cidx in eachindex(t_stops)
    ]
    return sols, sols_ss
end

# PEtab.jl's default ODE solver by model size
_get_odesolver(model_size) =
    model_size == :small  ? ODE.Rodas5P() :
    model_size == :medium ? ODE.FBDF()    : # NOTE: PEtab.jl uses QNDF
                            ODE.FBDF()      # NOTE: PEtab.jl uses KenCarp4

# PEtab.jl's default solve settings
function _solve(prob, solver; kwargs...)
    sol = ODE.solve(prob, solver; abstol = 1e-8, reltol = 1e-8, maxiters = 10^4, kwargs...)
    ODE.SciMLBase.successful_retcode(sol) || error("ODE solve failed with retcode $(sol.retcode)")
    return sol
end

# t_stops[cidx]: t = 0, measurement times, and event times of simulation condition cidx
function _get_t_stops(petab)
    t_stops = Vector{Vector{Float64}}(undef, length(petab.conditions))
    for cidx in eachindex(petab.conditions)
        t_meas = [measurement.time for measurement in petab.measurements if measurement.cidx == cidx]
        t_events = [event.event_time for event in petab.events if 0 < event.event_time < maximum(t_meas)]
        t_stops[cidx] = sort!(unique!([0.0; t_meas; t_events]))
    end
    t_pad = minimum(t_stops[cidx][end] for cidx in eachindex(t_stops) if t_stops[cidx][end] > 0)
    for cidx in eachindex(t_stops)
        t_stops[cidx][end] == 0 && push!(t_stops[cidx], t_pad)
    end
    return t_stops
end

# ODEProblem of simulation condition cidx at theta0
function _get_odeproblem(petab, theta0, cidx, sols_ss, t_stops)
    ssidx = petab.preeq_idxs[cidx]
    zss = ssidx == 0 ? nothing : sols_ss[ssidx].u[end]
    op = _get_op(petab, theta0, petab.conditions[cidx], zss)
    return ODE.ODEProblem(petab.model.sys, op, (0.0, t_stops[cidx][end]); build_initializeprob = false)
end

# Pre-equilibration steady-state simulations
function _get_sols_ss(petab, theta0, solver)
    return [
        _solve_steadystate(petab, ODE.ODEProblem(petab.model.sys, _get_op(petab, theta0, condition, nothing), (0.0, Inf); build_initializeprob = false), solver)
        for condition in petab.preeq_conditions
    ]
end

# PEtab.jl's steady state by simulation
function _solve_steadystate(petab, prob, solver; abstol = 1e-6, reltol = 1e-6)
    at_steadystate(u, t, integrator) = t >= 0.1 &&
        sqrt(sum(abs2, ODE.SciMLBase.get_du(integrator) ./ (reltol .* u .+ abstol)) / length(u)) < 1
    terminate = ODE.DiscreteCallback(at_steadystate, ODE.terminate!; save_positions = (false, true))
    return _solve(prob, solver; callback = ODE.CallbackSet(petab.model.callbacks, terminate))
end

# MTK operating point of one condition at theta0
function _get_op(petab, theta0, condition, zss)
    (; sys, speciemap, parametermap) = petab.model
    op = merge!(Dict{Any, Any}(speciemap), Dict{Any, Any}(parametermap))
    key = Dict(replace(string(symbol), "(t)" => "") => symbol for symbol in keys(op))
    scales = [parameter.scale for parameter in petab.parameters if parameter.estimate]
    value(cell) = cell isa Int ? _unscale(theta0[cell], scales[cell]) : cell
    i = 0
    for parameter in petab.parameters
        parameter.estimate && (i += 1)
        haskey(key, parameter.parameter_id) || continue
        op[key[parameter.parameter_id]] = parameter.estimate ? _unscale(theta0[i], parameter.scale) : parameter.value
    end
    if !isnothing(zss)
        for (v, state) in enumerate(MTK.unknowns(sys))
            op[state] = zss[v]
        end
    end
    for (id, cell) in zip(condition.target_ids, condition.target_values)
        isnan(cell) || (op[key[id]] = value(cell))
    end
    return op
end

# Mesh nodes per condition and K
function _determine_mesh(petab, sols, model_size)
    t_stops = _get_t_stops(petab)
    if model_size == :small
        every, K = 1, 4
    elseif model_size == :medium
        every, K = 2, 4
    elseif model_size == :large
        every, K = 4, 3
    end
    nodes = [_get_nodes(sols[cidx].t, t_stops[cidx], every) for cidx in eachindex(sols)]
    N = maximum(length.(nodes)) - 1
    for cidx in eachindex(nodes)
        _split_widest!(nodes[cidx], N)
    end
    return nodes, K
end

_determine_mesh(petab, sols::Nothing, model_size) = Vector{Float64}[], 0

# Fixed nodes t_stops plus one in every `every` integrator steps between them
function _get_nodes(t, t_stops, every; tol = 1e-8)
    nodes = Float64[t_stops[1]]
    for j in 1:(length(t_stops) - 1)
        between = filter(step -> t_stops[j] + tol < step < t_stops[j+1] - tol, t)
        for step in between[every:every:end]
            step - nodes[end] > tol && push!(nodes, step)
        end
        push!(nodes, t_stops[j+1])
    end
    return nodes
end

# Insert the midpoint of the widest interval until nodes has N intervals
function _split_widest!(nodes, N)
    while length(nodes) - 1 < N
        i = argmax(diff(nodes))
        insert!(nodes, i + 1, (nodes[i] + nodes[i+1]) / 2)
    end
    return nodes
end

# z0[v,cidx,i,k]: state v of condition cidx at node i and its collocation points
function _get_z0(sols, nodes, K)
    taus = EMC._get_taus(EMC.GaussRadau(), K)
    Nz, Nc, N = length(sols[1].u[1]), length(sols), length(nodes[1]) - 1
    z0 = Array{Float64, 4}(undef, Nz, Nc, N, K + 1)
    for cidx in 1:Nc, i in 1:N
        h = nodes[cidx][i+1] - nodes[cidx][i]
        z0[:,cidx,i,1] = sols[cidx](nodes[cidx][i])
        for k in 1:K
            z0[:,cidx,i,k+1] = sols[cidx](nodes[cidx][i] + h * taus[k])
        end
    end
    return z0
end

_get_z0(sols::Nothing, nodes, K) = zeros(0, 0, 0, 0)

# zss0: steady state of every pre-equilibration simulation
_get_zss0(sols_ss) = Vector{Float64}[sol.u[end] for sol in sols_ss]
