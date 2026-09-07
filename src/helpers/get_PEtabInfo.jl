# gets PEtabInfo by simulating the ODE system at the nominal guess theta0
# 1. obtain good initial guesses for the discretized state variables
# 2. determine mesh node placements
# 3. create PEtabInfo
function _get_PEtabInfo(filename)
    # Parse petab .yaml file
    petab = _parse_yaml(filename)

    # Determine model size
    model_size = _get_model_size(petab)

    # Simulate at nominal theta0
    sols, sols_ss = _initial_solve(petab, model_size)

    # Determine mesh nodes and K
    mesh_size = _get_mesh_size(petab, sols, _get_Nz(petab.model))
    nodes, K = _determine_mesh(petab, sols, mesh_size)

    # Initial guesses {z0, zss0}
    theta0 = _get_theta0(petab)
    z0 = _get_z0(sols, nodes, K)
    cv0 = _get_cv0(petab)
    zss0 = _get_zss0(sols_ss)

    # PEtabInfo: PEtabTables, 
    return PEtabInfo(petab..., nodes, K, theta0, z0, cv0, zss0)
end

# Parse the PEtab problem files
function _parse_yaml(filename)
    # Unpack .yaml file
    files = _read_yaml(filename)
    parameters = _get_parameters(_read_tsv(files.parameters))
    observables = _get_observables(_read_tsv(files.observables))
    conditions_table = _read_tsv(files.conditions)
    measurements_table = _read_tsv(files.measurements)

    # Unpack column simulation and pre-eqbm condition ids, cidx = 1:Nc and ssidx = 1:Nss
    sim_column = measurements_table.simulationConditionId
    preeq_column = _get_column(measurements_table, :preequilibrationConditionId, "")
    sim_ids = unique(sim_column)
    preeq_ids = filter(!isempty, unique(preeq_column))
    preeq_idxs = [
        _get_preeq_idx(sim_id, sim_column, preeq_column, preeq_ids) 
        for sim_id in sim_ids
    ]

    # Parse steady-state only models as simulation condition == pre-eqbm condition
    if all(time -> isinf(parse(Float64, time)), measurements_table.time)
        isempty(preeq_ids) || 
            throw(ArgumentError("pre-equilibration with time = inf measurements is not supported"))
        preeq_ids = sim_ids
        preeq_idxs = collect(eachindex(sim_ids))
    end

    # Return parsed petab info
    model = _get_model(files.sbml)
    conditions = [_get_condition(conditions_table, id, parameters) for id in sim_ids]
    events = _get_events(files.sbml)
    return (
        model            = model,
        parameters       = parameters,
        conditions       = conditions,
        preeq_conditions = [_get_condition(conditions_table, id, parameters) for id in preeq_ids],
        preeq_idxs       = preeq_idxs,
        observables      = observables,
        events           = events,
        event_times      = _get_event_times(events, conditions, parameters, model.parametermap),
        measurements     = _get_measurements(measurements_table, sim_ids, observables, parameters),
    )
end

# Return PEtabModel from SBMLImporter model
function _get_model(sbml_file)
    reactionsys, _ = SBMLImporter.load_SBML(sbml_file)
    sys = SBMLImporter.Catalyst.ode_model(reactionsys)
    speciemap, parametermap = SBMLImporter.get_u0_map(reactionsys), SBMLImporter.get_parameter_map(reactionsys)
    sys, speciemap, parametermap = _parse_dzdt0_as_parameters(sys, speciemap, parametermap)
    sys = MTK.mtkcompile(sys)
    model_SBML = SBMLImporter.parse_SBML(sbml_file, false; model_as_string = false, inline_assignment_rules = false)
    callbacks = SBMLImporter.create_callbacks(sys, model_SBML, model_SBML.name)
    return PEtabModel(sys, speciemap, parametermap, callbacks)
end

# Unknowns with dz/dt = 0 become parameters of the same name, their species-map entries parameter-map entries
function _parse_dzdt0_as_parameters(sys, speciemap, parametermap)
    # dz/dt = 0
    isdzdt0(equation) = MTK.isdifferential(equation.lhs) && isequal(Symbolics.value(equation.rhs), 0)
    parameter = Dict(
        _get_id(x) => MTK.toparam(Symbolics.variable(Symbolics.getname(x)))
        for x in [
            only(Symbolics.arguments(Symbolics.unwrap(equation.lhs))) 
            for equation in MTK.equations(sys) if isdzdt0(equation)
        ]
    )
    isconstant(x) = haskey(parameter, _get_id(x))
    replace(expr) = Symbolics.substitute(
        expr, 
        Dict(
            x => parameter[_get_id(x)] 
            for x in Symbolics.get_variables(expr) if isconstant(x)
        )
    )
    equations = [
        replace(equation) 
        for equation in MTK.equations(sys) if !isdzdt0(equation)
    ]
    unknowns = filter(!isconstant, MTK.unknowns(sys))
    parameters = [
        MTK.parameters(sys); 
        [parameter[_get_id(x)] for x in filter(isconstant, MTK.unknowns(sys))]
    ]
    sys = MTK.System(
        equations, 
        only(MTK.independent_variables(sys)), 
        unknowns, 
        parameters; 
        name = nameof(sys)
    )
    speciemap, parametermap = [x => replace(v) for (x, v) in speciemap], [x => replace(v) for (x, v) in parametermap]
    parametermap = [parametermap; [parameter[_get_id(x)] => v for (x, v) in speciemap if isconstant(x)]]
    speciemap = filter(pair -> !isconstant(pair.first), speciemap)
    return sys, speciemap, parametermap
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
            event.trigger,
        )
        for event in values(events)
    ]
end

# get event times for each event variable u per simulation condition cidx
# event_times[uidx,cidx]
_get_event_times(events, conditions, parameters, parametermap) = Float64[
    _get_event_time(event.event_formula, condition, parameters, parametermap)
    for event in events, condition in conditions
]

# Event time of a `t <op> expression` event formula, the expression evaluated in the condition
_get_event_time(event_formula, condition, parameters, parametermap) =
    _evaluate_event(_split_event_formula(event_formula)[2], event_formula, condition, parameters, parametermap)

# (op, expression) of a `t <op> expression` event formula, `-(t, c) <op> 0` and `expression <op> t` rewritten
function _split_event_formula(event_formula)
    parsed = Meta.parse(event_formula)
    op, sides = parsed.args[1], parsed.args[2:3]
    if sides[2] == 0 && sides[1] isa Expr && sides[1].args[1:2] == [:-, :t]
        sides = sides[1].args[2:3]
    end
    :t in sides || throw(ArgumentError("event '$event_formula': only time events are supported"))
    sides[1] == :t && return op, sides[2]
    mirrored = Dict(:≤ => :≥, :<= => :>=, :< => :>, :≥ => :≤, :>= => :<=, :> => :<, :(==) => :(==))
    return mirrored[op], sides[1]
end

# Evaluate an event expression: numbers, condition cells, fixed parameters, SBML constants
_evaluate_event(number::Number, event_formula, condition, parameters, parametermap) = Float64(number)

function _evaluate_event(id::Symbol, event_formula, condition, parameters, parametermap)
    i = findfirst(==(string(id)), condition.target_ids)
    if !isnothing(i)
        cell = condition.target_values[i]
        cell isa Int && throw(ArgumentError("event '$event_formula': '$id' is estimated in condition '$(condition.condition_id)'"))
        return cell
    end
    i = findfirst(parameter -> parameter.parameter_id == string(id), parameters)
    if !isnothing(i)
        parameters[i].estimate && throw(ArgumentError("event '$event_formula': '$id' is estimated"))
        return parameters[i].value
    end
    i = findfirst(pair -> string(pair.first) == string(id), parametermap)
    value = isnothing(i) ? nothing : Symbolics.value(parametermap[i].second)
    value isa Number || throw(ArgumentError("event '$event_formula': '$id' is not a fixed number"))
    return Float64(value)
end

function _evaluate_event(expression::Expr, event_formula, condition, parameters, parametermap)
    operations = Dict(:+ => +, :- => -, :* => *, :/ => /)
    haskey(operations, expression.args[1]) ||
        throw(ArgumentError("event '$event_formula': unsupported operation '$(expression.args[1])'"))
    return operations[expression.args[1]](
        (_evaluate_event(arg, event_formula, condition, parameters, parametermap) for arg in expression.args[2:end])...
    )
end

_get_id(symbol) = replace(string(symbol), "(t)" => "")

# Event targets
_get_u_ids(petab) = unique(id for event in petab.events for id in event.target_ids)

# Default value of a model species or parameter
function _get_default(model, id)
    for (symbol, value) in Iterators.flatten((model.speciemap, model.parametermap))
        _get_id(symbol) == id && return Float64(Symbolics.value(value))
    end
    throw(ArgumentError("'$id' is not in the model"))
end

# Value of event target id at time t, nothing when no event on id has fired
function _get_u_value(petab, id, times, t)
    value = nothing
    for uidx in sortperm(times)
        event = petab.events[uidx]
        j = findfirst(==(id), event.target_ids)
        isnothing(j) && continue
        times[uidx] <= t && (value = parse(Float64, event.target_values[j]))
    end
    return value
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

_is_steadystate(petab::Union{NamedTuple, PEtabInfo}) = all(measurement -> isinf(measurement.time), petab.measurements)

# theta0: nominal values of the estimated parameters on their scale
_get_theta0(petab) = [
    _logscale(parameter.value == 0 && parameter.scale != :lin ? parameter.lb : parameter.value, parameter.scale)
    for parameter in petab.parameters if parameter.estimate
]

# Number of ODE states
_get_Nz(model) = length(MTK.unknowns(model.sys))

# PEtab.jl's n_xdynamic_sys
function _get_Ntheta_per_cond(petab)
    sys_ids = string.(MTK.parameters(petab.model.sys))
    n_sys = count(parameter -> parameter.estimate && parameter.parameter_id in sys_ids, petab.parameters)
    n_condition = maximum(
        count(cell -> cell isa Int, condition.target_values) 
        for condition in [petab.conditions; petab.preeq_conditions]; 
        init = 0
    )
    return n_sys + n_condition
end

# PEtab.jl's model size by the state count and the estimated dynamic parameter count
function _get_model_size(petab)
    Nz, Ntheta_per_cond = _get_Nz(petab.model), _get_Ntheta_per_cond(petab)
    return Nz <= 15 && Ntheta_per_cond <= 20 ? :small  :
           Nz <= 50 && Ntheta_per_cond <= 70 ? :medium : :large
end

# Simulate every condition at theta0
_initial_solve(petab, model_size) = _get_sols(petab, _get_theta0(petab), _get_odesolver(model_size))

# Simulate every condition at theta, sols[cidx] and sols_ss[ssidx]
function _get_sols(petab, theta, solver)
    sols_ss = _get_sols_ss(petab, theta, solver)
    _is_steadystate(petab) && return nothing, sols_ss
    t_stops = _get_t_stops(petab)
    sols = [
        _solve(
            _get_odeproblem(petab, theta, cidx, sols_ss, t_stops), 
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
        t_events = [time for time in petab.event_times[:,cidx] if 0 < time < maximum(t_meas)]
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
        _solve_steadystate(
            petab,
            ODE.ODEProblem(
                petab.model.sys,
                _get_op(petab, theta0, condition, nothing),
                (0.0, Inf);
                build_initializeprob = false
            ),
            solver
        )
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
    value(cell) = cell isa Int ? _linscale(theta0[cell], scales[cell]) : cell
    i = 0
    for parameter in petab.parameters
        parameter.estimate && (i += 1)
        haskey(key, parameter.parameter_id) || continue
        op[key[parameter.parameter_id]] = parameter.estimate ? _linscale(theta0[i], parameter.scale) : parameter.value
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
function _determine_mesh(petab, sols, mesh_size)
    t_stops = _get_t_stops(petab)
    if mesh_size == :small
        every, K = 2, 4
    elseif mesh_size == :medium
        every, K = 4, 4
    elseif mesh_size == :large
        every, K = 8, 3
    elseif mesh_size == :massive
        every, K = 16, 3
    end
    nodes = [_get_nodes(sols[cidx].t, t_stops[cidx], every) for cidx in eachindex(sols)]
    N = maximum(length.(nodes)) - 1
    for cidx in eachindex(nodes)
        _split_widest!(nodes[cidx], N)
    end
    return nodes, K
end

_determine_mesh(petab, sols::Nothing, mesh_size) = Vector{Float64}[], 0

# Mesh size by the variable count of the finest mesh, a node at every integrator step with K = 4
function _get_mesh_size(petab, sols, Nz)
    t_stops = _get_t_stops(petab)
    N1 = maximum(length(_get_nodes(sols[cidx].t, t_stops[cidx], 1)) for cidx in eachindex(sols)) - 1
    nvar1 = Nz * length(petab.conditions) * N1 * 5
    return nvar1 < 5e5   ? :small  :
           nvar1 < 2.5e6 ? :medium :
           nvar1 < 1e7   ? :large  : :massive
end

_get_mesh_size(petab, sols::Nothing, Nz) = :small

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

_get_zss0(sols_ss) = Vector{Float64}[sol.u[end] for sol in sols_ss]

# Condition targets that some condition sets to an unknown parameter
function _get_cv_ids(petab)
    for condition in petab.preeq_conditions, (id, cell) in zip(condition.target_ids, condition.target_values)
        cell isa Int && throw(ArgumentError(
            "condition '$(condition.condition_id)' sets '$id' to an estimated parameter in pre-equilibration, which is not supported"
        ))
    end
    cv_ids = String[]
    for condition in petab.conditions, (id, cell) in zip(condition.target_ids, condition.target_values)
        cell isa Int && !(id in cv_ids) && push!(cv_ids, id)
    end
    return cv_ids
end

# cv0[cvidx,cidx]: value of condition target cv_ids[cvidx] in condition cidx at theta0
_get_cv0(petab) = _get_cv(petab, _get_theta0(petab))

# cv[cvidx,cidx]: value of condition target cv_ids[cvidx] in condition cidx at theta
function _get_cv(petab, theta)
    cv_ids = _get_cv_ids(petab)
    scales = [parameter.scale for parameter in petab.parameters if parameter.estimate]
    value(cell) = cell isa Int ? _linscale(theta[cell], scales[cell]) : cell
    cv = Matrix{Float64}(undef, length(cv_ids), length(petab.conditions))
    for (cidx, condition) in enumerate(petab.conditions), (cvidx, id) in enumerate(cv_ids)
        i = findfirst(==(id), condition.target_ids)
        isnothing(i) && throw(ArgumentError("condition '$(condition.condition_id)' leaves '$id' unset"))
        cv[cvidx,cidx] = value(condition.target_values[i])
    end
    return cv
end
