# Creates the collocation + continuity / initial conition / cv auxiliary variable constraints
function _create_constraints(
        core::EMC.CollocationExaCore,
        PEinfo::PEtabInfo
    )
    core = _create_collocation_constraints(core, PEinfo)

    core = _create_ic_constraints(core, PEinfo)

    core = _create_cv_constraints(core, PEinfo)

    if _has_zss(PEinfo)
        core = _create_steadystate_constraints(core, PEinfo)
    end

    return core
end

# Create collocation and continuity constraints
function _create_collocation_constraints(
        core::EMC.CollocationExaCore,
        PEinfo::PEtabInfo
    )
    # Unpack variables and model functions
    z, theta = core.z, core.theta
    cvof = _get_cvof(core, PEinfo)
    f = _get_f(PEinfo)

    # Create colloation constraint iterator
    # TODO callbacks -> creating u(t) for the iterator core.mesh.t[i,1] in u(t)
    # TODO
    # TODO (REVIEW) u[g,cidx,i] is constant per interval since the event times are nodes, so it
    # is read off nodes[cidx][i] in _get_u and carried in the row as an NTuple with data[d,cidx].
    data, u = _get_data(PEinfo, PEinfo.conditions), _get_u(PEinfo)
    Nd, Nu = size(data, 1), size(u, 1)
    itr = [
        (m, ntuple(d -> data[d,m], Nd), ntuple(g -> u[g,m,i], Nu), i, k)
        for m in 1:_get_Nc(PEinfo), i in 1:core.N, k in 1:core.K
    ]

    # Create collocation constraints
    # TODO f[v](theta,z[:,m,i,k],cv,u,t) ??? where u is any fixed time event variable with a known profile
    # TODO (REVIEW) one call per state since every f[v] is its own expression, the arguments are
    # (theta, z, cv, data, u, t) with data and u read off the row.
    for v in 1:_get_Nz(PEinfo)
        EMC.@add_con_collocation(core, z[v,m],
            f[v](theta[:], z[:,m,i,k], cvof(m), dm, um, t)
            for (m, dm, um, i, k) in itr
        )
    end

    # Create continuity constraints
    EMC.@add_con_continuity(core, z)

    return core
end

# Create initial condition constraints
function _create_ic_constraints(
        core::EMC.CollocationExaCore,
        PEinfo::PEtabInfo
    )
    # z[:,:,1,0] = fixed value, theta, or fz0(...)
    z, theta = core.z, core.theta
    cvof = _get_cvof(core, PEinfo)
    fz0 = _get_fz0(PEinfo)
    data = _get_data(PEinfo, PEinfo.conditions)
    Nz, Nc, Nd = _get_Nz(PEinfo), _get_Nc(PEinfo), size(data, 1)

    # States carried over from the pre-equilibration steady state
    state_ids = _get_state_ids(PEinfo.model)
    carried(v, cidx) = PEinfo.preeq_idxs[cidx] != 0 && !(state_ids[v] in PEinfo.conditions[cidx].target_ids)
    itr_zss = [(v, cidx, PEinfo.preeq_idxs[cidx]) for v in 1:Nz, cidx in 1:Nc if carried(v, cidx)]
    if !isempty(itr_zss)
        zss = core.zss
        ExaModels.@add_con(core,
            z[v,cidx,1,0] - zss[v,ssidx]
            for (v, cidx, ssidx) in itr_zss
        )
    end

    # Remaining states from the species map or the condition target
    for v in 1:Nz
        itr = [(cidx, ntuple(d -> data[d,cidx], Nd)) for cidx in 1:Nc if !carried(v, cidx)]
        isempty(itr) && continue
        ExaModels.@add_con(core,
            z[v,cidx,1,0] - fz0[v](theta[:], cvof(cidx), dc)
            for (cidx, dc) in itr
        )
    end

    return core
end

# Create cv auxiliary variable constraints
function _create_cv_constraints(
        core::EMC.CollocationExaCore,
        PEinfo::PEtabInfo
    )
    # cv[:,:] = fixed value or theta
    _has_cv(PEinfo) || return core
    cv, theta = core.cv, core.theta
    cv_ids = _get_cv_ids(PEinfo)
    scales = [parameter.scale for parameter in PEinfo.parameters if parameter.estimate]
    cells = [
        (cvidx, cidx, condition.target_values[_get_index(id, condition.target_ids)])
        for (cvidx, id) in enumerate(cv_ids), (cidx, condition) in enumerate(PEinfo.conditions)
    ]

    # Numeric cells
    itr = [(cvidx, cidx, cell) for (cvidx, cidx, cell) in cells if cell isa Float64]
    if !isempty(itr)
        ExaModels.@add_con(core,
            cv[cvidx,cidx] - value
            for (cvidx, cidx, value) in itr
        )
    end

    # theta cells, one call per scale so no kernel branches on it
    for scale in (:log10, :log, :lin)
        itr = [(cvidx, cidx, cell) for (cvidx, cidx, cell) in cells if cell isa Int && scales[cell] == scale]
        isempty(itr) && continue
        ExaModels.@add_con(core,
            cv[cvidx,cidx] - _linscale(theta[j], scale)
            for (cvidx, cidx, j) in itr
        )
    end

    return core
end

# cv[:,cidx] as a tuple, or () without cv
_get_cvof(core, PEinfo) = _has_cv(PEinfo) ? cidx -> core.cv[:,cidx] : cidx -> ()

# Symbolic arguments (theta, z, cv, data, u, t) of the model functions
function _get_arguments(PEinfo)
    Ntheta, Nz, Ncv = _get_Ntheta(PEinfo), _get_Nz(PEinfo), _get_Ncv(PEinfo)
    Nd, Nu = length(_get_data_ids(PEinfo)), length(_get_u_ids(PEinfo))
    Symbolics.@variables theta[1:Ntheta] z[1:Nz] cv[1:Ncv] data[1:Nd] u[1:Nu]
    t = only(MTK.independent_variables(PEinfo.model.sys))
    return (; theta, z, cv, data, u, t)
end

# Model symbol => its expression in the arguments: states, assignment rules, initial
# assignments, parameters-table values, cv rows, condition data, event targets
function _get_substitutions(PEinfo, arguments)
    (; sys, speciemap, parametermap) = PEinfo.model
    (; theta, z, cv, data, u) = arguments
    key = Dict(_get_id(symbol) => symbol for symbol in MTK.parameters(sys))
    rules = Dict{Any, Any}()
    for (v, state) in enumerate(MTK.unknowns(sys))
        rules[state] = z[v]
    end
    for equation in MTK.observed(sys)
        rules[equation.lhs] = equation.rhs
    end
    for (symbol, value) in parametermap
        rules[symbol] = value
    end
    j = 0
    for parameter in PEinfo.parameters
        parameter.estimate && (j += 1)
        haskey(key, parameter.parameter_id) || continue
        rules[key[parameter.parameter_id]] = parameter.estimate ? _linscale(theta[j], parameter.scale) : parameter.value
    end
    for (cvidx, id) in enumerate(_get_cv_ids(PEinfo))
        haskey(key, id) && (rules[key[id]] = cv[cvidx])
    end
    for (d, id) in enumerate(_get_data_ids(PEinfo))
        haskey(key, id) && (rules[key[id]] = data[d])
    end
    for (g, id) in enumerate(_get_u_ids(PEinfo))
        rules[key[id]] = u[g]
    end
    return rules
end

# Substitute until nothing changes, so nested assignment rules and initial assignments resolve
function _substitute(expr, rules)
    for _ in 1:100
        next = Symbolics.substitute(expr, rules)
        isequal(next, expr) && return next
        expr = next
    end
    throw(ArgumentError("model expression does not resolve: $expr"))
end

# f[v](theta, z, cv, data, u, t): right-hand side of state v
function _get_f(PEinfo)
    arguments = _get_arguments(PEinfo)
    rules = _get_substitutions(PEinfo, arguments)
    return [
        Symbolics.build_function(_substitute(equation.rhs, rules), arguments...; expression = Val{false})
        for equation in MTK.equations(PEinfo.model.sys)
    ]
end

# fz0[v](theta, cv, data): initial value of state v, a condition target on the state first
function _get_fz0(PEinfo)
    arguments = _get_arguments(PEinfo)
    rules = _get_substitutions(PEinfo, arguments)
    cv_ids, data_ids = _get_cv_ids(PEinfo), _get_data_ids(PEinfo)
    z0 = Dict(_get_id(state) => value for (state, value) in PEinfo.model.speciemap)
    fz0 = []
    for id in _get_state_ids(PEinfo.model)
        expr = id in cv_ids   ? arguments.cv[_get_index(id, cv_ids)]     :
               id in data_ids ? arguments.data[_get_index(id, data_ids)] : z0[id]
        push!(fz0, Symbolics.build_function(
            _substitute(expr, rules), arguments.theta, arguments.cv, arguments.data; expression = Val{false}
        ))
    end
    return fz0
end

# Model symbol name without (t)
_get_id(symbol) = replace(string(symbol), "(t)" => "")

_get_state_ids(model) = [_get_id(state) for state in MTK.unknowns(model.sys)]

# Numeric value of a species map or parameter map entry
function _get_default(model, id)
    for (symbol, value) in Iterators.flatten((model.speciemap, model.parametermap))
        _get_id(symbol) == id || continue
        value isa Number && return Float64(value)
        throw(ArgumentError("'$id' has no numeric default, got $value"))
    end
    throw(ArgumentError("'$id' is not in the model"))
end

# Condition targets that are numbers in every condition
function _get_data_ids(PEinfo)
    cv_ids = _get_cv_ids(PEinfo)
    data_ids = String[]
    for condition in [PEinfo.conditions; PEinfo.preeq_conditions], id in condition.target_ids
        id in cv_ids || id in data_ids || push!(data_ids, id)
    end
    return data_ids
end

# data[d,cidx]: value of condition target data_ids[d] in condition cidx
function _get_data(PEinfo, conditions)
    data_ids = _get_data_ids(PEinfo)
    data = Matrix{Float64}(undef, length(data_ids), length(conditions))
    for (cidx, condition) in enumerate(conditions), (d, id) in enumerate(data_ids)
        i = findfirst(==(id), condition.target_ids)
        isnothing(i) && throw(ArgumentError("condition '$(condition.condition_id)' leaves '$id' unset"))
        data[d,cidx] = condition.target_values[i]
    end
    return data
end

# Event targets, model parameters only
function _get_u_ids(PEinfo)
    parameter_ids = _get_id.(MTK.parameters(PEinfo.model.sys))
    u_ids = String[]
    for event in PEinfo.events, id in event.target_ids
        id in parameter_ids || throw(ArgumentError("event target '$id' is not a model parameter, which is not supported"))
        id in u_ids || push!(u_ids, id)
    end
    return u_ids
end

# u[g,cidx,i]: event target u_ids[g] on interval i of condition cidx, constant since event times are nodes
function _get_u(PEinfo)
    u_ids = _get_u_ids(PEinfo)
    Nc, N = _get_Nc(PEinfo), length(PEinfo.nodes[1]) - 1
    u = Array{Float64, 3}(undef, length(u_ids), Nc, N)
    for cidx in 1:Nc, i in 1:N, (g, id) in enumerate(u_ids)
        u[g,cidx,i] = _get_u_value(PEinfo, id, cidx, PEinfo.nodes[cidx][i])
    end
    return u
end

# Value of event target id at time t in condition cidx: a `t ≥ c` event holds from c on, a `t ≤ c` event until c
function _get_u_value(PEinfo, id, cidx, t)
    value = _get_default(PEinfo.model, id)
    for uidx in sortperm(PEinfo.event_times[:,cidx])
        event = PEinfo.events[uidx]
        j = findfirst(==(id), event.target_ids)
        isnothing(j) && continue
        time = PEinfo.event_times[uidx,cidx]
        holds = _holds_before(event.event_formula) ? t < time : t >= time
        holds && (value = parse(Float64, event.target_values[j]))
    end
    return value
end
