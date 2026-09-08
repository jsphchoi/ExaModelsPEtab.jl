# Creates the collocation + continuity / initial conition / cv auxiliary variable constraints
function _create_constraints(
        core::EMC.CollocationExaCore,
        PEinfo::PEtabInfo
    )
    core = _create_collocation_constraints(core, PEinfo)

    core = _create_ic_constraints(core, PEinfo)

    core = _create_cv_constraints(core, PEinfo)

    if _has_zss(PEinfo)
        core = _create_zss_constraints(core, PEinfo)
    end

    return core
end

# Create collocation and continuity constraints
function _create_collocation_constraints(
        core::EMC.CollocationExaCore,
        PEinfo::PEtabInfo
    )
    # Unpack variables and model functions
    z, theta, cv = core.z, core.theta, core.cv
    Nz, Nc, N, K = _get_Nz(PEinfo), _get_Nc(PEinfo), core.N, core.K

    # One kernel per term form of the right-hand sides when that is fewer than one per state
    forms, arguments = _analyze_rhs(core, PEinfo)
    if length(forms) + 1 < Nz
        # Create base rows sum_j A[j,k] z[v,m,i,j] = 0, the terms of f are added below
        rows = [(v, m, i, k) for v in 1:Nz, m in 1:Nc, i in 1:N, k in 1:K]
        EMC.@add_con_collocation(core, coll, z[v,m], 0 for (v, m, i, k) in rows)
        pos = LinearIndices(rows)

        # Add -h f to the rows of every state containing the term form, one call per form
        h, t = core.mesh.h, core.mesh.t
        for (expr, terms) in forms
            f = Symbolics.build_function(
                expr, 
                arguments...; 
                expression = Val{false}, 
                nanmath = false
            )
            itr = [
                (pos[v,m,i,k], h[m,i], m, i, k, js, vs, cvidxs, data[m,i], t[m,i,k])
                for (v, js, vs, cvidxs, data) in terms, m in 1:Nc, i in 1:N, k in 1:K
            ]
            # Constraint augmentation append reaction terms 
            ExaModels.@add_con!(core, coll,
                row => -h_mi * f(theta, z, cv, m, i, k, js, vs, cvidxs, data_mi, t_mik)
                for (row, h_mi, m, i, k, js, vs, cvidxs, data_mi, t_mik) in itr
            )
        end
    else
        # Create colloation constraint iterator
        cvfixed = _get_cvfixed(PEinfo, PEinfo.conditions)
        u = _get_u(PEinfo)
        itr = [
            (m, Tuple(cvfixed[:,m]), Tuple(u[:,m,i]), i, k)
            for m in 1:_get_Nc(PEinfo), i in 1:core.N, k in 1:core.K
        ]

        # Create collocation constraints
        f = _get_f(PEinfo) # get RHS functions f[v](theta,z,cv,cvfixed,u,t)
        for v in 1:_get_Nz(PEinfo)
            EMC.@add_con_collocation(core, z[v,m],
                f[v](theta[:], z[:,m,i,k], cv[:,m], cvfixed_m, u_mi, t)
                for (m, cvfixed_m, u_mi, i, k) in itr
            )
        end
    end

    # Create continuity constraints
    EMC.@add_con_continuity(core, z)

    return core
end

# Create initial condition constraints, z(t=0) = {fixed value, cv, theta, zss, fzic(...)}
function _create_ic_constraints(
        core::EMC.CollocationExaCore,
        PEinfo::PEtabInfo
    )
    # Unpack variables
    z, theta, cv = core.z, core.theta, core.cv

    # Parse z(t=0) = ???
    arguments = _get_arguments(PEinfo)
    zic_sym = _get_zic_sym(PEinfo, arguments)
    cvfixed = _get_cvfixed(PEinfo, PEinfo.conditions)
    scales = [parameter.scale for parameter in PEinfo.parameters if parameter.estimate]
    thetas = [_linscale(arguments.theta[j], scale) for (j, scale) in enumerate(scales)]
    state_ids = _get_state_ids(PEinfo.model)
    carried(v, cidx) = PEinfo.preeq_idxs[cidx] != 0 && !(state_ids[v] in PEinfo.conditions[cidx].target_ids)
    itr_fixed, itr_cv, itr_theta = Tuple{Int, Int, Float64}[], Tuple{Int, Int, Int}[], Tuple{Int, Int, Int}[]
    itr_zss, itr_fzic = Tuple{Int, Int, Int}[], Tuple{Int, Int}[]
    for v in 1:_get_Nz(PEinfo)
        value = Symbolics.value(zic_sym[v])
        cvfixedidx = findfirst(idx -> isequal(zic_sym[v], arguments.cvfixed[idx]), 1:size(cvfixed, 1))
        cvidx = findfirst(idx -> isequal(zic_sym[v], arguments.cv[idx]), 1:_get_Ncv(PEinfo))
        j = findfirst(candidate -> isequal(zic_sym[v], candidate), thetas)
        for cidx in 1:_get_Nc(PEinfo)
            if carried(v, cidx)             push!(itr_zss, (v, cidx, PEinfo.preeq_idxs[cidx]))
            elseif value isa Number         push!(itr_fixed, (v, cidx, Float64(value)))
            elseif !isnothing(cvfixedidx)   push!(itr_fixed, (v, cidx, cvfixed[cvfixedidx,cidx]))
            elseif !isnothing(cvidx)        push!(itr_cv, (v, cidx, cvidx))
            elseif !isnothing(j)            push!(itr_theta, (v, cidx, j))
            else                            push!(itr_fzic, (v, cidx))
            end
        end
    end

    # Create constraints for z(t=0) = fixed value
    if !isempty(itr_fixed)
        ExaModels.@add_con(core,
            z[v,cidx,1,0] - value
            for (v, cidx, value) in itr_fixed
        )
    end

    # Create constraints for z(t=0) = cv
    if !isempty(itr_cv)
        ExaModels.@add_con(core,
            z[v,cidx,1,0] - cv[cvidx,cidx]
            for (v, cidx, cvidx) in itr_cv
        )
    end

    # Create constraints for z(t=0) = theta
    for scale in (:log10, :log, :lin)
        itr = [(v, cidx, j) for (v, cidx, j) in itr_theta if scales[j] == scale]
        isempty(itr) && continue
        ExaModels.@add_con(core,
            z[v,cidx,1,0] - _linscale(theta[j], scale)
            for (v, cidx, j) in itr
        )
    end

    # Create constraints for z(t=0) = zss
    if !isempty(itr_zss)
        zss = core.zss
        ExaModels.@add_con(core,
            z[v,cidx,1,0] - zss[v,ssidx]
            for (v, cidx, ssidx) in itr_zss
        )
    end

    # Create constraints for z(t=0) = fzic
    for v in unique(first.(itr_fzic))
        fzic = Symbolics.build_function(
            zic_sym[v], 
            arguments.theta, 
            arguments.z, 
            arguments.cv, 
            arguments.cvfixed;
            expression = Val{false}, 
            nanmath = false
        )
        itr = [(cidx, Tuple(cvfixed[:,cidx])) for (vv, cidx) in itr_fzic if vv == v]
        ExaModels.@add_con(core,
            z[v,cidx,1,0] - fzic(theta[:], z[:,cidx,1,0], cv[:,cidx], cvfixed_c)
            for (cidx, cvfixed_c) in itr
        )
    end

    return core
end

# Create cv auxiliary variable constraints, cv = {fixed value, theta}
function _create_cv_constraints(
        core::ExaCore,
        PEinfo::PEtabInfo
    )
    # Do nothing if there is no cv
    _has_cv(PEinfo) || return core

    # Unpack variables
    cv, theta = core.cv, core.theta

    # Parse cv table values
    cv_ids = _get_cv_ids(PEinfo)
    scales = [parameter.scale for parameter in PEinfo.parameters if parameter.estimate]
    cells = [
        (cvidx, cidx, condition.target_values[_get_index(id, condition.target_ids)])
        for (cvidx, id) in enumerate(cv_ids), (cidx, condition) in enumerate(PEinfo.conditions)
    ]

    # Create constraints for cv = fixed value
    itr = [(cvidx, cidx, cell) for (cvidx, cidx, cell) in cells if cell isa Float64]
    if !isempty(itr)
        ExaModels.@add_con(core,
            cv[cvidx,cidx] - value
            for (cvidx, cidx, value) in itr
        )
    end

    # Create constraints for cv = _linscale(theta)
    for scale in (:log10, :log, :lin)
        itr = [
            (cvidx, cidx, cell) 
            for (cvidx, cidx, cell) in cells if cell isa Int && scales[cell] == scale
        ]
        isempty(itr) && continue
        ExaModels.@add_con(core,
            cv[cvidx,cidx] - _linscale(theta[j], scale)
            for (cvidx, cidx, j) in itr
        )
    end

    return core
end

# ----- helper functinos -----

# cv[:,cidx] as a tuple, or () without cv
# Symbolic arguments (theta, z, cv, cvfixed, u, t) of the model functions
function _get_arguments(PEinfo)
    Ntheta, Nz, Ncv = _get_Ntheta(PEinfo), _get_Nz(PEinfo), _get_Ncv(PEinfo)
    Ncvfixed, Nu = length(_get_cvfixed_ids(PEinfo)) + length(_get_ifelses(PEinfo)), length(_get_u_ids(PEinfo))
    Symbolics.@variables theta[1:Ntheta] z[1:Nz] cv[1:Ncv] cvfixed[1:Ncvfixed] u[1:Nu]
    t = only(MTK.independent_variables(PEinfo.model.sys))
    return (; theta, z, cv, cvfixed, u, t)
end

# Model symbol => its expression in the arguments: states, assignment rules, initial
# assignments, parameters-table values, cv rows, cvfixed rows, event targets
function _get_substitutions(PEinfo, arguments)
    (; sys, speciemap, parametermap) = PEinfo.model
    (; theta, z, cv, cvfixed, u) = arguments
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
    for (cvfixedidx, id) in enumerate(_get_cvfixed_ids(PEinfo))
        haskey(key, id) && (rules[key[id]] = cvfixed[cvfixedidx])
    end
    for (q, term) in enumerate(_get_ifelses(PEinfo))
        c, a, b = Symbolics.arguments(term)
        flag = cvfixed[length(_get_cvfixed_ids(PEinfo)) + q]
        rules[term] = flag * a + (1 - flag) * b
    end
    for (uidx, id) in enumerate(_get_u_ids(PEinfo))
        haskey(key, id) || throw(ArgumentError("event target '$id' is not a model parameter, which is not supported"))
        rules[key[id]] = u[uidx]
    end
    return rules
end

# f[v](theta, z, cv, cvfixed, u, t): right-hand side of state v
function _get_f(PEinfo)
    arguments = _get_arguments(PEinfo)
    rules = _get_substitutions(PEinfo, arguments)
    return [
        Symbolics.build_function(
            Symbolics.fixpoint_sub(equation.rhs, rules; fold = Val(true)), 
            arguments...; 
            expression = Val{false}, 
            nanmath = false
        )
        for equation in MTK.equations(PEinfo.model.sys)
    ]
end

# Resolved initial value of every state in the arguments, a condition target on the state first
function _get_zic_sym(PEinfo, arguments)
    rules = _get_substitutions(PEinfo, arguments)
    cv_ids, cvfixed_ids = _get_cv_ids(PEinfo), _get_cvfixed_ids(PEinfo)
    z0 = Dict(_get_id(state) => value for (state, value) in PEinfo.model.speciemap)
    return [
        Symbolics.fixpoint_sub(
            id in cv_ids      ? arguments.cv[_get_index(id, cv_ids)] :
            id in cvfixed_ids ? arguments.cvfixed[_get_index(id, cvfixed_ids)] : z0[id],
            rules;
            fold = Val(true)
        )
        for id in _get_state_ids(PEinfo.model)
    ]
end

_get_state_ids(model) = [_get_id(state) for state in MTK.unknowns(model.sys)]

# Condition targets that are numbers in every condition
function _get_cvfixed_ids(PEinfo)
    cv_ids = _get_cv_ids(PEinfo)
    cvfixed_ids = String[]
    for condition in [PEinfo.conditions; PEinfo.preeq_conditions], id in condition.target_ids
        id in cv_ids || id in cvfixed_ids || push!(cvfixed_ids, id)
    end
    return cvfixed_ids
end

# cvfixed[cvfixedidx,cidx]: value of condition target cvfixed_ids[cvfixedidx] in condition cidx
function _get_cvfixed(PEinfo, conditions)
    cvfixed_ids = _get_cvfixed_ids(PEinfo)
    cvfixed = Matrix{Float64}(undef, length(cvfixed_ids), length(conditions))
    for (cidx, condition) in enumerate(conditions), (cvfixedidx, id) in enumerate(cvfixed_ids)
        i = findfirst(==(id), condition.target_ids)
        isnothing(i) && throw(ArgumentError("condition '$(condition.condition_id)' leaves '$id' unset"))
        cvfixed[cvfixedidx,cidx] = condition.target_values[i]
    end
    key = Dict(_get_id(symbol) => symbol for symbol in MTK.parameters(PEinfo.model.sys))
    flags = Matrix{Float64}(undef, length(_get_ifelses(PEinfo)), length(conditions))
    for (q, term) in enumerate(_get_ifelses(PEinfo))
        f = Symbolics.build_function(Symbolics.arguments(term)[1], (key[id] for id in cvfixed_ids)...; expression = Val{false})
        flags[q,:] = [f(cvfixed[:,cidx]...) for cidx in eachindex(conditions)]
    end
    return [cvfixed; flags]
end

# ifelse terms of the right-hand sides whose condition reads only cvfixed values, each a flag row of cvfixed
function _get_ifelses(PEinfo)
    cvfixed_ids = _get_cvfixed_ids(PEinfo)
    isifelse(x) = Symbolics.iscall(x) && Symbolics.operation(x) === ifelse
    terms = unique([term for equation in MTK.equations(PEinfo.model.sys) for term in Symbolics.filterchildren(isifelse, equation.rhs)])
    return filter(term -> all(v -> _get_id(v) in cvfixed_ids, Symbolics.get_variables(Symbolics.arguments(term)[1])), terms)
end

# u[uidx,cidx,i]: value of event target u_ids[uidx] on interval i of condition cidx
function _get_u(PEinfo)
    u_ids = _get_u_ids(PEinfo)
    Nc, N = _get_Nc(PEinfo), length(PEinfo.nodes[1]) - 1
    u = Array{Float64, 3}(undef, length(u_ids), Nc, N)
    for cidx in 1:Nc, i in 1:N, (uidx, id) in enumerate(u_ids)
        times, t = PEinfo.event_times[:,cidx], PEinfo.nodes[cidx][i]
        u[uidx,cidx,i] = something(_get_u_value(PEinfo, id, times, t), _get_default(PEinfo.model, id))
    end
    return u
end

# Right-hand-side terms grouped by form: forms = [(expr, terms)] with expr written over the slot
# arguments (theta, z, cv, m, i, k, js, vs, cvidxs, data, t) and terms = [(v, js, vs, cvidxs, data[m,i])]
function _analyze_rhs(core, PEinfo)
    arguments = _get_arguments(PEinfo)
    rules = _get_substitutions(PEinfo, arguments)
    cvfixed, u = _get_cvfixed(PEinfo, PEinfo.conditions), _get_u(PEinfo)
    Nc, N = _get_Nc(PEinfo), core.N
    terms = [
        (v, term)
        for (v, equation) in enumerate(MTK.equations(PEinfo.model.sys))
        for term in _get_terms(Symbolics.fixpoint_sub(equation.rhs, rules; fold = Val(true)))
    ]
    slots = _get_slots(core, PEinfo, maximum(_count_slots(term) for (v, term) in terms))
    exprs, occurrences = Dict{String, Any}(), []
    for (v, term) in terms
        form = _get_form(term)
        expr, leaves = _get_expr(term, arguments, slots)
        haskey(exprs, form) || (exprs[form] = expr)
        data = [Tuple(_get_data(leaf, cvfixed, u, m, i) for leaf in leaves.data) for m in 1:Nc, i in 1:N]
        push!(occurrences, (form, v, Tuple(leaves.js), Tuple(leaves.vs), Tuple(leaves.cvidxs), data))
    end
    forms = [
        (exprs[form], [(v, js, vs, cvidxs, data) for (other, v, js, vs, cvidxs, data) in occurrences if other == form])
        for form in unique(first.(occurrences))
    ]
    return forms, (arguments.theta, slots.z, slots.cv, slots.m, slots.i, slots.k, slots.js, slots.vs, slots.cvidxs, slots.data, arguments.t)
end

# Slot arrays of a form: z[v,m,i,k] and cv[cvidx,m] with the mesh indices, index tuples js, vs, cvidxs, data
function _get_slots(core, PEinfo, n; Nsum = 0)
    Nz, Nc, N, K, Ncv = _get_Nz(PEinfo), _get_Nc(PEinfo), core.N, core.K, _get_Ncv(PEinfo)
    Symbolics.@variables z[1:Nz, 1:Nc, 1:N, 1:(K + 1)] cv[1:Ncv, 1:Nc] zsum[1:Nsum] m::Int i::Int k::Int js[1:n]::Int vs[1:n]::Int cvidxs[1:n]::Int qs[1:n]::Int data[1:n]
    return (; z, cv, zsum, m, i, k, js, vs, cvidxs, qs, data)
end

_count_leaves(x) = isnothing(_get_leaf(x)) ? sum(_count_leaves, SymbolicUtils.arguments(x)) : 1

_count_slots(x) = isnothing(_get_leaf(x)) ? sum(_count_slots, last(_get_children(x))) : 1

# Top-level + terms of a right-hand side
function _get_terms(rhs)
    rhs = SymbolicUtils.unwrap_const(Symbolics.unwrap(rhs))
    rhs isa Number && return [rhs]
    return SymbolicUtils.isadd(rhs) ? collect(SymbolicUtils.arguments(rhs)) : [rhs]
end

# Leaf of a term as (kind, index or value), or nothing for an inner node
function _get_leaf(x)
    x = SymbolicUtils.unwrap_const(x)
    x isa Number && return (:data, x)
    SymbolicUtils.issym(x) && return (:t, 0)
    if SymbolicUtils.iscall(x) && SymbolicUtils.operation(x) === getindex
        array, index = SymbolicUtils.arguments(x)
        return (Symbol(string(array)), SymbolicUtils.unwrap_const(index))
    end
    return nothing
end

# Integer power of a leaf as repeated factors, else the operation and its arguments
function _get_children(x)
    op, args = SymbolicUtils.operation(x), collect(SymbolicUtils.arguments(x))
    if op === (^) && !isnothing(_get_leaf(args[1]))
        n = SymbolicUtils.unwrap_const(args[2])
        n isa Integer && 2 <= n <= 4 && return (*), fill(args[1], n)
    end
    op === (*) || return op, args
    args = [f for arg in args for f in _get_factors(arg)]
    return op, any(arg -> SymbolicUtils.unwrap_const(arg) isa Number, args) ? args : [1; args]
end

# Factors an argument of a product stands for
function _get_factors(x)
    SymbolicUtils.iscall(x) || return [x]
    op, args = _get_children(x)
    return op === (*) ? args : [x]
end

# Form of a term: every leaf occurrence its kind, children of + and * sorted by form
function _get_form(x)
    leaf = _get_leaf(x)
    isnothing(leaf) || return leaf[1] in (:cvfixed, :u) ? "data" : string(leaf[1])
    op, args = _get_children(x)
    if op === (^) && SymbolicUtils.unwrap_const(args[2]) isa Integer
        return string("^(", _get_form(args[1]), ",", SymbolicUtils.unwrap_const(args[2]), ")")
    end
    kids = [_get_form(a) for a in args]
    op in (+, *) && sort!(kids)
    return string(nameof(op), "(", join(kids, ","), ")")
end

# Term written over the slots, leaves numbered in the order of the form
function _get_expr(x, arguments, slots)
    leaves = (; js = Int[], vs = Int[], cvidxs = Int[], qs = Int[], data = Tuple{Symbol, Any}[])
    return _get_expr!(x, arguments, slots, leaves), leaves
end

function _get_expr!(x, arguments, slots, leaves)
    leaf = _get_leaf(x)
    if !isnothing(leaf)
        kind, index = leaf
        kind === :t && return arguments.t
        kind === :theta && (push!(leaves.js, index); return arguments.theta[slots.js[length(leaves.js)]])
        kind === :z && (push!(leaves.vs, index); return slots.z[slots.vs[length(leaves.vs)], slots.m, slots.i, slots.k])
        kind === :cv && (push!(leaves.cvidxs, index); return slots.cv[slots.cvidxs[length(leaves.cvidxs)], slots.m])
        kind === :zsum && (push!(leaves.qs, index); return slots.zsum[slots.qs[length(leaves.qs)]])
        push!(leaves.data, leaf)
        return slots.data[length(leaves.data)]
    end
    op, args = _get_children(x)
    if op === (^) && SymbolicUtils.unwrap_const(args[2]) isa Integer
        return _get_expr!(args[1], arguments, slots, leaves)^SymbolicUtils.unwrap_const(args[2])
    end
    order = op in (+, *) ? sortperm([_get_form(a) for a in args]) : eachindex(args)
    kids = [_get_expr!(args[q], arguments, slots, leaves) for q in order]
    return op(kids...)
end

# Value of a data leaf in condition m, interval i
_get_data(leaf, cvfixed, u, m, i) = Float64(
    leaf[1] === :data    ? leaf[2] :
    leaf[1] === :cvfixed ? cvfixed[leaf[2],m] :
    leaf[1] === :u       ? u[leaf[2],m,i] :
    throw(ArgumentError("unsupported leaf $(leaf[1]) in a right-hand side"))
)
