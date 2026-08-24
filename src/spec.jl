# Compiled problem spec: index orders, scales, condition pairs, compiled functions

# A condition cell or initial value referencing the estimated parameter p[pidx]
struct ParamRef
    pidx::Int
end

# A condition cell or initial value referencing the condition variable cv[cvidx, :]
struct CvRef
    cvidx::Int
end

"""
    PEtabSpec

Everything derived once from the tables and the symbolic model: index orders, scales,
condition (preeq, sim) pairs, measurement maps, compiled RHS/rule functions, control
profiles, and the flat-direction bookkeeping.
"""
struct PEtabSpec
    # counts
    Np::Int
    Nz::Int
    Nc::Int
    Ncc::Int
    Ncv::Int
    Nm::Int
    Nu::Int
    # parameters, parameters-table order over estimate == 1 rows, on estimation scale
    pnames::Vector{String}
    pscale::Vector{Symbol}
    plb::Vector{Float64}
    pub::Vector{Float64}
    pnom::Vector{Float64}
    pidx_of::Dict{String, Int}
    # condition (preeq, sim) pairs, first appearance in the measurements table
    conds::Vector{NamedTuple{(:preeq, :sim), Tuple{String, String}}}
    cvcol_ids::Vector{String}
    simcv::Vector{Int}
    precv::Vector{Int}
    cv_names::Vector{String}
    cv_cells::Matrix{Any}
    # measurements
    meas_cidx::Vector{Int}
    meas_time::Vector{Float64}
    meas_transform::Vector{Symbol}
    # symbols
    z_syms::Vector{Symbolics.Num}
    z_syms_bare::Vector{Symbolics.Num}
    p_syms::Vector{Symbolics.Num}
    cv_syms::Vector{Symbolics.Num}
    u_syms::Vector{Symbolics.Num}
    # compiled once
    rhs_exprs::Vector{Any}
    rhs_fns::Vector{Any}
    subst_t::Function
    subst_bare::Function
    rules::NamedTuple
    fixed_vals::Dict{Any, Any}
    z0::Vector{Any}
    controls::Matrix{ControlProfile}
    used_p::BitVector
end

function _compile_spec(tables::PEtabTables, modelsys::PEtabModelSys)::PEtabSpec
    _assert_normal_noise(tables)
    sys = modelsys.sys

    # Parameters: table order over estimate == 1, bounds/nominal moved to estimation scale
    pnames, pscale, plb, pub, pnom = _estimated_parameters(tables)
    Np = length(pnames)
    pidx_of = Dict(name => m for (m, name) in enumerate(pnames))

    # Condition (preeq, sim) pairs in measurement first-appearance order
    conds, meas_cidx = _condition_pairs(tables)
    Nc = length(conds)

    # cv columns: one per distinct conditionId referenced, in conditions-table order
    cv_names = _cv_names(tables)
    Ncv = length(cv_names)
    cvcol_ids, simcv, precv = _cv_columns(tables, conds)
    Ncc = length(cvcol_ids)

    # Symbols, states in (t)-form and bare form
    z_syms = Symbolics.Num.(MTK.unknowns(sys))
    Nz = length(z_syms)
    z_syms_bare = _strip_t.(z_syms)
    p_syms = Symbolics.Num.(Symbolics.variable.(Symbol.(pnames)))
    cv_syms = Symbolics.Num.(Symbolics.variable.(Symbol.(cv_names)))
    u_syms = _u_syms(sys)
    Nu = length(u_syms)

    # Fixed numeric values: resolved parametermap, parameters-table nominals override
    fixed_vals = _fixed_values(tables, modelsys, pidx_of, cv_names, p_syms, cv_syms)
    fixed_names = Dict{String, Float64}(
        string(k) => Float64(Symbolics.value(v))
        for (k, v) in fixed_vals if Symbolics.value(v) isa Number
    )

    # Condition cells parsed once: Float64 | ParamRef
    cv_cells = _condition_cells(tables, cv_names, cvcol_ids, pidx_of, fixed_names)

    # Measurements
    Nm = _nrows(tables.measurements)
    meas_time = [_cellfloat(cell, "measurements time") for cell in tables.measurements.time]
    meas_transform = _measurement_transforms(tables)

    # Assignment substitutors and the rule table, built once
    subst_t = _assignment_substitutor(sys, z_syms; remove_t = false)
    subst_bare = _assignment_substitutor(sys, z_syms; remove_t = true)
    rules = _rule_table(sys, z_syms)

    # ODE RHS compiled once
    rhs_exprs, rhs_fns = _compile_rhs(modelsys, subst_t, fixed_vals,
                                      z_syms, p_syms, cv_syms, u_syms)

    # Initial-state classification: Float64 | ParamRef | CvRef | compiled f(p, cv)
    z0 = _initial_state_classes(modelsys, fixed_vals, z_syms, p_syms, cv_syms, cv_names)

    # Control profiles per (u, cv column), estimated trigger times error here
    controls = _control_profiles(modelsys.events, string.(u_syms),
                                 _trigger_columns(cv_names, cv_cells, Ncc, pnames, fixed_names))

    # Flat-direction bookkeeping, extended later by the objective phase
    used_p = falses(Np)
    mark = expr -> _mark_used_p!(used_p, pidx_of, expr)
    foreach(mark, rhs_exprs)
    for class in z0
        class isa ParamRef && (used_p[class.pidx] = true)
        class isa NamedTuple && mark(class.expr)
    end
    for cell in cv_cells
        cell isa ParamRef && (used_p[cell.pidx] = true)
    end

    return PEtabSpec(Np, Nz, Nc, Ncc, Ncv, Nm, Nu,
                     pnames, pscale, plb, pub, pnom, pidx_of,
                     conds, cvcol_ids, simcv, precv, cv_names, cv_cells,
                     meas_cidx, meas_time, meas_transform,
                     z_syms, z_syms_bare, p_syms, cv_syms, u_syms,
                     rhs_exprs, rhs_fns, subst_t, subst_bare, rules, fixed_vals, z0,
                     controls, used_p)
end

# Estimation-scale value of a linear-scale table entry
_scale_to_estimation(x::Float64, scale::Symbol) =
    scale === :log10 ? log10(x) : scale === :log ? log(x) : x

# (!!!) Parameters-table order over estimate == 1 rows defines p[1:Np]
function _estimated_parameters(tables::PEtabTables)
    params = tables.parameters
    estimate_col = _getcol(params, :estimate)
    scale_col = _getcol(params, :parameterScale, "lin")
    pnames = String[]
    pscale = Symbol[]
    plb, pub, pnom = Float64[], Float64[], Float64[]
    for i in 1:_nrows(params)
        strip(estimate_col[i]) == "1" || continue
        pid = String(params.parameterId[i])
        scale = _norm_cell(scale_col[i], :lin)
        scale in (:lin, :log, :log10) ||
            error("ExaModelsPEtab: unsupported parameterScale ':$scale' for '$pid'.")
        push!(pnames, pid)
        push!(pscale, scale)
        push!(plb, _scale_to_estimation(_cellfloat(params.lowerBound[i], "lowerBound ($pid)"), scale))
        push!(pub, _scale_to_estimation(_cellfloat(params.upperBound[i], "upperBound ($pid)"), scale))
        push!(pnom, _scale_to_estimation(_cellfloat(params.nominalValue[i], "nominalValue ($pid)"), scale))
    end
    isempty(pnames) && error("ExaModelsPEtab: no estimated parameters (estimate == 1).")
    return pnames, pscale, plb, pub, pnom
end

function _resolve_theta0(spec::PEtabSpec, p0)
    theta0 = p0 === nothing ? copy(spec.pnom) : Float64.(collect(p0))
    length(theta0) == spec.Np ||
        error("ExaModelsPEtab: p0 has length $(length(theta0)), expected Np = $(spec.Np).")
    @assert all(spec.plb .<= theta0 .<= spec.pub) "Nominal θ values fall outside estimation-scale bounds."
    return theta0
end

# (!!!) Condition pairs (preeq, sim) in measurement first-appearance order define cidx = 1:Nc
function _condition_pairs(tables::PEtabTables)
    measurements = tables.measurements
    sim_col = _getcol(measurements, :simulationConditionId)
    preeq_col = _getcol(measurements, :preequilibrationConditionId)
    conds = NamedTuple{(:preeq, :sim), Tuple{String, String}}[]
    idx_of = Dict{Tuple{String, String}, Int}()
    meas_cidx = Int[]
    for i in 1:_nrows(measurements)
        pair = (String(strip(preeq_col[i])), String(strip(sim_col[i])))
        isempty(pair[2]) &&
            error("ExaModelsPEtab: measurements row $i has no simulationConditionId.")
        cidx = get!(idx_of, pair) do
            push!(conds, (preeq = pair[1], sim = pair[2]))
            length(conds)
        end
        push!(meas_cidx, cidx)
    end
    return conds, meas_cidx
end

# (!!!) Returns ::Vector{String} of condition-dependent variable {cv} names (column names)
function _cv_names(tables::PEtabTables)::Vector{String}
    exclude = (:conditionId, :conditionName) # Exclude non-cv (metadata) columns
    return [String(col) for col in keys(tables.conditions) if !(col in exclude)]
end

# (!!!) cv columns = distinct referenced conditionIds in conditions-table order, with the
# cidx -> column maps for the simulation (simcv) and pre-equilibration (precv) condition
function _cv_columns(tables::PEtabTables, conds)
    condition_ids = String.(tables.conditions.conditionId)
    used = Set{String}()
    for cond in conds
        cond.sim in condition_ids || error(
            "ExaModelsPEtab: simulationConditionId '$(cond.sim)' not in the conditions table.")
        push!(used, cond.sim)
        isempty(cond.preeq) && continue
        cond.preeq in condition_ids || error(
            "ExaModelsPEtab: preequilibrationConditionId '$(cond.preeq)' not in the conditions table.")
        push!(used, cond.preeq)
    end
    cvcol_ids = [cid for cid in condition_ids if cid in used]
    col_of = Dict(cid => i for (i, cid) in enumerate(cvcol_ids))
    simcv = [col_of[cond.sim] for cond in conds]
    precv = [isempty(cond.preeq) ? 0 : col_of[cond.preeq] for cond in conds]
    return cvcol_ids, simcv, precv
end

# Parse every condition cell once: Float64 | ParamRef
function _condition_cells(tables::PEtabTables, cv_names, cvcol_ids, pidx_of, fixed_names)
    conditions = tables.conditions
    row_of = Dict(String(cid) => i for (i, cid) in enumerate(conditions.conditionId))
    cells = Matrix{Any}(undef, length(cv_names), length(cvcol_ids))
    for (col, cid) in enumerate(cvcol_ids)
        row = row_of[cid]
        for (cvidx, name) in enumerate(cv_names)
            cell = getproperty(conditions, Symbol(name))[row]
            cells[cvidx, col] = _parse_condition_cell(cell, pidx_of, fixed_names,
                                                      "condition '$cid', column '$name'")
        end
    end
    return cells
end

function _parse_condition_cell(cell::AbstractString, pidx_of, fixed_names, context::String)
    s = String(strip(cell))
    isempty(s) && error("ExaModelsPEtab: $context: empty cell unsupported.")
    val = tryparse(Float64, s)
    if val !== nothing
        isnan(val) &&
            error("ExaModelsPEtab: $context: NaN cell (inherit from model) unsupported.")
        return val
    end
    haskey(pidx_of, s) && return ParamRef(pidx_of[s])
    haskey(fixed_names, s) && return fixed_names[s]
    error("Condition variable '$s' not found in unknown parameter list.")
end

# Normalize a raw observableTransformation / noiseDistribution cell to a ::Symbol
# empty cells default to PEtab default
function _norm_cell(val, default::Symbol)::Symbol
    s = (ismissing(val) || isnothing(val)) ? "" : lowercase(strip(string(val)))
    return isempty(s) ? default : Symbol(s)
end

# (!!!) Returns ::Vector{Symbol} of per-measurement observable transformations
# (:lin/:log/:log10), aligned to measurement row index 1:Nm. The Gaussian noise acts
# on this scale, so the NLL residual is taken in transformed space (see _create_objective).
function _measurement_transforms(tables::PEtabTables)::Vector{Symbol}
    observables = tables.observables
    transform_col = _getcol(observables, :observableTransformation)
    dict_tr = Dict(
        String(observables.observableId[i]) => _norm_cell(transform_col[i], :lin)
        for i in 1:_nrows(observables)
    )
    return Symbol[get(dict_tr, String(oid), :lin) for oid in tables.measurements.observableId]
end

# GUARD: only supports Gaussian (:normal) noise. If empty, defaults to :normal.
# CATCHES: other noise models such as :laplace
function _assert_normal_noise(tables::PEtabTables)
    observables = tables.observables
    dist_col = _getcol(observables, :noiseDistribution)
    for i in 1:_nrows(observables)
        dist = _norm_cell(dist_col[i], :normal)
        dist === :normal || error("ExaModelsPEtab: unsupported noiseDistribution ':$dist' " *
                                   "(observable '$(observables.observableId[i])'); only :normal supported.")
    end
    return nothing
end

# In-lines the physical (linear) parameter value of p[m] as an ExaModels expression
# where p[m] is the PEtab-scaled decision variable
@inline function _p_phys(p, m::Integer, pscale::Vector{Symbol})
    s = pscale[m]
    return s === :log10 ? exp(log(10.0) * p[m]) :
           s === :log   ? exp(p[m])             :
                          p[m]                      # :lin
end

# In-lines the physical (linear) parameter value of theta[m] as a numeric value
# where theta[m] is the numeric value of a PEtab-scaled variable
@inline function _p_phys_val(theta, m::Integer, pscale::Vector{Symbol})
    s = pscale[m]
    return s === :log10 ? exp10(theta[m]) :
           s === :log   ? exp(theta[m])   :
                          theta[m]            # :lin
end

# Strip the MTK (t) from a variable, returning the bare symbol (e.g. x(t) -> x).
# Module-scope so both _assignment_substitutor and _rule_table can use them.
_strip_t(s) = Symbolics.Num(Symbolics.variable(Symbol(split(string(s), "(")[1])))
# Rewrite every variable in an expression to its bare (no (t)) form.
_rebare(e) = (vs = collect(Symbolics.get_variables(e));
            isempty(vs) ? e : Symbolics.substitute(e, Dict(v => _strip_t(v) for v in vs)))

# (!!!) Returns the piecewise(time) control parameters (__parameter_ifelseN)
# (SBMLImporter rewrites piecewise(time>T, …) into a MTK parameters updated by discrete_events block at t=T)
function _u_syms(sys)::Vector{Symbolics.Num}
    return Symbolics.Num[
        Symbolics.Num(pp)
        for pp in MTK.parameters(sys) if occursin("__parameter_ifelse", string(pp))
    ]
end

# Automatically substitutes/applies in "assignment rules" (MTK "observed" variable expressions)
# into existing expressions until only the core {p,z,cv,...} (ExaModels-scope variables) remain
function _assignment_substitutor(sys, z_syms::Vector{Symbolics.Num}; remove_t::Bool)
    # Flatten in (t)-form first (matches MTK.observed's variables)
    rules_t = Dict{Any,Any}()
    for eq in MTK.observed(sys)
        any(isequal(eq.lhs, z) for z in z_syms) && continue       # never rewrite a state alias
        rules_t[eq.lhs] = Symbolics.substitute(eq.rhs, rules_t)   # topological order ⇒ fully flattened
    end

    # optionally strip (t)
    rules = remove_t ? Dict(_strip_t(k) => _rebare(v) for (k, v) in rules_t) : rules_t

    # Match rule symbols by name
    keyset_str = Set(string(k) for k in keys(rules))

    return function (expr)
        isempty(rules) && return expr
        any(v -> string(v) in keyset_str, Symbolics.get_variables(expr)) || return expr
        return Symbolics.substitute(expr, rules) # rules pre-flattened ⇒ single pass suffices
    end
end

# Returns (ids, lhs, rhs, is_flat) encoding the SBML assignment rules (MTK observed variable expression)
# _assignment_substitutor for the SBML variables
function _rule_table(sys, z_syms::Vector{Symbolics.Num})
    ids = Symbol[]; lhs = Symbolics.Num[]; rhs = Any[]
    for eq in MTK.observed(sys)
        any(isequal(eq.lhs, z) for z in z_syms) && continue   # never treat a state alias as a rule
        push!(ids, Symbol(split(string(eq.lhs), "(")[1]))
        push!(lhs, _strip_t(eq.lhs))
        push!(rhs, _rebare(eq.rhs))
    end
    nr = length(ids)
    is_flat = Bool[
        !any(w -> any(k -> k != r && isequal(w, lhs[k]), 1:nr), Symbolics.get_variables(rhs[r]))
        for r in 1:nr
    ]
    return (ids = ids, lhs = lhs, rhs = rhs, is_flat = is_flat)
end

# Returns a mapping: parameters that are fixed numeric values => its numeric value
# as well as PEtab initialAssignment parameters and their numeric values.
# Parameters-table nominal values of non-estimated rows override the SBML defaults, and
# the control parameters are excluded so they stay symbolic in every expression.
function _fixed_values(tables::PEtabTables, modelsys::PEtabModelSys, pidx_of, cv_names,
                       p_syms, cv_syms)
    dict_all_val = Dict(modelsys.parametermap)
    defaults = Dict{Any,Any}(dict_all_val)

    # Recursively substitute initialAssignment expressions down to a numeric value
    for _ in 1:100
        all(Symbolics.value(v) isa Number for v in values(defaults)) && break
        for (k, v) in defaults
            Symbolics.value(v) isa Number && continue
            defaults[k] = Symbolics.substitute(v, defaults)
        end
    end

    # The symbolics variables which we know are fixed
    fixed_syms = setdiff(keys(dict_all_val), union(p_syms, cv_syms))
    dict_fixed_val = Dict{Any,Any}()
    # Create the dictionary mapping these fixed variables to their values
    for sym in fixed_syms
        occursin("__parameter_ifelse", string(sym)) && continue
        rv = Symbolics.value(defaults[sym])
        dict_fixed_val[sym] = rv isa Number ? Float64(rv) : dict_all_val[sym]
    end

    # Parameters-table nominalValue for every non-estimated, non-cv, non-control row
    params = tables.parameters
    nominal_col = _getcol(params, :nominalValue)
    cv_set = Set(cv_names)
    for i in 1:_nrows(params)
        pid = String(params.parameterId[i])
        (haskey(pidx_of, pid) || pid in cv_set || occursin("__parameter_ifelse", pid)) && continue
        val = tryparse(Float64, strip(nominal_col[i]))
        val === nothing && continue
        dict_fixed_val[Symbolics.value(Symbolics.variable(Symbol(pid)))] = val
    end

    return dict_fixed_val
end

# Returns a vector (indexed by v=1:Nz) for each ODE RHS equation, f[v]([z; p; cv; u; t]...)
function _compile_rhs(modelsys::PEtabModelSys, subst_t, fixed_vals,
                      z_syms, p_syms, cv_syms, u_syms)
    # Get the ODE RHS function expressions in its purest symbolic form
    f_exprs_raw = [eqn.rhs for eqn in MTK.equations(modelsys.sys)]

    # Substitutes (in-lines) in expressions and numeric values until the ODE RHS expression
    # is only left with the ExaModels decision variables or other known numeric values
    f_exprs = [Symbolics.substitute(subst_t(f_raw), fixed_vals) for f_raw in f_exprs_raw]

    # Every input (variables which may appear) of the ODE RHS function
    all_syms = [z_syms; p_syms; cv_syms; u_syms; modelsys.t_sym]

    # Build the numeric function for every ODE RHS function expression
    rhs_fns = [
        Symbolics.build_function(f_expr, all_syms..., expression = Val{false})
        for f_expr in f_exprs
    ]
    return f_exprs, rhs_fns
end

# Classify each state's initial value: Float64 | ParamRef | CvRef | (expr, fn) over (p, cv)
function _initial_state_classes(modelsys::PEtabModelSys, fixed_vals,
                                z_syms, p_syms, cv_syms, cv_names)
    # Get mapping of initial condition: symbolic state variable => number/var(p?cv?)/expr
    # (speciemap looked up by name, the reference reorder matches by name as well)
    z0_of = Dict(string(k) => v for (k, v) in modelsys.speciemap)

    state_name(s) = String(split(string(s), "(")[1])   # strip the MTK "(t)"
    z0 = Vector{Any}(undef, length(z_syms))
    for v in eachindex(z_syms)
        ov_cvidx = findfirst(==(state_name(z_syms[v])), cv_names)
        if ov_cvidx !== nothing
            # initial value overridden by a conditions-table column => use that cv
            z0[v] = CvRef(ov_cvidx)
            continue
        end
        haskey(z0_of, string(z_syms[v])) || error(
            "ExaModelsPEtab: no initial value for state '$(z_syms[v])' in the SBML species map.")
        # Substitute fixed constants
        val = Symbolics.simplify(Symbolics.substitute(z0_of[string(z_syms[v])], fixed_vals))
        if Symbolics.value(val) isa Number
            # if z0 is a numeric value...
            z0[v] = Float64(Symbolics.value(val))
        elseif (pidx = findfirst(x -> isequal(x, val), p_syms)) !== nothing
            # if z0 is an unknown parameter p...
            z0[v] = ParamRef(pidx)
        elseif (cvidx = findfirst(x -> isequal(x, val), cv_syms)) !== nothing
            # if z0 is a condition-dependent variable cv...
            z0[v] = CvRef(cvidx)
        else
            # if z0 is some arbitrary function of [p,cv]...
            z0[v] = (expr = val,
                     fn = Symbolics.build_function(val, [p_syms; cv_syms]...,
                                                   expression = Val{false}))
        end
    end
    return z0
end

# Per cv column, numeric trigger-parameter values and the estimated name behind the rest
function _trigger_columns(cv_names, cv_cells, Ncc::Int, pnames, fixed_names)
    columns = NamedTuple[]
    for col in 1:Ncc
        values = Dict{Symbolics.Num, Float64}(
            Symbolics.variable(Symbol(name)) => val for (name, val) in fixed_names
        )
        est_of = Dict{String, String}(name => name for name in pnames)
        for (cvidx, name) in enumerate(cv_names)
            cell = cv_cells[cvidx, col]
            if cell isa Float64
                values[Symbolics.variable(Symbol(name))] = cell
            elseif cell isa ParamRef
                est_of[name] = pnames[cell.pidx]
            end
        end
        push!(columns, (values = values, est_of = est_of))
    end
    return columns
end

function _mark_used_p!(used_p::BitVector, pidx_of, expr)
    for v in Symbolics.get_variables(expr)
        m = get(pidx_of, string(v), 0)
        m > 0 && (used_p[m] = true)
    end
    return nothing
end

# GUARD: every estimated parameter must be read by some emitted expression
# CATCHES: flat directions, e.g. an estimated parameter consumed only by a frozen trigger
function _assert_no_flat_p(spec::PEtabSpec)
    for m in 1:spec.Np
        spec.used_p[m] && continue
        error("ExaModelsPEtab: estimated parameter '$(spec.pnames[m])' appears in no model " *
              "expression, so it would be a flat direction. Unsupported model.")
    end
    return nothing
end

# Returns ::Vector{Float64} in the variable's own column-major index order of
# the start guess values of an ExaModels variable from an ExaCore
# basically just ExaModels.get_starts with ExaCore as input instead of ExaModel
_var_starts(c::ExaCore, v) = Array(view(c.x0, (v.offset + 1):(v.offset + v.length)))
