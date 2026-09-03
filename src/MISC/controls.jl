# Fixed-time control profiles from piecewise events
#
# SBMLImporter rewrites piecewise(time cond, a, b) into a boolean parameter
# __parameter_ifelseN inlined as ((1 - b)*deactivated + b*activated), plus an event whose
# trigger string flips b from 0 to 1. The branch values are ordinary RHS expressions, so
# only the trigger time itself must be fixed. Trigger times and predicates are resolved
# statically here, per condition column, with a hard error on any estimated trigger time.

# A control's trigger under one condition: switch time and the numeric trigger predicate
struct ControlProfile{F}
    ts::Float64            # switch time, NaN if the trigger is t-independent
    holds::F               # t -> Bool, the trigger predicate
end

# u(t) as iterator data: 1.0 where the trigger holds, 0.0 elsewhere
_control_value(profile::ControlProfile, t::Real) = profile.holds(t) ? 1.0 : 0.0

# Parse a placeholder-substituted PEtab table formula into a Symbolics expression
function _to_symbolic(str::AbstractString)
    parsed = Meta.parse(str)
    return parsed isa Symbol ? Symbolics.Num(Symbolics.variable(parsed)) :
                               Symbolics.parse_expr_to_symbolic(parsed, @__MODULE__)
end

# Longer tokens first so "<=" is never read as "<"
const _TRIGGER_OPS = ("<=" => <=, ">=" => >=, "≤" => <=, "≥" => >=, "==" => ==,
                      "<" => <, ">" => >)

# Split a trigger string into symbolic (lhs, op, rhs)
function _parse_trigger(trigger::String, event_name::String)
    for (token, op) in _TRIGGER_OPS
        occursin(token, trigger) || continue
        sides = split(trigger, token)
        length(sides) == 2 ||
            error("ExaModelsPEtab: cannot parse trigger '$trigger' of event '$event_name'.")
        return _to_symbolic(sides[1]), op, _to_symbolic(sides[2])
    end
    error("ExaModelsPEtab: no comparison operator in trigger '$trigger' of event '$event_name'.")
end

"""
    _control_profiles(events, u_names, columns)

Returns `Matrix{ControlProfile}` of size Nu x Ncc.

- `events`: `parse_SBML`'s event dict, keyed by `__parameter_ifelseN` name
- `u_names`: control parameter names aligned with `u_syms`
- `columns`: per cv column `(values, est_of)`, numeric trigger-parameter values and the
  estimated-parameter name behind each non-numeric one
"""
function _control_profiles(events::Dict, u_names::Vector{String}, columns::Vector)
    profiles = Matrix{ControlProfile}(undef, length(u_names), length(columns))
    for (g, name) in enumerate(u_names)
        haskey(events, name) ||
            error("ExaModelsPEtab: no event found for control parameter '$name'.")
        event = events[name]
        event.is_ifelse ||
            error("ExaModelsPEtab: event '$name' is not a piecewise(time) control.")
        lhs, op, rhs = _parse_trigger(event.trigger, name)
        for (col, column) in enumerate(columns)
            profiles[g, col] = _resolve_profile(lhs - rhs, op, column, name)
        end
    end
    return profiles
end

# Resolve one trigger under one condition column to a numeric ControlProfile
function _resolve_profile(diff, op, column, event_name::String)
    resolved = Symbolics.substitute(diff, column.values)
    tvar = nothing
    for v in Symbolics.get_variables(resolved)
        vname = string(v)
        if vname == "t"
            tvar = Symbolics.Num(v)
        elseif haskey(column.est_of, vname)
            error("ExaModelsPEtab: estimated parameter '$(column.est_of[vname])' sets the " *
                  "trigger time of event '$event_name'. Only fixed-time events are supported.")
        else
            error("ExaModelsPEtab: cannot resolve '$vname' in the trigger of event " *
                  "'$event_name' to a numeric value. Only fixed-time events are supported.")
        end
    end
    if tvar === nothing
        val = Float64(Symbolics.value(resolved))
        return ControlProfile(NaN, t -> op(val, 0.0))
    end
    ts_expr = try
        Symbolics.symbolic_linear_solve(resolved, tvar)
    catch
        error("ExaModelsPEtab: cannot solve the trigger of event '$event_name' for its " *
              "switch time.")
    end
    ts = Float64(Symbolics.value(ts_expr))
    difffn = Symbolics.build_function(resolved, tvar; expression = Val{false})
    return ControlProfile(ts, t -> op(difffn(t), 0.0))
end

# Switch times of one condition column falling strictly inside (t0, tend)
function _switch_times(profiles::Matrix{ControlProfile}, col::Int, t0::Real, tend::Real)
    ts = [profiles[g, col].ts for g in 1:size(profiles, 1)]
    return sort!(unique!(Float64[t for t in ts if !isnan(t) && t0 < t < tend]))
end
