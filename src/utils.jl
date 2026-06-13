# Key: (!!!) := determines index -> variable ordering/mapping

# Normalize a raw observableTransformation / noiseDistribution cell to a Symbol,
# defaulting blanks to the PEtab default.
function _norm_cell(v, default::Symbol)::Symbol
    s = (ismissing(v) || isnothing(v)) ? "" : lowercase(strip(string(v)))
    return isempty(s) ? default : Symbol(s)
end

# TODO make a PR to ExaModels for simple new tool?
# Returns ::Vector{Float64} in the variable's own column-major index order of 
# the start guess values of an ExaModels variable from an ExaCore
_var_starts(c::ExaCore, v) = Array(view(c.x0, (v.offset + 1):(v.offset + v.length)))

# (!!!) Returns ::Vector{Symbolics.Num} of state variables
# z[1:Nz,i,k,cidx]
function _get_z_syms(PEprob::PEtabODEProblem)::Vector{Symbolics.Num}
    sys = PEprob.model_info.model.sys
    return MTK.unknowns(sys)
end

# (!!!) Returns ::Vector{Symbolics.Num} of unknown parameters
# p[1:Np]
function _get_p_syms(PEprob::PEtabODEProblem)::Vector{Symbolics.Num}
    return Symbolics.Num.(Symbolics.variable.(PEprob.xnames)) # Converts variable name (::String) into symbolic variable (::Symbolics.Num)
end

# (!!!) Returns ::Vector{Symbol} of per-parameter PEtab estimation scales, aligned
# to PEprob.xnames (== decision-variable index 1:Np). Each entry is in {:log10, :log, :lin}
function _get_pscale(PEprob::PEtabODEProblem)::Vector{Symbol}
    xscale = PEprob.model_info.xindices.xscale # Dict{Symbol,Symbol}: param name => scale
    return Symbol[xscale[Symbol(name)] for name in PEprob.xnames]
end

# (!!!) Returns ::Vector{Symbol} of per-measurement observable transformations
# (:lin/:log/:log10), aligned to measurement row index 1:Nm. The Gaussian noise acts
# on this scale, so the NLL residual is taken in transformed space (see _create_objective).
function _get_meas_transforms(PEmodel::PEtabModel)::Vector{Symbol}
    measurements_df = PEmodel.petab_tables[:measurements]
    observables_df  = PEmodel.petab_tables[:observables]
    has_tr  = :observableTransformation in propertynames(observables_df)
    dict_tr = Dict(
        string(observables_df[i, :observableId]) =>
            (has_tr ? _norm_cell(observables_df[i, :observableTransformation], :lin) : :lin)
        for i in 1:size(observables_df, 1)
    )
    Nm = size(measurements_df, 1)
    return Symbol[get(dict_tr, string(measurements_df[midx, :observableId]), :lin) for midx in 1:Nm]
end

# GUARD: only supports Gaussian (:normal) noise. If empty, defaults to :normal.
# CATCHES: other noise models such as :laplace
function _assert_normal_noise(PEmodel::PEtabModel)
    observables_df = PEmodel.petab_tables[:observables]
    :noiseDistribution in propertynames(observables_df) || return nothing
    for i in 1:size(observables_df, 1)
        dist = _norm_cell(observables_df[i, :noiseDistribution], :normal)
        dist === :normal || error("Unsupported noiseDistribution '$dist' for observable " *
                                   "$(observables_df[i, :observableId]); only :normal is supported.")
    end
    return nothing
end

# GUARD: only support fixed-time discete events (variable 'x' changes to value 'y' at time 't')
# CATCHES: continous or conditional events, e.g., Liu, Smith
function _assert_supported_events(PEmodel::PEtabModel)
    sbml_path = get(PEmodel.paths, :SBML, nothing)
    (sbml_path === nothing || !isfile(sbml_path)) && return nothing
    nev = count(r"<event[ >]", read(sbml_path, String))
    nev == 0 && return nothing
    error("ExaModelsPEtab does not support SBML <event> elements ($nev found in " *
          "$(basename(sbml_path))). These encode state-triggered, state-jump, or parametric-time " *
          "events that the collocation transcription cannot represent and would otherwise be " *
          "SILENTLY IGNORED (wrong dynamics/objective). Fixed-time piecewise(time>T) gates ARE " *
          "supported. Known affected benchmark models: Liu, Smith.")
end

# In-lines the physical (linear) parameter value of p[m] as an ExaModels expression
# where p[m] is the PEtab-scaled decision variable
@inline function _p_phys(p, m::Integer, pscale::Vector{Symbol})
    s = pscale[m]
    return s === :log10 ? exp(log(10.0) * p[m]) :
           s === :log   ? exp(p[m])             :
                          p[m]                      # :lin
end

# In-lines the physical (linear) parameter value of θ[m] as a numeric value
# where θ[m] is the numeric value of a PEtab-scaled variable
@inline function _p_phys_val(θ, m::Integer, pscale::Vector{Symbol})
    s = pscale[m]
    return s === :log10 ? exp10(θ[m]) :
           s === :log   ? exp(θ[m])   :
                          θ[m]            # :lin
end

# (!!!) Returns ::Vector{String} of condition-dependent variable {cv} names (column names)
function _get_cv_names(PEmodel::PEtabModel)::Vector{String}
    conditions_df = PEmodel.petab_tables[:conditions] # DataFrame of conditions
    exclude = ["conditionId", "conditionName"] # Exclude non-cv (metadata) columns
    return [string(str) for str in names(conditions_df) if !(str in exclude)]
end

# (!!!) Returns ::Vector{Symbolics.Num} of condition-dependent variables
# cv[1:Ncv,cidx]
function _get_cv_syms(PEmodel::PEtabModel)::Vector{Symbolics.Num}
    cv_strings = _get_cv_names(PEmodel) # Names of condition-dependent variables (::String)
    return Symbolics.Num.(Symbolics.variable.(cv_strings)) # Converts variable name (::String) into symbolic variable (::Symbolics.Num)
end

# (!!!) Returns ::Vector{String} of the simulation conditionIds (conditions that appear as a simulationConditionId in the measurements table)
# cidx[1:Nc]
function _get_cids(PEmodel::PEtabModel)::Vector{String}
    PEtable       = PEmodel.petab_tables
    conditions_df = PEtable[:conditions]
    sim_set       = Set(string.(PEtable[:measurements][!, :simulationConditionId]))
    return [string(c) for c in conditions_df[!, :conditionId] if string(c) in sim_set]
end

# The Nc simulation conditions (positions 1:Nc, matching the cv column layout) followed by any
# DISTINCT pre-equilibration conditions. The extra columns let the steady-state residual f(zss)=0
# read the pre-eq inputs (e.g. Zheng dilution=1); without pre-eq this is exactly _get_cids.
function _get_cv_cids(PEmodel::PEtabModel, PEprob::PEtabODEProblem)::Vector{String}
    sim_cids = _get_cids(PEmodel)
    si = PEprob.model_info.simulation_info
    preeq = si.has_pre_equilibration ? unique(string.(si.conditionids[:pre_equilibration])) : String[]
    extra = [c for c in preeq if !(c in sim_cids)]
    return [sim_cids; extra]
end

function _get_cv_cid_rows(PEmodel::PEtabModel, PEprob::PEtabODEProblem)::Vector{Int}
    conditions_df = PEmodel.petab_tables[:conditions]
    dict_cid_row = Dict(string(conditions_df[row, :conditionId]) => row for row in 1:size(conditions_df, 1))
    return [dict_cid_row[cid] for cid in _get_cv_cids(PEmodel, PEprob)]
end

# (!!!) Returns ::Vector{String} of observableIds
# y[1:Nm]
function _get_obsids(PEmodel::PEtabModel)::Vector{String}
    PEtable = PEmodel.petab_tables # :measurements, :observables, :parameters, :conditions
    observables_df = PEtable[:observables]
    return observables_df[!,:observableId]
end

# Automatically substitutes/applies in "assignment rules" (MTK "observed" variable expressions)
# into existing expressions until only the core {p,z,cv,...} ExaModels-scope variables remain
function _assignment_substitutor(PEprob::PEtabODEProblem; remove_t::Bool)
    sys    = PEprob.model_info.model.sys
    z_syms = _get_z_syms(PEprob)
    strip_t(s) = Symbolics.Num(Symbolics.variable(Symbol(split(string(s), "(")[1])))
    rebare(e) = (vs = collect(Symbolics.get_variables(e));
                 isempty(vs) ? e : Symbolics.substitute(e, Dict(v => strip_t(v) for v in vs)))
    rules = Dict{Any,Any}()
    for eq in MTK.observed(sys)
        any(isequal(eq.lhs, z) for z in z_syms) && continue   # never rewrite a state alias
        # if remove_t == true, remove the (t) appended to variable name. else keep the (t)
        rules[remove_t ? strip_t(eq.lhs) : eq.lhs] = remove_t ? rebare(eq.rhs) : eq.rhs
    end
    keyset = collect(keys(rules))
    return function (expr)
        isempty(rules) && return expr
        # Substitute while any assignment-rule symbol still appears among expr's variables.
        # (Use isequal-membership, not `intersect`/`in`: Symbolics `==` returns a symbolic
        # equation, not a Bool, so set ops on Nums misbehave.) Capped against cyclic rules.
        for _ in 1:100
            # need to resubstitute because observed variables may be defined in terms of each other
            vars = Symbolics.get_variables(expr)
            any(v -> any(k -> isequal(v, k), keyset), vars) || return expr
            expr = Symbolics.substitute(expr, rules)
        end
        return expr
    end
end

# SBML assignment rules (`MTK.observed(sys)` minus state aliases) as bare `(t)`-free symbols,
# returned as `(ids, lhs, rhs, is_flat)`; is_flat[r] is true iff rhs[r] references no other rule.
# The objective binds each occurring flat rule to an auxiliary "observed variable" (ov) once per
# node instead of inlining it everywhere (inlining blows up codegen — e.g. SalazarCavazos's
# 72-species EGFRtot across 4 observables); nested rules fall back to inlining. Mirrors _assignment_substitutor.
function _rule_table(PEprob::PEtabODEProblem)
    sys    = PEprob.model_info.model.sys
    z_syms = _get_z_syms(PEprob)
    strip_t(s) = Symbolics.Num(Symbolics.variable(Symbol(split(string(s), "(")[1])))
    rebare(e) = (vs = collect(Symbolics.get_variables(e));
                 isempty(vs) ? e : Symbolics.substitute(e, Dict(v => strip_t(v) for v in vs)))
    ids = Symbol[]; lhs = Symbolics.Num[]; rhs = Any[]
    for eq in MTK.observed(sys)
        any(isequal(eq.lhs, z) for z in z_syms) && continue   # never treat a state alias as a rule
        push!(ids, Symbol(split(string(eq.lhs), "(")[1]))
        push!(lhs, strip_t(eq.lhs))
        push!(rhs, rebare(eq.rhs))
    end
    nr = length(ids)
    is_flat = Bool[
        !any(w -> any(k -> k != r && isequal(w, lhs[k]), 1:nr), Symbolics.get_variables(rhs[r]))
        for r in 1:nr
    ]
    return ids, lhs, rhs, is_flat
end

# Numeric values for fixed parameter-table params (NOT estimated, condition (cv), or gate vars).
# Crucially includes observable/noise-only params (e.g. a fixed scale `s_…` or noise sd `sd_…`) that
# never appear in the SBML model — absent from the parametermap, they would otherwise leak as unbound
# symbols in the observable/noise build_function. Returns Dict(bare symbol => Float64), keyed via
# Symbolics.variable(name) to match the formula parser's bare symbols for substitution.
function _get_table_fixed_vals(PEmodel::PEtabModel, PEprob::PEtabODEProblem)::Dict{Any,Any}
    params_df = PEmodel.petab_tables[:parameters]
    estimated = Set(string.(PEprob.xnames))                                  # decision vars p
    cv_names  = Set(string(Symbolics.value(s)) for s in _get_cv_syms(PEmodel))   # condition vars
    out = Dict{Any,Any}()
    for row in eachrow(params_df)
        pid = string(row[:parameterId])
        (pid in estimated || pid in cv_names || occursin("__parameter_ifelse", pid)) && continue
        nv  = row[:nominalValue]
        (ismissing(nv) || (nv isa AbstractString && isempty(strip(nv)))) && continue
        val = nv isa AbstractString ? tryparse(Float64, strip(nv)) : Float64(nv)
        val === nothing && continue
        out[Symbolics.value(Symbolics.variable(Symbol(pid)))] = val
    end
    return out
end

# Numeric values for fixed (non-estimated, non-cv) params from the parametermap, as Dict(sym => value).
# An initialAssignment param has a symbolic parametermap value (e.g. Bertozzi `beta_N => R0_*gamma_/N_`);
# PEtab freezes it at the DEFAULT-valued expression, NOT the condition/estimated overrides, so we must
# too — else the collocation RHS uses a different constant than the warm-start ODE and the warm start is
# collocation-infeasible. Fixpoint-substitute the parametermap into itself until numeric, then read off
# the fixed params; fall back to the raw symbolic value if one can't reduce to a number.
function _resolve_fixed_vals(PEmodel::PEtabModel, PEprob::PEtabODEProblem)
    dict_all_val = Dict(PEprob.model_info.model.parametermap)
    defaults = Dict{Any,Any}(dict_all_val)
    for _ in 1:100
        all(Symbolics.value(v) isa Number for v in values(defaults)) && break
        for (k, v) in defaults
            Symbolics.value(v) isa Number && continue
            defaults[k] = Symbolics.substitute(v, defaults)
        end
    end
    fixed_syms = setdiff(keys(dict_all_val), union(_get_p_syms(PEprob), _get_cv_syms(PEmodel)))
    out = Dict{Any,Any}()
    for sym in fixed_syms
        rv = Symbolics.value(defaults[sym])
        out[sym] = rv isa Number ? Float64(rv) : dict_all_val[sym]
    end
    return out
end

# (!!!) piecewise(time) gating parameters (__parameter_ifelseN): SBMLImporter rewrites each
# `piecewise(time>T, …)` into a smooth form toggled by an MTK parameter at the fixed time T. Not
# estimated, not condition columns — we keep them live in the RHS (one extra arg each) and supply the
# per-(interval,condition) value as data rather than freezing them at the pre-trigger default. Empty
# if the model has no time events.
function _get_gate_syms(PEprob::PEtabODEProblem)::Vector{Symbolics.Num}
    sys = PEprob.model_info.model.sys
    return Symbolics.Num[Symbolics.Num(pp) for pp in MTK.parameters(sys)
                         if occursin("__parameter_ifelse", string(pp))]
end

# One RHS function per state, signature f[v]([z[:,i,k,cidx]; p; cv[:,cidx]; gates; t]...). t is
# ALWAYS the final arg, even for an autonomous RHS (build_function tolerates unused args), so every
# call site passes it uniformly (t_ij for collocation, 0.0 at steady state) with no has_t branching.
# Gate params stay symbolic trailing args; their per-(interval,condition) values are supplied as
# iterator data at the call site, like h[i]/t_ij. gate_syms empty => no-event behavior.
function _get_rhs_funcs(PEmodel::PEtabModel, PEprob::PEtabODEProblem,
                        gate_syms::Vector{Symbolics.Num} = Symbolics.Num[])
    sys = PEprob.model_info.model.sys

    f_exprs_raw = [eqn.rhs for eqn in MTK.equations(sys)]

    # Substitute in fixed constant values (initialAssignment-defined params resolved to the
    # constants PEtab freezes them at — see _resolve_fixed_vals). EXCLUDE the gate parameters: they
    # stay symbolic so they remain function arguments whose value is set per interval.
    dict_fixed_val = _resolve_fixed_vals(PEmodel, PEprob)
    gate_names = Set(string.(gate_syms))
    for k in collect(keys(dict_fixed_val))
        string(k) in gate_names && delete!(dict_fixed_val, k)
    end

    z_syms  = _get_z_syms(PEprob)
    p_syms  = _get_p_syms(PEprob)
    cv_syms = _get_cv_syms(PEmodel)

    # Recursively substitute ALL assignment rules (u(t,p) inputs, derived quantities, and any
    # nested ones) to a fixpoint, then the fixed numeric constants. After this the RHS is a
    # function of states / estimated params / condition vars / gates (and possibly t) only.
    subst_rules = _assignment_substitutor(PEprob; remove_t = false)
    f_exprs = [
        Symbolics.substitute(subst_rules(f_raw), dict_fixed_val)
        for f_raw in f_exprs_raw
    ]

    # The t argument: use the actual independent-variable object when the RHS is time-dependent (so
    # build_function binds it correctly), otherwise any unused `t` symbol — it is a trailing arg the
    # autonomous RHS never references. (String compare on the name is robust across Symbolics vers.)
    all_free = foldl(union, Symbolics.get_variables.(f_exprs))
    t_basic  = nothing
    for v in all_free
        if string(v) == "t"
            t_basic = v
            break
        end
    end
    t_sym = t_basic !== nothing ? Symbolics.Num(t_basic) : Symbolics.Num(Symbolics.variable(:t))

    all_syms = [z_syms; p_syms; cv_syms; gate_syms; [t_sym]]

    return [
        Symbolics.build_function(f_expr, all_syms..., expression = Val{false})
        for f_expr in f_exprs
    ]
end

# Fixed-time gate toggle times: the t at which any __parameter_ifelse gate flips, across all
# simulation conditions. _get_z_init forces these into the collocation mesh (as tstops) so every
# event lands on a mesh node — otherwise _get_gate_vals rejects a mid-interval toggle (Isensee's
# t=45 was a measurement-free time). Detected by stepping a callback-applied integrator and
# watching the gate parameters change: SBMLImporter's time callback forces a stop exactly at the
# toggle, so integ.t at the change IS the event time. Empty for non-event models (no mesh change).
function _get_event_times(PEmodel::PEtabModel, PEprob::PEtabODEProblem)::Vector{Float64}
    gate_syms = _get_gate_syms(PEprob)
    isempty(gate_syms) && return Float64[]
    si        = PEprob.model_info.simulation_info
    has_preeq = si.has_pre_equilibration
    sim_ids   = si.conditionids[:simulation]
    preeq_ids = si.conditionids[:pre_equilibration]
    p_nominal = PEtab.get_x(PEprob)
    solver    = PEprob.probinfo.solver.solver
    abstol    = PEprob.probinfo.solver.abstol
    reltol    = PEprob.probinfo.solver.reltol
    gate_raw  = [Symbolics.value(g) for g in gate_syms]
    read_gates(integ) = Float64[Float64(integ.ps[g]) for g in gate_raw]
    t_events = Float64[]
    for cid in Symbol.(_get_cids(PEmodel))
        cond_arg = cid
        if has_preeq
            pos = findfirst(==(cid), sim_ids); pos === nothing && continue
            cond_arg = preeq_ids[pos] => sim_ids[pos]
        end
        oprob, cbs = PEtab.get_odeproblem(p_nominal, PEprob; condition = cond_arg)
        integ = ODE.init(oprob, solver; callback = cbs, abstol = abstol, reltol = reltol)
        cur   = read_gates(integ)
        t_end  = oprob.tspan[2]
        while integ.t < t_end
            tprev = integ.t
            ODE.step!(integ)
            (isfinite(integ.t) && integ.t > tprev) || break   # integrator stalled/failed
            new = read_gates(integ)
            if new != cur
                push!(t_events, integ.t)
                cur = new
            end
        end
    end
    return sort(unique(t_events))
end

# Per-(interval, condition) values of the live gating parameters gate_syms, plus the per-condition
# value to use in the steady-state residual. SBMLImporter's init callback sets each gate at t=0 to
# its triggered value (e.g. 1.0), which the bare ODEProblem does NOT reflect (it carries the
# pre-trigger default) — so we read the gates off a stepped INTEGRATOR, not the problem.
#
# Returns (gate_vals, gate_vals_ss):
#   gate_vals[g,i,cidx]  : value of gate g on interval i of simulation condition cidx (collocation)
#   gate_vals_ss[g,cidx] : value of gate g for cidx's steady-state residual (pre-eq condition's gate
#                          if pre-equilibrated, else the condition's own t=0 value)
# Asserts every in-window gate toggle lands on a mesh node (t_nodes); a mid-interval toggle would
# make a single per-interval value wrong and is rejected with a clear error.
function _get_gate_vals(PEmodel::PEtabModel, PEprob::PEtabODEProblem,
                        gate_syms::Vector{Symbolics.Num}, h::Vector{Float64}, taus::Vector{Float64},
                        t_nodes::Vector{Float64})
    Ng   = length(gate_syms)
    cids = Symbol.(_get_cids(PEmodel))
    Nc   = length(cids)
    N    = length(h)
    gate_vals    = zeros(Float64, Ng, N, Nc)
    gate_vals_ss = zeros(Float64, Ng, Nc)
    Ng == 0 && return gate_vals, gate_vals_ss

    si        = PEprob.model_info.simulation_info
    has_preeq = si.has_pre_equilibration
    sim_ids   = si.conditionids[:simulation]
    preeq_ids = si.conditionids[:pre_equilibration]
    p_nominal = PEtab.get_x(PEprob)
    solver    = PEprob.probinfo.solver.solver
    abstol    = PEprob.probinfo.solver.abstol
    reltol    = PEprob.probinfo.solver.reltol

    # EXACT mesh nodes from t_nodes (T_0..T_N) — no cumsum(h) round-off. Interval i = [t_nodes[i],
    # t_nodes[i+1]]; bnds = the right-node times (where a toggle may live).
    bnds       = t_nodes[2:end]                              # mesh nodes (interval right endpoints)
    t_interior = [t_nodes[i] + taus[2] * h[i] for i in 1:N]  # strictly inside interval i (past left node)
    tstops     = sort(unique(vcat(t_interior, bnds)))        # stop at every interior sample and node

    gate_raw = [Symbolics.value(g) for g in gate_syms]   # unwrap Num for the .ps[] indexing interface
    read_gates(integ) = Float64[Float64(integ.ps[g]) for g in gate_raw]

    for (cidx, cid) in enumerate(cids)
        cond_arg = cid
        if has_preeq
            pos = findfirst(==(cid), sim_ids)
            pos === nothing && continue
            cond_arg = preeq_ids[pos] => sim_ids[pos]
        end
        oprob, cbs = PEtab.get_odeproblem(p_nominal, PEprob; condition = cond_arg)
        t_end  = max(maximum(bnds), oprob.tspan[2])
        oprob = ODE.remake(oprob; tspan = (oprob.tspan[1], t_end))
        integ = ODE.init(oprob, solver; callback = cbs, tstops = tstops, abstol = abstol, reltol = reltol)

        g0        = read_gates(integ)                        # gate at the (post-init) initial condition, t=0
        cur       = copy(g0)
        change_ts = Float64[]
        for i in 1:N
            tq = t_interior[i]
            while integ.t < tq
                tprev = integ.t
                ODE.step!(integ)
                (isfinite(integ.t) && integ.t > tprev) || break   # integrator stalled/failed
                new = read_gates(integ)
                if new != cur
                    push!(change_ts, integ.t)
                    cur = new
                end
            end
            gate_vals[:, i, cidx] = cur
        end
        # Every toggle must coincide EXACTLY with a mesh node — guaranteed because _get_event_times
        # forces each event time into the mesh tstops (so it appears bit-identically in t_nodes).
        # Exact `==` (no isapprox): both tc and t_nodes derive from the same fixed callback time.
        for tc in change_ts
            any(==(tc), t_nodes) || error(
                "Condition '$cid' has a fixed-time event at t=$tc not on a collocation mesh node. " *
                "This should be impossible (events are forced into the mesh by _get_event_times); " *
                "indicates the toggle time wasn't captured at mesh-build time.")
        end

        # Steady-state residual / initial-condition gate value: the gate at the IC (t=0+), i.e. the
        # SAME value interval 1 uses. The continuity constraint sets z[:,1,0,cidx] = zss[:,cidx] and
        # the residual pins f(zss)=0; using g0 keeps the residual consistent with the first
        # collocation interval (and with PEtab's pre-equilibrated u0 — verified ‖f(u0; g0)‖ ≈ 0,
        # e.g. Brannmark Dose_0 4.7e-6). Pre-eq conditions cannot be addressed standalone (only as
        # the pair preeq=>sim), so there is no separate pre-eq integrator to read.
        gate_vals_ss[:, cidx] = g0
    end
    return gate_vals, gate_vals_ss
end

# Per-condition steady-state gate values for the PURE steady-state path (no collocation mesh).
# Each gate is constant at the equilibrium, so we read it off the integrator at t=0 (post-init).
function _get_gate_vals_ss(PEmodel::PEtabModel, PEprob::PEtabODEProblem, gate_syms::Vector{Symbolics.Num})
    Ng   = length(gate_syms)
    cids = Symbol.(_get_cids(PEmodel))
    Nc   = length(cids)
    out  = zeros(Float64, Ng, Nc)
    Ng == 0 && return out
    p_nominal = PEtab.get_x(PEprob)
    solver    = PEprob.probinfo.solver.solver
    abstol    = PEprob.probinfo.solver.abstol
    reltol    = PEprob.probinfo.solver.reltol
    gate_raw  = [Symbolics.value(g) for g in gate_syms]   # unwrap Num for the .ps[] indexing interface
    for (cidx, cid) in enumerate(cids)
        oprob, cbs = PEtab.get_odeproblem(p_nominal, PEprob; condition = cid)
        integ = ODE.init(oprob, solver; callback = cbs, abstol = abstol, reltol = reltol)
        out[:, cidx] = Float64[Float64(integ.ps[g]) for g in gate_raw]
    end
    return out
end

# Returns ::Dictionary{} of p::String => p[pidx] index
function _get_dict_pstr_pidx(PEprob::PEtabODEProblem)::Dict{String, Int64}
    return Dict(pstr => pidx for (pidx,pstr) in enumerate(String.(PEprob.xnames)))
end

# Returns a Vector mapping each canonical condition index cidx (1:Nc) to the canonical
# index of its pre-equilibration condition (sscidx). Conditions that are not simulation
# conditions map to themselves.
function _get_dict_cidx_sscidx(PEmodel::PEtabModel, PEprob::PEtabODEProblem)::Vector{Int64}
    cids        = _get_cids(PEmodel)                          # simulation conditions, cv cols 1:Nc
    cv_cond_ids = _get_cv_cids(PEmodel, PEprob)           # [sim; distinct extra pre-eq]
    pos_of      = Dict(cv_cond_ids[i] => i for i in eachindex(cv_cond_ids))
    sim_ids = PEprob.model_info.simulation_info.conditionids[:simulation]
    ssc_ids = PEprob.model_info.simulation_info.conditionids[:pre_equilibration]
    # Map each simulation condition cidx -> the cv COLUMN of its pre-equilibration condition, so
    # the steady-state residual f(zss)=0 is evaluated under the PRE-EQ inputs. Because cv now
    # carries a column for every distinct pre-eq condition (see _get_cv_cids), pos_of always
    # resolves it; the cidx fallback only triggers for a (nonexistent) non-simulation cidx.
    return map(eachindex(cids)) do cidx
        sim_idx = findfirst(==(Symbol(cids[cidx])), sim_ids)
        sim_idx === nothing && return cidx
        get(pos_of, string(ssc_ids[sim_idx]), cidx)
    end
end

####################################################
# UTILS FOR CHECKING MODEL FEATURES
####################################################
# True if the model uses steady-state pre-equilibration (x0SSpre) initial conditions.
function _check_x0SSpre(PEprob::PEtabODEProblem)::Bool
    return PEprob.model_info.simulation_info.has_pre_equilibration
end

####################################################
# UTILS FOR OBJECTIVE FUNCTION
####################################################
# Returns dictionary: condition id (::String) => condition index, cidx in [Nc] (::Int64)
function _get_dict_cid_cidx(PEmodel::PEtabModel)::Dict{String, Int64}
    return Dict(
        cid => cidx
        for (cidx, cid) in enumerate(_get_cids(PEmodel))
    )
end

# Returns dictionary: measurement time (::Float64) => mesh point index k in 0:N (::Int64)
# where T_k = t_nodes[k+1] is the right endpoint of interval k (T_0 = t_nodes[1]). A measurement
# at time T_k corresponds to the state at that mesh point; consumers map k to a z node:
# k=0 -> initial node z[·,1,0,·]; 1<=k<=N-1 -> z[·,k+1,0,·]; k=N -> L1 endpoint of interval N.
function _get_dict_t_tidx(t_nodes::AbstractVector, t_meas)::Dict{Float64, Int64}
    # EXACT map. t_nodes = the original mesh boundary times sol.t (T_k = t_nodes[k+1], T_0 = tstart).
    # Every measurement time was a solver tstop, so it appears verbatim in t_nodes ⇒ match with `==`
    # (no isapprox). Previously this matched against cumsum(diff(sol.t)), whose rounding drift grows
    # with node index and, on a fine mesh, can match the WRONG adjacent node.
    d = Dict{Float64, Int64}(0.0 => 0)   # t = T_0 = 0 -> initial-condition mesh point
    for t_data in t_meas
        k = findfirst(==(t_data), t_nodes)
        k === nothing && error("measurement time t=$t_data is not a mesh node — it should have been " *
            "forced as a solver tstop in _get_z_init. (t_nodes range $(first(t_nodes))..$(last(t_nodes)).)")
        d[t_data] = k - 1   # 1-based position -> 0-based mesh-node index k in 0:N
    end
    return d
end