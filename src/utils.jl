# Key: (!!!) := determines index -> variable ordering/mapping

# Reads the numeric start (initial-guess) values of a variable handle back out of the
# ExaCore start buffer (c.x0). Returns a host Vector{Float64} in the variable's own
# column-major index order, so it can be reshaped to the variable's dimensions.
# (ExaModels' get_start is only defined on a built ::ExaModel, not on ::ExaCore.)
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
# to PEprob.xnames (== decision-variable index 1:Np). Each entry is :log10, :log, or :lin.
function _get_pscale(PEprob::PEtabODEProblem)::Vector{Symbol}
    xscale = PEprob.model_info.xindices.xscale # Dict{Symbol,Symbol}: param name => scale
    return Symbol[xscale[Symbol(name)] for name in PEprob.xnames]
end

# Normalize a raw observableTransformation / noiseDistribution cell to a Symbol,
# defaulting blanks to the PEtab default.
function _norm_cell(v, default::Symbol)::Symbol
    s = (ismissing(v) || isnothing(v)) ? "" : lowercase(strip(string(v)))
    return isempty(s) ? default : Symbol(s)
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

# Guard: this formulation only supports Gaussian (:normal) noise. Error early on
# anything else (e.g. :laplace) rather than silently emitting the wrong likelihood.
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

# Physical parameter value of p[m] as an ExaModels expression, where the decision
# variable p[m] := θ lives on PEtab's estimation scale. The native ODE RHS and all
# observable/noise formulas use physical parameters, so this inverse-transform is
# applied wherever p enters such a formula. (10^θ via exp(log(10)·θ) keeps it a
# single SIMD-friendly exp node.)
@inline function _p_phys(p, m::Integer, pscale::Vector{Symbol})
    s = pscale[m]
    return s === :log10 ? exp(log(10.0) * p[m]) :
           s === :log   ? exp(p[m])             :
                          p[m]                      # :lin
end

# Numeric physical value from an estimation-scale vector θ (for warm-start computations).
@inline function _p_phys_val(θ, m::Integer, pscale::Vector{Symbol})
    s = pscale[m]
    return s === :log10 ? exp10(θ[m]) :
           s === :log   ? exp(θ[m])   :
                          θ[m]            # :lin
end

# (!!!) Returns ::Vector{String} of condition-dependent variable column names, cv
# These are all conditions-table columns except the metadata columns. This is the
# single source of truth for both the cv count/order and column lookups, so we never
# assume a fixed positional offset (conditionName is optional in PEtab).
function _get_cv_colnames(PEmodel::PEtabModel)::Vector{String}
    conditions_df = PEmodel.petab_tables[:conditions] # DataFrame of conditions
    exclude = ["conditionId", "conditionName"] # Exclude non-cv (metadata) columns
    return [string(str) for str in names(conditions_df) if !(str in exclude)]
end

# (!!!) Returns ::Vector{Symbolics.Num} of condition-dependent variables, cv
# cv[1:Ncv,cidx]
function _get_cv_syms(PEmodel::PEtabModel)::Vector{Symbolics.Num}
    cv_strings = _get_cv_colnames(PEmodel) # Variable names of condition-dependent (::String)
    return Symbolics.Num.(Symbolics.variable.(cv_strings)) # Converts variable name (::String) into symbolic variable (::Symbolics.Num)
end

# (!!!) Returns ::Vector{String} of the SIMULATION conditionIds (conditions that appear as a
# simulationConditionId in the measurements table), in conditions-table order. This is THE
# canonical cidx[1:Nc] ordering used for z / Nc / cv / objective. Pre-equilibration-ONLY
# conditions (conditions-table rows that are never simulated, e.g. Zheng's `preequilibration`)
# are excluded — they would otherwise be (wrongly) collocated and break _get_z_init. For
# models without pre-eq-only rows this is every conditions-table row, so nothing changes.
function _get_cids(PEmodel::PEtabModel)::Vector{String}
    PEtable       = PEmodel.petab_tables
    conditions_df = PEtable[:conditions]
    sim_set       = Set(string.(PEtable[:measurements][!, :simulationConditionId]))
    return [string(c) for c in conditions_df[!, :conditionId] if string(c) in sim_set]
end

# Row index into the conditions table for each canonical cidx. Since `_get_cids` may be a
# subset of the conditions-table rows (pre-eq-only rows dropped), cidx is no longer the row
# number — condition-dependent values must be looked up by conditionId, not by row position.
function _get_cond_rows(PEmodel::PEtabModel)::Vector{Int}
    conditions_df = PEmodel.petab_tables[:conditions]
    cid2row = Dict(string(conditions_df[r, :conditionId]) => r for r in 1:size(conditions_df, 1))
    return [cid2row[cid] for cid in _get_cids(PEmodel)]
end

# (!!!) Returns ::Vector{String} of observableIds
# "y","[1:Ny]"
function _get_obsids(PEmodel::PEtabModel)::Vector{String}
    PEtable = PEmodel.petab_tables # :measurements, :observables, :parameters, :conditions
    observables_df = PEtable[:observables]
    return observables_df[!,:observableId]
end

# Builds a fixpoint substitutor for SBML assignment rules (= `MTK.observed(sys)`, excluding
# any state aliases). These derived/algebraic variables (e.g. total_pop = Σstates, or a
# condition/trigger-dependent rate) can be referenced anywhere — in the ODE RHS, the
# observable formula, or NESTED inside other assignment rules — so a single substitution
# pass is not enough; we iterate to a fixpoint until no assignment-rule symbol remains.
#
# `bare=true` strips the `(t)` from every variable so the rules match the objective's
# bare-symbol convention (`_create_objective` parses table formulas into `(t)`-free
# variables); `bare=false` keeps MTK's native `u(t)` form, used for the ODE RHS.
function _assignment_substitutor(PEprob::PEtabODEProblem; bare::Bool)
    sys    = PEprob.model_info.model.sys
    z_syms = _get_z_syms(PEprob)
    strip_t(s) = Symbolics.Num(Symbolics.variable(Symbol(split(string(s), "(")[1])))
    rebare(e) = (vs = collect(Symbolics.get_variables(e));
                 isempty(vs) ? e : Symbolics.substitute(e, Dict(v => strip_t(v) for v in vs)))
    rules = Dict{Any,Any}()
    for eq in MTK.observed(sys)
        any(isequal(eq.lhs, z) for z in z_syms) && continue   # never rewrite a state alias
        rules[bare ? strip_t(eq.lhs) : eq.lhs] = bare ? rebare(eq.rhs) : eq.rhs
    end
    keyset = collect(keys(rules))
    return function (expr)
        isempty(rules) && return expr
        # Substitute while any assignment-rule symbol still appears among expr's variables.
        # (Use isequal-membership, not `intersect`/`in`: Symbolics `==` returns a symbolic
        # equation, not a Bool, so set ops on Nums misbehave.) Capped against cyclic rules.
        for _ in 1:100
            vars = Symbolics.get_variables(expr)
            any(v -> any(k -> isequal(v, k), keyset), vars) || return expr
            expr = Symbolics.substitute(expr, rules)
        end
        return expr
    end
end

# Table of SBML assignment rules (= `MTK.observed(sys)`, excluding state aliases), as BARE
# symbols matching the objective's `(t)`-free convention. Returns `(ids, lhs, rhs, is_flat)`:
#   ids[r]     :: Symbol  — the rule's name
#   lhs[r]     :: Num     — the bare leaf symbol that stands for the rule
#   rhs[r]     :: <expr>  — the bare RHS (a function of states/params/cv, possibly nested)
#   is_flat[r] :: Bool    — true iff rhs[r] references NO other rule symbol
# Instead of inlining a rule everywhere it appears (which makes build_function expand the rule
# into every formula and blows up codegen — e.g. SalazarCavazos's EGFRtot = Σ72 species divided
# into 4 observables), the objective binds each occurring FLAT rule to an auxiliary "observed
# variable" (ov) defined once per evaluation node. Nested rules (is_flat == false) are left to
# the inlining fallback. Mirrors the bare-symbol construction in `_assignment_substitutor`.
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

# Resolves every FIXED (neither estimated nor condition-dependent) parameter to a numeric
# constant, returning Dict(sym::Num => value). A parameter can be defined by an SBML
# initialAssignment, i.e. its parametermap value is a SYMBOLIC expression of other parameters
# (e.g. Bertozzi's `beta_N => (R0_*gamma_)/N_`). PEtab freezes such a parameter at the value of
# that expression evaluated with the model's DEFAULT parameter values — it does NOT re-evaluate
# it with the per-condition / estimated overrides (verified against PEtab's ODEProblem parameter
# vector: beta_N = 0.1*0.1/1 = 0.01 even though R0_/gamma_/N_ carry their condition values). We
# MUST freeze it the same way, otherwise the collocation RHS uses a different constant than the
# ODE that produced the warm start, leaving the warm start grossly collocation-infeasible.
#
# We get there by fixpoint-substituting the parametermap into itself until every default is
# numeric (resolves nested initialAssignments), then reading off the fixed params. If a fixed
# param cannot be reduced to a number, we fall back to its raw parametermap value (the previous
# behavior) so models that relied on the symbolic form are unaffected.
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

# (!!!) Returns ::Vector{Symbolics.Num} of piecewise(time) gating parameters, __parameter_ifelseN.
# SBMLImporter rewrites every `piecewise(time>T, a, b)` in an SBML rule into the smooth algebraic
# form `b*a + (1-b)*c` where `b = __parameter_ifelseN` is an MTK parameter toggled by a callback at
# the fixed time T. These are NOT estimated and NOT condition columns; we keep them live in the RHS
# (one extra function argument each) and supply their per-(interval,condition) value as data, rather
# than freezing them at the parametermap default (= the pre-trigger value). Empty for models with no
# time events, in which case _get_rhs_funcs is byte-identical to the no-gate behavior.
function _get_gate_syms(PEprob::PEtabODEProblem)::Vector{Symbolics.Num}
    sys = PEprob.model_info.model.sys
    return Symbolics.Num[Symbolics.Num(pp) for pp in MTK.parameters(sys)
                         if occursin("__parameter_ifelse", string(pp))]
end

# Returns ::Vector{Function}, one RHS function per state, with signature
#   f[v]([z[:,i,k,cidx]; p[:]; cv[:,cidx]; gates; t]...)
# The independent variable t is ALWAYS the final argument, even for an autonomous RHS that ignores
# it: build_function tolerates unused arguments (z/p/cv are already passed in full for every state),
# so always carrying t lets every call site pass it uniformly (t_ij for collocation, 0.0 at steady
# state) with no has_t branching. The live piecewise(time) gates gate_syms (see _get_gate_syms) are
# kept as trailing args (the smooth form b*a+(1-b)*c stays symbolic in b); their per-(interval,
# condition) values are supplied as iterator data at the call site, like h[i]/t_ij. ONE
# build_function per state regardless of the gate pattern. gate_syms empty => no-event behavior.
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
    subst_rules = _assignment_substitutor(PEprob; bare = false)
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

# Per-(interval, condition) values of the live gating parameters gate_syms, plus the per-condition
# value to use in the steady-state residual. SBMLImporter's init callback sets each gate at t=0 to
# its triggered value (e.g. 1.0), which the bare ODEProblem does NOT reflect (it carries the
# pre-trigger default) — so we read the gates off a stepped INTEGRATOR, not the problem.
#
# Returns (gate_vals, gate_vals_ss):
#   gate_vals[g,i,cidx]  : value of gate g on interval i of simulation condition cidx (collocation)
#   gate_vals_ss[g,cidx] : value of gate g for cidx's steady-state residual (pre-eq condition's gate
#                          if pre-equilibrated, else the condition's own t=0 value)
# Asserts every in-window gate toggle lands on a mesh node (cumsum(h)); a mid-interval toggle would
# make a single per-interval value wrong and is rejected with a clear error.
function _get_gate_vals(PEmodel::PEtabModel, PEprob::PEtabODEProblem,
                        gate_syms::Vector{Symbolics.Num}, h::Vector{Float64}, taus::Vector{Float64})
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

    h_cum      = cumsum(h) .- h
    bnds       = cumsum(h)                                   # mesh nodes = where events may live
    t_interior = [h_cum[i] + taus[2] * h[i] for i in 1:N]    # strictly inside interval i (past left node)
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
        tend  = max(maximum(bnds), oprob.tspan[2])
        oprob = ODE.remake(oprob; tspan = (oprob.tspan[1], tend))
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
        # Reject any toggle that does not coincide with a mesh node (would split an interval).
        for tc in change_ts
            any(b -> isapprox(b, tc; atol = 1.0e-6, rtol = 1.0e-6), bnds) || error(
                "Condition '$cid' has a fixed-time event at t=$tc that does not coincide with a " *
                "collocation mesh node; mid-interval events are not supported. (Mesh nodes come " *
                "from measurement times; add a measurement/tstop at t=$tc to support it.)")
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
    cids = _get_cids(PEmodel)
    sim_ids = PEprob.model_info.simulation_info.conditionids[:simulation]
    ssc_ids = PEprob.model_info.simulation_info.conditionids[:pre_equilibration]
    dict_cid_cidx = Dict(cids[i] => i for i in eachindex(cids))
    return map(eachindex(cids)) do cidx
        sim_idx = findfirst(==(Symbol(cids[cidx])), sim_ids)
        # Map cidx -> the canonical index of its pre-equilibration condition. If that pre-eq
        # condition is not itself a simulation condition (so it's not in `cids`), fall back to
        # cidx — the steady-state residual then uses this condition's own cv. (Exact only when
        # the pre-eq condition's cv equals the sim condition's, or the cv is absent from the
        # ODE RHS; the pre-eq steady-state *start* is always correct via PEtab's u0.)
        sim_idx === nothing ? cidx : get(dict_cid_cidx, string(ssc_ids[sim_idx]), cidx)
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
# where T_k = cumsum(h)[k] is the right endpoint of interval k (T_0 = 0). A measurement
# at time T_k corresponds to the state at that mesh point; consumers map k to a z node:
# k=0 -> initial node z[·,1,0,·]; 1<=k<=N-1 -> z[·,k+1,0,·]; k=N -> L1 endpoint of interval N.
function _get_dict_t_tidx(h,t_meas)::Dict{Float64, Int64}
    return merge(
        Dict(0.0 => 0),   # t = T_0 = 0 -> initial-condition mesh point
        Dict(
            t_data => findfirst(x -> isapprox(x, t_data; rtol = 1e-10), cumsum(h))
            for t_data in t_meas
        )
    )
end