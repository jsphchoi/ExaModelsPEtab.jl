# Key: (!!!) := determines index -> variable ordering/mapping

# Normalize a raw observableTransformation / noiseDistribution cell to a ::Symbol
# empty cells default to PEtab default
function _norm_cell(val, default::Symbol)::Symbol
    s = (ismissing(val) || isnothing(val)) ? "" : lowercase(strip(string(val)))
    return isempty(s) ? default : Symbol(s)
end

# Returns ::Vector{Float64} in the variable's own column-major index order of 
# the start guess values of an ExaModels variable from an ExaCore
# (basically ExaModels.get_starts but with ExaCore as input instead of ExaModel)
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

# (!!!) Returns ::Vector{String} of the simulation conditionIds 
# (conditions that appear as a simulationConditionId in the measurements table)
# cidx[1:Nc]
function _get_cids(PEmodel::PEtabModel)::Vector{String}
    PEtable       = PEmodel.petab_tables
    conditions_df = PEtable[:conditions]
    sim_set       = Set(string.(PEtable[:measurements][!, :simulationConditionId]))
    return [string(c) for c in conditions_df[!, :conditionId] if string(c) in sim_set]
end

# Returns the list of simulation conditions (index 1:Nc) appended by unique pre-eqbm conditions
# (if there are no pre-eqbm conditions, then this is same as _get_cids)
function _get_cv_cids(PEmodel::PEtabModel, PEprob::PEtabODEProblem)::Vector{String}
    sim_cids = _get_cids(PEmodel)
    si = PEprob.model_info.simulation_info
    preeq = si.has_pre_equilibration ? unique(string.(si.conditionids[:pre_equilibration])) : String[]
    preeq_cids = [cid for cid in preeq if !(cid in sim_cids)]
    return [sim_cids; preeq_cids]
end

# Returns the row for the condition-dependent variables for a given simulation condition
function _get_cv_cid_rows(PEmodel::PEtabModel, PEprob::PEtabODEProblem)::Vector{Int}
    conditions_df = PEmodel.petab_tables[:conditions]
    dict_cid_row = Dict(
        string(conditions_df[row, :conditionId]) => row 
        for row in 1:size(conditions_df, 1)
    )
    return [dict_cid_row[cid] for cid in _get_cv_cids(PEmodel, PEprob)]
end

# Strip the MTK (t) from a variable, returning the bare symbol (e.g. x(t) -> x).
# Module-scope so both _assignment_substitutor and _rule_table can use them.
_strip_t(s) = Symbolics.Num(Symbolics.variable(Symbol(split(string(s), "(")[1])))
# Rewrite every variable in an expression to its bare (no (t)) form.
_rebare(e) = (vs = collect(Symbolics.get_variables(e));
            isempty(vs) ? e : Symbolics.substitute(e, Dict(v => _strip_t(v) for v in vs)))

# Automatically substitutes/applies in "assignment rules" (MTK "observed" variable expressions)
# into existing expressions until only the core {p,z,cv,...} (ExaModels-scope variables) remain
function _assignment_substitutor(PEprob::PEtabODEProblem; remove_t::Bool)
    sys    = PEprob.model_info.model.sys
    z_syms = _get_z_syms(PEprob)

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
# (basically the same as _assignment_substitutor but for the SBML variables)
function _rule_table(PEprob::PEtabODEProblem)
    sys    = PEprob.model_info.model.sys
    z_syms = _get_z_syms(PEprob)
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
    return ids, lhs, rhs, is_flat
end

# Parse the parameters PEtab file and return mapping: observable/noise parameters properly scaled
# Dict(:{bare symbol} => Float64)
function _get_table_fixed_vals(PEmodel::PEtabModel, PEprob::PEtabODEProblem)::Dict{Any,Any}
    params_df = PEmodel.petab_tables[:parameters]
    estimated = Set(string.(PEprob.xnames))                                  # decision vars p
    cv_names  = Set(string(Symbolics.value(s)) for s in _get_cv_syms(PEmodel))   # condition vars
    table_fixed_vals = Dict{Any,Any}()
    for row in eachrow(params_df)
        # for each row in the :parameters DataFrame...
        pid = string(row[:parameterId])
        (pid in estimated || pid in cv_names || occursin("__parameter_ifelse", pid)) && continue
        nv  = row[:nominalValue]
        (ismissing(nv) || (nv isa AbstractString && isempty(strip(nv)))) && continue
        val = nv isa AbstractString ? tryparse(Float64, strip(nv)) : Float64(nv)
        val === nothing && continue
        table_fixed_vals[Symbolics.value(Symbolics.variable(Symbol(pid)))] = val
    end
    
    return table_fixed_vals
end

# Returns a mapping: parameters that are fixed numeric values => its numeric value
# as well as PEtab initialAssignment parameters and their numeric values
function _get_dict_fixed_val(PEmodel::PEtabModel, PEprob::PEtabODEProblem)
    dict_all_val = Dict(PEprob.model_info.model.parametermap)
    defaults = Dict{Any,Any}(dict_all_val)

    # Recursively substitute initialAssignment expressions until it is just
    # a numeric value
    for _ in 1:100
        all(Symbolics.value(v) isa Number for v in values(defaults)) && break
        for (k, v) in defaults
            Symbolics.value(v) isa Number && continue
            defaults[k] = Symbolics.substitute(v, defaults)
        end
    end

    # The symbolics variables which we know are fixed
    fixed_syms = setdiff(keys(dict_all_val), union(_get_p_syms(PEprob), _get_cv_syms(PEmodel)))
    dict_fixed_val = Dict{Any,Any}()
    # Create the dictionary mapping these fixed variables to their values
    for sym in fixed_syms
        rv = Symbolics.value(defaults[sym])
        dict_fixed_val[sym] = rv isa Number ? Float64(rv) : dict_all_val[sym]
    end

    return dict_fixed_val
end

# (!!!) Returns the piecewise(time) gating parameters (__parameter_ifelseN) 
# (SBMLImporter rewrites piecewise(time>T, …) into a MTK parameters updated by discrete_events block at t=T)
function _get_gate_syms(PEprob::PEtabODEProblem)::Vector{Symbolics.Num}
    sys = PEprob.model_info.model.sys
    return Symbolics.Num[
        Symbolics.Num(pp) 
        for pp in MTK.parameters(sys) if occursin("__parameter_ifelse", string(pp))
    ]
end

# Returns a vector (indexed by v=1:Nz) for each ODE RHS equation, f[v]([z[:,i,k,cidx]; p; cv[:,cidx]; gates; t]...)
function _get_rhs_funcs(PEmodel::PEtabModel, PEprob::PEtabODEProblem)
    sys = PEprob.model_info.model.sys

    # Get the ODE RHS function expressions in its purest symbolic form
    f_exprs_raw = [eqn.rhs for eqn in MTK.equations(sys)]

    # Get dictionary mapping for model parameters which are fixed numeric values
    dict_fixed_val = _get_dict_fixed_val(PEmodel, PEprob)

    # Get rid of the gate_syms from the fixed value dictionary mapping
    gate_syms = _get_gate_syms(PEprob)
    gate_names = Set(string.(gate_syms))
    for k in collect(keys(dict_fixed_val))
        string(k) in gate_names && delete!(dict_fixed_val, k)
    end

    # Obtains assignment rules for MTK eliminated variables (MTK.observed() variables)
    # (until they are expressed explicilty in terms of z,p,cv,gate_vals,time, or other numeric values)
    subst_rules = _assignment_substitutor(PEprob; remove_t = false)

    # Substitutes (in-lines) in expressions and numeric values until the ODE RHS expression
    # is only left with the ExaModels decision variables or other known numeric values
    f_exprs = [
        Symbolics.substitute(subst_rules(f_raw), dict_fixed_val)
        for f_raw in f_exprs_raw
    ]

    # Handles u(t), u(t,p): if t appears, then we parse it as its own variable and give it its own
    # input in the ODE RHS function so we can simply evaluate it at every collocation point
    all_free = foldl(union, Symbolics.get_variables.(f_exprs))
    t_basic  = nothing
    for v in all_free
        if string(v) == "t"
            t_basic = v
            break
        end
    end
    t_sym = t_basic !== nothing ? Symbolics.Num(t_basic) : Symbolics.Num(Symbolics.variable(:t))

    # Every input (variables which may appear) of the ODE RHS function
    z_syms   = _get_z_syms(PEprob)
    p_syms   = _get_p_syms(PEprob)
    cv_syms  = _get_cv_syms(PEmodel)
    all_syms = [z_syms; p_syms; cv_syms; gate_syms; [t_sym]]

    # Build the numeric function for every ODE RHS function expression
    return [
        Symbolics.build_function(
            f_expr, 
            all_syms...,
            expression = Val{false}
        )
        for f_expr in f_exprs
    ]
end

# Returns t_events, times at which a fixed-time event occurs, so we can force the initial solve
# for the mesh generation to include interval nodes at every event
function _get_event_times(PEmodel::PEtabModel, PEprob::PEtabODEProblem)::Vector{Float64}
    gate_syms = _get_gate_syms(PEprob)
    isempty(gate_syms) && return Float64[] # if there are no gate variables, return nothing
    
    # Extract ODE solving info from PEtab
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

    # for every simulation condition...
    for cid in Symbol.(_get_cids(PEmodel))
        cond_arg = cid
        if has_preeq
            # if x0SSpre, simulate the ODE with the correct pre-eqbm conditions
            pos = findfirst(==(cid), sim_ids); pos === nothing && continue
            cond_arg = preeq_ids[pos] => sim_ids[pos]
        end

        # create and initialize the ODEproblem to obtain gate value profile
        oprob, cbs = PEtab.get_odeproblem(p_nominal, PEprob; condition = cond_arg)
        integ = ODE.init(oprob, solver; callback = cbs, abstol = abstol, reltol = reltol)
        cur   = read_gates(integ)
        t_end  = oprob.tspan[2]

        # sweep the solution profile to identify at which times the gate values change
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

# Returns the gate values for every time interval and every condition gate_vals[g=1:Ng,i,cidx] 
# and steady-state ODE value gate_vals_ss[g,cidx]
function _get_gate_vals(
        PEmodel::PEtabModel,
        PEprob::PEtabODEProblem,
        h::Vector{Float64},
        taus::Vector{Float64},
        t_nodes::Vector{Float64}
    )
    # Get problem info
    gate_syms = _get_gate_syms(PEprob)  # symbolic gate variables
    Ng   = length(gate_syms)            # Number of gate variables
    cids = Symbol.(_get_cids(PEmodel))  # simulation conditionIds
    Nc   = length(cids)                 # number of simulation conditions
    N    = length(h)                    # number of intervals
    gate_vals    = zeros(Float64, Ng, N, Nc)    
    gate_vals_ss = zeros(Float64, Ng, Nc)
    Ng == 0 && return gate_vals, gate_vals_ss # If there are no gate variables, return nothing

    # Extract ODE solving info from PEtab
    si        = PEprob.model_info.simulation_info
    has_preeq = si.has_pre_equilibration
    sim_ids   = si.conditionids[:simulation]
    preeq_ids = si.conditionids[:pre_equilibration]
    p_nominal = PEtab.get_x(PEprob)
    solver    = PEprob.probinfo.solver.solver
    abstol    = PEprob.probinfo.solver.abstol
    reltol    = PEprob.probinfo.solver.reltol

    # Keep track of useful time points
    bnds       = t_nodes[2:end]                                 # mesh nodes (interval right-side endpoints)
    t_interior = [t_nodes[i] + taus[2] * h[i] for i in 1:N]     # strictly inside interval i (past left node)
    tstops     = sort(unique(vcat(t_interior, bnds)))           # stop at every interior sample and node

    gate_raw = [Symbolics.value(g) for g in gate_syms]          # unwrap Num for the .ps[] indexing interface
    read_gates(integ) = Float64[Float64(integ.ps[g]) for g in gate_raw]

    # for every simulation condition...
    for (cidx, cid) in enumerate(cids)
        cond_arg = cid
        if has_preeq
            # if x0SSpre, simulate the ODE with the correct pre-eqbm conditions
            pos = findfirst(==(cid), sim_ids)
            pos === nothing && continue
            cond_arg = preeq_ids[pos] => sim_ids[pos]
        end

        # create and initialize the ODEproblem to obtain gate value profile
        oprob, cbs = PEtab.get_odeproblem(p_nominal, PEprob; condition = cond_arg)
        t_end  = max(maximum(bnds), oprob.tspan[2]) # the last time point in integration
        oprob = ODE.remake(oprob; tspan = (oprob.tspan[1], t_end))
        integ = ODE.init(oprob, solver; callback = cbs, tstops = tstops, abstol = abstol, reltol = reltol)

        g0        = read_gates(integ) # gate at the (post-init) initial condition, t=0
        cur       = copy(g0)
        change_ts = Float64[]
        for i in 1:N
            # sweep the solution profile over every interval to see how the gate_vals change
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

        # Make sure every t_event hits an interval node in the mesh
        for tc in change_ts
            any(==(tc), t_nodes) || error(
                "Condition '$cid' has a fixed-time event at t=$tc not on a collocation mesh node."
            )
        end

        # initial gate value for steady-state models is simply the gate value at t=0
        gate_vals_ss[:, cidx] = g0
    end

    return gate_vals, gate_vals_ss
end

# Returns the steady-state gate values for the steady-state model path (no collocation mesh)
# (essentially just a constant numeric value, extracted from PEtab ODEProblem at t=0)
function _get_gate_vals_ss(PEmodel::PEtabModel, PEprob::PEtabODEProblem)
    gate_syms = _get_gate_syms(PEprob)
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
    # for every simulation condition...
    for (cidx, cid) in enumerate(cids)
        # extract the gate values at the initial step
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

# Returns a vector indexed by cidx (1:Nc) => index of the steady-state pre-eqbm condition (sscidx)
function _get_dict_cidx_sscidx(PEmodel::PEtabModel, PEprob::PEtabODEProblem)::Vector{Int64}
    cids        = _get_cids(PEmodel)            # simulation conditions, cv cols 1:Nc
    cv_cids = _get_cv_cids(PEmodel, PEprob) # [sim; distinct extra pre-eq]
    pos_of      = Dict(cv_cids[i] => i for i in eachindex(cv_cids))
    sim_ids = PEprob.model_info.simulation_info.conditionids[:simulation]
    ssc_ids = PEprob.model_info.simulation_info.conditionids[:pre_equilibration]
    # Map every simulation condition (cidx) => cv column of the pre-eqbm condition
    return map(eachindex(cids)) do cidx
        sim_idx = findfirst(==(Symbol(cids[cidx])), sim_ids) # which simulation condition index
        sim_idx === nothing && return cidx # if not a pre-eqbm condition, cidx = sscidx
        get(pos_of, string(ssc_ids[sim_idx]), cidx)
    end
end

# True if the model uses steady-state pre-equilibration (x0SSpre) initial conditions
function _check_x0SSpre(PEprob::PEtabODEProblem)::Bool
    return PEprob.model_info.simulation_info.has_pre_equilibration
end

# Returns dictionary: condition id (::String) => condition index, cidx in [Nc] (::Int64)
function _get_dict_cid_cidx(PEmodel::PEtabModel)::Dict{String, Int64}
    return Dict(
        cid => cidx
        for (cidx, cid) in enumerate(_get_cids(PEmodel))
    )
end

# Returns dictionary: floating point time (::Float64) => index which corresponds to that time in the mesh (::Int64)
function _get_dict_t_tidx(t_nodes::AbstractVector, t_meas)::Dict{Float64, Int64}
    d = Dict{Float64, Int64}(0.0 => 0) # at t=0.0, tidx=0
    # for every time point in the t_stops...
    for t_data in t_meas
        k = findfirst(==(t_data), t_nodes) # find at which index 'k' t_nodes[k] matches this t_data
        k === nothing && error(
            "measurement time t=$t_data is not a mesh node — it should have been " *
            "forced as a solver tstop in _get_z_init. (t_nodes range $(first(t_nodes))..$(last(t_nodes)).)"
        )
        d[t_data] = k - 1 # store the time to index mapping in the Dict. shifted by 1 because k in 0:K
    end
    return d
end