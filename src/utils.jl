#######################################################
# STATE AS OF: 05/20/26
# TODO: create _get_sigma_funcs (measurement error), _get_y_funcs (model observable)
#######################################################

# Key: (!!!) := determines index -> variable ordering/mapping

# (!!!) Returns ::Vector{Symbolics.Num} of state variables
# [z[:,i,k,cidx]...]
function _get_z_syms(PEprob::PEtabODEProblem)::Vector{Symbolics.Num}
    sys = PEprob.model_info.model.sys
    return MTK.unknowns(sys)
end

# (!!!) Returns ::Vector{Symbolics.Num} of unknown parameters
# [p[:]...]
function _get_p_syms(PEprob::PEtabODEProblem)::Vector{Symbolics.Num}
    return Symbolics.Num.(Symbolics.variable.(PEprob.xnames)) # Converts variable name (::String) into symbolic variable (::Symbolics.Num)
end

# (!!!) Returns ::Vector{Symbolics.Num} of condition-dependent variables, cv
# [cv[:,cidx]...]
function _get_cv_syms(PEmodel::PEtabModel)::Vector{Symbolics.Num}
    PEtable = PEmodel.petab_tables # :measurements, :observables, :parameters, :conditions
    conditions_df = PEtable[:conditions] # DataFrame of conditions
    cond_var_strings = names(conditions_df)[3:end] # Variable names of condition-dependent (::String)
    return Symbolics.Num.(Symbolics.variable.(cond_var_strings)) # Converts variable name (::String) into symbolic variable (::Symbolics.Num)
end

# (!!!) Returns ::Vector{String} of conditionIds
function _get_cids(PEmodel::PEtabModel)::Vector{String}
    PEtable = PEmodel.petab_tables # :measurements, :observables, :parameters, :conditions
    conditions_df = PEtable[:conditions] # DataFrame of different conditions and properties for each condition
    return conditions_df[!,:conditionId]
end

# Returns ::Vector{String} of observableIds
function _get_obsids(PEmodel::PEtabModel)::Vector{String}
    PEtable = PEmodel.petab_tables # :measurements, :observables, :parameters, :conditions
    observables_df = PEtable[:observables]
    return observables_df[!,:observableId]
end

# Returns ::Vector{(Function)} of ODE RHS equations
# f[v=1:Nz]([z[:,i,k,cidx]; p[:]; cv[:,cidx]]...)
function _get_rhs_funcs(PEmodel, PEprob)
    # Get symbolic ODE RHS expressions
    sys = PEprob.model_info.model.sys # ODESystem from PEprob
    f_exprs_raw = [ # Vector of raw symbolic ODE RHS expressions
        eqn.rhs for eqn in MTK.equations(sys)
    ]

    # Substitute in fixed constant values
    dict_all_val = Dict(PEprob.model_info.model.parametermap) # Mapping: symbolics of all parameters => nominal values
    fixed_syms = setdiff( # Symbolics of fixed constants
        keys(Dict(dict_all_val)), 
        union(_get_p_syms(PEprob), _get_cv_syms(PEmodel))
    )
    dict_fixed_val = Dict(sym => val for (sym,val) in dict_all_val if (sym in fixed_syms)) # Mapping: symbolics of fixed constants => values
    f_exprs = [ # Substitute fixed values
        Symbolics.substitute(f_expr_raw, dict_fixed_val)
        for f_expr_raw in f_exprs_raw
    ]

    # Convert symbolic RHS expression into numeric function
    return [ 
        Symbolics.build_function(
            f_expr,
            [_get_z_syms(PEprob); _get_p_syms(PEprob); _get_cv_syms(PEmodel)]...,
            expression = Val{false}
        )
        for f_expr in f_exprs
    ]
end

# Returns ::Dictionary{} of p::String => p[pidx] index
function _get_dict_pstr_pidx(PEprob::PEtabODEProblem)
    return Dict(pstr => pidx for (pidx,pstr) in enumerate(String.(PEprob.xnames)))
end

# Returns ::Dictionary{} of cidx => steady-state cidx
function _get_dict_cidx_sscidx(PEmodel::PEtabModel, PEprob::PEtabODEProblem)
    cids = _get_cids(PEmodel)
    sim_ids = PEprob.model_info.simulation_info.conditionids[:simulation]
    ssc_ids = PEprob.model_info.simulation_info.conditionids[:pre_equilibration]
    dict_cid_cidx = Dict(cids[i] => i for i in eachindex(cids))
    dict_cidx_sscidx = map(eachindex(cids)) do cidx
        sim_idx = findfirst(==(Symbol(cids[cidx])), sim_ids)
        dict_cid_cidx[string(ssc_ids[sim_idx])]
    end
    return dict_cidx_sscidx
end

####################################################
# UTILS FOR CHECKING MODEL FEATURES
####################################################
function _check_x0SSpre(PEprob::PEtabODEProblem)::Bool
    return PEprob.model_info.simulation_info.has_pre_equilibration
end
# check if steadysteate preeq x0 type (Claude Sonnet 4.6)
# function _check_x0SSpre(path_yaml::String)::Bool
#     petab_version = PEtab._get_version(path_yaml)

#     if petab_version == "1.0.0"
#         tables = PEtab.read_tables_v1(path_yaml)
#         measurements_df = tables[:measurements]
#         :preequilibrationConditionId in propertynames(measurements_df) || return false
#         return any(!ismissing, measurements_df.preequilibrationConditionId)
#     else
#         # v2: check experiments table for time == -Inf rows (pre-eq marker)
#         tables = PEtab.read_tables_v2(path_yaml)
#         experiments_df = tables[:experiments]
#         isempty(experiments_df) && return false
#         return any(isinf.(experiments_df.time) .& (experiments_df.time .< 0))
#     end
# end

####################################################
# UTILS FOR OBJECTIVE FUNCTION
####################################################
# Returns a dictionary mapping: condition id (::String) => condition index, cidx in [Nc] (::Int64)
function _cid_to_cidx(PEmodel::PEtabModel)
    return Dict(
        cid => cidx
        for (cidx, cid) in enumerate(_get_cids(PEmodel))
    )
end

# Returns a dictionary mapping: time of measurement (::Float64) => interval index, i in [N] (::Int64)
function _t_to_tidx(h,t_meas)
    return Dict(
        t_data => findfirst(x -> isapprox(x,t_data; rtol = 1e-10), cumsum(h)) 
        for t_data in t_meas
    )
end

# Returns ::Dictionary{} of obsid => ovfidx observable variable function index
function _get_dict_obsid_ovfidx(PEmodel::PEtabModel, PEprob::PEtabODEProblem)
    # TODO
    return
end

# get obs function
function _get_y_funcs(PEmodel::PEtabModel, PEprob::PEtabODEProblem, PEinfo::PEInfo)

    return
end

# get sigma function
function _get_sigma_funcs(PEmodel::PEtabModel, PEprob::PEtabODEProblem, PEinfo::PEInfo)

    return
end

####################################################
# UTILS FOR 
####################################################
function get_u0_all_experiments(x, prob::PEtabODEProblem)
    experiments_df = prob.model_info.model.petab_tables[:experiments]
    experiment_ids = Symbol.(unique(experiments_df.experimentId))

    u0_per_experiment = Dict{Symbol, Vector{Pair}}()
    for exp_id in experiment_ids
        u0_per_experiment[exp_id] = get_u0(x, prob; experiment = exp_id)
    end
    return u0_per_experiment
end

# Returns ::Dict{(condition id)::Symbol, (solution)}
function _get_dict_cid_z0expr(p_nominal, PEmodel::PEtabModel, PEprob::PEtabODEProblem)
    sols = Dict{Symbol, Any}()
    for cid in Symbol.(_get_cids(PEmodel))
        odesys, ~ = PEtab.get_odeproblem(p_nominal, PEprob; condition = cid)
    end
    return Dict(

    )
end

function get_u0_symbolic_per_condition(prob::PEtabODEProblem)
    model = prob.model_info.model
    conditions_df = model.petab_tables[:conditions]
    state_ids = PEtab._get_state_ids(model.sys_mutated)

    # Build string-name → Symbolics.Num lookup from the original (un-mutated) system
    ps_dict = Dict{String, Any}()
    if !(model.sys isa ODE.ODEProblem)
        for p in MTK.parameters(model.sys)
            ps_dict[replace(string(p), "(t)" => "")] = p
        end
        for s in MTK.unknowns(model.sys)
            ps_dict[replace(string(s), "(t)" => "")] = s
        end
    end

    # Base u0 from speciemap: state_id → symbolic or numeric default
    # Note: states that appear in conditions_df get an __init__xxx__ placeholder in
    # speciemap_problem (the mutated one). We use model.speciemap which is the original.
    base_u0 = Dict{String, Any}()
    if !isnothing(model.speciemap)
        for pair in model.speciemap
            sid = replace(string(first(pair)), "(t)" => "")
            val = last(pair)
            # If val is an __init__ symbolic parameter, skip (conditions table takes over)
            if val isa Symbolics.Num && occursin("__init__", string(val))
                base_u0[sid] = nothing  # will be filled from conditions table
            else
                base_u0[sid] = val
            end
        end
    end

    u0_per_condition = Dict{Symbol, Dict{Symbol, Any}}()

    for row in eachrow(conditions_df)
        cid = Symbol(row.conditionId)
        u0 = Dict{Symbol, Any}()

        # Fill defaults
        for sid in state_ids
            val = get(base_u0, sid, 0.0)
            u0[Symbol(sid)] = isnothing(val) ? 0.0 : val
        end

        # Apply condition-specific overrides
        for col in names(conditions_df)
            col in ("conditionId", "conditionName") && continue
            !(col in state_ids) && continue

            raw = row[col]
            ismissing(raw) && continue

            resolved = _resolve_condition_value(raw, ps_dict)
            isnothing(resolved) && continue  # NaN / pre-eq placeholder
            u0[Symbol(col)] = resolved
        end

        u0_per_condition[cid] = u0
    end

    return u0_per_condition
end

function _resolve_condition_value(val, ps_dict::Dict)
    # Numeric: keep as-is (NaN signals pre-eq, skip it)
    if val isa Real
        isnan(val) && return nothing
        return val
    end

    val isa String || return val   # already symbolic or other type

    # Pre-eq placeholder
    val == "NaN" && return nothing

    # Plain number encoded as string
    PEtab.is_number(val) && return parse(Float64, val)

    # Parameter or state name → look up symbolic variable
    haskey(ps_dict, val) && return ps_dict[val]

    # Symbolic expression string → try to parse (e.g. "2*k1")
    try
        expr = Meta.parse(val)
        # substitute known symbols
        return eval(expr)   # works if all names are in scope; otherwise return string
    catch
        return val   # fall back to string
    end
end

u0_map = get_u0_symbolic_per_condition(PEprob)
