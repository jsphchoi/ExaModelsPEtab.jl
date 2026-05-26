#######################################################
# STATE AS OF: 05/25/26
# TODO: create _get_sigma_funcs (measurement error), _get_y_funcs (model observable)
#######################################################

# Key: (!!!) := determines index -> variable ordering/mapping

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

# (!!!) Returns ::Vector{Symbolics.Num} of condition-dependent variables, cv
# cv[1:Ncv,cidx]
function _get_cv_syms(PEmodel::PEtabModel)::Vector{Symbolics.Num}
    PEtable = PEmodel.petab_tables # :measurements, :observables, :parameters, :conditions
    conditions_df = PEtable[:conditions] # DataFrame of conditions
    exclude = ["conditionId", "conditionName"] # Exclude non-cv columns
    cv_strings = [ # Variable names of condition-dependent (::String)
        string(str) for str in names(conditions_df) if !(str in exclude)
    ]
    return Symbolics.Num.(Symbolics.variable.(cv_strings)) # Converts variable name (::String) into symbolic variable (::Symbolics.Num)
end

# (!!!) Returns ::Vector{String} of conditionIds
# cidx[1:Nc]
function _get_cids(PEmodel::PEtabModel)::Vector{String}
    PEtable = PEmodel.petab_tables # :measurements, :observables, :parameters, :conditions
    conditions_df = PEtable[:conditions] # DataFrame of different conditions and properties for each condition
    return conditions_df[!,:conditionId]
end

# (!!!) Returns ::Vector{String} of observableIds
# "y","[1:Ny]"
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
# Returns dictionary: condition id (::String) => condition index, cidx in [Nc] (::Int64)
function _get_dict_cid_cidx(PEmodel::PEtabModel)::Dict{String, Int64}
    return Dict(
        cid => cidx
        for (cidx, cid) in enumerate(_get_cids(PEmodel))
    )
end

# Returns dictionary: time of measurement (::Float64) => interval index, i in [N] (::Int64)
function _get_dict_t_tidx(h,t_meas)::Dict{Float64, Int64}
    return Dict(
        t_data => findfirst(x -> isapprox(x,t_data; rtol = 1e-10), cumsum(h)) 
        for t_data in t_meas
    )
end

# Returns dictionary: observableId (::String) => observableExpression (::Num)
function _get_dict_obsid_obsexpr(PEmodel::PEtabModel)
    PEtable = PEmodel.petab_tables # :measurements, :observables, :parameters, :conditions
    observables_df = PEtable[:observables] # :observableId, :observableName, :observableFormula, :noiseFormula, :observableTransformation, :noiseDistribution
    return Dict(
        obsid => begin
            isempty(obsexpr) &&
                error("Empty observableFormula for observableId = $obsid")

            parsed = Meta.parse(obsexpr)
            if parsed isa Symbol
                # if the formula is a single variable, directly parse it as a Num
                Symbolics.Num(Symbolics.variable(parsed))
            else
                # else the formula is a general expression so parse as an expression tree
                Symbolics.parse_expr_to_symbolic(parsed, @__MODULE__)
            end
        end
        for (obsid, obsexpr) in zip(
            observables_df.observableId,
            observables_df.observableFormula
        )
    )
end

# Returns dictionary: observableId (::String) => yidx, i in [Ny] (::Int64) observable function index
function _get_dict_obids_yidx(PEmodel::PEtabModel)::Dict{String, Int64}
    return Dict(
        obsid => yidx for (yidx, obsid) in enumerate(_get_obsids(PEmodel))
    )
end


# TODO: instead just loop and make function in iterator.
# Returns ::Vector{(Function)} of observable variable equations, y
# yf[yidx=1:Ny]([z[:,i,k,cidx]; p[:]; cv[:,cidx]]...)
function _get_y_funcs(PEmodel::PEtabModel, PEprob::PEtabODEProblem)
    # Get symbolic observable variable expressions
    PEtable = PEmodel.petab_tables # :measurements, :observables, :parameters, :conditions
    observables_df = PEtable[:observables]

    y_exprs_raw = [ # Vector of raw ::String in DataFrame column observableFormula
        begin
            idx = findfirst(==(obsid), observables_df.observableId)
            observables_df[idx, :observableFormula]
        end
        for obsid in _get_obsids(PEmodel)
    ]

    y_exprs_sym = [ # Parse raw ::String as Symbolics.Num expression
        Symbolics.parse_expr_to_symbolic(Meta.parse(raw_str), @__MODULE__)
        for raw_str in y_exprs_raw
    ]

    # Substitute in fixed constant values
    dict_all_val = Dict(PEprob.model_info.model.parametermap) # Mapping: symbolics of all parameters => nominal values
    fixed_syms = setdiff( # Symbolics of fixed constants
        keys(Dict(dict_all_val)), 
        union(_get_p_syms(PEprob), _get_cv_syms(PEmodel))
    )
    dict_fixed_val = Dict(sym => val for (sym,val) in dict_all_val if (sym in fixed_syms)) # Mapping: symbolics of fixed constants => values
    y_exprs = [ # Substitute fixed values
        Symbolics.substitute(y_expr_sym, dict_fixed_val)
        for y_expr_sym in y_exprs_sym
    ]

    # Convert expression into numeric function
    return [
        Symbolics.build_function(
            y_expr,
            [_get_z_syms(PEprob); _get_p_syms(PEprob); _get_cv_syms(PEmodel)]..., # TODO: find observableParameter variables and correct mapping!
            expression = Val{false}
        )
        for y_expr in y_exprs
    ]
end

# Returns ::Vector{(Function)} of observable variable noise equations, sigma
# sigmaf[yidx=1:Ny]([z[:,i,k,cidx]; p[:]; cv[:,cidx]]...)
function _get_sigma_funcs(PEmodel::PEtabModel, PEprob::PEtabODEProblem)
    # Get symbolic observable variable expressions
    PEtable = PEmodel.petab_tables # :measurements, :observables, :parameters, :conditions
    observables_df = PEtable[:observables]

    sigma_exprs_raw = [ # Vector of raw ::String in DataFrame column observableFormula
        begin
            idx = findfirst(==(obsid), observables_df.observableId)
            observables_df[idx, :observableFormula]
        end
        for obsid in _get_obsids(PEmodel)
    ]

    y_exprs_sym = [ # Parse raw ::String as Symbolics.Num expression
        Symbolics.parse_expr_to_symbolic(Meta.parse(raw_str), @__MODULE__)
        for raw_str in y_exprs_raw
    ]

    # Substitute in fixed constant values
    dict_all_val = Dict(PEprob.model_info.model.parametermap) # Mapping: symbolics of all parameters => nominal values
    fixed_syms = setdiff( # Symbolics of fixed constants
        keys(Dict(dict_all_val)), 
        union(_get_p_syms(PEprob), _get_cv_syms(PEmodel))
    )
    dict_fixed_val = Dict(sym => val for (sym,val) in dict_all_val if (sym in fixed_syms)) # Mapping: symbolics of fixed constants => values
    y_exprs = [ # Substitute fixed values
        Symbolics.substitute(y_expr_sym, dict_fixed_val)
        for y_expr_sym in y_exprs_sym
    ]

    # Convert expression into numeric function
    return [
        Symbolics.build_function(
            y_expr,
            [_get_z_syms(PEprob); _get_p_syms(PEprob); _get_cv_syms(PEmodel)]..., # TODO: find observableParameter variables and correct mapping!
            expression = Val{false}
        )
        for y_expr in y_exprs
    ]
    # TODO: observableTransform: "lin" => keep, "log" => ..., "log10" => ...
end
