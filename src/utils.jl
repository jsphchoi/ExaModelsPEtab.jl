#######################################################
# STATE AS OF: 05/28/26
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

# Returns (::Vector{Function}, ::Bool) where bool = has_t (ODE depends on time after substitution)
# Without time: f[v=1:Nz]([z[:,i,k,cidx]; p[:]; cv[:,cidx]]...)
# With time:    f[v=1:Nz]([z[:,i,k,cidx]; p[:]; cv[:,cidx]; t]...)
function _get_rhs_funcs(PEmodel::PEtabModel, PEprob::PEtabODEProblem)
    sys = PEprob.model_info.model.sys

    f_exprs_raw = [eqn.rhs for eqn in MTK.equations(sys)]

    # Substitute in fixed constant values
    dict_all_val  = Dict(PEprob.model_info.model.parametermap)
    fixed_syms    = setdiff(keys(Dict(dict_all_val)), union(_get_p_syms(PEprob), _get_cv_syms(PEmodel)))
    dict_fixed_val = Dict(sym => val for (sym,val) in dict_all_val if (sym in fixed_syms))

    # Identify u(t,p) inputs: observed vars that appear in the ODE but are NOT state variables.
    # get_variables returns Set{BasicSymbolic}; observed eq.lhs is also BasicSymbolic → isequal works.
    # z_syms are Num-wrapped; isequal(BasicSymbolic, Num) = true for same variable.
    z_syms  = _get_z_syms(PEprob)
    p_syms  = _get_p_syms(PEprob)
    cv_syms = _get_cv_syms(PEmodel)
    ode_free = foldl(union, Symbolics.get_variables.(f_exprs_raw))  # Set{BasicSymbolic}
    dict_utp = Dict(
        eq.lhs => eq.rhs
        for eq in MTK.observed(sys)
        if  any(isequal(eq.lhs, v) for v in ode_free) &&
           !any(isequal(eq.lhs, z) for z in z_syms)
    )

    f_exprs = [
        Symbolics.substitute(Symbolics.substitute(f_raw, dict_utp), dict_fixed_val)
        for f_raw in f_exprs_raw
    ]

    # Detect time-dependence: check if MTK's independent variable 't' is a free variable
    # after substituting u(t,p) inputs. Use string comparison — robust across Symbolics versions.
    all_free = foldl(union, Symbolics.get_variables.(f_exprs))
    t_basic  = nothing
    for v in all_free
        if string(v) == "t"
            t_basic = v
            break
        end
    end
    has_t = t_basic !== nothing
    t_sym = has_t ? Symbolics.Num(t_basic) : nothing

    all_syms = has_t ? [z_syms; p_syms; cv_syms; [t_sym]] : [z_syms; p_syms; cv_syms]

    return [
        Symbolics.build_function(f_expr, all_syms..., expression = Val{false})
        for f_expr in f_exprs
    ], has_t
end

# Returns ::Dictionary{} of p::String => p[pidx] index
function _get_dict_pstr_pidx(PEprob::PEtabODEProblem)::Dict{String, Int64}
    return Dict(pstr => pidx for (pidx,pstr) in enumerate(String.(PEprob.xnames)))
end

# Returns ::Dictionary{} of cidx => steady-state cidx
function _get_dict_cidx_sscidx(PEmodel::PEtabModel, PEprob::PEtabODEProblem)::Dict{Int64, Int64}
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
    return merge(
        Dict(0.0 => 1),
        Dict(
            t_data => findfirst(x -> isapprox(x, t_data; rtol = 1e-10), cumsum(h))
            for t_data in t_meas
        )
    )
end