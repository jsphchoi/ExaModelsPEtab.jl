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
function _get_cv_syms(PEmodel::PEtabModel)
    PEtable = PEmodel.petab_tables # :measurements, :observables, :parameters, :conditions
    conditions_df = PEtable[:conditions] # DataFrame of conditions
    cond_var_strings = names(conditions_df)[3:end] # Variable names of condition-dependent (::String)
    return Symbolics.Num.(Symbolics.variable.(cond_var_strings)) # Converts variable name (::String) into symbolic variable (::Symbolics.Num)
end

# Returns ::Vector{Symbolics.Num} of state variables and unknown parameters
# [z[:,i,k,cidx]...; p[:]...]
function _get_zp_syms(PEprob::PEtabODEProblem)::Vector{Symbolics.Num}
    return [_get_z_syms(PEprob); _get_p_syms(PEprob)] 
end

# (!!!) Returns ::Vector{String} of conditionIds
function _get_cids(PEmodel::PEtabModel)
    PEtable = PEmodel.petab_tables # :measurements, :observables, :parameters, :conditions
    conditions_df = PEtable[:conditions] # DataFrame of different conditions and properties for each condition
    return conditions_df[!,:conditionId]
end