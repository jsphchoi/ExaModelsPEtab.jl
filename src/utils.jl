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

# Returns ::Vector{(Function)} of ODE RHS equations
# f[v=1:Nz]([z[:,i,k,cidx]; p[:]; cv[:,cidx]]...)
function _get_rhs_funcs(PEmodel, PEprob)
    # Get symbolic ODE RHS expressions
    sys = PEprob.model_info.model.sys # ODESystem from PEprob
    f_exprs_raw = [ # Vector of raw symbolic ODE RHS expressions
        eqn.rhs for eqn in MTK.equations(sys)
    ]

    # Substitute in fixed constant values
    p_map = Dict(PEprob.model_info.model.parametermap) # Mapping: symbolics of all parameters => nominal values
    fixed_syms = setdiff( # Symbolics of fixed constants
        keys(Dict(p_map)), 
        union(_get_p_syms(PEprob), _get_cv_syms(PEmodel))
    )
    dict_fixed = Dict(sym => val for (sym,val) in p_map if (sym in fixed_syms)) # Mapping: symbolics of fixed constants => values
    f_exprs = [ # Substitute fixed values
        Symbolics.substitute(f_expr_raw, dict_fixed)
        for f_expr_raw in f_exprs_raw
    ]

    # Convert symbolic RHS expression into numeric function
    return [ 
        Symbolics.build_function(
            f_expr,
            [_get_zp_syms(PEprob); _get_cv_syms(PEmodel)]...,
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

# Returns ::Vector{(Function)} of ODE RHS equations
# f[v=1:Nz]([z[:,i,k,cidx]; p[:]; cv[:,cidx]]...)

# Returns ::Dictionary{} of obsid => ovfidx observable variable function index
function _get_dict_obsid_ovfidx(PEmodel::PEtabModel, PEprob::PEtabODEProblem)

    return
end