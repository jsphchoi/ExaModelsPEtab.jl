#
function _create_collocation(
    c::ExaCore,
    PEmodel::PEtabModel,
    PEprob::PEtabODEProblem,
    PEinfo::PEInfo
)
    # Unpack problem info
    (; N, K, Np, Nc, Nz, Ncv, h, taus) = PEinfo
    z = c.z
    p = c.p
    cv = c.cv

    # Get ODE RHS functions
    fs = _get_rhs_funcs(PEmodel, PEprob)

    # Constraint
    itr_coll = [(i,k,cidx,h[i]) for i in 1:N, k in 1:K, cidx in 1:Nc]
    c_coll = [
        ExaModels.@add_con(c,
            -hi*f( # inputs of the ODE RHS function
                ntuple(v -> z[v,i,k,cidx], Nz)...,
                ntuple(m -> p[m], Np)...,
                ntuple(m -> cv[m], Ncv)...
            )
            for (i,k,cidx,hi) in itr_coll
        )
        for f in fs
    ]
    
    # Constraint augmentation
    DLDTAU = [_eval_dldtau(j,k,taus) for j in 0:K, k in 1:K]
    itr_coll! = [(i,j,k,cidx,DLDTAU[j+1,k]) for i in 1:N, j in 0:K, k in 1:K, cidx in 1:Nc]
    for v in eachindex(c_coll)
        ExaModels.@add_con!(c, 
            c_coll[v],
            (i,k,cidx) => z[v,i,j,cidx]*DLDTAU
            for (i,j,k,cidx,DLDTAU) in itr_coll!
        )
    end

    # Condition-dependent variable constraints
    conditions_df = PEmodel.petab_tables[:conditions]
    dict_pstr_pidx = _get_dict_pstr_pidx(PEprob)
    # Create iterators
    itr_cv_fixed = Tuple{Int, Int, Float64}[]
    itr_cv_p = Tuple{Int, Int, Int}[]
    for cidx in 1:Nc
        for cvidx in 1:Ncv
            val = conditions_df[cidx,cvidx+2]
            if val isa Number
                push!(itr_cv_fixed, (cvidx, cidx, Float64(val)))
            elseif val isa String || val isa Symbol
                str_val = String(val)
                parsed_val = tryparse(Float64, str_val)
                if parsed_val !== nothing
                    push!(itr_cv_fixed, (cvidx, cidx, parsed_val))
                else
                    if haskey(dict_pstr_pidx, str_val)
                        pidx = dict_pstr_pidx[str_val]
                        push!(itr_cv_p, (cvidx, cidx, pidx))
                    else
                        error("Condition variable '$str_val' not found in unknown parameter list.")
                    end
                end
            end
        end
    end
    # Create constraints
    if !isempty(itr_cv_fixed)
        ExaModels.@add_con(c,
            cv[cvidx,cidx] - val
            for (cvidx, cidx, val) in itr_cv_fixed
        )
    end
    if !isempty(itr_cv_p)
        ExaModels.@add_con(c,
            cv[cvidx,cidx] - p[pidx]
            for (cvidx, cidx, pidx) in itr_cv_p
        )
    end

    return c
end