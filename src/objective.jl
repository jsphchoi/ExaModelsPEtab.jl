#######################################################
# STATE AS OF: 05/20/26
# COMPLETE: >=1 experimental conditions
# NOTE: only supporting noiseDistribution = normal. Laplace noise nondifferentiable.
# TODO automatically transform observable scale, {lin, log, log10}
# TODO: turn objectiveFormula -> function
#######################################################

# (*) Main function for creating ExaModels objective function (*)
function _create_objective(
        c::ExaCore,
        PEmodel::PEtabModel,
        PEprob::PEtabODEProblem,
        PEinfo::PEInfo
    )
    ###############################################
    # Unpack problem info
    ###############################################
    (; Np, Ncv, Nz, Nm, N, K, t_meas, h, L1) = PEinfo
    z = c.z
    p = c.p
    y = c.y
    sigma = c.sigma
    if Ncv >= 1
        cv = c.cv
    end

    PEtable = PEmodel.petab_tables # :measurements, :observables, :parameters, :conditions
    measurements_df = PEtable[:measurements] # :observableId, :preequilibrationConditionId, :simulationConditionId, :measurement, :time, :observableParameters, :noiseParameters, :datasetId
    observables_df = PEtable[:observables] # :observableId, :observableName, :observableFormula, :noiseFormula, :observableTransformation, :noiseDistribution
    
    noiseParameters = measurements_df[!,:noiseParameters]

    ###############################################
    # Objective function
    ###############################################
    # Create objective function (negative log-likelihood): min ∑ₘ(y - yₘ)²/σ²ₘ
    itr_obj = [(midx, Float64(measurements_df[midx, :measurement])) for midx in 1:Nm]
    ExaModels.@add_obj(c,
        (y[midx] - ymeas)^2/sigma[midx]
        for (midx, ymeas) in itr_obj
    )
    
    ###############################################
    # Auxiliary variable constraints for y, sigma
    ###############################################
    
    # Utility mappings
    dict_cid_cidx   = _get_dict_cid_cidx(PEmodel)
    dict_t_tidx     = _get_dict_t_tidx(h, t_meas)
    dict_obsid_yidx = _get_dict_obsid_yidx(PEmodel)
    
    # Create iterators
    itr_y       =
    itr_sigma_fix   = Tuple{Int, Float64}[]
    itr_sigma_func  = Tuple{Int, Any}[]

    for midx in 1:Nm
        # Iterator for obvservable variable, y
        y_val = 

        # Iterator for observable error, sigma
        sigma_val = noiseParameter[midx]
        if sigma_val isa Number
            # If sigma is a numeric value...
            push!(itr_sigma_fix, (midx, sigma_val))
        elseif (pidx = findfirst(x -> isequal(x, val), p_syms)) !== nothing
            # If z0 is an unknown parameter p...
            append!(itr_z0_p, ((v, cidx, pidx) for cidx in 1:Nc))
        else
            # If sigma is an arbitrary function of TODO ???...

        end
    end

    # Create auxiliary variable constraints
    if !isempty(itr_y_z)
        # Observable 'y' is a state variable, 'z'
        aux_y_z = ExaModels.@add_con(c,
            -y[midx]
            for (midx) in itr_y_z
        )
        ExaModels.@add_con!(c,
            aux_y_z,
            midx => z[v,i,j,cidx]
            for (midx,i,j,cidx) in itr_y_z!
        )
    end
    # if !isempty(itr_y_func)
    #     # Observable 'y' is an arbitrary function
    #     for () in itr_y_func
    #         ExaModels.@add_con(c,
    #             y[midx] - y_func(
                    
    #             )
    #             for () in itr
    #         )
    #     end
    # end
    if !isempty(itr_sigma_fix)
        # Measurement error 'sigma' is a fixed value
        ExaModels.@add_con(c,
            sigma[midx] - val
            for (midx, val) in itr_sigma_fix
        )
    end
    # if !isempty(itr_sigma_func)
    #     # Measurement error 'sigma' is an arbitrary function, f(...)
    #     for () in itr_sigma_func
    #         ExaModels.@add_con(c,
    #             sigma[midx] - sigma_func(
    #                 
    #             )
    #             for 
    #         )
    #     end
    # end
    return c
end