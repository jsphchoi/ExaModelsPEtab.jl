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
    dict_obsid_yidx = _get_dict_obsid_yidx(PEmodel) # TODO, verify claude
    
    # Create iterators
    itr_y       =
    itr_sigma   =

    # Create auxiliary variable constraints
    if !isempty(itr_y_z)
        # Observable 'y' is a state variable, 'z'
        ExaModels.@add_con(c,

        )
    end
    if !isempty(itr_y_func)
        # Observable 'y' is an arbitrary function
        for () in itr_y_func
            ExaModels.@add_con(c,
                y[midx] - y_func(
                    
                )
                for () in itr
            )
        end
    end
    if !isempty(itr_sigma_fix)
        # Measurement error 'sigma' is a fixed value
        ExaModels.@add_con(c,
            sigma[midx] - val
            for (v, cidx, val) in itr_sigma_fix
        )
    end
    
    return c
end