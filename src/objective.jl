#
function _create_objective(
    c::ExaCore,
    PEmodel::PEtabModel,
    PEprob::PEtabODEProblem,
    PEinfo::PEInfo
)
    # Unpack problem info
    (; Np, Nz, N, K, t_meas, h, L1) = PEinfo

    # 1. Parse observables: obsid -> observableformula: is single state variable? is function? -> zidx : ovfidx.
    # :observableTransformation, :noiseDistribution, :noiseFormula

    # for each measurement, create iterators
    


    # Condition-dependent variable constraints
    conditions_df = PEmodel.petab_tables[:conditions]
    dict_pstr_pidx = _get_dict_pstr_pidx(PEprob)
    # Create iterators
    itr_ = Tuple{Int, Int, Float64}[]
    itr_ = Tuple{Int, Int, Int}[]

    # Create constraints
    if !isempty(itr_obj_z_square)
        # for observableFormula = state variable, just use obsid -> state id
        ExaModels.@add_obj(c,
            
            for () in itr_obj_z_square
        )
        ExaModels.@add_obj(c,
            
            for () in itr_obj_z_linear
        )
        # if error is not fixed,
        ExaModels.@add_con(c,
            # sigma[sidx] = sigmav[sixd](z,p,cv...)
        )
    end
    if !isempty(itr_obj_ov_square)
        # for observableFormula = function, created var ov. (meas_val - ov[meas_idx])
        ExaModels.@add_obj(c,
            
            for () in itr_obj_ov_square
        )
        ExaModels.@add_obj(c,
            
            for () in itr_obj_ov_linear
        )
        ExaModels.@add_con(c,
            # ov[ovidx] = ovf[ovidx](... idk yet)
        )
        # if error is not fixed,
        ExaModels.@add_con(c,
            # sigma[sidx] = sigmav[sixd](z,p,cv...)
        )
    end

    return c
end