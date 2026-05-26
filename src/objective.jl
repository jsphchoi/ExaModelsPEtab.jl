#######################################################
# STATE AS OF: 05/26/26
# TODO: automatically transform observable scale, {lin, log, log10}
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
    # Parsed table values => ExaModels variable index mappings
    dict_obsid_obsexpr  = _get_dict_obsid_obsexpr(PEmodel)
    dict_cid_cidx       = _get_dict_cid_cidx(PEmodel)
    dict_t_tidx         = _get_dict_t_tidx(h, t_meas)

    # Symbolics of variables that may appear in parsed table formulas
    z_syms = [
        Symbolics.Num(Symbolics.variable(Symbol(split(string(z_sym), "(")[1])))
        for z_sym in _get_z_syms(PEprob)
    ]
    p_syms = _get_p_syms(PEprob)
    cv_syms = _get_cv_syms(PEmodel)
    
    # Create iterators
    itr_y_z         = Int[]
    itr_y_z!        = Tuple{Int, Int, Int, Int, Int, Float64}[]
    itr_y_func      = Tuple{Any}[]
    itr_y_func      = Tuple{Any, Int, Int, Int, Int, Float64}[]
    itr_sigma_fix   = Tuple{Int, Float64}[]
    itr_sigma_func  = Tuple{Int, Any}[]

    for midx in 1:Nm
        # For every measurement...
        row = measurements_df[midx,:]

        # Iterator for obvservable variable, y
        obsid = string(row[:observableId]) # ::String
        cid = string(row[:simulationConditionId]) # ::String
        time = Float64(row[:time]) # ::Float64
        obsexpr = dict_obsid_obsexpr[obsid]
        cidx = dict_cid_cidx[cid]
        idx = dict_t_tidx[time]
        if (zidx = findfirst(x -> isequal(x, obsexpr), z_syms)) !== nothing
            # If observable expression is a state variable 'z'...
            push!(itr_y_z, midx)
            append!(itr_y_z!, (midx, zidx, idx, cidx, j, L1[j+1]) for j in 0:K)
        else
            # If observable expression is an abstract function...
            # TODO
            # create 
        end

        
    end

    for yidx in 1:Ny
        # Iterator for observable error, sigma
        sigma_val = row[:noiseParameters] # TODO fix. SIGMA = INDEX WITH OBSID TO GET THE CORRECT NOISE FORMULA. A PRIORI CATEGORIZE NOISE
        if sigma_val isa Number
            # If sigma is a numeric value...
            push!(itr_sigma_fix, (midx, sigma_val))
        else
            # If sigma is an arbitrary function of TODO ???...
        end
    end

    # Create auxiliary variable constraints
    if !isempty(itr_y_z)
        # Observable 'y' is a state variable, 'z'
        con_y_z = ExaModels.@add_con(c,
            -y[midx]
            for midx in itr_y_z
        )
        ExaModels.@add_con!(c,
            con_y_z,
            midx => L1j*z[v,i,j,cidx]
            for (midx, v, i, cidx, j, L1j) in itr_y_z!
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
                    
    #             )
    #             for 
    #         )
    #     end
    # end
    return c
end