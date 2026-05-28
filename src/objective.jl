#######################################################
# STATE AS OF: 05/28/26
# TODO: verify implementation
# TODO: resolve idx+1 issue for itr_y_func
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
    (; Np, Ncv, Nz, Nm, Ny, N, K, t_meas, h, L1) = PEinfo
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
    dict_cid_cidx       = _get_dict_cid_cidx(PEmodel)
    dict_t_tidx         = _get_dict_t_tidx(h, t_meas)

    # Substitute in fixed constant values
    dict_all_val = Dict(PEprob.model_info.model.parametermap) # Mapping: symbolics of all parameters => nominal values
    fixed_syms = setdiff( # Symbolics of fixed constants
        keys(Dict(dict_all_val)), 
        union(_get_p_syms(PEprob), _get_cv_syms(PEmodel))
    )
    dict_fixed_val = Dict(sym => val for (sym,val) in dict_all_val if (sym in fixed_syms)) # Mapping: symbolics of fixed constants => values

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
    itr_sigma_fix   = Tuple{Int, Float64}[]
    itr_sigma_p     = Tuple{Int, Int}[]
    itr_sigma_ppy   = Tuple{Int, Int, Int}[]

    for yidx in 1:Ny
        # For every observable...
        row_obs     = observables_df[yidx,:]
        obs_id      = row_obs[:observableId]

        # Parse observableFormula
        obs_expr_raw = row_obs[:observableFormula]

        # Parse noiseFormula
        sigma_expr_raw = row_obs[:noiseFormula]

        if isempty(obs_expr_raw)
            error("Missing observableFormula for observableId: $obs_id")
        elseif isempty(sigma_expr_raw)
            error("Missing noiseFormula for observableId: $obs_id")
        else
            # Find which midx(s) correspond to this obs_id
            obs_midxs = [
                midx for midx in 1:Nm
                if string(measurements_df[midx, :observableId]) == obs_id
            ]

            begin # observableFormula parsing
                # If obs_expr contains observableParameter${n}_${obs_id}, then lookup
                # measurements_df indexed at this obs_id, replace with entries in column :observableParameters
                # this will be the new obs_expr_sub.
                meas_rows = filter(r -> r[:observableId] == obs_id, eachrow(measurements_df))
                obs_params_str = ""
                for row in meas_rows
                    v = row[:observableParameters]
                    if !ismissing(v) && !isempty(string(v))
                        obs_params_str = string(v)
                        break
                    end
                end

                # Replace observableParameter${n}_${obs_id} with the n-th semicolon-delimited
                # entry, which is itself a parameter ID or numeric literal.
                obs_expr_sub = obs_expr_raw
                if !isempty(obs_params_str)
                    replacements = strip.(split(obs_params_str, ";"))
                    replace_pairs = [
                        "observableParameter$(n)_$(obs_id)" => replace 
                        for (n, replace) in enumerate(replacements)
                    ]
                    obs_expr_sub = replace(obs_expr_sub, replace_pairs...)
                end

                # Now parse the observableFormula string
                parsed = Meta.parse(obs_expr_sub)
                if parsed isa Symbol
                    # If the observableFormula is a state variable, directly parse it as a Num
                    obs_expr = Symbolics.Num(Symbolics.variable(parsed))
                    
                    # Since y = z, create itr_y_z for this obs_id
                    zidx = findfirst(x -> isequal(x, obs_expr), z_syms) 
                    append!(itr_y_z, obs_midxs)
                    
                    for midx in obs_midxs
                        # For every measurement... (add these as well)
                        row_meas = measurements_df[midx,:]
                        cid = string(row_meas[:simulationConditionId]) # ::String
                        time = Float64(row_meas[:time]) # ::Float64
                        cidx = dict_cid_cidx[cid]
                        idx = dict_t_tidx[time]
                        append!(itr_y_z!, (midx, zidx, idx, cidx, j, L1[j+1]) for j in 0:K)
                    end    
                else
                    # Otherwise, the formula is an arbitrary expression, so parse it as an expression tree
                    obs_expr_parsed = Symbolics.parse_expr_to_symbolic(parsed, @__MODULE__)
                    obs_expr = Symbolics.substitute(obs_expr_parsed, dict_fixed_val)
                    obs_func = Symbolics.build_function(
                        obs_expr,
                        [z_syms; p_syms; cv_syms]...,
                        expression = Val{false}
                    )

                    # create itrs based on this obs_id => midx mapping. create util, midxes of obs_ids
                    # within each func, create itrs again so just make the @add_con in here
                    itr_y_func = Tuple{Int, Int, Int}[]
                    for midx in obs_midxs
                        # For every measurement... (add these as well)
                        row_meas = measurements_df[midx, :]
                        cid  = string(row_meas[:simulationConditionId])
                        time = Float64(row_meas[:time])
                        cidx = dict_cid_cidx[cid]
                        idx  = dict_t_tidx[time]
                        push!(itr_y_func, (midx, idx, cidx))
                    end
                    ExaModels.@add_con(c,
                        y[midx] - obs_func(
                            ntuple(v -> z[v,idx+1,0,cidx], Nz)..., # use i+1 instead of sum Lj zij
                            ntuple(m -> p[m], Np)...,
                            ntuple(m -> cv[m,cidx], Ncv)
                        )
                        for (midx, idx, cidx) in itr_y_func
                    )
                end
            end
            begin # noiseFormula parsing
                # Check if noiseFormula is a fixed numeric value
                sigma_val_raw = tryparse(Float64, strip(sigma_expr_raw))
                if sigma_val_raw !== nothing
                    for midx in obs_midxs
                        push!(itr_sigma_fix, (midx, sigma_val_raw))
                    end
                else
                    # If not, noiseFormula must have noiseParameters
                    first_noise_params_str = ""
                    for row in meas_rows
                        v = row[:noiseParameters]
                        if !ismissing(v) && !isempty(string(v))
                            first_noise_params_str = string(v)
                            break
                        end
                    end
                    sigma_expr_probe = sigma_expr_raw
                    if !isempty(first_noise_params_str)
                        replacements = strip.(split(first_noise_params_str, ";"))
                        replace_pairs = ["noiseParameter$(n)_$(obs_id)" => repl for (n, repl) in enumerate(replacements)]
                        sigma_expr_probe = replace(sigma_expr_raw, replace_pairs...)
                    end

                    if tryparse(Float64, strip(sigma_expr_probe)) !== nothing
                        # Case B: noiseParameter placeholder resolves to a numeric per-row —
                        # values may differ across rows so must be resolved individually
                        for midx in obs_midxs
                            row_meas = measurements_df[midx, :]
                            noise_params_row = string(row_meas[:noiseParameters])
                            replacements_row = strip.(split(noise_params_row, ";"))
                            replace_pairs_row = ["noiseParameter$(n)_$(obs_id)" => repl for (n, repl) in enumerate(replacements_row)]
                            sigma_expr_row = replace(sigma_expr_raw, replace_pairs_row...)
                            push!(itr_sigma_fix, (midx, parse(Float64, strip(sigma_expr_row))))
                        end
                    else
                        # Cases C/D: noiseParameters resolves to parameter IDs — same across all
                        # rows for this observable, so the probe substitution is authoritative
                        sigma_parsed = Meta.parse(sigma_expr_probe)
                        sigma_expr_sym = if sigma_parsed isa Symbol
                            Symbolics.Num(Symbolics.variable(sigma_parsed))
                        else
                            Symbolics.parse_expr_to_symbolic(sigma_parsed, @__MODULE__)
                        end
                        sigma_expr_final = Symbolics.substitute(sigma_expr_sym, dict_fixed_val)

                        sigma_free   = Symbolics.get_variables(sigma_expr_final)
                        sigma_p_vars = filter(v -> any(isequal(v, p) for p in p_syms), sigma_free)
                        sigma_z_vars = filter(v -> any(isequal(v, z) for z in z_syms), sigma_free)

                        if length(sigma_p_vars) == 1 && isempty(sigma_z_vars)
                            # Form C: sigma = p[pidx]
                            pidx = findfirst(p -> isequal(p, only(sigma_p_vars)), p_syms)
                            for midx in obs_midxs
                                push!(itr_sigma_p, (midx, pidx))
                            end
                        elseif length(sigma_p_vars) == 2
                            # Form D: sigma = p[pidx1] + p[pidx2]*y[midx]
                            dict_z_zero = Dict(z => 0.0 for z in z_syms)
                            p1_expr = Symbolics.substitute(sigma_expr_final, dict_z_zero)
                            pidx1   = findfirst(p -> isequal(p, p1_expr), p_syms)
                            p2_sym  = only(filter(v -> !isequal(v, p_syms[pidx1]), sigma_p_vars))
                            pidx2   = findfirst(p -> isequal(p, p2_sym), p_syms)
                            for midx in obs_midxs
                                push!(itr_sigma_ppy, (midx, pidx1, pidx2))
                            end
                        else
                            error("Unrecognized noiseFormula form for observableId: $obs_id — got \"$sigma_expr_probe\"")
                        end
                    end
                end
            end
        end
    end

    # Create auxiliary variable constraints for fixed values
    if !isempty(itr_y_z)
        # Observable 'y' is a state variable 'z'
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
    if !isempty(itr_sigma_fix)
        # Measurement error 'sigma' is a fixed value
        ExaModels.@add_con(c,
            sigma[midx] - val
            for (midx, val) in itr_sigma_fix
        )
    end
    if !isempty(itr_sigma_p)
        # Measurement error 'sigma' is an unknown parameter 'p'
        ExaModels.@add_con(c,
            sigma[midx] - p[pidx]
            for (midx, pidx) in itr_sigma_fix
        )
    end
    if !isempty(itr_sigma_ppy)
        # Measurement error 'sigma' is an error model of the form p + p*y
        ExaModels.@add_con(c,
            sigma[midx] - (
                p[pidx1] + p[pidx2]*y[midx]
            )
            for (midx, pidx1, pidx2) in itr_sigma_fix
        )
    end

    return c
end