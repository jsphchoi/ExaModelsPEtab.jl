#######################################################
# STATE AS OF: 05/29/26
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
    dict_cid_cidx = _get_dict_cid_cidx(PEmodel)
    dict_t_tidx   = _get_dict_t_tidx(h, t_meas)

    # Substitute in fixed constant values
    dict_all_val = Dict(PEprob.model_info.model.parametermap)
    fixed_syms = setdiff(
        keys(Dict(dict_all_val)),
        union(_get_p_syms(PEprob), _get_cv_syms(PEmodel))
    )
    dict_fixed_val = Dict(sym => val for (sym,val) in dict_all_val if (sym in fixed_syms))

    # Symbolics of variables that may appear in parsed table formulas
    z_syms = [
        Symbolics.Num(Symbolics.variable(Symbol(split(string(z_sym), "(")[1])))
        for z_sym in _get_z_syms(PEprob)
    ]
    p_syms  = _get_p_syms(PEprob)
    cv_syms = _get_cv_syms(PEmodel)

    # Fast lookup: observableId => row in observables_df
    dict_obsid_obsrow = Dict(
        string(observables_df[i, :observableId]) => observables_df[i, :]
        for i in 1:size(observables_df, 1)
    )

    # Helper: normalize a raw table cell to String (handle missing/nothing)
    _safe_str(v) = (ismissing(v) || isnothing(v)) ? "" : strip(string(v))

    ###############################################
    # Observable formula (y) constraints
    # Group measurements by (obsId, observableParameters) so that each unique
    # formula gets its own compiled ExaModels constraint.
    ###############################################
    itr_y_z  = Int[]
    itr_y_z! = Tuple{Int, Int, Int, Int, Int, Float64}[]

    obs_y_groups = Dict{Tuple{String,String}, Vector{Int}}()
    for midx in 1:Nm
        row     = measurements_df[midx, :]
        obs_id  = string(row[:observableId])
        obs_key = _safe_str(row[:observableParameters])
        push!(get!(obs_y_groups, (obs_id, obs_key), Int[]), midx)
    end

    for ((obs_id, obs_params_str), group_midxs) in obs_y_groups
        obs_expr_raw = string(dict_obsid_obsrow[obs_id][:observableFormula])

        # Substitute observableParameter${n}_${obs_id} placeholders with the
        # n-th semicolon-delimited entry from this group's observableParameters.
        obs_expr_sub = obs_expr_raw
        if !isempty(obs_params_str)
            parts        = strip.(split(obs_params_str, ";"))
            replace_pairs = ["observableParameter$(n)_$(obs_id)" => parts[n] for n in eachindex(parts)]
            obs_expr_sub = replace(obs_expr_sub, replace_pairs...)
        end

        parsed = Meta.parse(obs_expr_sub)

        if parsed isa Symbol
            # Observable is a single state variable: y[midx] = z[zidx, i, j, cidx]
            obs_sym = Symbolics.Num(Symbolics.variable(parsed))
            zidx    = findfirst(x -> isequal(x, obs_sym), z_syms)
            append!(itr_y_z, group_midxs)
            for midx in group_midxs
                row  = measurements_df[midx, :]
                cid  = string(row[:simulationConditionId])
                time = Float64(row[:time])
                cidx = dict_cid_cidx[cid]
                idx  = dict_t_tidx[time]
                append!(itr_y_z!, [(midx, zidx, idx, cidx, j, L1[j+1]) for j in 0:K])
            end
        else
            # Observable is an arbitrary expression: compile obs_func for this group
            obs_parsed_sym = Symbolics.parse_expr_to_symbolic(parsed, @__MODULE__)
            obs_expr_final = Symbolics.substitute(obs_parsed_sym, dict_fixed_val)
            obs_func = Symbolics.build_function(
                obs_expr_final,
                [z_syms; p_syms; cv_syms]...,
                expression = Val{false}
            )

            itr_y_func = Tuple{Int, Int, Int}[]
            for midx in group_midxs
                row  = measurements_df[midx, :]
                cid  = string(row[:simulationConditionId])
                time = Float64(row[:time])
                cidx = dict_cid_cidx[cid]
                idx  = dict_t_tidx[time]
                push!(itr_y_func, (midx, idx, cidx))
            end

            ExaModels.@add_con(c,
                y[midx] - obs_func(
                    ntuple(v -> z[v,idx+1,0,cidx], Nz)...,
                    ntuple(m -> p[m], Np)...,
                    ntuple(m -> cv[m,cidx], Ncv)
                )
                for (midx, idx, cidx) in itr_y_func
            )
        end
    end

    ###############################################
    # Noise formula (sigma) constraints
    # Group measurements by (obsId, noiseParameters) so that each unique
    # formula gets its own compiled ExaModels constraint.
    ###############################################
    itr_sigma_fix = Tuple{Int, Float64}[]  # sigma = numeric literal
    itr_sigma_p   = Tuple{Int, Int}[]      # sigma = p[pidx]

    obs_sigma_groups = Dict{Tuple{String,String}, Vector{Int}}()
    for midx in 1:Nm
        row        = measurements_df[midx, :]
        obs_id     = string(row[:observableId])
        noise_key  = _safe_str(row[:noiseParameters])
        push!(get!(obs_sigma_groups, (obs_id, noise_key), Int[]), midx)
    end

    for ((obs_id, noise_params_str), group_midxs) in obs_sigma_groups
        sigma_expr_raw = string(dict_obsid_obsrow[obs_id][:noiseFormula])

        # Substitute noiseParameter${n}_${obs_id} placeholders.
        sigma_expr_sub = sigma_expr_raw
        if !isempty(noise_params_str)
            parts        = strip.(split(noise_params_str, ";"))
            replace_pairs = ["noiseParameter$(n)_$(obs_id)" => parts[n] for n in eachindex(parts)]
            sigma_expr_sub = replace(sigma_expr_raw, replace_pairs...)
        end

        # Case A: substituted formula is a numeric literal
        sigma_val = tryparse(Float64, strip(sigma_expr_sub))
        if sigma_val !== nothing
            for midx in group_midxs
                push!(itr_sigma_fix, (midx, sigma_val))
            end
            continue
        end

        # Parse as symbolic expression
        sigma_parsed     = Meta.parse(sigma_expr_sub)
        sigma_parsed_sym = sigma_parsed isa Symbol ?
            Symbolics.Num(Symbolics.variable(sigma_parsed)) :
            Symbolics.parse_expr_to_symbolic(sigma_parsed, @__MODULE__)
        sigma_expr_final = Symbolics.substitute(sigma_parsed_sym, dict_fixed_val)

        sigma_free   = Symbolics.get_variables(sigma_expr_final)
        sigma_p_vars = filter(v -> any(isequal(v, pv) for pv in p_syms), sigma_free)

        if length(sigma_p_vars) == 1 && isempty(filter(v -> any(isequal(v, zv) for zv in z_syms), sigma_free))
            # Case B: sigma = p[pidx]
            pidx = findfirst(pv -> isequal(pv, only(sigma_p_vars)), p_syms)
            for midx in group_midxs
                push!(itr_sigma_p, (midx, pidx))
            end
        else
            # Case C: sigma is a general expression — compile sigma_func for this group
            sigma_func = Symbolics.build_function(
                sigma_expr_final,
                [z_syms; p_syms; cv_syms]...,
                expression = Val{false}
            )

            itr_sigma_func = Tuple{Int, Int, Int}[]
            for midx in group_midxs
                row  = measurements_df[midx, :]
                cid  = string(row[:simulationConditionId])
                time = Float64(row[:time])
                cidx = dict_cid_cidx[cid]
                idx  = dict_t_tidx[time]
                push!(itr_sigma_func, (midx, idx, cidx))
            end

            ExaModels.@add_con(c,
                sigma[midx] - sigma_func(
                    ntuple(v -> z[v,idx+1,0,cidx], Nz)...,
                    ntuple(m -> p[m], Np)...,
                    ntuple(m -> cv[m,cidx], Ncv)
                )
                for (midx, idx, cidx) in itr_sigma_func
            )
        end
    end

    ###############################################
    # Emit batched constraints for accumulated iterators
    ###############################################
    if !isempty(itr_y_z)
        # y[midx] = Σ_{j=0}^{K} L_j(1) * z[zidx, i, j, cidx]
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
        ExaModels.@add_con(c,
            sigma[midx] - val
            for (midx, val) in itr_sigma_fix
        )
    end

    if !isempty(itr_sigma_p)
        ExaModels.@add_con(c,
            sigma[midx] - p[pidx]
            for (midx, pidx) in itr_sigma_p
        )
    end

    return c
end
