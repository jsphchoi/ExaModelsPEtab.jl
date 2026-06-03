#######################################################
# STATE AS OF: 05/30/26
# TODO: verify implementation
# TODO: y0 initial guess just idx+1, if idx=N then use weighted sum.
#######################################################

# Reduce a noise (σ) expression so that its dependence on model states enters ONLY through
# the observable. Every PEtab benchmark noise form (c | θ | β·y | α+β·y | √(α²+(β·y)²)) is a
# function of the observable y and parameters, and the observable is affine in the states.
# We solve O = Y (the observable placeholder) for ONE observable-state and substitute it into
# σ; if σ's state dependence is genuinely only through O, every other state cancels. This is
# robust to Symbolics flattening `β*(state*param)` into one product (which defeats a direct
# substitute(O => Y)) and cancels any observableParameter coefficients that ride along.
# Returns (reduced_expr, ok) where ok=true means no model state remains in reduced_expr.
function _reduce_sigma_to_obs(sigma_expr, obs_expr, Y_sym, z_syms)
    has_z(e) = any(zv -> any(isequal(v, zv) for v in Symbolics.get_variables(e)), z_syms)
    obs_states = [zv for zv in z_syms
                  if any(isequal(v, zv) for v in Symbolics.get_variables(obs_expr))]
    isempty(obs_states) && return (sigma_expr, !has_z(sigma_expr))  # σ has no state via obs
    # require the observable affine in its states: each ∂O/∂sᵢ must itself be state-free
    coeffs = [Symbolics.expand_derivatives(Symbolics.Differential(s)(obs_expr)) for s in obs_states]
    any(has_z, coeffs) && return (sigma_expr, false)               # observable nonlinear in states
    a0     = Symbolics.substitute(obs_expr, Dict(s => 0 for s in obs_states))  # O at states=0
    s1, b1 = obs_states[1], coeffs[1]
    rest   = length(obs_states) > 1 ?
             sum(coeffs[i] * obs_states[i] for i in 2:length(obs_states)) : 0
    s1_sol  = (Y_sym - a0 - rest) / b1                              # solve O = Y for s₁
    reduced = Symbolics.expand(Symbolics.substitute(sigma_expr, Dict(s1 => s1_sol)))
    has_z(reduced) && (reduced = Symbolics.expand(Symbolics.simplify(reduced)))
    return (reduced, !has_z(reduced))
end

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
    (; Np, Ncv, Nz, Nc, Nm, Ny, N, K, t_meas, h, L1, pscale) = PEinfo
    z = c.z
    p = c.p
    y = c.y
    sigma = c.sigma
    if Ncv >= 1
        cv = c.cv
    end

    # ---- Warm-start support -------------------------------------------------
    # y and sigma are auxiliary variables defined entirely by z, p (and cv), which
    # already carry good PEtab initial guesses. We evaluate their defining formulas
    # at the initial point here so y/sigma can be given matching (feasible) starts
    # via set_start! after the model is built — critical for the IPM solver.
    z0  = reshape(_var_starts(c, z), Nz, N, K + 1, Nc) # z0[v, i, j+1, cidx]
    θ0  = _var_starts(c, p)                            # decision var p := θ (estimation scale)
    p0  = [_p_phys_val(θ0, m, pscale) for m in 1:Np]   # PHYSICAL parameter starts (10^θ)
    cv0 = Ncv >= 1 ? reshape(_var_starts(c, cv), Ncv, Nc) : zeros(Float64, 0, Nc)
    y0     = zeros(Float64, Nm) # computed observable values at the initial guess
    sigma0 = zeros(Float64, Nm) # computed noise (std) values at the initial guess

    PEtable = PEmodel.petab_tables # :measurements, :observables, :parameters, :conditions
    measurements_df = PEtable[:measurements] # :observableId, :preequilibrationConditionId, :simulationConditionId, :measurement, :time, :observableParameters, :noiseParameters, :datasetId
    observables_df = PEtable[:observables] # :observableId, :observableName, :observableFormula, :noiseFormula, :observableTransformation, :noiseDistribution

    ###############################################
    # Objective function
    ###############################################
    # Objective: Gaussian negative log-likelihood, matching PEtab.jl. The noise acts on
    # the observable's PEtab `observableTransformation` scale (lin/log/log10), so the
    # residual is taken in transformed space, with the change-of-variables Jacobian:
    #   lin   : 0.5(y-ymeas)²/σ²            + log σ + 0.5log2π
    #   log   : 0.5(ln y - ln ymeas)²/σ²    + log σ + 0.5log2π + ln ymeas
    #   log10 : 0.5(log10 y - log10 ymeas)²/σ² + log σ + 0.5log2π + ln ymeas + ln(ln10)
    # (= -logpdf(Normal/LogNormal/Log10Normal(transform(y), σ), ymeas)). y[midx],
    # sigma[midx] are the aux observable / noise-std vars bound to z,p by the constraints
    # below; ymeas is data so all the trailing terms are per-measurement constants.
    _assert_normal_noise(PEmodel)
    transforms = _get_meas_transforms(PEmodel)
    HALF_LOG2PI = 0.5 * log(2π)
    LN10        = log(10.0)
    itr_obj_lin   = Tuple{Int, Float64, Float64}[]  # (midx, ymeas,     const)
    itr_obj_log   = Tuple{Int, Float64, Float64}[]  # (midx, ln(ymeas), const)
    itr_obj_log10 = Tuple{Int, Float64, Float64}[]  # (midx, log10(ymeas), const)
    for midx in 1:Nm
        ymeas = Float64(measurements_df[midx, :measurement])
        tr    = transforms[midx]
        if tr === :lin
            push!(itr_obj_lin, (midx, ymeas, HALF_LOG2PI))
        elseif tr === :log
            @assert ymeas > 0 "log-transformed observable needs ymeas>0 (midx=$midx)"
            push!(itr_obj_log, (midx, log(ymeas), HALF_LOG2PI + log(ymeas)))
        elseif tr === :log10
            @assert ymeas > 0 "log10-transformed observable needs ymeas>0 (midx=$midx)"
            push!(itr_obj_log10, (midx, log10(ymeas), HALF_LOG2PI + log(ymeas) + log(LN10)))
        else
            error("Unsupported observableTransformation '$tr' (midx=$midx)")
        end
    end
    if !isempty(itr_obj_lin)
        ExaModels.@add_obj(c,
            0.5*(y[midx] - ymeas)^2/sigma[midx]^2 + log(sigma[midx]) + cst
            for (midx, ymeas, cst) in itr_obj_lin
        )
    end
    if !isempty(itr_obj_log)
        ExaModels.@add_obj(c,
            0.5*(log(y[midx]) - lnym)^2/sigma[midx]^2 + log(sigma[midx]) + cst
            for (midx, lnym, cst) in itr_obj_log
        )
    end
    if !isempty(itr_obj_log10)
        ExaModels.@add_obj(c,
            0.5*(log(y[midx])/LN10 - l10ym)^2/sigma[midx]^2 + log(sigma[midx]) + cst
            for (midx, l10ym, cst) in itr_obj_log10
        )
    end

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

    # Resolves SBML assignment rules (derived/algebraic variables, e.g. pY1173 = Σspecies/c1)
    # that appear inside observable / noise formulas, to a fixpoint. No-op for models without
    # assignment rules. Bare-symbol form to match the parsed (t-free) table formulas.
    apply_rules = _assignment_substitutor(PEprob; bare = true)

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

    # Some PEtab models omit these columns entirely when no row uses them
    has_obs_params_col   = :observableParameters in propertynames(measurements_df)
    has_noise_params_col = :noiseParameters      in propertynames(measurements_df)

    ###############################################
    # Observable formula (y) constraints
    # Group measurements by (obsId, observableParameters) so that each unique
    # formula gets its own compiled ExaModels constraint.
    ###############################################
    itr_y_z    = Int[]                                       # midx values (row p holds -y[itr_y_z[p]])
    itr_y_z!   = Tuple{Int, Int, Int, Int, Int, Float64}[]   # (pos, zidx, idx, cidx, j, L1) — pos indexes into itr_y_z
    itr_y_z_ic = Tuple{Int, Int, Int}[]  # state observable at t=0 -> initial-condition node

    obs_y_groups = Dict{Tuple{String,String}, Vector{Int}}()
    for midx in 1:Nm
        row     = measurements_df[midx, :]
        obs_id  = string(row[:observableId])
        obs_key = has_obs_params_col ? _safe_str(row[:observableParameters]) : ""
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
        obs_sym = parsed isa Symbol ? Symbolics.Num(Symbolics.variable(parsed)) :
                                      Symbolics.parse_expr_to_symbolic(parsed, @__MODULE__)
        # Resolve SBML assignment rules (e.g. pY1173 = Σspecies/c1) so the observable becomes
        # a function of states / params only.
        obs_sym = apply_rules(obs_sym)

        zidx = findfirst(x -> isequal(x, obs_sym), z_syms)  # observable is a single state?
        if zidx !== nothing
            # Observable is a single state variable: y[midx] = state at the measurement time.
            for midx in group_midxs
                row  = measurements_df[midx, :]
                cid  = string(row[:simulationConditionId])
                time = Float64(row[:time])
                cidx = dict_cid_cidx[cid]
                idx  = dict_t_tidx[time]
                if idx == 0
                    # t = 0: state is the initial-condition node z[zidx,1,0,cidx] directly
                    # (the L1 interval-interpolation has no interval 0 to extrapolate).
                    push!(itr_y_z_ic, (midx, zidx, cidx))
                    y0[midx] = z0[zidx, 1, 1, cidx]
                else
                    # y[midx] = Σ_j L1[j+1] * z[zidx, idx, j, cidx] (τ=1 endpoint of interval idx)
                    push!(itr_y_z, midx)
                    pos = length(itr_y_z)   # this midx's row position in con_y_z
                    append!(itr_y_z!, [(pos, zidx, idx, cidx, j, L1[j+1]) for j in 0:K])
                    y0[midx] = sum(L1[j+1] * z0[zidx, idx, j+1, cidx] for j in 0:K)
                end
            end
        else
            # Observable is an arbitrary expression (assignment rules already resolved):
            # compile obs_func for this group.
            obs_expr_final = Symbolics.substitute(obs_sym, dict_fixed_val)
            obs_func = Symbolics.build_function(
                obs_expr_final,
                [z_syms; p_syms; cv_syms]...,
                expression = Val{false}
            )

            itr_y_func   = Tuple{Int, Int, Int}[]  # idx < N : state = z[·,idx+1,0,·] (one node)
            itr_y_func_N = Tuple{Int, Int}[]       # idx = N : state = L1 endpoint of interval N
            for midx in group_midxs
                row  = measurements_df[midx, :]
                cid  = string(row[:simulationConditionId])
                time = Float64(row[:time])
                cidx = dict_cid_cidx[cid]
                idx  = dict_t_tidx[time]
                # warm start: evaluate obs_func at the initial guess, mirroring the constraint
                if idx == N
                    push!(itr_y_func_N, (midx, cidx))
                    zatt = ntuple(v -> sum(L1[jj+1] * z0[v, N, jj+1, cidx] for jj in 0:K), Nz)
                else
                    push!(itr_y_func, (midx, idx, cidx))
                    zatt = ntuple(v -> z0[v, idx+1, 1, cidx], Nz)
                end
                y0[midx] = obs_func(
                    zatt...,
                    ntuple(m -> p0[m], Np)...,
                    ntuple(m -> cv0[m, cidx], Ncv)...
                )
            end

            # idx < N: single-node kernel (state = node 0 of the next interval, by continuity)
            if !isempty(itr_y_func)
                ExaModels.@add_con(c,
                    y[midx] - obs_func(
                        ntuple(v -> z[v,idx+1,0,cidx], Nz)...,
                        ntuple(m -> _p_phys(p,m,pscale), Np)...,
                        ntuple(m -> cv[m,cidx], Ncv)...
                    )
                    for (midx, idx, cidx) in itr_y_func
                )
            end
            # idx == N: final-time group; the L1 (τ=1) endpoint sum is inlined here only.
            if !isempty(itr_y_func_N)
                ExaModels.@add_con(c,
                    y[midx] - obs_func(
                        ntuple(v -> sum(L1[jj+1]*z[v,N,jj,cidx] for jj in 0:K), Nz)...,
                        ntuple(m -> _p_phys(p,m,pscale), Np)...,
                        ntuple(m -> cv[m,cidx], Ncv)...
                    )
                    for (midx, cidx) in itr_y_func_N
                )
            end
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
        noise_key  = has_noise_params_col ? _safe_str(row[:noiseParameters]) : ""
        push!(get!(obs_sigma_groups, (obs_id, noise_key), Int[]), midx)
    end

    # Placeholder symbol standing in for the observable inside a noise formula, plus a
    # memo of each observable's (fixed-value-substituted) symbolic expression by obs_id.
    Y_sym = Symbolics.Num(Symbolics.variable(:__sigma_obs_Y__))
    dict_obsid_obssym = Dict{String, Any}()

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
                sigma0[midx] = sigma_val # warm start
            end
            continue
        end

        # Parse as symbolic expression
        sigma_parsed     = Meta.parse(sigma_expr_sub)
        sigma_parsed_sym = sigma_parsed isa Symbol ?
            Symbolics.Num(Symbolics.variable(sigma_parsed)) :
            Symbolics.parse_expr_to_symbolic(sigma_parsed, @__MODULE__)
        sigma_parsed_sym = apply_rules(sigma_parsed_sym)   # resolve SBML assignment rules
        sigma_expr_final = Symbolics.substitute(sigma_parsed_sym, dict_fixed_val)

        sigma_free   = Symbolics.get_variables(sigma_expr_final)
        sigma_p_vars = filter(v -> any(isequal(v, pv) for pv in p_syms), sigma_free)

        if length(sigma_p_vars) == 1 && isempty(filter(v -> any(isequal(v, zv) for zv in z_syms), sigma_free))
            # Case B: sigma = p[pidx]
            pidx = findfirst(pv -> isequal(pv, only(sigma_p_vars)), p_syms)
            for midx in group_midxs
                push!(itr_sigma_p, (midx, pidx))
                sigma0[midx] = p0[pidx] # warm start
            end
        else
            # σ couples to model states ONLY through the observable y (true for every PEtab
            # benchmark noise form: c | θ | β·y | α+β·y | sqrt(α²+(β·y)²)). Reduce σ to a
            # function of the observable placeholder Y_sym and parameters (see
            # _reduce_sigma_to_obs); if it succeeds, σ = σ_fun(y, p, cv) and the constraint
            # references the existing y[midx] variable directly — sparser than re-deriving
            # the state expression, and σ never couples to z. Otherwise warn and fall back to
            # a general state-dependent expression compiled over (z, p, cv).
            obs_sym = get!(dict_obsid_obssym, obs_id) do
                obs_raw = string(dict_obsid_obsrow[obs_id][:observableFormula])
                op = Meta.parse(obs_raw)
                s  = op isa Symbol ? Symbolics.Num(Symbolics.variable(op)) :
                                     Symbolics.parse_expr_to_symbolic(op, @__MODULE__)
                Symbolics.substitute(apply_rules(s), dict_fixed_val)
            end
            sigma_reduced, reduced_ok = _reduce_sigma_to_obs(sigma_expr_final, obs_sym, Y_sym, z_syms)
            # Conforming iff the reduction left no state AND only Y / parameters / cv remain
            # (a stray symbol, e.g. an unreduced observableParameter, routes to the fallback).
            allowed    = [Y_sym; p_syms; cv_syms]
            conforming = reduced_ok && all(
                rv -> any(isequal(rv, a) for a in allowed),
                Symbolics.get_variables(sigma_reduced)
            )

            if conforming
                # Expected form: σ as a function of the observable y and parameters/cv.
                sigma_fun = Symbolics.build_function(
                    sigma_reduced,
                    [Y_sym; p_syms; cv_syms]...,
                    expression = Val{false}
                )
                itr_sigma_obs = Tuple{Int, Int}[]  # (midx, cidx)
                for midx in group_midxs
                    cidx = dict_cid_cidx[string(measurements_df[midx, :simulationConditionId])]
                    push!(itr_sigma_obs, (midx, cidx))
                    # warm start: σ at the initial guess, reusing the already-computed y0[midx]
                    sigma0[midx] = sigma_fun(y0[midx], p0..., cv0[:, cidx]...)
                end
                ExaModels.@add_con(c,
                    sigma[midx] - sigma_fun(
                        y[midx],
                        ntuple(m -> _p_phys(p,m,pscale), Np)...,
                        ntuple(m -> cv[m,cidx], Ncv)...
                    )
                    for (midx, cidx) in itr_sigma_obs
                )
            else
                @warn "Noise model for observable '$obs_id' references model states " *
                      "outside the observable formula; it does not follow an expected " *
                      "PEtab noise form (σ as a function of parameters and the observable " *
                      "y). Falling back to a general state-dependent expression."
                # Fallback: general expression compiled over (z, p, cv).
                sigma_func = Symbolics.build_function(
                    sigma_expr_final,
                    [z_syms; p_syms; cv_syms]...,
                    expression = Val{false}
                )

                itr_sigma_func   = Tuple{Int, Int, Int}[]  # idx < N : one node
                itr_sigma_func_N = Tuple{Int, Int}[]       # idx = N : L1 endpoint of interval N
                for midx in group_midxs
                    row  = measurements_df[midx, :]
                    cid  = string(row[:simulationConditionId])
                    time = Float64(row[:time])
                    cidx = dict_cid_cidx[cid]
                    idx  = dict_t_tidx[time]
                    # warm start: evaluate sigma_func at the initial guess, mirroring the constraint
                    if idx == N
                        push!(itr_sigma_func_N, (midx, cidx))
                        zatt = ntuple(v -> sum(L1[jj+1] * z0[v, N, jj+1, cidx] for jj in 0:K), Nz)
                    else
                        push!(itr_sigma_func, (midx, idx, cidx))
                        zatt = ntuple(v -> z0[v, idx+1, 1, cidx], Nz)
                    end
                    sigma0[midx] = sigma_func(
                        zatt...,
                        ntuple(m -> p0[m], Np)...,
                        ntuple(m -> cv0[m, cidx], Ncv)...
                    )
                end

                # idx < N: single-node kernel
                if !isempty(itr_sigma_func)
                    ExaModels.@add_con(c,
                        sigma[midx] - sigma_func(
                            ntuple(v -> z[v,idx+1,0,cidx], Nz)...,
                            ntuple(m -> _p_phys(p,m,pscale), Np)...,
                            ntuple(m -> cv[m,cidx], Ncv)...
                        )
                        for (midx, idx, cidx) in itr_sigma_func
                    )
                end
                # idx == N: final-time group; L1 (τ=1) endpoint sum inlined here only.
                if !isempty(itr_sigma_func_N)
                    ExaModels.@add_con(c,
                        sigma[midx] - sigma_func(
                            ntuple(v -> sum(L1[jj+1]*z[v,N,jj,cidx] for jj in 0:K), Nz)...,
                            ntuple(m -> _p_phys(p,m,pscale), Np)...,
                            ntuple(m -> cv[m,cidx], Ncv)...
                        )
                        for (midx, cidx) in itr_sigma_func_N
                    )
                end
            end
        end
    end

    ###############################################
    # Emit batched constraints for accumulated iterators
    ###############################################
    if !isempty(itr_y_z)
        # y[midx] = Σ_{j=0}^{K} L_j(1) * z[zidx, i, j, cidx]
        # NOTE: @add_con! indexes the base constraint by ROW POSITION (offset0 = o0 + key),
        # NOT by the loop value. Base row p holds -y[itr_y_z[p]], so the augmentation must be
        # keyed by that position p — keying by midx would mis-attach terms whenever itr_y_z
        # is not the contiguous identity 1:Nm (e.g. multi-observable / multi-condition models).
        con_y_z = ExaModels.@add_con(c,
            -y[midx]
            for midx in itr_y_z
        )
        ExaModels.@add_con!(c,
            con_y_z,
            pos => L1j*z[v,i,j,cidx]
            for (pos, v, i, cidx, j, L1j) in itr_y_z!
        )
    end

    if !isempty(itr_y_z_ic)
        # t=0 state observables: y[midx] = z[zidx, 1, 0, cidx] (initial-condition node)
        ExaModels.@add_con(c,
            y[midx] - z[zidx,1,0,cidx]
            for (midx, zidx, cidx) in itr_y_z_ic
        )
    end

    if !isempty(itr_sigma_fix)
        ExaModels.@add_con(c,
            sigma[midx] - val
            for (midx, val) in itr_sigma_fix
        )
    end

    if !isempty(itr_sigma_p)
        # sigma[midx] equals the PHYSICAL value of p[pidx]. pidx is a per-entry data
        # index, so partition by scale and emit one fixed-form constraint per scale.
        for sc in (:log10, :log, :lin)
            grp = [t for t in itr_sigma_p if pscale[t[2]] === sc]
            isempty(grp) && continue
            if sc === :log10
                ExaModels.@add_con(c, sigma[midx] - exp(log(10.0)*p[pidx]) for (midx,pidx) in grp)
            elseif sc === :log
                ExaModels.@add_con(c, sigma[midx] - exp(p[pidx])           for (midx,pidx) in grp)
            else
                ExaModels.@add_con(c, sigma[midx] - p[pidx]                for (midx,pidx) in grp)
            end
        end
    end

    return c, y0, sigma0
end
