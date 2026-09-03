# Unified observable/noise constraints and Gaussian NLL objective
#
# Runs in phases, in this order, because each one feeds the next:
#   1. the Gaussian nll objective and any parameter priors  (_add_nll_objective/_add_prior_objective)
#   2. observable (y) formulas   -> direct constraints, or a deferred row if the formula needs an
#                                   aux var that does not exist yet
#   3. noise (sigma) formulas    -> same, and may reuse y0 from phase 2
#   4. the batched iterators phases 2-3 accumulated
#   5. ov, the bound assignment-rule aux vars, then the pending_ov rows that needed them
#
# Phases 2 and 3 are the same dispatch over a parsed table formula, specialized per side:
#
#   formula is...            | y                        | sigma
#   -------------------------|--------------------------|---------------------------
#   a numeric literal        | -                        | itr_sigma_fix
#   a single state z         | itr_y1                   | -
#   a single parameter p     | -                        | itr_sigma_p
#   only FLAT rule leaves    | pending_ov               | pending_ov
#   reducible to σ(y,p,cv)   | -                        | direct constraint on y[midx]
#   anything else            | general fn per class     | general fn per class
#
# The node a measurement lands on is abstracted by a NodeCtx: the collocation context has an
# initial-condition class z[·,·,1,0] and an interval-endpoint class z[·,·,idx,K] (the τ=1 right
# endpoint Radau collocates), the steady-state context has the single class zss[·,cidx].
# Node-dependent constraints are emitted through the context's traced accessors, so both
# pipelines share this one implementation.

# One node kind: membership test plus traced state accessors at fixed row positions
#   zref  rows: (lhs index, col, s, node...)   node fields from position 4
#   z1ref rows: (midx, zidx, node...)          node fields from position 3
struct NodeClass{M, Z, Z1}
    member::M
    zref::Z
    z1ref::Z1
end

# The node abstraction of one pipeline: measurement -> node, warm-start states at a node,
# cv column of a node, and the node classes
struct NodeCtx{N, A, C}
    node_of::N
    z0at::A
    col_of::C
    classes::Vector{NodeClass}
end

# Collocation node context: idx == 0 is the initial-condition node, idx >= 1 the interval
# idx τ=1 right endpoint
function _collocation_ctx(c::ExaCore, spec::PEtabSpec, mesh::PEtabMesh, z0arr)
    (; Nz) = spec
    K = c.K
    z = c.z
    endpoint = NodeClass(
        n -> n[2] >= 1,
        r -> ntuple(v -> z[v, r[4], r[5], K], Nz),
        r -> z[r[2], r[3], r[4], K],
    )
    ic = NodeClass(
        n -> n[2] == 0,
        r -> ntuple(v -> z[v, r[4], 1, 0], Nz),
        r -> z[r[2], r[3], 1, 0],
    )
    return NodeCtx(
        midx -> (spec.meas_cidx[midx], mesh.meas_iidx[midx]),
        n -> n[2] == 0 ? ntuple(v -> z0arr[v, n[1], 1, 1], Nz) :
                         ntuple(v -> z0arr[v, n[1], n[2], K + 1], Nz),
        n -> spec.simcv[n[1]],
        NodeClass[endpoint, ic],
    )
end

# Emit lhs(row) - fn(states@node, p_phys, cv[:, col], ov leaves) over rows of one class
function _emit_node_rows(c::ExaCore, class::NodeClass, spec::PEtabSpec, p, cv, lhs, fn, rows;
                         ov = nothing, upos::Vector{Int} = Int[])
    isempty(rows) && return c
    (; Np, Ncv, pscale) = spec
    nleaf = length(upos)
    g = r -> lhs(r) - fn(class.zref(r)...,
                         ntuple(m -> _p_phys(p, m, pscale), Np)...,
                         ntuple(m -> cv[m, r[2]], Ncv)...,
                         ntuple(k -> ov[upos[k], r[3]], nleaf)...)
    c, _ = ExaModels.add_con(c, Base.Generator(g, rows))
    return c
end

# (*) Main function for creating ExaModels objective function (*)
function _create_objective(c::ExaCore, spec::PEtabSpec, tables::PEtabTables, ctx::NodeCtx)
    (; Np, Ncv, Nz, Nm, pscale) = spec
    p = c.p
    y = c.y
    sigma = c.sigma
    if Ncv >= 1
        cv = c.cv
    end

    # Warm start: evaluate y/sigma formulas at the p,cv initial point for feasible set_start!
    # values, with node states through ctx.z0at
    theta0 = _var_starts(c, p)                            # decision var p (estimation scale)
    p0  = [_p_phys_val(theta0, m, pscale) for m in 1:Np]  # PHYSICAL parameter starts
    cv0 = Ncv >= 1 ? reshape(_var_starts(c, cv), Ncv, :) : zeros(Float64, 0, spec.Ncc)
    y0     = zeros(Float64, Nm) # computed observable values at the initial guess
    sigma0 = zeros(Float64, Nm) # computed noise (std) values at the initial guess

    measurements = tables.measurements
    observables = tables.observables

    ###############################################
    # Objective function (Gaussian negative log-likelihood)
    ###############################################
    # @add_obj/@add_con rebind the core, so capture the returned core from each helper
    c = _add_nll_objective(c, spec, tables)
    c = _add_prior_objective(c, spec, tables)   # MAP: add -log prior(θ) terms (matches PEtab.nllh)

    ###############################################
    # Auxiliary variable constraints for y, sigma
    ###############################################
    mark = expr -> _mark_used_p!(spec.used_p, spec.pidx_of, expr)
    apply_rules = spec.subst_bare
    z_syms  = spec.z_syms_bare
    p_syms  = spec.p_syms
    cv_syms = spec.cv_syms
    fixed_vals = spec.fixed_vals

    # Fast lookup: observableId => (observableFormula, noiseFormula)
    dict_obsid_formulas = Dict(
        String(observables.observableId[i]) =>
            (obs = String(observables.observableFormula[i]),
             noise = String(observables.noiseFormula[i]))
        for i in 1:_nrows(observables)
    )

    obs_params_col   = _getcol(measurements, :observableParameters)
    noise_params_col = _getcol(measurements, :noiseParameters)

    # Bind each FLAT assignment rule to an aux var ov[·] once per node instead of inlining a large
    # rule into every formula, which inflates compile time. Nested rules fall back to inlining
    (; ids, lhs, rhs, is_flat) = spec.rules
    rule_lhs, rule_rhs, rule_is_flat = lhs, rhs, is_flat
    n_rules = length(ids)
    # rules present (as leaf symbols) in a parsed, not-yet-inlined formula
    _used_rules(expr) = Int[r for r in 1:n_rules
                            if any(v -> isequal(v, rule_lhs[r]), Symbolics.get_variables(expr))]
    # compiled rule RHS over [z; p; cv] with fixed constants frozen, memoized and usable on Floats
    rule_func_cache = Dict{Int, Any}()
    _rule_func(r) = get!(rule_func_cache, r) do
        Symbolics.build_function(
            Symbolics.substitute(rule_rhs[r], fixed_vals),
            [z_syms; p_syms; cv_syms]..., expression = Val{false})
    end
    relevant_rules = Set{Int}()          # rule indices that get bound to ov
    ov_nodes       = Tuple[]             # evaluation nodes needing ov
    # Deferred obs/noise constraints referencing ov leaves: (aux, func, used, [(midx, node)])
    pending_ov = Tuple{Any, Any, Vector{Int}, Vector{Tuple{Int, Tuple}}}[]

    ###############################################
    # Observable formula (y) constraints
    # Group measurements by (obsId, observableParameters) so that each unique
    # formula gets its own compiled ExaModels constraint.
    ###############################################
    itr_y1 = Tuple{Int, Int, Tuple}[]   # (midx, zidx, node): observable is a single state

    obs_y_groups = Dict{Tuple{String, String}, Vector{Int}}()
    for midx in 1:Nm
        obs_id  = String(measurements.observableId[midx])
        obs_key = String(strip(obs_params_col[midx]))
        push!(get!(obs_y_groups, (obs_id, obs_key), Int[]), midx)
    end

    for ((obs_id, obs_params_str), group_midxs) in obs_y_groups
        obs_expr_raw = dict_obsid_formulas[obs_id].obs

        obs_expr_sub = _sub_placeholders(obs_expr_raw, obs_params_str, "observable", obs_id)
        obs_sym      = _to_symbolic(obs_expr_sub)
        # FLAT rules are bound to ov aux vars in the branch below, or inlined by substitution
        used = _used_rules(obs_sym)
        bound = !isempty(used) && all(r -> rule_is_flat[r], used)
        bound || (obs_sym = apply_rules(obs_sym))
        mark(obs_sym)
        for r in used
            mark(rule_rhs[r])
        end

        zidx = bound ? nothing : findfirst(x -> isequal(x, obs_sym), z_syms)  # single state?
        if zidx !== nothing
            # Observable is a single state variable: y[midx] = state at the measurement time.
            for midx in group_midxs
                node = ctx.node_of(midx)
                push!(itr_y1, (midx, zidx, node))
                y0[midx] = ctx.z0at(node)[zidx]
            end
        elseif bound
            # Compile obs_func over [z;p;cv;rule_leaves] with each leaf fed its ov var per node. Defer
            leaves     = [rule_lhs[r] for r in used]
            obs_efinal = Symbolics.substitute(obs_sym, fixed_vals)   # rule leaves survive
            obs_func   = Symbolics.build_function(
                obs_efinal, [z_syms; p_syms; cv_syms; leaves]..., expression = Val{false})
            rows = Tuple{Int, Tuple}[]
            for midx in group_midxs
                node  = ctx.node_of(midx)
                col   = ctx.col_of(node)
                zatt0 = ctx.z0at(node)   # node state at the initial guess
                rvals = ntuple(k -> _rule_func(used[k])(zatt0..., p0..., cv0[:, col]...), length(used))
                y0[midx] = obs_func(zatt0..., p0..., cv0[:, col]..., rvals...)
                push!(rows, (midx, node)); push!(ov_nodes, node)
            end
            for r in used; push!(relevant_rules, r); end
            push!(pending_ov, (y, obs_func, used, rows))
        else
            # Observable is an arbitrary expression (assignment rules already resolved):
            # compile obs_func for this group.
            obs_expr_final = Symbolics.substitute(obs_sym, fixed_vals)
            obs_func = Symbolics.build_function(
                obs_expr_final,
                [z_syms; p_syms; cv_syms]...,
                expression = Val{false}
            )

            for midx in group_midxs
                node  = ctx.node_of(midx)
                col   = ctx.col_of(node)
                # warm start: evaluate obs_func at the initial guess, mirroring the constraint
                y0[midx] = obs_func(ctx.z0at(node)...,
                                    ntuple(m -> p0[m], Np)...,
                                    ntuple(m -> cv0[m, col], Ncv)...)
            end
            for class in ctx.classes
                rows = [(midx, ctx.col_of(ctx.node_of(midx)), 0, ctx.node_of(midx)...)
                        for midx in group_midxs if class.member(ctx.node_of(midx))]
                c = _emit_node_rows(c, class, spec, p, Ncv >= 1 ? c.cv : nothing,
                                    r -> y[r[1]], obs_func, rows)
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

    obs_sigma_groups = Dict{Tuple{String, String}, Vector{Int}}()
    for midx in 1:Nm
        obs_id    = String(measurements.observableId[midx])
        noise_key = String(strip(noise_params_col[midx]))
        push!(get!(obs_sigma_groups, (obs_id, noise_key), Int[]), midx)
    end

    # Placeholder symbol standing in for the observable inside a noise formula, plus a
    # memo of each observable's (fixed-value-substituted) symbolic expression by obs_id.
    Y_sym = Symbolics.Num(Symbolics.variable(:__sigma_obs_Y__))
    dict_obsid_obssym = Dict{String, Any}()

    for ((obs_id, noise_params_str), group_midxs) in obs_sigma_groups
        sigma_expr_raw = dict_obsid_formulas[obs_id].noise

        sigma_expr_sub = _sub_placeholders(sigma_expr_raw, noise_params_str, "noise", obs_id)

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
        sigma_parsed_sym = _to_symbolic(sigma_expr_sub)
        # FLAT rules in the noise formula are bound to ov aux vars (parallels the observable branch)
        used_sig = _used_rules(sigma_parsed_sym)
        if !isempty(used_sig) && all(r -> rule_is_flat[r], used_sig)
            mark(sigma_parsed_sym)
            for r in used_sig
                mark(rule_rhs[r])
            end
            leaves   = [rule_lhs[r] for r in used_sig]
            sig_efin = Symbolics.substitute(sigma_parsed_sym, fixed_vals)   # rule leaves survive
            sigma_func = Symbolics.build_function(
                sig_efin, [z_syms; p_syms; cv_syms; leaves]..., expression = Val{false})
            rows = Tuple{Int, Tuple}[]
            for midx in group_midxs
                node  = ctx.node_of(midx)
                col   = ctx.col_of(node)
                zatt0 = ctx.z0at(node)
                rvals = ntuple(k -> _rule_func(used_sig[k])(zatt0..., p0..., cv0[:, col]...), length(used_sig))
                sigma0[midx] = sigma_func(zatt0..., p0..., cv0[:, col]..., rvals...)
                push!(rows, (midx, node)); push!(ov_nodes, node)
            end
            for r in used_sig; push!(relevant_rules, r); end
            push!(pending_ov, (sigma, sigma_func, used_sig, rows))
            continue
        end
        sigma_parsed_sym = apply_rules(sigma_parsed_sym)   # resolve SBML assignment rules
        mark(sigma_parsed_sym)
        sigma_expr_final = Symbolics.substitute(sigma_parsed_sym, fixed_vals)

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
            # Reduce σ to a function of Y_sym + params so the constraint references y[midx] directly
            # (sparser). Fall back to a general (z,p,cv) expression if σ couples to states elsewhere
            obs_sym = get!(dict_obsid_obssym, obs_id) do
                obs_raw = dict_obsid_formulas[obs_id].obs
                Symbolics.substitute(apply_rules(_to_symbolic(obs_raw)), fixed_vals)
            end
            sigma_reduced, reduced_ok = _reduce_sigma_to_obs(sigma_expr_final, obs_sym, Y_sym, z_syms)
            # Conforming iff only Y / params / cv remain (a stray symbol routes to the fallback)
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
                itr_sigma_obs = Tuple{Int, Int}[]  # (midx, col)
                for midx in group_midxs
                    col = ctx.col_of(ctx.node_of(midx))
                    push!(itr_sigma_obs, (midx, col))
                    # warm start: σ at the initial guess, reusing the already-computed y0[midx]
                    sigma0[midx] = sigma_fun(y0[midx], p0..., cv0[:, col]...)
                end
                ExaModels.@add_con(c,
                    sigma[midx] - sigma_fun(
                        y[midx],
                        ntuple(m -> _p_phys(p,m,pscale), Np)...,
                        ntuple(m -> cv[m,col], Ncv)...
                    )
                    for (midx, col) in itr_sigma_obs
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

                for midx in group_midxs
                    node = ctx.node_of(midx)
                    col  = ctx.col_of(node)
                    # warm start: evaluate sigma_func at the initial guess, mirroring the constraint
                    sigma0[midx] = sigma_func(ctx.z0at(node)...,
                                              ntuple(m -> p0[m], Np)...,
                                              ntuple(m -> cv0[m, col], Ncv)...)
                end
                for class in ctx.classes
                    rows = [(midx, ctx.col_of(ctx.node_of(midx)), 0, ctx.node_of(midx)...)
                            for midx in group_midxs if class.member(ctx.node_of(midx))]
                    c = _emit_node_rows(c, class, spec, p, Ncv >= 1 ? c.cv : nothing,
                                        r -> sigma[r[1]], sigma_func, rows)
                end
            end
        end
    end

    ###############################################
    # Emit batched constraints for accumulated iterators
    ###############################################
    for class in ctx.classes
        # y[midx] = the single observed state at the measurement node
        rows = [(midx, zidx, node...) for (midx, zidx, node) in itr_y1 if class.member(node)]
        isempty(rows) && continue
        g = r -> y[r[1]] - class.z1ref(r)
        c, _ = ExaModels.add_con(c, Base.Generator(g, rows))
    end

    if !isempty(itr_sigma_fix)
        ExaModels.@add_con(c,
            sigma[midx] - val
            for (midx, val) in itr_sigma_fix
        )
    end

    if !isempty(itr_sigma_p)
        # sigma[midx] = physical value of p[pidx], partitioned by scale
        for (_, pidx) in itr_sigma_p
            spec.used_p[pidx] = true
        end
        c = _add_scaled_cons(c, p, itr_sigma_p, pscale, r -> sigma[r[1]]; pat = 2)
    end

    ###############################################
    # Observed-variable aux variables, ov (bound assignment rules)
    ###############################################
    # Bind each FLAT rule to ov[relpos,nodeslot] = rule(state@node,p,cv), one per (rule,node) and
    # shared across formulas so a large rule is differentiated once instead of inlined everywhere.
    # Rectangular rule×node grid. Node feeding matches the node class's own accessor
    if !isempty(pending_ov)
        rels     = sort(collect(relevant_rules))
        rel_pos  = Dict(r => i for (i, r) in enumerate(rels))
        nodes    = sort(unique(ov_nodes))                 # distinct nodes
        node_slot = Dict(nd => s for (s, nd) in enumerate(nodes))
        nrel = length(rels); nnode = length(nodes)

        # warm start ov0[i,s] = rule_func[rels[i]](state@node at the initial guess)
        ov0 = Matrix{Float64}(undef, nrel, nnode)
        for (s, node) in enumerate(nodes)
            zatt0 = ctx.z0at(node)
            col   = ctx.col_of(node)
            for (i, r) in enumerate(rels)
                ov0[i, s] = _rule_func(r)(zatt0..., p0..., cv0[:, col]...)
            end
        end
        ExaModels.@add_var(c, ov, 1:nrel, 1:nnode; start = ov0, lvar = -Inf, uvar = Inf)
        ov = c.ov

        # ov defining constraints: ov[i,s] - rule_r(state@node) = 0, grouped by (rule, node class)
        for (i, r) in enumerate(rels)
            rf = _rule_func(r)
            for class in ctx.classes
                rows = [(i, ctx.col_of(nd), node_slot[nd], nd...)
                        for nd in nodes if class.member(nd)]
                c = _emit_node_rows(c, class, spec, p, Ncv >= 1 ? c.cv : nothing,
                                    r -> ov[r[1], r[3]], rf, rows)
            end
        end

        # Re-emit the deferred bound obs/noise constraints over single nodes + ov leaves.
        # aux[midx] - func(node_z..., p_phys..., cv..., ov[relpos(used[k]), nodeslot]...) = 0
        for (aux, func, used, mrows) in pending_ov
            upos = [rel_pos[r] for r in used]   # captured literal rule positions
            for class in ctx.classes
                rows = [(midx, ctx.col_of(nd), node_slot[nd], nd...)
                        for (midx, nd) in mrows if class.member(nd)]
                c = _emit_node_rows(c, class, spec, p, Ncv >= 1 ? c.cv : nothing,
                                    r -> aux[r[1]], func, rows; ov = ov, upos = upos)
            end
        end
    end

    return c, y0, sigma0
end

# Fill the observableParameter${n}_${obs_id} / noiseParameter${n}_${obs_id} placeholders with the
# n-th semicolon-delimited entry from this group's observableParameters/noiseParameters cell
function _sub_placeholders(raw::AbstractString, params_str::AbstractString, kind::String, obs_id::String)
    isempty(params_str) && return String(raw)
    parts = strip.(split(params_str, ";"))
    return replace(String(raw), ["$(kind)Parameter$(n)_$(obs_id)" => parts[n] for n in eachindex(parts)]...)
end

# Reduce a noise σ expression so its state-dependence enters only through the observable. Solve O=Y
# for one observable-state (requires O affine in states) and substitute into σ. Returns
# (reduced_expr, ok) where ok=true means no state remains
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

# Gaussian negative log-likelihood, matching PEtab.jl. Residual on the observableTransformation
# scale, with the change-of-variables Jacobian in the trailing per-measurement constants:
#   lin   : 0.5(y-ymeas)²/σ²               + log σ + 0.5log2π
#   log   : 0.5(ln y - ln ymeas)²/σ²       + log σ + 0.5log2π + ln ymeas
#   log10 : 0.5(log10 y - log10 ymeas)²/σ² + log σ + 0.5log2π + ln ymeas + ln(ln10)
function _add_nll_objective(c::ExaCore, spec::PEtabSpec, tables::PEtabTables)
    y = c.y
    sigma = c.sigma

    HALF_LOG2PI = 0.5 * log(2π)
    itr_obj_lin   = Tuple{Int, Float64, Float64}[]  # (midx, ymeas,     const)
    itr_obj_log   = Tuple{Int, Float64, Float64}[]  # (midx, ln(ymeas), const)
    itr_obj_log10 = Tuple{Int, Float64, Float64}[]  # (midx, log10(ymeas), const)
    for midx in 1:spec.Nm
        ymeas = _cellfloat(tables.measurements.measurement[midx], "measurements row $midx")
        tr    = spec.meas_transform[midx]
        if tr === :lin
            push!(itr_obj_lin, (midx, ymeas, HALF_LOG2PI))
        elseif tr === :log
            @assert ymeas > 0 "log-transformed observable needs ymeas>0 (midx=$midx)"
            push!(itr_obj_log, (midx, log(ymeas), HALF_LOG2PI + log(ymeas)))
        elseif tr === :log10
            @assert ymeas > 0 "log10-transformed observable needs ymeas>0 (midx=$midx)"
            push!(itr_obj_log10, (midx, log10(ymeas), HALF_LOG2PI + log(ymeas) + log(log(10.0))))
        else
            error("ExaModelsPEtab: unsupported observableTransformation ':$tr'.")
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
            0.5*(log(y[midx])/log(10.0) - l10ym)^2/sigma[midx]^2 + log(sigma[midx]) + cst
            for (midx, l10ym, cst) in itr_obj_log10
        )
    end
    return c
end

# Parameter priors (MAP objective), matching PEtab.nllh with normalization constants. Supports
# parameterScaleNormal and parameterScaleLaplace on the estimation-scale p[pidx], and laplace on
# the physical value by scale. Uniform adds only a constant and is omitted. Unsupported types error out
function _add_prior_objective(c::ExaCore, spec::PEtabSpec, tables::PEtabTables)
    params = tables.parameters
    prior_col = _getcol(params, :objectivePriorType)
    all(isempty ∘ strip, prior_col) && return c   # no priors in this model
    p           = c.p
    HALF_LOG2PI = 0.5 * log(2π)
    prior_params_col = _getcol(params, :objectivePriorParameters)
    scale_col        = _getcol(params, :parameterScale, "lin")
    # parameterScale* priors act on p[pidx] in estimation scale, and laplace acts on the physical value
    psnorm = Tuple{Int,Float64,Float64}[]    # parameterScaleNormal:  0.5((p-μ)/σ)² + logσ + ½log2π
    pslap  = Tuple{Int,Float64,Float64}[]    # parameterScaleLaplace: |p-μ|/b + log2b
    lap_li = Tuple{Int,Float64,Float64}[]    # laplace, lin-scale param:  |p-μ|/b + log2b
    lap_10 = Tuple{Int,Float64,Float64}[]    # laplace, log10-scale:      |10^p-μ|/b + log2b
    lap_e  = Tuple{Int,Float64,Float64}[]    # laplace, log-scale:        |e^p-μ|/b + log2b
    for i in 1:_nrows(params)
        ptype = _norm_cell(prior_col[i], Symbol(""))
        ptype === Symbol("") && continue                # blank => no prior
        pid = String(params.parameterId[i])
        haskey(spec.pidx_of, pid) || continue           # only estimated params are decision vars
        idx = spec.pidx_of[pid]
        spec.used_p[idx] = true
        pp  = strip.(split(prior_params_col[i], ";"))
        a   = parse(Float64, pp[1]); b = length(pp) >= 2 ? parse(Float64, pp[2]) : NaN
        if     ptype === :parameterscalenormal  ; push!(psnorm, (idx, a, b))
        elseif ptype === :parameterscalelaplace ; push!(pslap,  (idx, a, b))
        elseif ptype === :uniform               ; nothing   # constant offset with no effect on the optimum
        elseif ptype === :normal
            error("ExaModelsPEtab: unsupported objectivePriorType 'normal' (linear-scale Gaussian) for param $pid.")
        elseif ptype === :laplace
            sc = _norm_cell(scale_col[i], :lin)
            sc === :log10 ? push!(lap_10, (idx, a, b)) :
            sc === :log   ? push!(lap_e,  (idx, a, b)) :
                            push!(lap_li, (idx, a, b))   # :lin (and any non-log scale)
        else
            error("ExaModelsPEtab: unsupported objectivePriorType '$ptype' for param $pid.")
        end
    end

    # Create objective function constraint
    isempty(psnorm) || ExaModels.@add_obj(c, 0.5*((p[i]-mu)/sg)^2 + log(sg) + HALF_LOG2PI for (i,mu,sg) in psnorm)
    isempty(pslap)  || ExaModels.@add_obj(c, abs(p[i]-mu)/b + log(2*b)               for (i,mu,b) in pslap)
    isempty(lap_li) || ExaModels.@add_obj(c, abs(p[i]-mu)/b + log(2*b)               for (i,mu,b) in lap_li)
    isempty(lap_10) || ExaModels.@add_obj(c, abs(exp(log(10.0)*p[i])-mu)/b + log(2*b)     for (i,mu,b) in lap_10)
    isempty(lap_e)  || ExaModels.@add_obj(c, abs(exp(p[i])-mu)/b + log(2*b)          for (i,mu,b) in lap_e)
    return c
end
