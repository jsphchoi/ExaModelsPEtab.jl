# get initial guess for z and a lot of other things
function _get_z_init(PEmodel::PEtabModel, PEprob::PEtabODEProblem, K::Int)
    # Get unique t_meas
    PEtable = PEmodel.petab_tables
    t_meas = sort(unique(filter(t -> !iszero(t), PEtable[:measurements][!,:time])))

    # Fixed-time event toggles must land on a mesh node (else _get_gate_vals rejects a mid-interval
    # toggle). Force them in by making the mesh tstops the UNION of measurement times and event
    # times. t_events is empty for non-event models, so the mesh is byte-identical there.
    t_events = _get_event_times(PEmodel, PEprob)
    t_stops  = sort(unique(vcat(t_meas, t_events)))

    # Solve all experimental conditions at nominal values (tstops = measurements ∪ events ⇒ nodes)
    p_nominal = PEtab.get_x(PEprob)
    sol = _solve_conds(p_nominal, PEmodel, PEprob, t_stops)

    # Build the shared collocation mesh by TILING per simulation-condition tspan. Every condition is
    # integrated over the SAME full span (see _solve_conds) — giving a rectangular warm start (one
    # shared N for all conditions) — but each condition's DATA only extends to its own last
    # measurement time t_end[cidx]. We split the timeline at the distinct t_end breakpoints and, in
    # each segment [a,b], (a) take the base nodes from the condition that actually terminates there
    # (the "owner") and (b) cap the interval width at h_cap = the SMALLEST of the per-condition
    # largest steps among the conditions whose data still covers the segment (t_end ≥ b). Any base
    # interval wider than h_cap is split into equal sub-intervals. This resolves every region at
    # least as finely as the most-demanding condition covering it, rather than inheriting a single
    # (finest) condition's adaptive nodes everywhere.
    #
    # Because tstops = the UNION of all conditions' measurement+event times is forced on every solve,
    # the owner's sol.t contains every measurement/event time in its segment bit-exactly; segment
    # subdivision only inserts interior points and re-emits the exact right boundary, so every t_meas
    # still lands verbatim on a node (letting _get_dict_t_tidx map by exact `==`, not drifting cumsum).
    cids     = _get_cids(PEmodel)                                    # canonical cidx order
    sol_t    = [collect(Float64, sol[Symbol(cid)].t) for cid in cids]  # each condition's node times
    t_start  = minimum(first, sol_t)                                 # ODE t0 (shared across conditions)
    T_global = maximum(last,  sol_t)                                 # full solved span (shared)

    # Per-condition actual end = own last (nonzero) measurement time; 0 for t=0-only conditions —
    # those then cover/own no segment and drop out of the mesh logic entirely.
    meas_df = PEtable[:measurements]
    t_end   = zeros(Float64, length(cids))
    for row in eachrow(meas_df)
        cidx = findfirst(==(string(row[:simulationConditionId])), cids)
        cidx === nothing && continue
        t_end[cidx] = max(t_end[cidx], Float64(row[:time]))
    end

    # Segment boundaries: t_start, each distinct positive t_end, and the full solved span.
    B = sort(unique(vcat(t_start, filter(>(t_start), t_end), T_global)))

    t_nodes = Float64[]
    for k in 1:length(B)-1
        a, b = B[k], B[k+1]
        covering = [cidx for cidx in eachindex(cids) if t_end[cidx] >= b - _MESH_TOL]
        isempty(covering) && (covering = [argmax(t_end)])           # tail beyond all data: longest cond
        h_cap = minimum(_seg_maxwidth(sol_t[cidx], a, b) for cidx in covering)
        owner = _pick_owner(covering, t_end, sol_t, a, b)
        seg   = _subdivide_segment(sol_t[owner], a, b, h_cap)
        append!(t_nodes, isempty(t_nodes) ? seg : @view seg[2:end]) # drop the shared boundary `a`
    end
    h = diff(t_nodes)
    N = length(h) # number of intervals
    taus = _taus(K) # get interpolation points
    t_mesh = [t_nodes[i] + taus[j+1]*h[i] for i in 1:N, j in 0:K] # t_ij mesh; t_nodes[i]=exact left node (no cumsum drift)
    t_vec_mesh = Array(reshape(t_mesh',N*(K+1))) # vectorize t_ij mesh

    # Get constants
    Nz = Int64(PEprob.model_info.nstates) # number of state variables
    Nc = sol.count # number of experimental conditions
    L1 = [_eval_l(j,1.0,taus) for j in 0:K] # for interval continuity constraints later

    # Interpolate each condition's solution at every mesh node. Every condition was integrated
    # over the full mesh span (see _solve_conds), so sol[cid](t) is valid at every node — no
    # clamping/extrapolation needed. (Conditions observed only on a sub-window still carry the
    # true ODE profile over the rest of the mesh, matching what the collocation enforces.)
    sol_at_mesh= [
        sol[cid](t)
        for t in t_vec_mesh, cid in Symbol.(_get_cids(PEmodel))
    ]
    # t_vec_mesh orders nodes with j (collocation index) varying fastest within each
    # interval i, so the flattened mesh dimension reshapes as (K+1, N) — K+1 fastest —
    # NOT (N, K+1). Reshape to that layout, then permute to the (Nz, N, K+1, Nc) order
    # used everywhere else. (The previous identity permutedims left i and j transposed.)
    z_init = permutedims(reshape(stack(sol_at_mesh), Nz, K+1, N, Nc), (1, 3, 2, 4))

    return z_init, Nz, N, K, Nc, t_meas, h, taus, L1, t_nodes
end

# Absolute tolerance for mesh-time comparisons (segment coverage and width caps).
const _MESH_TOL = 1e-9

# Largest gap among `tvec`'s nodes restricted to [a,b] (the segment edges bound the gaps). This is
# the condition's coarsest adaptive step within the segment; the per-segment h_cap is the min of
# this over the covering conditions ("smallest largest hmax").
function _seg_maxwidth(tvec, a, b)
    nodes = sort!(unique!(Float64[a; b; filter(t -> a < t < b, tvec)]))
    return maximum(diff(nodes))
end

# Owner of a segment = the covering condition that terminates soonest at/after it (smallest t_end);
# ties are broken by the FINER condition (most own nodes inside [a,b]). Its mesh supplies the base
# nodes for the segment.
function _pick_owner(covering, t_end, sol_t, a, b)
    tmin  = minimum(t_end[cidx] for cidx in covering)
    cands = [cidx for cidx in covering if abs(t_end[cidx] - tmin) <= _MESH_TOL]
    return argmax(cidx -> count(t -> a <= t <= b, sol_t[cidx]), cands)
end

# Owner's base nodes clamped to [a,b], with any interval wider than h_cap split into
# ceil(width/h_cap) EQUAL sub-intervals. The right boundary b is emitted EXACTLY (not lo+n*w/n) so
# measurement/event times falling on segment boundaries stay bit-exact mesh nodes.
function _subdivide_segment(tvec, a, b, h_cap)
    base = sort!(unique!(Float64[a; b; filter(t -> a < t < b, tvec)]))
    out  = Float64[base[1]]
    for i in 1:length(base)-1
        lo, hi = base[i], base[i+1]
        w = hi - lo
        n = max(1, ceil(Int, w / h_cap - _MESH_TOL))
        for j in 1:n-1
            push!(out, lo + j * w / n)
        end
        push!(out, hi)   # exact boundary — keeps measurement/event nodes bit-exact
    end
    return out
end

# Returns ::Dict{(condition id)::Symbol, (solution)}
function _solve_conds(p_nominal, PEmodel::PEtabModel, PEprob::PEtabODEProblem, tstops)
    sols = Dict{Symbol, Any}()
    si        = PEprob.model_info.simulation_info
    has_preeq = si.has_pre_equilibration
    sim_ids   = si.conditionids[:simulation]
    preeq_ids = si.conditionids[:pre_equilibration]
    for cid in Symbol.(_get_cids(PEmodel))
        # For pre-equilibration models PEtab keys a simulation by the pair
        # (pre_eq_id => simulation_id); passing a plain condition id errors. The
        # returned ODEProblem's u0 is the pre-equilibrated steady state.
        cond_arg = cid
        if has_preeq
            pos = findfirst(==(cid), sim_ids)
            pos === nothing && continue        # condition not simulated (pre-eq only)
            cond_arg = preeq_ids[pos] => sim_ids[pos]
        end
        odesys, callbacks = PEtab.get_odeproblem(p_nominal, PEprob; condition = cond_arg)
        # Integrate over the FULL mesh span [tstart, max(t_meas)] — not just this condition's
        # own tmax — so z_init carries a real ODE profile at every shared-mesh node. The
        # collocation enforces the ODE over the whole mesh for every condition regardless, and
        # later (out-of-window) nodes carry no measurements for this condition, so this only
        # improves the warm start. (max() guards against ever shortening a longer tspan.)
        tend = isempty(tstops) ? odesys.tspan[2] : max(maximum(tstops), odesys.tspan[2])
        odesys = ODE.remake(odesys; tspan = (odesys.tspan[1], tend))
        solver = PEprob.probinfo.solver.solver
        sol = ODE.solve(
            odesys, solver;
            tstops = tstops,
            callback = callbacks,
            abstol = PEprob.probinfo.solver.abstol,
            reltol = PEprob.probinfo.solver.reltol
        )
        sols[cid] = sol
    end
    return sols
end

# Get the start value for the steady-state pre-equilibration variable zss.
#
# zss is indexed by SIMULATION condition (zss[:, cidx]); the continuity constraints
# set z[:,1,0,cidx] = zss[:,cidx] and enforce the steady-state residual f(zss)=0 under
# the pre-equilibration condition's inputs. PEtab's get_odeproblem(condition =
# pre_eq_id => sim_id) internally pre-equilibrates (solve_pre_equilibrium!!) and returns
# an ODEProblem whose u0 is exactly that steady-state initial condition — so we read u0
# directly rather than re-solving a SteadyStateProblem.
function _get_zss_init(PEmodel::PEtabModel, PEprob::PEtabODEProblem, PEinfo::PEInfo)
    (; Nz, Nc) = PEinfo

    si        = PEprob.model_info.simulation_info
    cids      = Symbol.(_get_cids(PEmodel))
    sim_ids   = si.conditionids[:simulation]
    preeq_ids = si.conditionids[:pre_equilibration]
    p_nominal = PEtab.get_x(PEprob)

    zss_inits = zeros(Nz, Nc)
    for (cidx, cid) in enumerate(cids)
        pos = findfirst(==(cid), sim_ids)
        pos === nothing && continue   # condition not simulated (pre-eq only)
        oprob, _ = PEtab.get_odeproblem(p_nominal, PEprob;
                                        condition = preeq_ids[pos] => sim_ids[pos])
        zss_inits[:, cidx] = oprob.u0[1:Nz]
    end
    return zss_inits
end