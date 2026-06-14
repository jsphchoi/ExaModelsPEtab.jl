# Absolute tolerance for mesh-time comparisons (segment coverage and width caps).
const _MESH_TOL = 1e-9

# get initial guess for z and a lot of other things
function _get_z_init(PEmodel::PEtabModel, PEprob::PEtabODEProblem, K::Int)
    # Get unique experimental measurement values
    PEtable = PEmodel.petab_tables
    t_meas = sort(unique(filter(t -> !iszero(t), PEtable[:measurements][!,:time])))

    # Get unique fixed-time event values
    t_events = _get_event_times(PEmodel, PEprob)

    # Force integrator to stop at t_meas and t_events
    t_stops  = sort(unique(vcat(t_meas, t_events)))

    # Solve all experimental conditions at nominal values (tstops = measurements ∪ events ⇒ nodes)
    p_nominal = PEtab.get_x(PEprob)
    sol = _solve_conds(p_nominal, PEmodel, PEprob, t_stops)

    # Construct the mesh
    cids     = _get_cids(PEmodel)                                       # canonical cidx order
    sol_t    = [collect(Float64, sol[Symbol(cid)].t) for cid in cids]   # each condition's node times
    t_start  = minimum(first, sol_t)                                    # absolute minimum start time
    T_global = maximum(last,  sol_t)                                    # absolute maximum end time

    # for each condition, find last measurement time t_end (defines integration time horizon)
    meas_df = PEtable[:measurements]
    t_end   = zeros(Float64, length(cids))
    for row in eachrow(meas_df)
        cidx = findfirst(==(string(row[:simulationConditionId])), cids)
        cidx === nothing && continue
        t_end[cidx] = max(t_end[cidx], Float64(row[:time]))
    end

    # Identify simulation time horizon boundaries 
    # (only really matters if different conditions have different spans)
    B = sort(unique(vcat(t_start, filter(>(t_start), t_end), T_global)))

    # Global mesh hueristic: within each (B)oundary, adopt the finest mesh
    # over the conditions which runs over this interval, while imposing the
    # smallest largest stepsize over every condition (min max h cap)
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

    # Interpolate each condition's solution at every collocation point
    sol_at_mesh= [
        sol[cid](t)
        for t in t_vec_mesh, cid in Symbol.(_get_cids(PEmodel))
    ]
    # Reshpae vectorized solution to match ExaModels variable box z[v,i,k,cidx]
    z_init = permutedims(reshape(stack(sol_at_mesh), Nz, K+1, N, Nc), (1, 3, 2, 4))

    return z_init, Nz, N, K, Nc, t_meas, h, taus, L1, t_nodes
end

# Finds the largest interval width within a segment of time [a,b]
function _seg_maxwidth(tvec, a, b)
    nodes = sort!(unique!(Float64[a; b; filter(t -> a < t < b, tvec)]))
    return maximum(diff(nodes))
end

# If there are multiple conditions which span some boundary interval, choose the finest mesh
function _pick_owner(covering, t_end, sol_t, a, b)
    tmin  = minimum(t_end[cidx] for cidx in covering)
    cands = [cidx for cidx in covering if abs(t_end[cidx] - tmin) <= _MESH_TOL]
    return argmax(cidx -> count(t -> a <= t <= b, sol_t[cidx]), cands)
end

# If the owner mesh violates the min max h cap, then divide it into two equal subintervals
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
        push!(out, hi)
    end
    return out
end

# Returns ::Dict{(condition id)::Symbol, (solution)} of solution profiles for every simulation condition
function _solve_conds(p_nominal, PEmodel::PEtabModel, PEprob::PEtabODEProblem, tstops)
    sols = Dict{Symbol, Any}()
    si        = PEprob.model_info.simulation_info
    has_preeq = si.has_pre_equilibration
    sim_ids   = si.conditionids[:simulation]
    preeq_ids = si.conditionids[:pre_equilibration]
    # for every simulation condition...
    for cid in Symbol.(_get_cids(PEmodel))
        cond_arg = cid
        if has_preeq
            pos = findfirst(==(cid), sim_ids)
            pos === nothing && continue # condition not simulated (pre-eq only)
            cond_arg = preeq_ids[pos] => sim_ids[pos]
        end
        odesys, callbacks = PEtab.get_odeproblem(p_nominal, PEprob; condition = cond_arg)

        # Integrate every condition over the entire mesh span so we can obtain z_init for the whole mesh
        t_end = isempty(tstops) ? odesys.tspan[2] : max(maximum(tstops), odesys.tspan[2])
        odesys = ODE.remake(odesys; tspan = (odesys.tspan[1], t_end))
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

# Returns steady-state pre-eq values at nominal p (for x0SSpre and pure steady-state models)
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