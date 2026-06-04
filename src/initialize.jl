# get initial guess for z and a lot of other things
function _get_z_init(PEmodel::PEtabModel, PEprob::PEtabODEProblem, K::Int)
    # Get unique t_meas
    PEtable = PEmodel.petab_tables
    t_meas = sort(unique(filter(t -> !iszero(t), PEtable[:measurements][!,:time])))

    # Solve all experimental conditions at nominal values
    p_nominal = PEtab.get_x(PEprob)
    sol = _solve_conds(p_nominal, PEmodel, PEprob, t_meas)

    # Construct the mesh from the finest (most-points) condition's solution. Every condition is
    # integrated over the SAME span [tstart, max(t_meas)] (see _solve_conds), so whichever
    # condition we pick spans the full mesh and — because tstops = the union of all measurement
    # times was forced on every solve — contains every measurement time as a node, so every
    # t_meas lands on a mesh boundary.
    h = diff(sol[argmax(k -> length(sol[k].t), keys(sol))].t)
    N = length(h) # number of intervals
    taus = _taus(K) # get interpolation points
    t_mesh = [(cumsum(h) .- h)[i] + taus[j+1]*h[i] for i in 1:N, j in 0:K] # construct t_ij mesh
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

    return z_init, Nz, N, K, Nc, t_meas, t_vec_mesh, h, taus, L1
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