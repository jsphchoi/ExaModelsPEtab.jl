# get initial guess for z and a lot of other things
function _get_z_init(PEmodel::PEtabModel, PEprob::PEtabODEProblem, K::Int)
    # Get unique t_meas
    PEtable = PEmodel.petab_tables
    t_meas = sort(unique(filter(t -> !iszero(t), PEtable[:measurements][!,:time])))

    # Solve all experimental conditions at nominal values
    sol = _solve_conds(get_x(PEprob), PEmodel, PEprob, t_meas)

    # Construct mesh
    h = diff(sol[argmax(k -> length(sol[k].t), keys(sol))].t) # choose mesh based on finest sol.t
    N = length(h) # number of intervals
    taus = _taus(K) # get interpolation points
    t_mesh = [(cumsum(h) .- h)[i] + taus[j+1]*h[i] for i in 1:N, j in 0:K] # construct t_ij mesh
    t_vec_mesh = Array(reshape(t_mesh',N*(K+1))) # vectorize t_ij mesh

    # Get constants
    Nz = Int64(PEprob.model_info.nstates) # number of state variables
    Nc = sol.count # number of experimental conditions
    L1 = [_eval_l(j,1.0,taus) for j in 0:K] # for interval continuity constraints later

    # Interpolate solution at every point in the mesh
    sol_at_mesh= [
        sol[cid](t) 
        for t in t_vec_mesh, cid in Symbol.(_get_cids(PEmodel))
    ]
    z_init = permutedims(reshape(stack(sol_at_mesh), Nz, N, K+1, Nc), (1, 2, 3, 4))

    return z_init, Nz, N, K, Nc, t_meas, t_vec_mesh, h, taus, L1
end

# Returns ::Dict{(condition id)::Symbol, (solution)}
function _solve_conds(p_nominal, PEmodel::PEtabModel, PEprob::PEtabODEProblem, tstops)
    sols = Dict{Symbol, Any}()
    for cid in Symbol.(_get_cids(PEmodel))
        odesys, callbacks = PEtab.get_odeproblem(p_nominal, PEprob; condition = cid) # TODO LOOK! gets callbacks for each condition
        solver = PEprob.probinfo.solver.solver
        sol = OrdinaryDiffEq.solve(
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

# get initial guess for zss by solving steady-state model for each condition
function _get_zss_init(PEmodel::PEtabModel, PEprob::PEtabODEProblem, PEinfo::PEInfo)
    # TODO
    return zss_init
end