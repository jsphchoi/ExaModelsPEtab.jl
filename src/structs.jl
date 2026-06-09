# Parameter Estimation Info (examodels nlp problem details)
struct PEInfo{T <: Number}
    Np::Int     # number of parameters
    Nz::Int     # (v = 1,...,Nz) number of state variables
    Nc::Int     # (cidx = 1,...,Nc) number of experimental conditions
    Ncv::Int    # (cv = 1,...,Ncv) number of condition-dependent variables
    Nm::Int     # number of data measurements
    Ny::Int     # number of model observables
    N::Int      # (i = 1,...,N) number of intervals
    K::Int      # (j/k = 0/1,...,K) number of interpolation points within each interval (k = 0,...,K)
    t_meas::Vector{T}
    t_vec_mesh::Vector{Float64}
    h::Vector{Float64}
    taus::Vector{Float64}
    L1::Vector{Float64}
    pscale::Vector{Symbol} # (m = 1,...,Np) per-parameter PEtab estimation scale (:log10/:log/:lin)
    # Fixed-time event support. SBMLImporter folds piecewise(time>T,…) into the RHS as the
    # smooth form b*a+(1-b)*c, where b is a gating parameter (__parameter_ifelseN) toggled by a
    # callback at a fixed time. We keep these gates LIVE in the RHS (see _get_rhs_funcs) and pass
    # their per-(interval,condition) values as data, instead of freezing them at the parametermap
    # default (which is the pre-trigger value, not the value PEtab actually simulates with).
    gate_syms::Vector{Symbolics.Num}  # (g = 1,...,Ng) gating parameters kept live in the RHS
    gate_vals::Array{Float64,3}       # Ng×N×Nc gate value on each (interval i, condition cidx)
    gate_vals_ss::Array{Float64,2}    # Ng×Nc gate value for the steady-state residual per condition
    t_nodes::Vector{Float64}          # ORIGINAL mesh boundary times T_0..T_N (sol.t of the finest
                                      # condition); measurement/event tstops appear bit-exactly, so
                                      # _get_dict_t_tidx maps times -> node index by exact `==`.
end
# PEinfo = PEInfo(Np, Nz, Nc, Ncv, Nm, Ny, N, K, t_meas, t_vec_mesh, h, taus, L1, pscale, gate_syms, gate_vals, gate_vals_ss, t_nodes)
# (; Np, Nz, Nc, Ncv, Nm, Ny, N, K, t_meas, t_vec_mesh, h, taus, L1, pscale, gate_syms, gate_vals, gate_vals_ss, t_nodes) = PEinfo