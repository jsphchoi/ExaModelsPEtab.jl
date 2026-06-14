# Parameter Estimation Info (problem details for formulating the ExaModels build)
struct PEInfo{T <: Number}
    # Basic counts
    Np::Int     # number of parameters
    Nz::Int     # (v = 1,...,Nz) number of state variables
    Nc::Int     # (cidx = 1,...,Nc) number of experimental conditions
    Ncv::Int    # (cv = 1,...,Ncv) number of condition-dependent variables
    Nm::Int     # number of data measurements
    N::Int      # (i = 1,...,N) number of intervals
    K::Int      # (j/k = 0/1,...,K) number of interpolation points within each interval (k = 0,...,K)

    # Pre-compute once mesh-related values
    t_meas::Vector{T}               # all of the unique measurement times
    t_nodes::Vector{Float64}        # interval mesh boundary times t_i
    h::Vector{Float64}              # interval widths, t_i+1 - t_i
    taus::Vector{Float64}           # relative positions scaled to [0,1] of the interpolation points (roots of shifted Gauss-Legendre polynomials of order K)
    L1::Vector{Float64}             # evaluations of the lagrange basis polynomial at tau=1 for each j in K

    # Parameter scaling tags
    pscale::Vector{Symbol} # (m = 1,...,Np) per-parameter PEtab estimation scale (:log10/:log/:lin)

    # Event handling
    gate_vals::Array{Float64,3}         # Ng×N×Nc gate value on each (interval i, condition cidx)
    gate_vals_ss::Array{Float64,2}      # Ng×Nc gate value for the steady-state residual per condition
end

# Create PEInfo with:
# PEinfo = PEInfo(Np, Nz, Nc, Ncv, Nm, N, K, t_meas, t_nodes, h, taus, L1, pscale, gate_vals, gate_vals_ss)

# Unpack PEInfo with:
# (; Np, Nz, Nc, Ncv, Nm, N, K, t_meas, t_nodes, h, taus, L1, pscale, gate_vals, gate_vals_ss) = PEinfo