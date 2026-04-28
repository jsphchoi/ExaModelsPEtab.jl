# Parameter Estimation Info (examodels nlp problem details)
struct PEInfo{T <: Number}
    Np::Int     # number of parameters
    Nz::Int     # (v = 1,...,Nz) number of state variables
    Nc::Int     # (cidx = 1,...,Nc) number of experimental conditions
    Ncv::Int    # (cv = 1,...,Ncv) number of condition-dependent variables
    Nm::Int     # number of data measurements
    N::Int      # (i = 1,...,N) number of intervals 
    K::Int      # (j/k = 0/1,...,K) number of interpolation points within each interval (k = 0,...,K)
    t_meas::Vector{T}
    t_vec_mesh::Vector{Float64}
    h::Vector{Float64}
    taus::Vector{Float64}
    L1::Vector{Float64}
end
# PEinfo = PEInfo(Np, Nz, Nc, Ncv, Nm, N, K, t_meas, t_vec_mesh, h, taus, L1)