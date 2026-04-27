# Parameter Estimation Info (examodels nlp problem details)
struct PEInfo
    Np::Int32           # number of parameters
    Nz::Int32           # (v = 1,...,Nz) number of state variables
    Nc::Int32           # (cidx = 1,...,Nc) number of experimental conditions
    N::Int32            # (i = 1,...,N) number of intervals 
    K::Int32            # (j/k = 0/1,...,K) number of interpolation points within each interval (k = 0,...,K)
    t_meas::Vector{T} where T<:Number
    t_vec_mesh::Vector{Float64}
    h::Vector{Float64}
    taus::Vector{Float64}
    L1::Vector{Float64}
end