# Parameter Estimation Info (problem details for formulating the ExaModels build)
struct PEInfo{T <: Number}
    # Basic counts
    Np::Int     # number of parameters
    Nz::Int     # (v = 1,...,Nz) number of state variables
    Nc::Int     # (cidx = 1,...,Nc) number of experimental conditions
    Ncv::Int    # (cv = 1,...,Ncv) number of condition-dependent variables
    Nm::Int     # number of data measurements

    # Pre-compute once measurement times
    t_meas::Vector{T}               # all of the unique measurement times

    # Parameter scaling tags
    pscale::Vector{Symbol} # (m = 1,...,Np) per-parameter PEtab estimation scale (:log10/:log/:lin)

    # Event handling
    gate_vals::Array{Float64,3}         # Ng×N×Nc gate value on each (interval i, condition cidx)
    gate_vals_ss::Array{Float64,2}      # Ng×Nc gate value for the steady-state residual per condition
end
