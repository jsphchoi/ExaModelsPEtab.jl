# define PEtabInfo data structures

# SBMLImporter output
struct PEtabModel{S, C}
    sys::S              # ModelingToolkit system
    speciemap
    parametermap
    callbacks::C
end

# One parameters-table row
struct PEtabParameter
    parameter_id::String
    estimate::Bool
    value::Float64      # nominalValue
    lb::Float64
    ub::Float64
    scale::Symbol       # :lin, :log, :log10
    prior_type::Symbol  # objectivePriorType, :none without
    prior_parameters::Vector{Float64}
end

# One conditions-table row, target_values resolved to a number, a theta index, or NaN (pre-equilibration value)
struct PEtabCondition
    condition_id::String
    target_ids::Vector{String}
    target_values::Vector{Union{Float64, Int}}
end

# One observables-table row
struct PEtabObservable
    observable_id::String
    observable_formula::String
    noise_formula::String
    transform::Symbol   # :lin, :log, :log10
end

# One SBML event at a fixed event time
struct PEtabEvent
    target_ids::Vector{String}
    target_values::Vector{String}
    event_time::Float64
end

# One measurements-table row
struct PEtabMeasurement
    time::Float64
    measurement::Float64
    observable_parameters::Vector{Union{Float64, Int}}
    noise_parameters::Vector{Union{Float64, Int}}
    cidx::Int           # simulation condition
    yidx::Int           # observable
end

# Everything the ExaModel build reads
struct PEtabInfo{M <: PEtabModel}
    model::M
    parameters::Vector{PEtabParameter}
    conditions::Vector{PEtabCondition}          # simulation conditions, cidx = 1:Nc
    preeq_conditions::Vector{PEtabCondition}    # pre-equilibration conditions, ssidx = 1:Nss
    preeq_idxs::Vector{Int}                     # cidx => ssidx, 0 without pre-equilibration
    observables::Vector{PEtabObservable}
    events::Vector{PEtabEvent}
    measurements::Vector{PEtabMeasurement}
    nodes::Vector{Vector{Float64}}              # nodes[cidx]
    K::Int
    theta0::Vector{Float64}                     # theta0[]
    z0::Array{Float64, 4}                       # z0[v,cidx,i,k]
    cv0::Matrix{Float64}                        # cv0[cvidx,cidx]
    zss0::Vector{Vector{Float64}}               # zss0[ssidx]
end
