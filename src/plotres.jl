# PEtab labelling carried on the built model in the core's `refs` (`tag` holds the mesh)
struct PEtabMeta{TB}
    petab_tables::TB            # PEmodel.petab_tables (measurements/observables/conditions/parameters)
    transforms::Vector{Symbol}  # (midx = 1,...,Nm) observableTransformation of each measurement row
    cids::Vector{String}        # (cidx = 1,...,Nc) simulation conditionId
    pnames::Vector{String}      # (m = 1,...,Np) estimated parameter name
    pscale::Vector{Symbol}      # (m = 1,...,Np) estimation scale (:log10/:log/:lin)
end

# Attaches the labelling to the core, so ExaModel(c) carries it over
function _attach_petab_meta(c::ExaCore, PEmodel::PEtabModel, PEprob::PEtabODEProblem, PEinfo::PEInfo)
    meta = PEtabMeta(
        PEmodel.petab_tables,
        _get_meas_transforms(PEmodel),
        _get_cids(PEmodel),
        string.(PEprob.xnames),
        PEinfo.pscale,
    )
    return ExaCore(c; refs = (; getfield(c, :refs)..., petab = meta))
end

"""
    PEtabSolution

A solved model read back in PEtab terms, built by [`plotsol`](@ref).

- `meta` measurement/observable/condition labelling
- `p` estimated parameters, on the PEtab estimation scale (`meta.pscale`)
- `y`, `sigma` simulated observable and error of each measurement row
- `objective` negative log-likelihood at `p`
"""
struct PEtabSolution{T, TB}
    meta::PEtabMeta{TB}
    p::Vector{T}
    y::Vector{T}
    sigma::Vector{T}
    objective::T
end

"""
    plotsol(model, result)

Reads a solver `result` back against `model` as a [`PEtabSolution`](@ref).

# Example
```julia
using Plots
plot(plotsol(m, madnlp(m)); plot_type = :model_fit, condition = "model1_data1")
```
"""
function plotsol(model, result)
    meta = model.petab   # PEtabMeta, off refs
    p = Array(ExaModels.solution(result, model.p))
    return PEtabSolution(
        meta,
        p,
        Array(ExaModels.solution(result, model.y)),
        Array(ExaModels.solution(result, model.sigma)),
        eltype(p)(result.objective),
    )
end

"""
    petab_obs_comparison_plots(sol::PEtabSolution; kwargs...)

One plot per (condition, observable), as a `Dict` keyed by condition then observableId.
Requires `Plots`.
"""
function petab_obs_comparison_plots end

# (!!!) Returns the measurement rows (midx, == the y/sigma index) of one observable on one
# condition, given as a conditionId, a pre-equilibration => simulation Pair, or nothing for all
function _petab_index_data(meta::PEtabMeta, condition, obs_id::String)::Vector{Int}
    df   = meta.petab_tables[:measurements]
    keep = _petab_cells(df, :observableId) .== obs_id
    condition === nothing && return findall(keep)

    preeq_id, sim_id = condition isa Pair ? (string(first(condition)), string(last(condition))) :
                                            (nothing, string(condition))
    keep .&= _petab_cells(df, :simulationConditionId) .== sim_id
    if preeq_id !== nothing
        :preequilibrationConditionId in propertynames(df) || error(
            "ExaModelsPEtab: condition '$preeq_id => $sim_id' given, but the measurements " *
            "table has no preequilibrationConditionId column."
        )
        keep .&= _petab_cells(df, :preequilibrationConditionId) .== preeq_id
    end
    return findall(keep)
end

# A conditionId column as trimmed strings; a missing cell must not read as the id "missing"
_petab_cells(df, col::Symbol)::Vector{String} =
    [(ismissing(v) || isnothing(v)) ? "" : strip(string(v)) for v in df[!, col]]

# Every condition in the measurements table, as a plot key => condition argument pair
function _petab_conditions(meta::PEtabMeta)::Vector{Pair{String, Any}}
    df   = meta.petab_tables[:measurements]
    sims = _petab_cells(df, :simulationConditionId)
    if !(:preequilibrationConditionId in propertynames(df))
        return Pair{String, Any}[cid => cid for cid in unique(sims)]
    end
    preeqs = _petab_cells(df, :preequilibrationConditionId)
    out    = Pair{String, Any}[]
    for (preeq, sim) in unique(collect(zip(preeqs, sims)))
        # a blank cell means no pre-equilibration for that row
        isempty(preeq) ? push!(out, sim => sim) : push!(out, "$preeq=>$sim" => (preeq => sim))
    end
    return out
end

# Measured and fitted values of one observable on one condition, sorted along t. Fitted values
# sit at the measurement times only, since y exists nowhere else
function _petab_observable(sol::PEtabSolution, condition, obs_id::String)
    idata = _petab_index_data(sol.meta, condition, obs_id)
    isempty(idata) && return (t_obs = Float64[], h_obs = Float64[], t_mod = Float64[], h_mod = Float64[])

    df = sol.meta.petab_tables[:measurements]
    t  = Float64.(df[idata, :time])
    ord = sortperm(t)
    return (
        t_obs = t[ord],
        h_obs = Float64.(df[idata, :measurement])[ord],
        t_mod = t[ord],
        h_mod = sol.y[idata][ord],
    )
end

# Residual of one observable on one condition, on the observableTransformation scale, both
# sides transformed. `standardized` divides by sigma
function _petab_residuals(sol::PEtabSolution, condition, obs_id::String, standardized::Bool)
    idata = _petab_index_data(sol.meta, condition, obs_id)
    isempty(idata) && return (t = Float64[], r = Float64[])

    df = sol.meta.petab_tables[:measurements]
    t  = Float64.(df[idata, :time])
    r  = [
        _transform_meas(sol.y[midx], sol.meta.transforms[midx]) -
        _transform_meas(Float64(df[midx, :measurement]), sol.meta.transforms[midx])
        for midx in idata
    ]
    standardized && (r ./= sol.sigma[idata])
    ord = sortperm(t)
    return (t = t[ord], r = r[ord])
end

# Puts a value on its observableTransformation scale
@inline _transform_meas(val, tr::Symbol) =
    tr === :log   ? log(val)   :
    tr === :log10 ? log10(val) :
                    val            # :lin

# The observableIds, in observables-table order
_petab_observable_ids(meta::PEtabMeta)::Vector{String} =
    string.(meta.petab_tables[:observables][!, :observableId])

# An observable's formula, or its id when observable_id_label is set
function _petab_observable_label(meta::PEtabMeta, obs_id::String, observable_id_label::Bool)
    observable_id_label && return obs_id
    df   = meta.petab_tables[:observables]
    iobs = findfirst(==(obs_id), string.(df[!, :observableId]))
    return iobs === nothing ? obs_id : string(df[iobs, :observableFormula])
end
