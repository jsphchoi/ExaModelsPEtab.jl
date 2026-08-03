module ExaModelsPEtabPlotsExt

import ExaModelsPEtab: ExaModelsPEtab, PEtabSolution
import Plots

const ALLOWED_SOLUTION_PLOTS = [
    :model_fit,
    :residuals,
    :standardized_residuals,
]

# Plots a solved model against the data, one condition per plot
Plots.@recipe function f(
        sol::PEtabSolution; plot_type = :model_fit, observable_ids = nothing,
        condition = nothing, observable_id_label = false
    )
    if !in(plot_type, ALLOWED_SOLUTION_PLOTS)
        error("Argument plot_type have an unrecognized value ($(plot_type)). Allowed \
               values are: $(ALLOWED_SOLUTION_PLOTS).")
    end

    if isnothing(observable_ids)
        observable_ids = ExaModelsPEtab._petab_observable_ids(sol.meta)
    else
        observable_ids = string.(observable_ids)
    end

    # No condition given plots the first one
    condition = isnothing(condition) ? last(first(ExaModelsPEtab._petab_conditions(sol.meta))) :
                                       condition
    title = condition isa Pair ?
            "Condition: $(first(condition)) => $(last(condition))" :
            "Condition: $(condition)"

    if plot_type == :model_fit
        plot_info = _plot_model_fit(sol, condition, observable_ids, observable_id_label)
    else
        plot_info = _plot_residuals(sol, condition, observable_ids, plot_type, observable_id_label)
    end

    seriestype --> plot_info.seriestype
    color --> plot_info.color
    label --> plot_info.label
    xlabel --> "Time"
    ylabel --> plot_info.y_label
    title --> title
    return plot_info.x, plot_info.y
end

function ExaModelsPEtab.petab_obs_comparison_plots(sol::PEtabSolution; kwargs...)
    comparison_dict = Dict()
    for (experiment_id, condition) in ExaModelsPEtab._petab_conditions(sol.meta)
        comparison_dict[experiment_id] = Dict()
        for observable_id in ExaModelsPEtab._petab_observable_ids(sol.meta)
            comparison_dict[experiment_id][observable_id] = Plots.plot(
                sol; observable_ids = [observable_id], condition = condition, kwargs...
            )
        end
    end
    return comparison_dict
end

# Measured points and fitted values, one scatter and one line per observable
function _plot_model_fit(sol, condition, observable_ids, observable_id_label)
    _seriestype = []
    _color = []
    _label = []
    x_vals = []
    y_vals = []

    for (obs_idx, obs_id) in enumerate(observable_ids)
        model_fits = ExaModelsPEtab._petab_observable(sol, condition, obs_id)
        # Plot args.
        append!(_seriestype, [:scatter, :line])
        append!(_color, [obs_idx, obs_idx])
        obs_label = ExaModelsPEtab._petab_observable_label(sol.meta, obs_id, observable_id_label)
        append!(_label, ["$(obs_label) ($type)" for type in ["measured", "fitted"]])

        # Measured plot values.
        push!(x_vals, model_fits.t_obs)
        push!(y_vals, model_fits.h_obs)
        # Fitted plot values.
        push!(x_vals, model_fits.t_mod)
        push!(y_vals, model_fits.h_mod)
    end

    # Set reshaped plot arguments
    n_obs = length(observable_ids)
    _seriestype = reshape(_seriestype, 1, 2n_obs)
    _color = reshape(_color, 1, 2n_obs)
    _label = reshape(_label, 1, 2n_obs)
    return (
        x = x_vals, y = y_vals, color = _color, label = _label,
        seriestype = _seriestype, y_label = "Model and observed values",
    )
end

# Residuals, on the observableTransformation scale
function _plot_residuals(sol, condition, observable_ids, plot_type, observable_id_label)
    standardized = plot_type == :standardized_residuals
    _y_label = standardized ? "Standardized residuals" : "Residuals (simulated output - data)"

    _seriestype = []
    _color = []
    _label = []
    x_vals = []
    y_vals = []

    for (obs_idx, obs_id) in enumerate(observable_ids)
        res = ExaModelsPEtab._petab_residuals(sol, condition, obs_id, standardized)
        isempty(res.t) && continue

        push!(x_vals, res.t)
        push!(y_vals, res.r)
        append!(_seriestype, [:scatter])
        append!(_color, [obs_idx])
        append!(_label, [ExaModelsPEtab._petab_observable_label(sol.meta, obs_id, observable_id_label)])
    end

    # Set reshaped plot arguments
    n_obs = length(x_vals)
    _seriestype = reshape(_seriestype, 1, n_obs)
    _color = reshape(_color, 1, n_obs)
    _label = reshape(_label, 1, n_obs)
    return (
        x = x_vals, y = y_vals, color = _color, label = _label,
        seriestype = _seriestype, y_label = _y_label,
    )
end

end
