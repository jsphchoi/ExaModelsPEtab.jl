# the ExaModelsPEtab API function evaluate_objective
"""
    evaluate_objective(filename, theta)

Evaluate the objective function value at `theta` from a [PEtab](https://github.com/PEtab-dev/PEtab) `.yaml` file.

# Arguments
- `filename` : PEtab `.yaml` file
- `theta` : estimated parameter vector in its `parameterScale`

# Example
```julia
using ExaModelsPEtab, MadNLP

theta = [1.5, 2.0]

obj = evaluate_objective("path/to/petab.yaml", theta)
```
"""
evaluate_objective(filename::String, theta) = _evaluate_objective(_parse_yaml(filename), theta)

# Objective at theta from a forward ODE solve at the initial-solve settings
# the nllh and prior of the ExaModel on the parsed tables or a PEtabInfo
function _evaluate_objective(
        petab,
        theta
    )
    # Simulate every condition at theta
    sols, sols_ss = _get_sols(petab, theta, _get_odesolver(_get_model_size(petab)))

    # Resolve the observable and noise formula of every measurement
    arguments, yvalue = _get_arguments(petab), _get_yvalue()
    exprs = _get_exprs(petab, arguments, yvalue, [Meta.parse(observable.observable_formula) for observable in petab.observables])
    sigmas, _ = _get_sigmas(petab, arguments, yvalue)
    cv, cvfixed = _get_cv(petab, theta), _get_cvfixed(petab, petab.conditions)

    # Sum the negative log-likelihood over the measurements, one compile per distinct formula
    fys, fsigmas = Dict{Any, Any}(), Dict{Any, Any}()
    nllh = 0.0
    for (m, measurement) in enumerate(petab.measurements)
        cidx, transform = measurement.cidx, _get_transform(petab, m)
        z = isinf(measurement.time) ? sols_ss[petab.preeq_idxs[cidx]].u[end] : sols[cidx](measurement.time)
        fy = get!(() -> _get_fy(exprs[m], arguments), fys, exprs[m])
        fsigma = get!(() -> _get_fsigma(sigmas[m], arguments, yvalue), fsigmas, sigmas[m])
        y = fy(theta, z, cv[:,cidx], cvfixed[:,cidx])
        sigma = fsigma(theta, cv[:,cidx], cvfixed[:,cidx], (y,))
        h = transform == :log ? log(y) : transform == :log10 ? log10(y) : y
        nllh += 0.5 * ((h - _get_ymeas(measurement, transform)) / sigma)^2 + log(sigma) + _get_constant(measurement, transform)
    end

    # Add the negative log prior of every estimated parameter
    estimated = [parameter for parameter in petab.parameters if parameter.estimate]
    return nllh + sum(_get_prior(parameter, theta[j]) for (j, parameter) in enumerate(estimated); init = 0.0)
end

# Negative log prior of an estimated parameter at theta_j, the formulas of _create_prior
function _get_prior(parameter::PEtabParameter, theta_j)
    (; prior_type, prior_parameters, scale) = parameter
    prior_type == :none && return 0.0
    prior_type in (:parameterScaleNormal, :parameterScaleLaplace, :normal, :laplace, :uniform) ||
        throw(ArgumentError("objectivePriorType '$prior_type' of '$(parameter.parameter_id)' is not supported"))
    prior_type == :uniform && return log(prior_parameters[2] - prior_parameters[1])
    value = prior_type in (:parameterScaleNormal, :parameterScaleLaplace) ? theta_j : _linscale(theta_j, scale)
    if prior_type in (:parameterScaleNormal, :normal)
        mu, sd = prior_parameters
        return 0.5 * ((value - mu) / sd)^2 + log(sd) + 0.5 * log(2pi)
    else
        mu, b = prior_parameters
        return abs(value - mu) / b + log(2 * b)
    end
end
