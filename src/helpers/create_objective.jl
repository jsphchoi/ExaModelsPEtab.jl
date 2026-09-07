# Creates objective function / y, sigma auxiliary variable constraints
# Creates the collocation + continuity / initial conition / cv auxiliary variable constraints
function _create_objective(
        core::ExaModels.ExaCore,
        PEinfo::PEtabInfo
    )
    core = _create_y(core, PEinfo)

    core = _create_sigma(core, PEinfo)

    core = _create_nllh(core, PEinfo)

    core = _create_prior(core, PEinfo)

    return core
end

# TODO only need to actually create sigma if sigma is not a fixed variable

# Create y (observables at the measurements), observable_parameters (their observableParameter cells) and constraints
# y = fy(theta, z, cv, cvfixed, observable_parameters)
function _create_y(
        core::ExaModels.ExaCore,
        PEinfo::PEtabInfo
    )
    # Unpack PEtabInfo and model functions
    Nm = _get_Nm(PEinfo)
    cells = _get_cells([measurement.observable_parameters for measurement in PEinfo.measurements])
    fy = _get_fy(PEinfo, size(cells, 1)) # get observable functions fy[yidx](theta,z,cv,cvfixed,observable_parameters)

    # Create ExaModels variables
    observable_parameters0 = _get_values(PEinfo, cells)
    ExaModels.@add_var(core,
        observable_parameters,
        1:size(cells, 1), 1:Nm;
        start = observable_parameters0,
    )
    ExaModels.@add_var(core,
        y,
        1:Nm;
        start = _get_f0(PEinfo, fy, observable_parameters0),
    )

    # Create constraints
    core = _create_placeholder_constraints(core, PEinfo, observable_parameters, cells)
    core = _create_measurement_constraints(core, PEinfo, y, fy, observable_parameters)

    return core
end

# Create sigma (noise at the measurements), noise_parameters (their noiseParameter cells) and constraints
# sigma = fsigma(theta, z, cv, cvfixed, noise_parameters)
function _create_sigma(
        core::ExaModels.ExaCore,
        PEinfo::PEtabInfo
    )
    # Unpack PEtabInfo and model functions
    Nm = _get_Nm(PEinfo)
    cells = _get_cells([measurement.noise_parameters for measurement in PEinfo.measurements])
    fsigma = _get_fsigma(PEinfo, size(cells, 1)) # get noise functions fsigma[yidx](theta,z,cv,cvfixed,noise_parameters)

    # Create ExaModels variables
    noise_parameters0 = _get_values(PEinfo, cells)
    ExaModels.@add_var(core,
        noise_parameters,
        1:size(cells, 1), 1:Nm;
        start = noise_parameters0,
    )
    ExaModels.@add_var(core,
        sigma,
        1:Nm;
        start = _get_f0(PEinfo, fsigma, noise_parameters0),
    )

    # Create constraints
    core = _create_placeholder_constraints(core, PEinfo, noise_parameters, cells)
    core = _create_measurement_constraints(core, PEinfo, sigma, fsigma, noise_parameters)

    return core
end

# Create the Gaussian negative log-likelihood, residuals on the observableTransformation scale with PEtab.jl's constants
function _create_nllh(
        core::ExaModels.ExaCore,
        PEinfo::PEtabInfo
    )
    # Unpack variables
    y, sigma = core.y, core.sigma

    # Create objective iterator, (m, ymeas, transform)
    rows = [
        (m, measurement.measurement, PEinfo.observables[measurement.yidx].transform)
        for (m, measurement) in enumerate(PEinfo.measurements)
    ]

    # Create objective for lin
    itr = [(m, ymeas, 0.5 * log(2pi)) for (m, ymeas, transform) in rows if transform == :lin]
    if !isempty(itr)
        ExaModels.@add_obj(core,
            0.5 * ((y[m] - ymeas) / sigma[m])^2 + log(sigma[m]) + constant
            for (m, ymeas, constant) in itr
        )
    end

    # Create objective for log
    itr = [(m, log(ymeas), 0.5 * log(2pi) + log(ymeas)) for (m, ymeas, transform) in rows if transform == :log]
    if !isempty(itr)
        ExaModels.@add_obj(core,
            0.5 * ((log(y[m]) - log_ymeas) / sigma[m])^2 + log(sigma[m]) + constant
            for (m, log_ymeas, constant) in itr
        )
    end

    # Create objective for log10
    itr = [(m, log10(ymeas), 0.5 * log(2pi) + log(ymeas) + log(log(10))) for (m, ymeas, transform) in rows if transform == :log10]
    if !isempty(itr)
        ExaModels.@add_obj(core,
            0.5 * ((log10(y[m]) - log10_ymeas) / sigma[m])^2 + log(sigma[m]) + constant
            for (m, log10_ymeas, constant) in itr
        )
    end

    return core
end

# Create the negative log prior of every estimated parameter with an objectivePriorType
function _create_prior(
        core::ExaModels.ExaCore,
        PEinfo::PEtabInfo
    )
    # Unpack variables
    theta = core.theta

    # Create prior iterator, (j, parameter) per estimated parameter with a prior
    estimated = [parameter for parameter in PEinfo.parameters if parameter.estimate]
    rows = [(j, parameter) for (j, parameter) in enumerate(estimated) if parameter.prior_type != :none]
    for (j, parameter) in rows
        parameter.prior_type in (:parameterScaleNormal, :parameterScaleLaplace, :normal, :laplace, :uniform) ||
            throw(ArgumentError("objectivePriorType '$(parameter.prior_type)' of '$(parameter.parameter_id)' is not supported"))
    end

    # Create objective for parameterScaleNormal, on theta
    itr = [(j, parameter.prior_parameters[1], parameter.prior_parameters[2]) for (j, parameter) in rows if parameter.prior_type == :parameterScaleNormal]
    if !isempty(itr)
        ExaModels.@add_obj(core,
            0.5 * ((theta[j] - mu) / sd)^2 + log(sd) + 0.5 * log(2pi)
            for (j, mu, sd) in itr
        )
    end

    # Create objective for parameterScaleLaplace, on theta
    itr = [(j, parameter.prior_parameters[1], parameter.prior_parameters[2]) for (j, parameter) in rows if parameter.prior_type == :parameterScaleLaplace]
    if !isempty(itr)
        ExaModels.@add_obj(core,
            abs(theta[j] - mu) / b + log(2 * b)
            for (j, mu, b) in itr
        )
    end

    # Create objective for normal and laplace, on _linscale(theta)
    for scale in (:log10, :log, :lin)
        itr = [(j, parameter.prior_parameters[1], parameter.prior_parameters[2]) for (j, parameter) in rows if parameter.prior_type == :normal && parameter.scale == scale]
        if !isempty(itr)
            ExaModels.@add_obj(core,
                0.5 * ((_linscale(theta[j], scale) - mu) / sd)^2 + log(sd) + 0.5 * log(2pi)
                for (j, mu, sd) in itr
            )
        end
        itr = [(j, parameter.prior_parameters[1], parameter.prior_parameters[2]) for (j, parameter) in rows if parameter.prior_type == :laplace && parameter.scale == scale]
        if !isempty(itr)
            ExaModels.@add_obj(core,
                abs(_linscale(theta[j], scale) - mu) / b + log(2 * b)
                for (j, mu, b) in itr
            )
        end
    end

    # Create objective for uniform, the constant log(ub - lb)
    itr = [(j, parameter.prior_parameters[1], parameter.prior_parameters[2]) for (j, parameter) in rows if parameter.prior_type == :uniform]
    if !isempty(itr)
        ExaModels.@add_obj(core,
            log(ub - lb)
            for (j, lb, ub) in itr
        )
    end

    return core
end

# ----- helper functions -----

_get_Nm(PEinfo::PEtabInfo) = length(PEinfo.measurements)
_get_Ny(PEinfo::PEtabInfo) = length(PEinfo.observables)
_is_steadystate(PEinfo::PEtabInfo) = all(measurement -> isinf(measurement.time), PEinfo.measurements)

# fy[yidx](theta, z, cv, cvfixed, observable_parameters): observable formula of observable yidx
_get_fy(PEinfo, Nplaceholders) = _get_formula_functions(
    PEinfo, [observable.observable_formula for observable in PEinfo.observables], "observableParameter", Nplaceholders
)

# fsigma[yidx](theta, z, cv, cvfixed, noise_parameters): noise formula of observable yidx
_get_fsigma(PEinfo, Nplaceholders) = _get_formula_functions(
    PEinfo, [observable.noise_formula for observable in PEinfo.observables], "noiseParameter", Nplaceholders
)

# f[yidx](theta, z, cv, cvfixed, placeholders): formulas[yidx] compiled, {prefix}{n}_{observableId} as placeholders[n]
function _get_formula_functions(PEinfo, formulas, prefix, Nplaceholders)
    arguments = _get_arguments(PEinfo)
    rules = _get_substitutions(PEinfo, arguments)
    symbols = _get_symbols(PEinfo, arguments)
    Symbolics.@variables placeholders[1:Nplaceholders]
    return [
        Symbolics.build_function(
            Symbolics.fixpoint_sub(
                _parse_formula(
                    Meta.parse(formula),
                    merge(symbols, Dict("$(prefix)$(n)_$(observable.observable_id)" => placeholders[n] for n in 1:Nplaceholders))
                ),
                rules;
                fold = Val(true)
            ),
            arguments.theta, arguments.z, arguments.cv, arguments.cvfixed, placeholders;
            expression = Val{false},
            nanmath = false
        )
        for (observable, formula) in zip(PEinfo.observables, formulas)
    ]
end

# Symbol of every formula id: model states, assignment rules and parameters, and parameters-table ids outside the model as theta or values
function _get_symbols(PEinfo, arguments)
    (; sys) = PEinfo.model
    symbols = Dict{String, Any}(
        _get_id(symbol) => symbol
        for symbol in [MTK.unknowns(sys); MTK.parameters(sys); [equation.lhs for equation in MTK.observed(sys)]]
    )
    j = 0
    for parameter in PEinfo.parameters
        parameter.estimate && (j += 1)
        haskey(symbols, parameter.parameter_id) && continue
        symbols[parameter.parameter_id] = parameter.estimate ? _linscale(arguments.theta[j], parameter.scale) : parameter.value
    end
    return symbols
end

# Symbolic expression of a table formula, ids resolved through symbols
_parse_formula(number::Number, symbols) = Symbolics.Num(number)

function _parse_formula(id::Symbol, symbols)
    haskey(symbols, string(id)) || throw(ArgumentError("'$id' is not a model symbol, a parameter, or a placeholder"))
    return symbols[string(id)]
end

function _parse_formula(expression::Expr, symbols)
    expression.head == :call || throw(ArgumentError("formula '$expression' is not supported"))
    return getfield(Base, expression.args[1])((_parse_formula(argument, symbols) for argument in expression.args[2:end])...)
end

# points[m]: (i, k) of the collocation point z[:,cidx,i,k] at the time of measurement m, node 1 is (1, 0) and node i + 1 is (i, K)
function _get_measurement_points(PEinfo)
    points = Vector{Tuple{Int, Int}}(undef, _get_Nm(PEinfo))
    for (m, measurement) in enumerate(PEinfo.measurements)
        node = _get_index(measurement.time, PEinfo.nodes[measurement.cidx])
        points[m] = node == 1 ? (1, 0) : (node - 1, PEinfo.K)
    end
    return points
end

# z0_measurements[:,m]: initial guess of the states at measurement m, z0 at its collocation point or zss0 of its condition
function _get_z0_measurements(PEinfo)
    z0_measurements = Matrix{Float64}(undef, _get_Nz(PEinfo), _get_Nm(PEinfo))
    if _is_steadystate(PEinfo)
        for (m, measurement) in enumerate(PEinfo.measurements)
            z0_measurements[:,m] = PEinfo.zss0[PEinfo.preeq_idxs[measurement.cidx]]
        end
    else
        points = _get_measurement_points(PEinfo)
        for (m, measurement) in enumerate(PEinfo.measurements)
            i, k = points[m]
            z0_measurements[:,m] = PEinfo.z0[:,measurement.cidx,i,k+1]
        end
    end
    return z0_measurements
end

# cells[n,m]: cell n of measurement m, 0.0 past its cells
function _get_cells(measurement_cells)
    N = maximum(length, measurement_cells; init = 0)
    return Union{Float64, Int}[
        n <= length(cells) ? cells[n] : 0.0
        for n in 1:N, cells in measurement_cells
    ]
end

# Values of the cells at theta0
function _get_values(PEinfo, cells)
    scales = [parameter.scale for parameter in PEinfo.parameters if parameter.estimate]
    value(cell) = cell isa Int ? _linscale(PEinfo.theta0[cell], scales[cell]) : cell
    return Float64[value(cell) for cell in cells]
end

# f0[m]: f[yidx](theta, z, cv, cvfixed, placeholders) of measurement m at the initial guess
function _get_f0(PEinfo, f, placeholders0)
    z0_measurements = _get_z0_measurements(PEinfo)
    cvfixed = _get_cvfixed(PEinfo, PEinfo.conditions)
    return [
        f[measurement.yidx](PEinfo.theta0, z0_measurements[:,m], PEinfo.cv0[:,measurement.cidx], cvfixed[:,measurement.cidx], placeholders0[:,m])
        for (m, measurement) in enumerate(PEinfo.measurements)
    ]
end

# Create placeholder constraints, placeholder = {fixed value, theta}
function _create_placeholder_constraints(
        core::ExaModels.ExaCore,
        PEinfo::PEtabInfo,
        placeholder,
        cells
    )
    # Unpack variables
    theta = core.theta

    # Parse cells
    scales = [parameter.scale for parameter in PEinfo.parameters if parameter.estimate]
    rows = [(n, m, cells[n,m]) for n in axes(cells, 1), m in axes(cells, 2)]

    # Create constraints for placeholder = fixed value
    itr = [(n, m, cell) for (n, m, cell) in rows if cell isa Float64]
    if !isempty(itr)
        ExaModels.@add_con(core,
            placeholder[n,m] - value
            for (n, m, value) in itr
        )
    end

    # Create constraints for placeholder = _linscale(theta)
    for scale in (:log10, :log, :lin)
        itr = [(n, m, cell) for (n, m, cell) in rows if cell isa Int && scales[cell] == scale]
        isempty(itr) && continue
        ExaModels.@add_con(core,
            placeholder[n,m] - _linscale(theta[j], scale)
            for (n, m, j) in itr
        )
    end

    return core
end

# Create constraints variable[m] = f[yidx](theta, z, cv, cvfixed, placeholder) at the collocation point of every measurement
function _create_measurement_constraints(
        core::EMC.CollocationExaCore,
        PEinfo::PEtabInfo,
        variable,
        f,
        placeholder
    )
    # Unpack variables
    z, theta, cv = core.z, core.theta, core.cv

    # Create constraint iterator
    cvfixed = _get_cvfixed(PEinfo, PEinfo.conditions)
    points = _get_measurement_points(PEinfo)
    rows = [
        (m, measurement.yidx, measurement.cidx, points[m][1], points[m][2], Tuple(cvfixed[:,measurement.cidx]))
        for (m, measurement) in enumerate(PEinfo.measurements)
    ]

    # Create constraints, one per observable
    for yidx in 1:_get_Ny(PEinfo)
        itr = [(m, cidx, i, k, cvfixed_m) for (m, yidx_m, cidx, i, k, cvfixed_m) in rows if yidx_m == yidx]
        isempty(itr) && continue
        ExaModels.@add_con(core,
            variable[m] - f[yidx](theta[:], z[:,cidx,i,k], cv[:,cidx], cvfixed_m, placeholder[:,m])
            for (m, cidx, i, k, cvfixed_m) in itr
        )
    end

    return core
end

# Create constraints variable[m] = f[yidx](theta, zss, (), cvfixed, placeholder) at the steady state of every measurement
function _create_measurement_constraints(
        core::ExaModels.ExaCore,
        PEinfo::PEtabInfo,
        variable,
        f,
        placeholder
    )
    # Unpack variables
    zss, theta = core.zss, core.theta

    # Create constraint iterator
    cvfixed = _get_cvfixed(PEinfo, PEinfo.conditions)
    rows = [
        (m, measurement.yidx, PEinfo.preeq_idxs[measurement.cidx], Tuple(cvfixed[:,measurement.cidx]))
        for (m, measurement) in enumerate(PEinfo.measurements)
    ]

    # Create constraints, one per observable
    for yidx in 1:_get_Ny(PEinfo)
        itr = [(m, ssidx, cvfixed_m) for (m, yidx_m, ssidx, cvfixed_m) in rows if yidx_m == yidx]
        isempty(itr) && continue
        ExaModels.@add_con(core,
            variable[m] - f[yidx](theta[:], zss[:,ssidx], (), cvfixed_m, placeholder[:,m])
            for (m, ssidx, cvfixed_m) in itr
        )
    end

    return core
end
