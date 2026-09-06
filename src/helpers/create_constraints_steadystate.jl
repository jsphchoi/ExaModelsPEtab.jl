# Creates the steady-state / cv auxiliary variable constraints
function _create_constraints_steadystate(
        core::ExaCore,
        PEinfo::PEtabInfo
    )
    core = _create_zss_constraints(core, PEinfo)

    core = _create_cv_constraints(core, PEinfo)

    return core
end

# Create steady-state constraints f(zss) = 0 and conservation constraints W zss = W z(t=0) for every pre-equilibration condition
function _create_zss_constraints(
        core::ExaCore,
        PEinfo::PEtabInfo
    )
    _has_cv(PEinfo) && throw(ArgumentError("cv with pre-equilibration is not supported"))

    # Unpack variables and model functions
    zss, theta = core.zss, core.theta
    f = _get_f(PEinfo)

    # Create steady-state constraint iterator, u after every event
    cvfixed = _get_cvfixed(PEinfo, PEinfo.preeq_conditions)
    times = _get_event_times(PEinfo.events, PEinfo.preeq_conditions, PEinfo.parameters, PEinfo.model.parametermap)
    u = [
        something(_get_u_value(PEinfo, id, times[:,ssidx], Inf), _get_default(PEinfo.model, id))
        for id in _get_u_ids(PEinfo), ssidx in 1:_get_Nss(PEinfo)
    ]
    itr = [
        (ssidx, Tuple(cvfixed[:,ssidx]), Tuple(u[:,ssidx]))
        for ssidx in 1:_get_Nss(PEinfo)
    ]

    # Conservation laws W[ssidx] zss[:,ssidx] = b[ssidx] and the rows of f they do not replace
    W, b, keep_rows = _get_conservation_laws(PEinfo, cvfixed, u)

    # Create steady-state constraints
    for v in 1:_get_Nz(PEinfo)
        itr_v = [row for row in itr if v in keep_rows[row[1]]]
        isempty(itr_v) && continue
        ExaModels.@add_con(core,
            f[v](theta[:], zss[:,ssidx], (), cvfixed_ss0, u_ss, 0.0)
            for (ssidx, cvfixed_s, u_s) in itr_v
        )
    end

    # Create conservation constraints
    itr_b = [(ssidx, k, b[ssidx][k]) for ssidx in 1:_get_Nss(PEinfo) for k in eachindex(b[ssidx])]
    isempty(itr_b) && return core
    con = ExaModels.@add_con(core,
        -b_sk
        for (ssidx, k, b_sk) in itr_b
    )
    itr_W = [
        (pos, W[ssidx][k,v], v, ssidx)
        for (pos, (ssidx, k, _)) in enumerate(itr_b) for v in 1:_get_Nz(PEinfo)
    ]
    ExaModels.@add_con!(core, con,
        pos => W_skv * zss[v,ssidx]
        for (pos, W_skv, v, ssidx) in itr_W
    )

    return core
end

# ----- helper functions -----

# dfdz(theta, z, cv, cvfixed, u, t): Jacobian of the right-hand side in z
function _get_dfdz(PEinfo)
    arguments = _get_arguments(PEinfo)
    rules = _get_substitutions(PEinfo, arguments)
    rhs = [Symbolics.fixpoint_sub(equation.rhs, rules; fold = Val(true)) for equation in MTK.equations(PEinfo.model.sys)]
    return Symbolics.build_function(
        Symbolics.jacobian(rhs, collect(arguments.z)),
        arguments...;
        expression = Val{false},
        nanmath = false
    )[1]
end

# z(t=0) of the pre-equilibration condition with targets cvfixed_s, a number for every state
function _get_zic(arguments, zic_sym, cvfixed_s)
    rules = Dict(arguments.cvfixed[cvfixedidx] => value for (cvfixedidx, value) in enumerate(cvfixed_s))
    zic = [Symbolics.value(Symbolics.substitute(expr, rules)) for expr in zic_sym]
    all(value -> value isa Number, zic) || throw(ArgumentError("pre-equilibration initial state depends on theta, which is not supported"))
    return Float64.(zic)
end

# W[ssidx], b[ssidx], keep_rows[ssidx]: left null space of df/dz at (theta0, zss0), b = W z(t=0), rows of f kept
function _get_conservation_laws(PEinfo, cvfixed_ss, u_ss; tol = 1e-12)
    Nz = _get_Nz(PEinfo)
    arguments = _get_arguments(PEinfo)
    zic_sym = _get_zic_sym(PEinfo, arguments)
    dfdz = _get_dfdz(PEinfo)
    W, b, keep_rows = Matrix{Float64}[], Vector{Float64}[], Vector{Int}[]
    for ssidx in 1:_get_Nss(PEinfo)
        J = dfdz(PEinfo.theta0, PEinfo.zss0[ssidx], Float64[], cvfixed_ss[:,ssidx], u_ss[:,ssidx], 0.0)
        F = LinearAlgebra.svd(J)
        r = count(<(tol * F.S[1]), F.S)
        W_s = Matrix(transpose(F.U[:,end-r+1:end]))
        drop = LinearAlgebra.qr(W_s, LinearAlgebra.ColumnNorm()).p[1:r]
        push!(W, W_s)
        push!(b, W_s * _get_zic(arguments, zic_sym, cvfixed_ss[:,ssidx]))
        push!(keep_rows, setdiff(1:Nz, drop))
    end
    return W, b, keep_rows
end
