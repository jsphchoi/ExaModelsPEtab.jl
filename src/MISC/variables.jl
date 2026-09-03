# NLP variable creation

# (*) Main function for creating ExaModels decision variables (*)
# Creation order fixes the x layout: p, z, y, sigma, cv, zss
function _create_variables(c::ExaCore, spec::PEtabSpec, mesh::PEtabMesh,
                           theta0::Vector{Float64}, z_init, zss_init)
    c = _create_p(c, spec, theta0)
    c = _create_z(c, spec, mesh, z_init)
    c = _create_y(c, spec)
    c = _create_sigma(c, spec)
    if spec.Ncv >= 1
        c = _create_cv(c, spec, theta0)
    end
    if _has_preequilibration(spec)
        c = _create_zss(c, spec; init = zss_init)
    end
    return c
end

# Creates ExaModels decision variables for unknown parameters
# p[1:Np]
function _create_p(c::ExaCore, spec::PEtabSpec, theta0::Vector{Float64})
    ExaModels.@add_var(c,
        p,
        1:spec.Np;
        lvar  = spec.plb,
        uvar  = spec.pub,
        start = theta0
    )
    return c
end

# Creates ExaModels decision variables for discretized states
# z[1:Nz,1:Nc,1:N,0:K], the condition axis declared or carried by the mesh family
function _create_z(c::ExaCore, spec::PEtabSpec, mesh::PEtabMesh, z_init)
    if _mesh_axis(mesh, spec.Nc)
        EMC.@add_var_collocation(c,
            z,
            1:spec.Nz;
            start = z_init,
            lvar = -Inf,
            uvar = Inf
        )
    else
        EMC.@add_var_collocation(c,
            z,
            1:spec.Nz, 1:spec.Nc;
            start = z_init,
            lvar = -Inf,
            uvar = Inf
        )
    end
    return c
end

# Creates ExaModels (auxiliary) decision variables for observable model variable
# y[1:Nm]
function _create_y(c::ExaCore, spec::PEtabSpec)
    # Impose nonnegativity for log transformed variables
    y_LB = [t === :log || t === :log10 ? 0.0 : -Inf for t in spec.meas_transform]
    ExaModels.@add_var(c,
        y,
        1:spec.Nm;
        lvar = y_LB # set_start! added later
    )
    return c
end

# Creates ExaModels (auxiliary) decision variables for standard deviation of error
# sigma[1:Nm]
function _create_sigma(c::ExaCore, spec::PEtabSpec)
    # sigma is a standard deviation and the objective takes log(sigma), so it needs a floor
    # clear of MadNLP's bound_relax_factor (1e-8 by default), which relaxes a tighter one
    # straight through zero and makes log throw
    ExaModels.@add_var(c,
        sigma,
        1:spec.Nm; # set_start! added later
        lvar = 1e-6
    )
    return c
end

# Creates ExaModels decision variables for condition-dependent variables
# cv[1:Ncv,1:Ncc]
function _create_cv(c::ExaCore, spec::PEtabSpec, theta0::Vector{Float64})
    (; Ncv, Ncc) = spec
    # Only two possible paths: cv = fixed value or cv = p, so init with fixed val or p0
    cv_init = zeros(Float64, Ncv, Ncc)
    for col in 1:Ncc, cvidx in 1:Ncv
        cv_init[cvidx, col] = _cell_value(spec, theta0, cvidx, col)
    end
    ExaModels.@add_var(c,
        cv,
        1:Ncv, 1:Ncc;
        start = cv_init
    )
    return c
end

# Creates ExaModels decision variables for steady-state state (used for x0SSpre and pure steady-state only models)
# zss[1:Nz,1:Nc]
function _create_zss(c::ExaCore, spec::PEtabSpec; init)
    ExaModels.@add_var(c,
        zss,
        1:spec.Nz, 1:spec.Nc;
        start = init,
        lvar = -Inf,
        uvar = Inf
    )
    return c
end
