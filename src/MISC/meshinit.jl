# Integrator-step mesh initialization
#
# The nominal ODE solve, forced through the anchors (t0, measurement times, switch
# times) as tstops, already concentrates its accepted steps where the dynamics are
# stiff. Take those step times as mesh nodes, keeping every stride-th interior step,
# so stride = 2 halves the integrator's resolution while inheriting its placement.
# Conditions are topped up to a common N by splitting the widest intervals.
#
# default: mesh_init = :integrator, falling back to the uniform mesh with a warning
# when the nominal simulation fails.

# Model-size defaults, PEtab.jl style: the seam where benchmark-calibrated thresholds
# land. Overridable through examodel_petab's mesh_opts.
_default_mesh_init(spec::PEtabSpec) = :integrator
_meshinit_defaults(spec::PEtabSpec) = (; stride = 2)

# Keep every stride-th interior step of each anchor gap, anchors always kept exactly.
# Near-duplicate steps (the integrator landing next to a tstop) collapse first.
function _subsample_steps(steps::AbstractVector{Float64}, anchors::AbstractVector{Float64},
                          stride::Int)
    tol = 1e-10 * (last(anchors) - first(anchors))
    kept = Float64[first(anchors)]
    for j in 1:(length(anchors) - 1)
        lo, hi = anchors[j], anchors[j + 1]
        interior = Float64[]
        for t in steps
            lo + tol < t < hi - tol || continue
            (isempty(interior) || t - last(interior) > tol) && push!(interior, t)
        end
        append!(kept, interior[stride:stride:end])
        push!(kept, hi)
    end
    return kept
end

# Integrator-step mesh, numbered steps. Falls back to the uniform mesh on any nominal
# simulation failure, never errors out of the build.
function _init_mesh(spec::PEtabSpec, modelsys::PEtabModelSys, theta0::Vector{Float64},
                    K::Int; subdivide::Int, variant::Symbol, stride::Int)
    stride >= 1 || error("ExaModelsPEtab: stride must be >= 1.")

    # 1. Nominal dense solve per condition, tstops on the anchors, steps subsampled
    base  = _mesh_base(spec, variant)
    solver = _default_solver(spec.Nz, spec.Np)
    times = Vector{Vector{Float64}}(undef, spec.Nc)
    for cidx in 1:spec.Nc
        try
            sol, _ = _condition_solution(spec, modelsys, theta0, cidx, last(base[cidx]),
                                         solver; tstops = base[cidx])
            times[cidx] = _subsample_steps(sol.t, base[cidx], stride)
        catch err
            @warn "ExaModelsPEtab: nominal simulation failed, keeping the uniform mesh" cidx exception = err
            return _build_mesh(spec; subdivide = subdivide, variant = variant)
        end
    end
    if variant === :shared
        merged = sort!(reduce(vcat, times))
        tol = 1e-10 * (last(merged) - first(merged))
        dedup = Float64[merged[1]]
        for t in merged
            t - last(dedup) > tol && push!(dedup, t)
        end
        times = [dedup for _ in 1:spec.Nc]
    end

    # 2. Common N across conditions: each step gap is one interval, widest split first
    N = maximum(length(t) - 1 for t in times)
    nodes = Matrix{Float64}(undef, spec.Nc, N + 1)
    ref_of = Vector{Vector{Int}}(undef, spec.Nc)
    for cidx in 1:spec.Nc
        nodes[cidx, :], ref_of[cidx] = _refine(times[cidx], N, 1)
    end

    # 3. Measurement intervals by index (anchors are step nodes exactly), control values
    meas_iidx = _measurement_intervals(spec, times, ref_of)
    u_vals, u_vals_ss = _control_values(spec, nodes, N)
    return PEtabMesh(variant, nodes, N, meas_iidx, u_vals, u_vals_ss)
end
