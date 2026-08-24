# Requirement-driven collocation mesh

"""
    PEtabMesh

Per-condition mesh rows built from required times only: t0, measurement times, control
switch times, each base interval subdivided to a common N. Records measurement interval
indices as integers at construction.
"""
struct PEtabMesh
    variant::Symbol             # :percond | :shared
    nodes::Matrix{Float64}      # Nc x (N+1)
    N::Int
    meas_iidx::Vector{Int}      # midx -> interval index; 0 = the t0 node
    u_vals::Array{Float64, 3}   # Nu x N x Nc, constant per interval
    u_vals_ss::Matrix{Float64}  # Nu x Nc, evaluated at t = 0
end

function _build_mesh(spec::PEtabSpec; subdivide::Int, variant::Symbol)
    subdivide >= 1 || error("ExaModelsPEtab: subdivide must be >= 1.")
    variant in (:percond, :shared) || error("ExaModelsPEtab: unknown mesh variant :$variant.")
    base = [_required_times(spec, cidx) for cidx in 1:spec.Nc]
    # A condition measured only at t = 0 gets a padding horizon, keeping the family equal-N
    tends = last.(base)
    if any(==(0.0), tends)
        tpad = minimum((t for t in tends if t > 0.0); init = Inf)
        isfinite(tpad) || error("ExaModelsPEtab: no condition has a measurement after t = 0.")
        for b in base
            last(b) == 0.0 && push!(b, tpad)
        end
    end
    if variant === :shared
        merged = sort!(unique!(reduce(vcat, base)))
        base = [merged for _ in 1:spec.Nc]
    end

    # Common N across conditions, each base interval split subdivide ways at minimum
    N = subdivide * maximum(length(b) - 1 for b in base)
    nodes = Matrix{Float64}(undef, spec.Nc, N + 1)
    ref_of = Vector{Vector{Int}}(undef, spec.Nc)
    for cidx in 1:spec.Nc
        nodes_c, ref_of_c = _refine(base[cidx], N, subdivide)
        nodes[cidx, :] = nodes_c
        ref_of[cidx] = ref_of_c
    end

    meas_iidx = _measurement_intervals(spec, base, ref_of)
    u_vals, u_vals_ss = _control_values(spec, nodes, N)
    return PEtabMesh(variant, nodes, N, meas_iidx, u_vals, u_vals_ss)
end

# Required base times of one condition: t0 = 0, its measurement times, its switch times.
# A condition measured only at t = 0 returns [0.0] and is padded by _build_mesh
function _required_times(spec::PEtabSpec, cidx::Int)::Vector{Float64}
    t0 = 0.0
    times = Float64[t0]
    tend = t0
    for m in 1:spec.Nm
        spec.meas_cidx[m] == cidx || continue
        t = spec.meas_time[m]
        t > t0 && push!(times, t)
        tend = max(tend, t)
    end
    tend > t0 && append!(times, _switch_times(spec.controls, spec.simcv[cidx], t0, tend))
    return sort!(unique!(times))
end

# Refine base nodes to exactly N intervals: subdivide per base gap, then split the widest
# current subinterval (lowest index on ties). Base nodes land exactly on refined nodes.
function _refine(base::Vector{Float64}, N::Int, subdivide::Int)
    nb = length(base)
    widths = diff(base)
    counts = fill(subdivide, nb - 1)
    while sum(counts) < N
        counts[argmax(widths ./ counts)] += 1
    end

    nodes = Vector{Float64}(undef, N + 1)
    ref_of = Vector{Int}(undef, nb)
    nodes[1] = base[1]
    ref_of[1] = 1
    pos = 1
    for j in 1:(nb - 1)
        h = widths[j] / counts[j]
        for s in 1:counts[j]
            nodes[pos + s] = s == counts[j] ? base[j + 1] : base[j] + s * h
        end
        pos += counts[j]
        ref_of[j + 1] = pos
    end
    return nodes, ref_of
end

# midx -> interval index, by base-node position (measurement times are nodes by construction)
function _measurement_intervals(spec::PEtabSpec, base, ref_of)::Vector{Int}
    meas_iidx = Vector{Int}(undef, spec.Nm)
    for m in 1:spec.Nm
        cidx = spec.meas_cidx[m]
        t = spec.meas_time[m]
        j = searchsortedfirst(base[cidx], t)
        (j <= length(base[cidx]) && base[cidx][j] == t) || error(
            "ExaModelsPEtab: measurement row $m (t = $t) is not a mesh node of condition " *
            "'$(spec.conds[cidx].sim)'.")
        meas_iidx[m] = ref_of[cidx][j] - 1
    end
    return meas_iidx
end

# Control values per interval, evaluated at the interval midpoint (switches sit on nodes)
function _control_values(spec::PEtabSpec, nodes::Matrix{Float64}, N::Int)
    (; Nu, Nc, controls, simcv, precv) = spec
    u_vals = Array{Float64, 3}(undef, Nu, N, Nc)
    u_vals_ss = Matrix{Float64}(undef, Nu, Nc)
    for cidx in 1:Nc
        for i in 1:N, g in 1:Nu
            tmid = (nodes[cidx, i] + nodes[cidx, i + 1]) / 2
            u_vals[g, i, cidx] = _control_value(controls[g, simcv[cidx]], tmid)
        end
        sscol = precv[cidx] > 0 ? precv[cidx] : simcv[cidx]
        for g in 1:Nu
            u_vals_ss[g, cidx] = _control_value(controls[g, sscol], 0.0)
        end
    end
    return u_vals, u_vals_ss
end

# Nodes input for CollocationExaCore: mesh family when the mesh axis carries conditions
function _core_nodes(mesh::PEtabMesh, Nc::Int)
    _mesh_axis(mesh, Nc) || return mesh.nodes[1, :]
    return [mesh.nodes[cidx, :] for cidx in 1:Nc]
end

# The mesh axis doubles as the condition axis only for :percond with several conditions
_mesh_axis(mesh::PEtabMesh, Nc::Int) = mesh.variant === :percond && Nc > 1
