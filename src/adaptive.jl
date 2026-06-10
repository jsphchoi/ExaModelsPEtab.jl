# adaptive.jl — outer moving-mesh (r-refinement) loop for the collocation NLP.
#
# Motivation: the collocation mesh is built adaptive to the NOMINAL parameters (t_nodes =
# sol.t of the nominal ODE solve). As the optimizer moves θ, the stiff transients move with
# it and the frozen mesh no longer resolves them — the collocation residual blows up and
# MadNLP restoration-spirals (e.g. Fujita's post-event transient: warm |viol| 174 at K=4).
#
# Fix: after each NLP solve, relocate mesh nodes to equidistribute a solution-based monitor,
# then re-solve warm-started. Because h/t_ij/gate values are now ExaModels PARAMETERS (see
# collocation.jl _create_lagrange), the mesh is updated via set_value! WITHOUT rebuilding the
# model — provided node RELOCATION preserves structure. It does, under two invariants:
#   (1) measurement and event times stay ANCHORED as fixed nodes (objective maps measurements
#       to integer node indices; moving a measurement off its node would change that map), and
#   (2) the count of FREE nodes within each anchor-bounded segment is preserved (so downstream
#       measurement node indices never shift).
# Within those invariants we redistribute free-node POSITIONS only — real adaptivity (pull a
# segment's nodes toward its transient) with zero rebuild. Segments too sparse to resolve their
# transient are the h-refinement (rebuild) fallback, handled separately.

# de Boor equidistribution within each anchor-bounded segment.
#   t_nodes   : sorted N+1 node times (t_nodes[1]=t0, t_nodes[end]=tmax)
#   is_anchor : Bool N+1, true at endpoints + measurement/event nodes (fixed, never moved)
#   rho       : per-interval monitor density, length N, >= 0 (higher => more nodes pulled in)
# Returns new N+1 node times: anchors unchanged, free nodes within each segment relocated so the
# cumulative monitor mass is equidistributed. Per-segment free-node COUNT is preserved.
function _redistribute_nodes(t_nodes::Vector{Float64}, is_anchor::Vector{Bool}, rho::Vector{Float64})
    Np1 = length(t_nodes)
    @assert length(is_anchor) == Np1 "is_anchor length must match t_nodes"
    @assert length(rho) == Np1 - 1 "rho must have one entry per interval"
    @assert is_anchor[1] && is_anchor[end] "endpoints must be anchored"
    new_nodes = copy(t_nodes)
    anchor_idx = findall(is_anchor)
    for s in 1:length(anchor_idx) - 1
        ia = anchor_idx[s]; ib = anchor_idx[s + 1]
        M = ib - ia - 1                       # free (interior) nodes in this segment
        M == 0 && continue                    # nothing to move
        A = t_nodes[ia]; B = t_nodes[ib]
        sub_t   = t_nodes[ia:ib]              # current sub-node times (length M+2)
        widths  = diff(sub_t)                 # current sub-interval widths (length M+1)
        mass    = rho[ia:ib - 1] .* widths    # monitor mass per current sub-interval
        total   = sum(mass)
        cum     = vcat(0.0, cumsum(mass))     # cumulative mass at each sub-node, cum[end]=total
        if !(total > 0) || !isfinite(total)
            for m in 1:M                       # degenerate monitor -> uniform spacing
                new_nodes[ia + m] = A + (B - A) * m / (M + 1)
            end
        else
            for m in 1:M                       # place node at equal monitor-mass fractions
                target = total * m / (M + 1)
                k      = clamp(searchsortedlast(cum, target), 1, length(mass))
                frac   = mass[k] > 0 ? (target - cum[k]) / mass[k] : 0.0
                new_nodes[ia + m] = sub_t[k] + frac * widths[k]
            end
        end
    end
    return new_nodes
end

# Per-interval monitor density rho (length N) from a solved state trajectory z[Nz,N,K+1,Nc].
# Uses per-state-range-normalized slope magnitude (an arc-length / curvature proxy): intervals
# where the (normalized) state changes fast get a higher density and thus more nodes. A floor
# (fraction of the mean) keeps quiescent intervals from being stripped of all nodes.
function _state_monitor(z::Array{Float64,4}, h::Vector{Float64}; floor_frac = 0.05)
    Nz, N, Kp1, Nc = size(z)
    rng = [maximum(@view z[v, :, :, :]) - minimum(@view z[v, :, :, :]) for v in 1:Nz]
    rng = [r > 0 ? r : 1.0 for r in rng]
    rho = zeros(N)
    for i in 1:N
        act = 0.0
        for cidx in 1:Nc, v in 1:Nz
            slope = abs(z[v, i, Kp1, cidx] - z[v, i, 1, cidx]) / (rng[v] * h[i])
            act += slope^2
        end
        rho[i] = sqrt(act)
    end
    m = sum(rho) / N
    rho .+= floor_frac * (m > 0 ? m : 1.0)
    return rho
end
