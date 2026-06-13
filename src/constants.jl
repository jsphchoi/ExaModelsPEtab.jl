# Evaluates the Lagrange basis polynomial lⱼ(τ) given {τⱼ...}
function _eval_l(
        j::Int64,
        tau::Float64,
        taus::Vector{Float64}
    )::Float64
    # j ∈ {0,...,K} : Index for the Lagrange basis polynomial at the interpolation point τⱼ.
    # tau ∈ [0,1]   : Evaluate the Lagrange basis polynomial at τ.
    # taus          : Vector of shifted Gauss-Legendre roots (including τ₀ = 0)
    @assert 0 <= j <= length(taus)-1 "Index j must be in {0,...,K}."
    return prod(
        (tau - taus[k+1])/(taus[j+1] - taus[k+1]) 
        for k in 0:length(taus)-1 if k != j
    )
end

# Evaluates the τ derivative of the Lagrange basis polynomial dlⱼ(τₖ)/dτ given {τⱼ...}
function _eval_dldtau(
        j::Int64,
        k::Int64,
        taus::Vector{Float64}
    )::Float64
    # j ∈ {0,...,K} : Index for the Lagrange basis polynomial at the interpolation point τⱼ.
    # tau ∈ [0,1]   : Evaluate the Lagrange basis polynomial at τ.
    # taus          : Vector of shifted Gauss-Legendre roots (including τ₀ = 0)
    @assert 0 <= j <= length(taus)-1 "Index j must be in {0,...,K}."
    if j == k
        return sum(
            1/(taus[j+1] - taus[m+1])
            for m = 0:length(taus)-1 if m != j;
            init = 0.0
        )
    else
        return prod(
            (taus[k+1] - taus[m+1])/(taus[j+1] - taus[m+1]) 
            for m in 0:length(taus)-1 if m != j && m != k;
            init = 1.0
        )/(taus[j+1] - taus[k+1])
    end
end

# Compute n Gauss-Legendre nodes on [-1, 1] using Newton's method
# with Chebyshev initial guesses. Sufficient for moderate n (K ≤ ~50).
function _legendre_nodes(n::Int64)::Vector{Float64}
    n == 0 && return Float64[]
    n == 1 && return [0.0]

    m = (n + 1) ÷ 2
    x = Vector{Float64}(undef, n)

    for i in 1:m
        # Chebyshev-based initial guess for the i-th node from the left
        xi = -cos(π * (i - 0.25) / (n + 0.5))

        # Newton iterations on the Legendre polynomial P_n
        for _ in 1:16
            p_prev, p_curr = 1.0, xi
            for k in 1:(n - 1)
                p_prev, p_curr = p_curr, ((2k + 1) * xi * p_curr - k * p_prev) / (k + 1)
            end
            # Derivative via the recurrence: P_n' = n(P_{n-1} - x P_n)/(1 - x²)
            dp = n * (p_prev - xi * p_curr) / (1 - xi^2)
            Δ  = p_curr / dp
            xi -= Δ
            abs(Δ) < 2eps() && break
        end

        x[i]         = xi   # left half (negative)
        x[n + 1 - i] = -xi  # right half (mirror symmetry)
    end
    isodd(n) && (x[m] = 0.0)  # pin the central node exactly

    return x  # sorted ascending
end

# Returns K+1 interpolation points on [0, 1):
#  - τ[1] = 0 (left endpoint of the element)
#  - τ[2:K+1] = K Gauss-Legendre nodes shifted from [-1,1] to (0,1)
# K is the number of collocation points per finite element.
function _taus(K::Int64)::Vector{Float64}
    # K ∈ {1,...} : Number of interpolation points within each finite element
    @assert K >= 1 "Number of interpolation points must be a positive integer."
    nodes = _legendre_nodes(K)       # K nodes on [-1, 1], ascending
    taus  = Vector{Float64}(undef, K + 1)
    taus[1] = 0.0                    # left endpoint  ← (−1 + 1)/2
    for i in 1:K
        taus[i + 1] = (nodes[i] + 1) / 2   # shift to (0, 1)
    end
    return taus
end