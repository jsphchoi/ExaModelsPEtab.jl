###############################################################################
# ISOLATION TEST: Crauste WITHOUT auxiliary y / sigma variables.
#
# Hypothesis under test: the auxiliary-variable strategy (y[midx], sigma[midx]
# decision variables bound to z/p via equality constraints) is what broke
# Crauste convergence after it was introduced.
#
# This script builds the SAME collocation NLP for Crauste but:
#   * does NOT create the y or sigma auxiliary decision variables,
#   * does NOT create the y/sigma binding equality constraints,
#   * instead inlines the state-variable interpolation expression and the
#     numeric noise value directly into the objective.
#
# For Crauste every observable is a single state variable, every noise value
# is a fixed numeric literal (the measurement-table noiseParameters column),
# and every measurement time is nonzero. So each objective term is exactly
#       ( Σ_j L1[j+1] * z[zidx, idx, j, cidx]  -  ymeas )^2 / sigma_val
# which is precisely what the aux constraints would force y/sigma to equal.
#
# Run:  julia --project scratch_manual_crauste.jl [K]
###############################################################################

using ExaModels
using PEtab
import ModelingToolkitBase as MTK
using Symbolics
import OrdinaryDiffEq as ODE
import SteadyStateDiffEq as SSDE
using CUDA, MadNLPGPU
using CUDSS   # activates MadNLPGPU CUDSSSolver (GPU sparse linear solver)

const SRC = joinpath(@__DIR__, "src")
for f in ("structs.jl","constants.jl","utils.jl","initialize.jl",
          "variables.jl","collocation.jl","continuity.jl","objective.jl","userfuncs.jl")
    include(joinpath(SRC, f))
end

K = length(ARGS) >= 1 ? parse(Int, ARGS[1]) : 5

yaml = joinpath(@__DIR__, "examples", "Benchmark-Models",
                "Crauste_CellSystems2017", "Crauste_CellSystems2017.yaml")

PEmodel = PEtab.PEtabModel(yaml)
PEprob  = PEtab.PEtabODEProblem(PEmodel)

println("############# Crauste (MANUAL, no aux y/sigma), K = $K #############")

###############################################################################
# Build the ExaCore manually, skipping _create_y / _create_sigma.
###############################################################################
c = ExaModels.ExaCore(; backend = CUDABackend(), concrete = Val(true))

# --- decision variables: p and z ONLY (no y, no sigma) ----------------------
c, Np = _create_p(c, PEprob)
c, Nz, N, K, Nc, t_meas, t_vec_mesh, h, taus, L1 = _create_z(c, PEmodel, PEprob, K)

Ncv = length(_get_cv_syms(PEmodel))                                   # 0 for Crauste
Nm  = length(eachrow(PEmodel.petab_tables[:measurements]))
Ny  = length(_get_obsids(PEmodel))
PEinfo = PEInfo(Np, Nz, Nc, Ncv, Nm, Ny, N, K, t_meas, t_vec_mesh, h, taus, L1)
@assert Ncv == 0 "manual Crauste script assumes no condition-dependent variables"

# --- dynamics + continuity constraints (unchanged) --------------------------
c = _create_collocation(c, PEmodel, PEprob, PEinfo)
c = _create_continuity(c, PEmodel, PEprob, PEinfo)

###############################################################################
# Inlined objective (negative-log-likelihood-style weighted least squares),
# directly substituting state interpolation + numeric sigma. No aux vars.
###############################################################################
z = c.z
measurements_df = PEmodel.petab_tables[:measurements]
observables_df  = PEmodel.petab_tables[:observables]
dict_cid_cidx   = _get_dict_cid_cidx(PEmodel)
dict_t_tidx     = _get_dict_t_tidx(h, t_meas)

# state symbols (strip the "(t)" call wrapper), matched by name to obs formula
z_syms = [
    Symbolics.Num(Symbolics.variable(Symbol(split(string(zs), "(")[1])))
    for zs in _get_z_syms(PEprob)
]
dict_obsid_row = Dict(
    string(observables_df[i, :observableId]) => observables_df[i, :]
    for i in 1:size(observables_df, 1)
)

_num(v) = v isa Number ? Float64(v) : parse(Float64, strip(string(v)))

# Each measurement term is  ( Σ_j w_j*z[v,i,j,c] - ymeas )^2 / s   with w_j = L1[j+1].
# DO NOT nest sum() inside @add_obj — that kills ExaModels' SIMD kernel. Instead
# pre-expand the square into fixed-form per-term kernels via two iterators:
#   diagonal  (per j):     (w_j^2/s)*z_j^2 + (-2*ymeas*w_j/s)*z_j + ymeas^2/(s*(K+1))
#   bilinear  (per j<j'):  (2*w_j*w_j'/s)*z_j*z_j'
# (the ymeas^2/s constant is split evenly across the K+1 diagonal nodes)
itr_diag = Tuple{Int,Int,Int,Int,Float64,Float64,Float64}[]  # (v,i,j,c, c1,c2,c3)
itr_bil  = Tuple{Int,Int,Int,Int,Int,Float64}[]              # (v,i,j,jp,c, cb)
# t=0 single-node terms (initial-condition node) — none for Crauste, but handled
itr_ic   = Tuple{Int,Int,Float64,Float64}[]                  # (v,c, ymeas, s)

for midx in 1:Nm
    row     = measurements_df[midx, :]
    obs_id  = string(row[:observableId])
    formula = string(dict_obsid_row[obs_id][:observableFormula])

    parsed = Meta.parse(formula)
    @assert parsed isa Symbol "manual script only handles single-state observables; got $formula"
    obs_sym = Symbolics.Num(Symbolics.variable(parsed))
    v       = findfirst(x -> isequal(x, obs_sym), z_syms)
    @assert v !== nothing "could not match observable $formula to a state variable"

    cidx  = dict_cid_cidx[string(row[:simulationConditionId])]
    i     = dict_t_tidx[Float64(row[:time])]
    ymeas = _num(row[:measurement])
    s     = _num(row[:noiseParameters])   # numeric noise value, inlined directly
    @assert s > 0 "non-positive sigma value $s at measurement $midx"

    if i == 0
        push!(itr_ic, (v, cidx, ymeas, s))
    else
        for j in 0:K
            wj = L1[j+1]
            push!(itr_diag, (v, i, j, cidx,
                             wj^2 / s,                # c1: z_j^2
                             -2 * ymeas * wj / s,     # c2: z_j
                             ymeas^2 / (s * (K + 1))))# c3: constant share
            for jp in (j+1):K
                push!(itr_bil, (v, i, j, jp, cidx, 2 * wj * L1[jp+1] / s))
            end
        end
    end
end

# min Σ over expanded fixed-form kernels (SIMD-friendly: no sum() inside)
if !isempty(itr_diag)
    ExaModels.@add_obj(c,
        c1 * z[v,i,j,cidx]^2 + c2 * z[v,i,j,cidx] + c3
        for (v, i, j, cidx, c1, c2, c3) in itr_diag
    )
end
if !isempty(itr_bil)
    ExaModels.@add_obj(c,
        cb * z[v,i,j,cidx] * z[v,i,jp,cidx]
        for (v, i, j, jp, cidx, cb) in itr_bil
    )
end
if !isempty(itr_ic)
    # single-node t=0 term: (z - ymeas)^2/s = (1/s)z^2 + (-2ymeas/s)z + ymeas^2/s
    ExaModels.@add_obj(c,
        (1/s) * z[v,1,0,cidx]^2 + (-2*ymeas/s) * z[v,1,0,cidx] + ymeas^2/s
        for (v, cidx, ymeas, s) in itr_ic
    )
end

println("inlined ", Nm - length(itr_ic), " interp measurements -> ",
        length(itr_diag), " diagonal + ", length(itr_bil), " bilinear kernels + ",
        length(itr_ic), " t=0 kernels (no aux y/sigma, no sum() in objective)")

###############################################################################
# Build + solve on GPU.
###############################################################################
println("\nBuilding manual Crauste ExaModel on the GPU (CUDABackend)...")
@time m = ExaModels.ExaModel(c)
@show m.meta.nvar
@show m.meta.ncon
DoF = m.meta.nvar - m.meta.ncon
println("degrees of freedom (nvar - ncon) = ", DoF, "   (Np = ", Np, ")")

# warm-start objective (z/p already carry the good PEtab initial guesses)
x0 = m.meta.x0
println("objective at warm start = ", ExaModels.obj(m, x0))

println("\nRunning MadNLP (GPU, cuDSS), tol = 1e-6 ...")
result = madnlp(m; linear_solver = MadNLPGPU.CUDSSSolver,
                   tol = 1e-6, max_wall_time = 600.0, max_iter = 1_000_000)

println("\n=== RESULT (manual, no aux y/sigma) ===")
@show result.status
@show result.objective
@show result.iter
@show result.counters.total_time
