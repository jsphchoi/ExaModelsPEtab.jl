###############################################################################
# Crauste with PARAMETERS ON THE PEtab ESTIMATION SCALE (log10/log/lin per the
# parameter file), i.e. decision variable p := theta, physical param = 10^theta.
#
# Goal:
#   (1) verify the transformed build is mathematically consistent with the
#       linear build (identical warm-start objective + constraint residuals),
#   (2) solve the transformed problem WITHOUT aux y/sigma vars (manual obj),
#   (3) solve the transformed problem WITH aux y/sigma vars (package obj).
#
# Crauste specifics that keep this simple: 5 states, all-numeric initial
# conditions, no condition-dependent vars, autonomous ODE. So p appears ONLY in
# the collocation RHS — continuity and both objective variants never touch p.
#
# Run:  julia --project scratch_logp_crauste.jl [K] [phase]
#   phase = consistency | solve | all   (default all)
###############################################################################

using ExaModels
using PEtab
import ModelingToolkitBase as MTK
using Symbolics
import OrdinaryDiffEq as ODE
import SteadyStateDiffEq as SSDE
using CUDA, MadNLPGPU
using CUDSS

const SRC = joinpath(@__DIR__, "src")
for f in ("structs.jl","constants.jl","utils.jl","initialize.jl",
          "variables.jl","collocation.jl","continuity.jl","objective.jl","userfuncs.jl")
    include(joinpath(SRC, f))
end

K     = length(ARGS) >= 1 ? parse(Int, ARGS[1]) : 5
phase = length(ARGS) >= 2 ? ARGS[2] : "all"

yaml = joinpath(@__DIR__, "examples", "Benchmark-Models",
                "Crauste_CellSystems2017", "Crauste_CellSystems2017.yaml")
PEmodel = PEtab.PEtabModel(yaml)
PEprob  = PEtab.PEtabODEProblem(PEmodel)

###############################################################################
# Per-parameter estimation scale (Symbol :log10 / :log / :lin), aligned 1:Np
###############################################################################
function get_pscale(PEprob)
    xscale = PEprob.model_info.xindices.xscale       # Dict{Symbol,Symbol}
    return [xscale[name] for name in PEprob.xnames]
end
pscale = get_pscale(PEprob)
println("parameter scales: ", collect(zip(PEprob.xnames, pscale)))

###############################################################################
# Decision variable p := theta on the estimation scale.
# start/bounds come straight from PEtab (already estimation-scale); NO untransform.
###############################################################################
function create_p_trans(c::ExaCore, PEprob::PEtabODEProblem)
    (; lower_bounds, upper_bounds) = PEprob
    Np  = PEprob.nparameters_estimate
    θlb = Array(lower_bounds)            # estimation scale
    θub = Array(upper_bounds)            # estimation scale
    θ0  = Array(PEtab.get_x(PEprob))     # estimation-scale nominal (ODE-solve point)
    ExaModels.@add_var(c, p, 1:Np; lvar = θlb, uvar = θub, start = θ0)
    return c, Np
end

# physical value of parameter m as an ExaModels expression from theta = p[m]
@inline function p_phys(p, m, pscale)
    s = pscale[m]
    return s === :log10 ? exp(log(10.0) * p[m]) :
           s === :log   ? exp(p[m])             :
                          p[m]                      # :lin
end

###############################################################################
# Transformed collocation (Crauste: autonomous, no cv). Identical to
# _create_collocation but feeds p_phys(p,m) into f instead of the raw p[m].
###############################################################################
function create_collocation_trans(c::ExaCore, PEmodel, PEprob, PEinfo, pscale)
    (; N, K, Np, Nc, Nz, Ncv, h, taus) = PEinfo
    z = c.z; p = c.p
    @assert Ncv == 0 "transformed collocation scratch assumes no condition vars"
    fs, has_t = _get_rhs_funcs(PEmodel, PEprob)
    @assert !has_t "transformed collocation scratch assumes autonomous ODE"

    itr_coll = [(i,k,cidx,h[i]) for i in 1:N, k in 1:K, cidx in 1:Nc]
    c_coll = [
        ExaModels.@add_con(c,
            -hi*f(
                ntuple(v -> z[v,i,k,cidx], Nz)...,
                ntuple(m -> p_phys(p, m, pscale), Np)...,   # <-- transformed param
            )
            for (i,k,cidx,hi) in itr_coll
        )
        for f in fs
    ]
    DLDTAU  = [_eval_dldtau(j,k,taus) for j in 0:K, k in 1:K]
    itr_coll! = [(i,j,k,cidx,DLDTAU[j+1,k]) for i in 1:N, j in 0:K, k in 1:K, cidx in 1:Nc]
    for v in eachindex(c_coll)
        ExaModels.@add_con!(c,
            c_coll[v],
            (i,k,cidx) => z[v,i,j,cidx]*DLDTAU
            for (i,j,k,cidx,DLDTAU) in itr_coll!
        )
    end
    return c
end

###############################################################################
# Manual (aux-free) objective: Gaussian negative log-likelihood matching PEtab.jl,
#   nll_m = 0.5*log(2π) + log(σ_m) + 0.5*(S_m - ymeas_m)^2/σ_m^2,   S_m = Σ_j w_j z
# with the inlined state interpolation + numeric σ. Expanded into SIMD-friendly
# diagonal + bilinear kernels (no sum() in obj). (does not touch p for Crauste)
#   diagonal (per j):  c1*z^2 + c2*z + c3,  c1=0.5 w_j^2/σ^2, c2=-ymeas w_j/σ^2,
#                      c3 = (0.5 ymeas^2/σ^2 + log σ + 0.5 log 2π)/(K+1)
#   bilinear (per j<j'): cb*z_j*z_j',  cb = w_j w_j'/σ^2
###############################################################################
function add_manual_objective!(c::ExaCore, PEmodel, PEprob, PEinfo)
    (; Nm, K, h, t_meas, L1) = PEinfo
    z = c.z
    measurements_df = PEmodel.petab_tables[:measurements]
    observables_df  = PEmodel.petab_tables[:observables]
    dict_cid_cidx   = _get_dict_cid_cidx(PEmodel)
    dict_t_tidx     = _get_dict_t_tidx(h, t_meas)
    z_syms = [Symbolics.Num(Symbolics.variable(Symbol(split(string(zs), "(")[1])))
              for zs in _get_z_syms(PEprob)]
    dict_obsid_row = Dict(string(observables_df[i,:observableId]) => observables_df[i,:]
                          for i in 1:size(observables_df,1))
    _num(v) = v isa Number ? Float64(v) : parse(Float64, strip(string(v)))

    itr_diag = Tuple{Int,Int,Int,Int,Float64,Float64,Float64}[]
    itr_bil  = Tuple{Int,Int,Int,Int,Int,Float64}[]
    itr_ic   = Tuple{Int,Int,Float64,Float64}[]
    for midx in 1:Nm
        row     = measurements_df[midx, :]
        formula = string(dict_obsid_row[string(row[:observableId])][:observableFormula])
        parsed  = Meta.parse(formula)
        @assert parsed isa Symbol "manual obj only handles single-state observables; got $formula"
        v    = findfirst(x -> isequal(x, Symbolics.Num(Symbolics.variable(parsed))), z_syms)
        cidx = dict_cid_cidx[string(row[:simulationConditionId])]
        i    = dict_t_tidx[Float64(row[:time])]
        ym   = _num(row[:measurement]); s = _num(row[:noiseParameters])
        cst = log(s) + 0.5*log(2π)         # per-measurement NLL constant (incl. log σ)
        if i == 0
            push!(itr_ic, (v, cidx, ym, s))
        else
            for j in 0:K
                wj = L1[j+1]
                push!(itr_diag, (v,i,j,cidx,
                                 0.5*wj^2/s^2,                       # c1: z^2
                                 -ym*wj/s^2,                         # c2: z
                                 (0.5*ym^2/s^2 + cst)/(K+1)))        # c3: const share
                for jp in (j+1):K
                    push!(itr_bil, (v,i,j,jp,cidx, wj*L1[jp+1]/s^2)) # cb: z_j*z_j'
                end
            end
        end
    end
    if !isempty(itr_diag)
        ExaModels.@add_obj(c, c1*z[v,i,j,cidx]^2 + c2*z[v,i,j,cidx] + c3
                           for (v,i,j,cidx,c1,c2,c3) in itr_diag)
    end
    if !isempty(itr_bil)
        ExaModels.@add_obj(c, cb*z[v,i,j,cidx]*z[v,i,jp,cidx]
                           for (v,i,j,jp,cidx,cb) in itr_bil)
    end
    if !isempty(itr_ic)
        # single-node t=0 NLL: 0.5/σ² z² - ymeas/σ² z + (0.5 ymeas²/σ² + log σ + 0.5 log 2π)
        ExaModels.@add_obj(c,
            (0.5/s^2)*z[v,1,0,cidx]^2 + (-ym/s^2)*z[v,1,0,cidx] +
                (0.5*ym^2/s^2 + log(s) + 0.5*log(2π))
            for (v,cidx,ym,s) in itr_ic)
    end
    return c
end

###############################################################################
# Unified builder
###############################################################################
function build_crauste(PEmodel, PEprob, K, pscale; backend, aux::Bool, transform_p::Bool)
    c = ExaModels.ExaCore(; backend, concrete = Val(true))
    c, Np = transform_p ? create_p_trans(c, PEprob) : _create_p(c, PEprob)
    c, Nz, N, K, Nc, t_meas, t_vec_mesh, h, taus, L1 = _create_z(c, PEmodel, PEprob, K)
    Ncv = length(_get_cv_syms(PEmodel))
    Nm  = length(eachrow(PEmodel.petab_tables[:measurements]))
    Ny  = length(_get_obsids(PEmodel))
    PEinfo = PEInfo(Np,Nz,Nc,Ncv,Nm,Ny,N,K,t_meas,t_vec_mesh,h,taus,L1)
    if aux
        c = _create_y(c, PEinfo); c = _create_sigma(c, PEinfo)
    end
    c = transform_p ? create_collocation_trans(c, PEmodel, PEprob, PEinfo, pscale) :
                      _create_collocation(c, PEmodel, PEprob, PEinfo)
    c = _create_continuity(c, PEmodel, PEprob, PEinfo)
    y0 = sigma0 = nothing
    if aux
        c, y0, sigma0 = _create_objective(c, PEmodel, PEprob, PEinfo)
    else
        c = add_manual_objective!(c, PEmodel, PEprob, PEinfo)
    end
    m = ExaModels.ExaModel(c)
    if aux
        ExaModels.set_start!(m, c.y, y0)
        ExaModels.set_start!(m, c.sigma, sigma0)
    end
    return m, PEinfo
end

# max constraint violation at the warm start
function warm_start_stats(m)
    x0  = m.meta.x0
    cx  = similar(x0, m.meta.ncon); ExaModels.cons!(m, x0, cx)
    cx  = Array(cx); lcon = Array(m.meta.lcon); ucon = Array(m.meta.ucon)
    viol = maximum(max.(lcon .- cx, cx .- ucon, 0.0))
    return ExaModels.obj(m, x0), viol
end

###############################################################################
# (1) Consistency: transformed vs linear (CPU, math is backend-independent)
###############################################################################
if phase in ("consistency", "all")
    println("\n=========== CONSISTENCY CHECK (CPU) ===========")
    mlin, _ = build_crauste(PEmodel, PEprob, K, pscale; backend=nothing, aux=false, transform_p=false)
    mtra, _ = build_crauste(PEmodel, PEprob, K, pscale; backend=nothing, aux=false, transform_p=true)
    olin, vlin = warm_start_stats(mlin)
    otra, vtra = warm_start_stats(mtra)
    println("linear     : nvar=", mlin.meta.nvar, " ncon=", mlin.meta.ncon,
            "  warm-obj=", olin, "  max|viol|=", vlin)
    println("transformed: nvar=", mtra.meta.nvar, " ncon=", mtra.meta.ncon,
            "  warm-obj=", otra, "  max|viol|=", vtra)
    println("Δobj  = ", abs(olin - otra))
    println("Δviol = ", abs(vlin - vtra))
    @assert isapprox(olin, otra; rtol=1e-10) "warm-start objectives differ!"
    @assert max(vlin, vtra) < 1e-2 "warm start not feasible"
    println("✓ transformed build is mathematically consistent with linear build")

    # also check the manual (aux-free) NLL expansion equals the package aux NLL
    maux, _ = build_crauste(PEmodel, PEprob, K, pscale; backend=nothing, aux=true, transform_p=true)
    oaux, vaux = warm_start_stats(maux)
    println("transformed+aux : nvar=", maux.meta.nvar, " ncon=", maux.meta.ncon,
            "  warm-obj=", oaux, "  max|viol|=", vaux)
    println("Δobj(manual vs aux) = ", abs(otra - oaux))
    @assert isapprox(otra, oaux; rtol=1e-8) "manual NLL expansion != aux NLL formula!"
    println("✓ manual (aux-free) NLL expansion matches package aux NLL")
end

###############################################################################
# (2)+(3) Solve transformed problem on GPU, with and without aux vars
###############################################################################
function solve_gpu(label; aux)
    println("\n=========== SOLVE (GPU): transformed, ", label, " ===========")
    @time m, _ = build_crauste(PEmodel, PEprob, K, pscale; backend=CUDABackend(), aux=aux, transform_p=true)
    o0, v0 = warm_start_stats(m)
    println("nvar=", m.meta.nvar, " ncon=", m.meta.ncon,
            "  DoF=", m.meta.nvar-m.meta.ncon, "  warm-obj=", o0, "  max|viol|=", v0)
    res = madnlp(m; linear_solver = MadNLPGPU.CUDSSSolver,
                    tol = 1e-6, max_wall_time = 500.0, max_iter = 1_000_000)
    println(">>> ", label, ": status=", res.status, "  obj=", res.objective,
            "  iter=", res.iter, "  time=", res.counters.total_time)
    return res, m
end

Np = PEprob.nparameters_estimate
θstar = nothing  # estimation-scale params from my transformed no-aux solve

if phase in ("solve", "all")
    res_noaux, m_noaux = solve_gpu("WITHOUT aux y/sigma (manual obj)"; aux=false)
    res_aux,   _       = solve_gpu("WITH aux y/sigma (package obj)";    aux=true)
    θstar = Array(res_noaux.solution)[1:Np]   # p block = θ (estimation scale)
    println("\n=========== SUMMARY (transformed / log-param) ===========")
    println("no-aux : status=", res_noaux.status, "  obj=", res_noaux.objective, "  iter=", res_noaux.iter)
    println("aux    : status=", res_aux.status,   "  obj=", res_aux.objective,   "  iter=", res_aux.iter)
end

###############################################################################
# (4) Compare against PEtab.jl's own optimum (Optim IPNewton)
###############################################################################
if phase in ("petab", "all")
    using Optim
    println("\n=========== PEtab.jl calibration (Optim IPNewton) ===========")
    x0 = PEtab.get_x(PEprob)
    @time pres = PEtab.calibrate(PEprob, x0, Optim.IPNewton())
    xmin = Array(pres.xmin)
    println("PEtab IPNewton: fmin(nllh) = ", pres.fmin, "   converged = ", pres.converged)

    println("\nparam (estimation scale)   PEtab xmin        my θ*           |Δ|")
    for i in 1:Np
        mine = θstar === nothing ? NaN : θstar[i]
        println(rpad(string(PEprob.xnames[i]),22),
                rpad(string(round(xmin[i], sigdigits=8)),18),
                rpad(string(round(mine,    sigdigits=8)),16),
                θstar === nothing ? "" : string(round(abs(xmin[i]-mine), sigdigits=4)))
    end

    # Evaluate PEtab's accurate ODE-integrated nllh at BOTH optima (apples-to-apples,
    # independent of collocation discretization error in my objective value)
    nllh_petab_at_petab = PEprob.nllh(xmin)
    println("\nPEtab nllh at PEtab xmin = ", nllh_petab_at_petab)
    if θstar !== nothing
        nllh_petab_at_mine = PEprob.nllh(θstar)
        println("PEtab nllh at my θ*      = ", nllh_petab_at_mine)
        println("Δnllh (mine - PEtab)     = ", nllh_petab_at_mine - nllh_petab_at_petab)
    end
end
