# Validate fixed-time event support: build (GPU) + solve with MadNLP/cuDSS, compare to PEtab.
# Usage: julia --project=examples examples/validate_events.jl <ModelDir>
using ExaModelsPEtab, PEtab, ExaModels
using CUDA, MadNLPGPU, CUDSS

const TARGETS = Dict(
    "Oliveira_NatCommun2021" => 23542.290730692126,
    "Fujita_SciSignal2010"   => -53.083772377888465,
    "Giordano_Nature2020"    => -2490.6874494624285,
    "Brannmark_JBC2010"      => 141.88929766387372,
    "Boehm_JProteomeRes2014" => 138.2219977274927,   # non-event regression (prior exa=138.2195)
    "Zhao_QuantBiol2020"     => 503.1633273557346,   # non-event regression
)

model = length(ARGS) >= 1 ? ARGS[1] : "Fujita_SciSignal2010"
yaml  = joinpath(@__DIR__, "..", "Benchmark-Models", model, model * ".yaml")
isfile(yaml) || (yaml = first(filter(isfile, [joinpath(@__DIR__, "..", "Benchmark-Models",model,f)
            for f in readdir(joinpath(@__DIR__, "..", "Benchmark-Models",model)) if endswith(f,".yaml")])))

println("== building $model (CUDABackend) ==")
t0 = time()
m = petab_examodel(yaml; backend = CUDABackend(), K = 5)
println("  built in $(round(time()-t0; digits=1)) s :: nvar=$(m.meta.nvar) ncon=$(m.meta.ncon)")

println("== solving with MadNLP/cuDSS ==")
t1 = time()
res = madnlp(m; tol = 1e-6, max_iter = 3000, max_wall_time = 1200.0,
             linear_solver = MadNLPGPU.CUDSSSolver)
println("  solved in $(round(time()-t1; digits=1)) s")
obj = res.objective
println("  status     = ", res.status)
println("  objective  = ", obj)
if haskey(TARGETS, model)
    tgt = TARGETS[model]
    println("  petab obj  = ", tgt)
    println("  abs diff   = ", abs(obj - tgt))
    println("  rel diff   = ", abs(obj - tgt) / max(1.0, abs(tgt)))
end
println("== DONE ==")
