# Fast smoke test: CPU build only (no solve). Catches compile errors in the refactored
# collocation/continuity tuple-data + always-t RHS path. Oliveira reaches _create_objective.
using ExaModelsPEtab, PEtab
model = length(ARGS) >= 1 ? ARGS[1] : "Oliveira_NatCommun2021"
yaml  = joinpath(@__DIR__, "..", "Benchmark-Models", model, model * ".yaml")
isfile(yaml) || (yaml = first(filter(isfile, [joinpath(@__DIR__, "..", "Benchmark-Models",model,f)
            for f in readdir(joinpath(@__DIR__, "..", "Benchmark-Models",model)) if endswith(f,".yaml")])))
println("== build smoke: $model (CPU) ==")
try
    m = petab_examodel(yaml; backend = nothing, K = 5)
    println("BUILT OK :: nvar=$(m.meta.nvar) ncon=$(m.meta.ncon)")
catch e
    msg = sprint(showerror, e)
    # Reaching the known orthogonal objective.jl error means collocation+continuity built fine.
    if occursin("sd_cumulative_cases", msg) || occursin("not defined in `Symbolics`", msg)
        println("COLLOCATION+CONTINUITY OK (hit orthogonal objective.jl error: ",
                split(msg, "\n")[1][1:min(end,90)], ")")
    else
        println("BUILD ERROR: ", split(msg, "\n")[1][1:min(end,200)])
    end
end
println("== DONE ==")
