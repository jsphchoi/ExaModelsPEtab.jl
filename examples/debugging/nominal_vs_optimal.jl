# nominal_vs_optimal.jl — does PEtab's nominal get_x already sit at the optimum?
# For each model: eval nllh at nominal get_x, calibrate from it, compare objectives.
#
# Usage (from repo root):
#   julia --project=examples -t 1 examples/debugging/nominal_vs_optimal.jl [model ...]

using PEtab
import Optim

const MODELDIR = joinpath(@__DIR__, "..", "Benchmark-Models")

models = length(ARGS) >= 1 ? ARGS :
         ["Crauste_CellSystems2017", "Boehm_JProteomeRes2014"]

_yaml(m) = begin
    d = joinpath(MODELDIR, m)
    joinpath(d, first(filter(f -> endswith(lowercase(f), ".yaml"), readdir(d))))
end

for model in models
    println("\n============================== $model ==============================")
    PEmodel = PEtab.PEtabModel(_yaml(model))
    PEprob  = PEtab.PEtabODEProblem(PEmodel)

    xnom = PEtab.get_x(PEprob)          # nominal parameter vector (estimation scale)
    obj_nom = PEprob.nllh(xnom)

    pres = PEtab.calibrate(PEprob, xnom, Optim.IPNewton())
    obj_opt = pres.fmin

    println("nominal get_x        : ", collect(xnom))
    println("optimized x*         : ", collect(pres.xmin))
    println("nllh @ nominal       : ", obj_nom)
    println("nllh @ optimum (fmin): ", obj_opt)
    println("Δ (nominal - optimum): ", obj_nom - obj_opt)
    println("max|Δx|              : ", maximum(abs.(collect(xnom) .- collect(pres.xmin))))
end
