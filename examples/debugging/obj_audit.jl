# obj_audit.jl — objective-equivalence audit at nominal θ across models.
# obj-consistency is K-independent (warm start uses the true trajectory at measurement nodes),
# so build at small K for speed. Flags models where exa's NLL != PEtab's nllh (false equivalence).
using ExaModelsPEtab, PEtab, ExaModels
import ModelingToolkitBase as MTK
using Symbolics
import OrdinaryDiffEq as ODE
import SteadyStateDiffEq as SSDE
const SRCDIR = joinpath(@__DIR__, "..", "..", "src")
for f in ("structs.jl","constants.jl","utils.jl","initialize.jl",
          "variables.jl","collocation.jl","continuity.jl","objective.jl","steadystate.jl","userfuncs.jl")
    include(joinpath(SRCDIR, f))
end
const MODELDIR = joinpath(@__DIR__, "..", "Benchmark-Models")
K = length(ARGS) >= 1 ? parse(Int, ARGS[1]) : 2
# log/log10-transform models (suspect) + lin controls (Boehm/Crauste validated-good)
models = length(ARGS) >= 2 ? ARGS[2:end] : [
    "Boehm_JProteomeRes2014","Crauste_CellSystems2017",       # lin controls (should match ~1e-7)
    "Blasi_CellSystems2016",                                  # log control (validated)
    "Schwen_PONE2014","Perelson_Science1996",                 # log10 (Schwen known-bad)
    "Lucarelli_CellSystems2018","Laske_PLOSComputBiol2019",   # log10/log
]   # (Borghans/Elowitz/Bachmann excluded: NaN-spin / OOM nominal integration hangs the audit)
_yaml(m)=joinpath(MODELDIR,m,first(filter(f->endswith(lowercase(f),".yaml"),readdir(joinpath(MODELDIR,m)))))
println("MODEL                          transforms        PEtab_nllh        exa_obj           reldiff"); flush(stdout)
for m in models
    try
        PEmodel = PEtab.PEtabModel(_yaml(m)); PEprob = PEtab.PEtabODEProblem(PEmodel)
        trs = _get_meas_transforms(PEmodel)
        tcount = Dict{Symbol,Int}(); for t in trs; tcount[t]=get(tcount,t,0)+1; end
        tstr = join(["$k:$v" for (k,v) in tcount], ",")
        nllh = PEprob.nllh(PEtab.get_x(PEprob))
        mdl  = _build_petab_examodel(PEmodel, PEprob, nothing, K)
        eo   = ExaModels.obj(mdl, mdl.meta.x0)
        rd   = abs(eo-nllh)/abs(nllh)
        flag = rd > 1e-4 ? "  <<< MISMATCH" : ""
        println(rpad(split(m,'_')[1],30), rpad(tstr,17), rpad(round(nllh;digits=4),17), rpad(round(eo;digits=4),17), rpad(round(rd;sigdigits=3),12), flag); flush(stdout)
    catch e
        println(rpad(split(m,'_')[1],30), "BUILD/EVAL FAILED: ", sprint(showerror,e)[1:min(80,end)]); flush(stdout)
    end
end
