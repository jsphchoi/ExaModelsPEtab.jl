# model_features.jl — extract structural PEtab features for a set of models (no model builds).
# Parses the SBML xml + PEtab tsv files directly to characterize each model:
#   nspecies, nparam(est), nconditions, nmeas, SBML <event>s, piecewise(time) gates,
#   assignment/rate rules, pre-equilibration, observable transforms, noise-model kind.
const MD = joinpath(@__DIR__, "..", "Benchmark-Models")
const RD = joinpath(@__DIR__, "..", "Benchmarks", "results")

models = length(ARGS) >= 1 ? ARGS : [
    "Bertozzi_PNAS2020","Zhao_QuantBiol2020","Laske_PLOSComputBiol2019","Oliveira_NatCommun2021",
    "Fujita_SciSignal2010","Brannmark_JBC2010","Giordano_Nature2020","Raimundez_PCB2020",
    "Borghans_BiophysChem1997","Elowitz_Nature2000","Lucarelli_CellSystems2018","Zheng_PNAS2012",
    "Bachmann_MSB2011","Isensee_JCB2018",
]

find1(dir, pat) = (fs = filter(f -> occursin(pat, lowercase(f)), readdir(dir)); isempty(fs) ? nothing : joinpath(dir, first(fs)))
ncount(s, pat) = length(collect(eachmatch(pat, s)))

function cols(tsv)
    lines = readlines(tsv)
    isempty(lines) && return (String[], Vector{Vector{String}}())
    hdr = split(strip(lines[1]), '\t')
    rows = [split(strip(l), '\t') for l in lines[2:end] if !isempty(strip(l))]
    (String.(hdr), rows)
end
colidx(hdr, name) = findfirst(==(name), hdr)
getcol(hdr, rows, name) = (i = colidx(hdr, name); i === nothing ? String[] : [r[i] for r in rows if length(r) >= i])

function exa_nvar(m)
    p = joinpath(RD, "$(m)_results.txt"); isfile(p) || return "?"
    for l in eachline(p); startswith(l, "exa_nvar=") && return split(l, '=')[2]; end
    "?"
end

println(rpad("model",26), rpad("spec",5), rpad("par",5), rpad("cond",5), rpad("meas",6),
        rpad("nvar(K4)",11), rpad("evt",4), rpad("pw",4), rpad("aRule",6), rpad("rRule",6),
        rpad("preeq",6), rpad("obsTransf",14), "noise")
println("-"^120)
for m in models
    d = joinpath(MD, m); isdir(d) || (println(rpad(m,26), "MISSING"); continue)
    xmlf = find1(d, ".xml"); xml = xmlf === nothing ? "" : read(xmlf, String)
    nspec = ncount(xml, r"<species ")
    nevt  = ncount(xml, r"<event[ >]")
    npw   = ncount(xml, r"piecewise")
    naR   = ncount(xml, r"<assignmentRule")
    nrR   = ncount(xml, r"<rateRule")

    pf = find1(d, "parameter"); npar = "?"
    if pf !== nothing
        h, r = cols(pf); est = getcol(h, r, "estimate")
        npar = isempty(est) ? string(length(r)) : string(count(==("1"), est))
    end
    cf = find1(d, "condition"); ncond = cf === nothing ? "?" : string(length(cols(cf)[2]))
    mf = find1(d, "measurement"); nmeas = "?"; preeq = "no"
    if mf !== nothing
        h, r = cols(mf); nmeas = string(length(r))
        pe = getcol(h, r, "preequilibrationConditionId")
        preeq = (!isempty(pe) && any(x -> !isempty(x) && x != "NaN", pe)) ? "YES" : "no"
    end
    of = find1(d, "observable"); otr = "?"; noise = "?"
    if of !== nothing
        h, r = cols(of)
        tr = getcol(h, r, "observableTransformation")
        otr = isempty(tr) ? "lin?" : join(sort(unique(tr)), "/")
        nf = getcol(h, r, "noiseFormula")
        # classify: pure estimated-const (noiseParameterN only) vs proportional/affine vs numeric
        kinds = String[]
        for f in nf
            fs = strip(f)
            if occursin(r"^noiseParameter\d+_\w+$", fs); push!(kinds, "const")
            elseif tryparse(Float64, fs) !== nothing; push!(kinds, "num")
            elseif occursin("noiseParameter", fs); push!(kinds, "affine/prop")
            else; push!(kinds, "formula"); end
        end
        noise = join(sort(unique(kinds)), "/")
    end
    println(rpad(m,26), rpad(nspec,5), rpad(npar,5), rpad(ncond,5), rpad(nmeas,6),
            rpad(exa_nvar(m),11), rpad(nevt,4), rpad(npw,4), rpad(naR,6), rpad(nrR,6),
            rpad(preeq,6), rpad(otr,14), noise)
end
