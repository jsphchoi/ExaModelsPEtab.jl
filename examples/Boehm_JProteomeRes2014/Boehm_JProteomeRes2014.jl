using ExaModels, PEtab
using MadNLPGPU, CUDA

# Nc = 1, C, E(1), u(t,p), x0 = f(p), obsv = obsvf
filename = joinpath("Bruno_JExpBot2016","Bruno_JExpBot2016.yaml")

m = petab_examodel(filename; backend = CUDA.CUDABackend(), K = 10)

@time result = madnlp(m; tol=1e-6)