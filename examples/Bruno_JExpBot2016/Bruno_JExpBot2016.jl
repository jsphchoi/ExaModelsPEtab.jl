using ExaModels, PEtab
using MadNLPGPU, CUDA

# Ex, x0(p) + CRAZY OBJ EXPRESSION
filename = joinpath("Bruno_JExpBot2016","Bruno_JExpBot2016.yaml")

m = petab_examodel(filename; backend = CUDA.CUDABackend(), K = 10)

@time result = madnlp(m; tol=1e-6)