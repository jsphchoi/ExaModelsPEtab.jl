using ExaModels, PEtab
using MadNLPGPU, CUDA

# E(1), NI, x0(p)
filename = joinpath("Schwen_PONE2014","Schwen_PONE2014.yaml")

m = petab_examodel(filename; backend = CUDA.CUDABackend(), K = 10)

@time result = madnlp(m; tol=1e-6)