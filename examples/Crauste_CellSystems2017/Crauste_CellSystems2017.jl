using ExaModels, PEtab
using MadNLPGPU, CUDA

# Ex, NI, x0fix, Nc=1
filename = joinpath("Crauste_CellSystems2017","Crauste_CellSystems2017.yaml")

m = petab_examodel(filename; backend = CUDA.CUDABackend(), K = 10)

@time result = madnlp(m; tol=1e-6)