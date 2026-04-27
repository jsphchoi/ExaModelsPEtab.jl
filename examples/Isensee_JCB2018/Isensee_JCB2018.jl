using ExaModels, PEtab
using MadNLPGPU, CUDA

# C, E(1), ev, NI, u(t,p), x0SSpre(p)
filename = joinpath("Isensee_JCB2018","Isensee_JCB2018.yaml")

m = petab_examodel(filename; backend = CUDA.CUDABackend(), K = 10)

@time result = madnlp(m; tol=1e-6)