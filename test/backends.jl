const BACKENDS = []

if "EMC_NO_TEST_CPU" in ARGS
    @info "excluding CPU"
else
    push!(BACKENDS, nothing)
end

if "EMC_TEST_KA" in ARGS
    @eval using KernelAbstractions
    @eval push!(BACKENDS, CPU())
    @info "including KernelAbstractions"
else
    @info "excluding KernelAbstractions"
end

if "EMC_TEST_CUDA" in ARGS
    @eval using CUDA
    @eval push!(BACKENDS, CUDABackend())
    @info "including CUDA"
else
    @info "excluding CUDA"
end

if "EMC_TEST_AMDGPU" in ARGS
    @eval using AMDGPU
    @eval push!(BACKENDS, ROCBackend())
    @info "including AMDGPU"
else
    @info "excluding AMDGPU"
end