# GPU-BLAS-Offload-Benchmark
A tool to determine at what point it is worthwhile offloading BLAS computations to the GPU on heterogeneous systems.

All computations performed by a vendor BLAS library are assumed to be functionally correct. As such, no verification of correct results will be performed or displayed.

# BLAS Kernels Evaluated
The kernels listed below are computed by the benchmark for a wide range of problem sizes and shapes.

### Level 3 BLAS
 - GEMM
   - FP16 (where supported), FP32, FP64
   - Square, short-&-wide, tall-&-thin input sizes 

 - SpMM
   - FP16 (where supported), FP32, FP64
   - ...

### Level 2 BLAS
 - GEMV
   - FP16 (where supported), FP32, FP64
   - Square, short-&-wide, tall-&-thin input sizes 

 - SpMV
   - FP16 (where supported), FP32, FP64
   - ...

# ToDo:
 - [ ] Outline what kernels are included in the benchmark, along with how they will be run.
   - [ ] Finish Sparce function descriptions, including what inputs are evaluated and why.
 - [ ] Consider the suitability of including batched versions of the chosen BLAS kernels.
 - [ ] Consider whether including MAGMA(-Batched) would be a worthwhile addition.
   - Could provide a point of comparison to a heterogeneous library, given this benchmark will be testing CPU vs. GPU individually.
 - [ ] Create main file which contains functionality of:
   - [ ] Print system information such as CPU vendor & name, GPU vendor & name.
   - [ ] Running each BLAS kernel for all input types & shapes on CPU.
     - [ ] Increase each dimension by 1 each run until reached upper-limit (user defined?).
     - [ ] Each for `n` iterations (user defined?).
   - [ ] Running each BLAS kernel for all input types & shapes on GPU.
     - [ ] Increase each dimension by 1 each run until reached upper-limit (user defined?).
     - [ ] Each for `n` iterations (user defined?).
       - [ ] Offload data once at start, once at end.
       - [ ] Offload data each iteration.
   - [ ] Calculate GLFOPs achieved for each BLAS kernel run.
   - [ ] Saving all data to .csv file(s).
   - [ ] Calculate for each kernel at what problem size offloading the computation to the GPU becomes worthwhile.
   - [ ] ...
 - [ ] Create Makefile with options for:
   - [ ] Selecting the compiler + compiler specific flags.
   - [ ] Selecting the CPU library target (ArmPL, oneMKL, OpenBLAS) + relevant flags.
   - [ ] Selecting the GPU library target (cuBLAS, oneMKL) + relevant flags.
 - [ ] Add support for ArmPL.
 - [ ] Add support for cuBLAS.
 - [ ] Add support for oneMKL.
 - [ ] Add support for OpenBLAS.

# Future Work
 - [ ] Add support for Intel AMX.
 - [ ] Add support for IBM Power10 MMA.
 - [ ] Add support for Arm SME (no hardware available).