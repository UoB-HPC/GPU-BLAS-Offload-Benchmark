# GPU-BLOB: GPU BLas Offload Benchmark
GPU-BLOB is a benchmark tool which can be used to determine at what point (i.e. problem size) it is worthwhile to offload select BLAS computations to the GPU on a heterogeneous system.
Not only can this aid to help programmers understand the characteristics of the hardware they are optimising for, but also whether or not it would be useful for them to utilise the GPU at all for their specific problem.

For each supported BLAS kernel (listed below) GPU-BLOB will run `n` iterations of each kernel, gradually increasing the problem size to gather large amounts of performance data. Said data can then be used to determine at
what point does the GPU's performance advantage over the CPU outweigh the cost of offloading data to/from the GPU.\
Each BLAS kernel is tested with a range of different problem size designs to attempt to capture performance differences that occur between different problem sets when utilising the same underlying kernel.

All computations performed by a vendor BLAS library are assumed to be functionally correct. As such, no verification of correct results will be performed or displayed.

# Build Options
Select the compiler you wish to use
``` bash
make COMPILER=GNU
```
The supported compiler names are: `ARM`, `CLANG`, `GNU`, `INTEL`.\
This option defaults to `GNU`.


### <u>CPU BLAS Library</u>
Specify which CPU BLAS Library you are using:
```bash
make COMPILER=GNU CPU_LIBRARY=ARMPL
```
The supported Libraries are as follows:
 - Arm Performance Libraries : `ARMPL`
 - Intel OneMKL : `ONEMKL`
 - AMD Optimizing CPU libraries : `AOCL`
 - OpenBLAS : `OPENBLAS`

If no library is selected then a naive solution to each kernel will be performed.


### <u>GPU BLAS Library</u>
Specify which GPU BLAS Library you are using:
```bash
make COMPILER=GNU CPU_LIBRARY=ARMPL GPU_LIBRARY=CUBLAS
```
The supported Libraries are as follows:
 - NVIDIA cuBLAS : `CUBLAS`
 - Intel OneMKL : `ONEMKL`
 - AMD rocBLAS : `ROCBLAS`

If no library is selected then no GPU BLAS kernels will be executed.


### <u>Iterations</u>
Specify how many iterations of each kernel to run:
```bash
make COMPILER=GNU CPU_LIBRARY=ARMPL GPU_LIBRARY=CUBLAS ITERATIONS=100
```
The default value is `10`


### <u>Problem Size Limit</u>
Specify what the upper limit should be for the largest dimention in a problem size:
```bash
make COMPILER=GNU CPU_LIBRARY=ARMPL GPU_LIBRARY=CUBLAS ITERATIONS=100 UPPER_LIMIT=8000
```
The default value is `1000`.\
__Example:__ For a square GEMM, the problem size will iterate up to `M=N=K=UPPER_LIMIT`. \
__Example:__ For a rectangular GEMM where `M=N` and `K=M/4`, the probelm size will iterate up to`M=N=UPPER_LIMIT` and `K=UPPER_LIMIT/4`.

# Running
The benchmark takes no run-time options. However, for the CPU kernels it is likely beneficial to set the relevant environment variables. For example, for ArmPL setting `OMP_NUM_THREADS`, `OMP_PROC_BIND`, and `OMP_PLACES` can be beneficial.

Some pre-analysed build and run options which have been found to improve CPU kernel performance can be found in the `Configurations` directory.


# BLAS Kernels Supported
The kernels listed below are computed by the benchmark for a wide range of problem sizes and shapes.

### <u>Level 3 BLAS</u>
 - GEMM
   - FP32, FP64
   - Square, short-&-wide, tall-&-thin input sizes

 - SpMM
   - FP32, FP64
   - ...

### <u>Level 2 BLAS</u>
 - GEMV
   - FP32, FP64
   - Square, short-&-wide, tall-&-thin input sizes 

 - SpMV
   - FP32, FP64
   - ...

# ToDo:
 - [x] Outline what kernels are included in the benchmark, along with how they will be run.
   - [ ] Finish Sparce function descriptions, including what inputs are evaluated and why.
 - [x] Consider the suitability of including batched versions of the chosen BLAS kernels.
 - [ ] Consider whether including MAGMA(-Batched) would be a worthwhile addition.
   - Could provide a point of comparison to a heterogeneous library, given this benchmark will be testing CPU vs. GPU individually.
 - [x] Create main file which contains functionality of:
   - [x] Print system information such as CPU library used, GPU library used...
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
 - [x] Create Makefile with options for:
   - [x] Selecting the compiler + compiler specific flags.
   - [x] Selecting the CPU library target (ArmPL, oneMKL, OpenBLAS) + relevant flags.
   - [x] Selecting the GPU library target (cuBLAS, oneMKL) + relevant flags.
 - [ ] Add support for ArmPL.
   - [ ] GEMM 
   - [ ] GEMV 
   - [ ] SpMM 
   - [ ] SpMV 
 - [ ] Add support for cuBLAS.
 - [ ] Add support for oneMKL.
 - [ ] Add support for AOCL (AMD Optimizing CPU libraries).
 - [ ] Add support for rocBLAS.
 - [ ] Add support for OpenBLAS.
 - [ ] Add support for NVIDIA NVPL(?)
 - [ ] Add support for Apple Accelerate(?)
 - [ ] Add support for Apple Metal Performance Shaders(?)
 - [ ] Add batched versions of appropriate BLAS kernels

# Future Work
 - [ ] Add support for Intel AMX.
 - [ ] Add support for IBM Power10 MMA.
 - [ ] Add support for Arm SME (no hardware available).