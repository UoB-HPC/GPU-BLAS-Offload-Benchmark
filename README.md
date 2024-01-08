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
make COMPILER=ARM
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
   - Currently only usable with the `ARM` compiler build option.
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


# Running
The benchmark takes the following runtime arguments:
```bash
./gpu-blob --iterations I --dimension_limit D
```
Where `I` (default of `10`) specifies how many iterations each kernel will run, and `D` (default of `128`) specifies the the upper limit for the largest dimention in a problem size.\
__Example:__ For a square GEMM, the problem size will iterate up to `M=N=K=D`.\
__Example:__ For a rectangular GEMM where `M=N` and `K=M/4`, the probelm size will iterate up to`M=N=D` and `K=D/4`.


For the CPU kernels it is also likely beneficial to set the relevant environment variables. For example, when using ArmPL, setting `OMP_NUM_THREADS`, `OMP_PROC_BIND`, and `OMP_PLACES` can be beneficial.


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
   - [ ] Research how to fairly and properly evaluate sparce BLAS kernels 
   - [ ] Finish Sparce function descriptions, including what problems are evaluated and why.
 - [x] Consider the suitability of including batched versions of the chosen BLAS kernels.
 - [x] Create main file which contains functionality of:
   - [x] Print system information such as CPU library used, GPU library used...
   - [ ] Running each BLAS kernel for all input types & shapes on CPU.
     - [x] Increase each dimension by 1 each run until reached upper-limit (user defined?).
     - [x] Each for `n` iterations (user defined?).
   - [ ] Running each BLAS kernel for all input types & shapes on GPU.
     - [x] Increase each dimension by 1 each run until reached upper-limit (user defined?).
     - [x] Each for `n` iterations (user defined?).
       - [ ] Offload data once at start, once at end.
       - [ ] Offload data each iteration.
   - [x] Calculate GLFOPs achieved for each BLAS kernel run.
   - [x] Saving all data to .csv file(s).
   - [ ] Calculate for each kernel at what problem size offloading the computation to the GPU becomes worthwhile.
     - i.e. the time taken on CPU becomes longer than on GPU
 - [x] Create Makefile with options for:
   - [x] Selecting the compiler + compiler specific flags.
   - [x] Selecting the CPU library target (ArmPL, oneMKL, OpenBLAS) + relevant flags.
   - [x] Selecting the GPU library target (cuBLAS, oneMKL) + relevant flags.
 - [ ] Add naive implementations of kernels for Default CPU
   - [x] GEMM 
   - [ ] GEMV 
   - [ ] SpMM 
   - [ ] SpMV 
 - [ ] Add support for ArmPL.
   - [x] GEMM 
   - [ ] GEMV 
   - [ ] SpMM 
   - [ ] SpMV 
 - [ ] Add support for cuBLAS.
 - [ ] Add support for oneMKL.
 - [ ] Add support for AOCL (AMD Optimizing CPU libraries).
 - [ ] Add support for rocBLAS.
 - [ ] Add support for OpenBLAS.
 - [ ] Add support for NVIDIA NVPL(?) CPU Library
 - [ ] Add support for Apple Accelerate(?)
 - [ ] Add support for Apple Metal Performance Shaders(?)
 - [ ] Add batched versions of appropriate BLAS kernels

# Future Work
 - [ ] Add support for Intel AMX.
 - [ ] Add support for IBM Power10 MMA.
 - [ ] Add support for Arm SME (no hardware available).