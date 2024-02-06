# GPU-BLOB: GPU BLas Offload Benchmark
GPU-BLOB is a benchmark tool which can be used to determine at what point (i.e. problem size) it is worthwhile to offload select BLAS computations to the GPU on a heterogeneous system.
Not only can this aid to help programmers understand the characteristics of the hardware they are optimising for, but also whether or not it would be useful for them to utilise the GPU at all for their specific application.

For each supported BLAS kernel (listed below) GPU-BLOB will run `n` iterations of each kernel, gradually increasing the problem size to gather large amounts of performance data. Said data can then be used to determine at
what point does the GPU's performance advantage over the CPU outweigh the cost of offloading data to/from the GPU.\
Each BLAS kernel is tested with a range of different problem size designs in an attempt to capture the performance differences that can occur between different problem sets when utilising the same underlying kernel.

All computations performed by a vendor BLAS library are assumed to be functionally correct. As such, no verification of correct results will be performed or displayed.

# Build Options
Select the compiler you wish to use
``` bash
make COMPILER=ARM
```
The supported compiler names are: `ARM`, `CLANG`, `GNU`, `INTEL`, `NVIDIA`.\
This option defaults to `GNU`.


### <u>CPU BLAS Library</u>
Specify which CPU BLAS Library you are using:
```bash
make COMPILER=GNU CPU_LIBRARY=ARMPL
```
The supported Libraries are as follows:
 - Arm Performance Libraries : `ARMPL`
   - Currently only usable with the `ARM` compiler build option.
 <!-- - Intel OneMKL : `ONEMKL` -->
 <!-- - AMD Optimizing CPU libraries : `AOCL` -->
 <!-- - OpenBLAS : `OPENBLAS` -->

If no library is selected then a naive solution to each kernel will be performed.


### <u>GPU BLAS Library</u>
Specify which GPU BLAS Library you are using:
```bash
make COMPILER=GNU CPU_LIBRARY=ARMPL GPU_LIBRARY=CUBLAS
```
The supported Libraries are as follows:
 - NVIDIA cuBLAS : `CUBLAS`
 <!-- - Intel OneMKL : `ONEMKL` -->
 <!-- - AMD rocBLAS : `ROCBLAS` -->

If no library is selected then no GPU BLAS kernels will be executed.

### <u>Additional Flags</u>
Any additional flags can be passed to the Makefile using `CXXFLAGS=`:
```bash
make COMPILER=GNU CPU_LIBRARY=ARMPL GPU_LIBRARY=CUBLAS CXXFLAGS="-I/my/include/dir -g"
```


# Running
The benchmark takes the following runtime arguments:
```bash
./gpu-blob --iterations I --dimension_limit D
```
Where `I` (default of `10`) specifies how many iterations each kernel will run, and `D` (default of `128`) specifies the the upper limit for the largest dimention in a problem size.\
__Example:__ For a square GEMM, the problem size will iterate up to `M=N=K=D`.\
__Example:__ For a rectangular GEMM where `M=N` and `K=M/4`, the probelm size will iterate up to`M=N=D` and `K=D/4`.


# Environment Variables
It is recommended to set the relevant environment variables to ensure the best performance on host and device. 

### <u>Arm Performance Libraries</u>
When using ArmPL, setting the following environment variables is beneficial:
 - `OMP_NUM_THREADS` -- Setting to the core count of the host CPU should ensure the best performance
 - `OMP_PROC_BIND` -- `close` is often found to perform best
 - `OMP_PLACES` -- `cores` is often found to perform best

### <u>cuBLAS</u>
When using cuBLAS, it is important to pin the initialised data on the host to the correct NUMA domain to ensure data-offload is done optimally:
 1. Use `nvidia-smi topo -m` to find out what the device's NUMA affinaty is.
 2. Prefix the run command with `numactl -Na -Ma` where `a` is the NUMA node the device is connected to.
 3. If a device cannot be found, ensure `CUDA_VISIBLE_DEVICES` is set correctly.


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
   - [x] Running each BLAS kernel for all input types & shapes on CPU.
     - [x] Increase each dimension by 1 each run until reached upper-limit (user defined?).
     - [x] Each for `n` iterations (user defined?).
   - [ ] Running each BLAS kernel for all input types & shapes on GPU.
     - [x] Increase each dimension by 1 each run until reached upper-limit (user defined?).
     - [x] Each for `n` iterations (user defined?).
       - [x] Offload data once at start, once at end.
       - [x] Offload data each iteration.
       - [ ] Unified memory solution.
   - [x] Calculate GLFOPs achieved for each BLAS kernel run.
   - [x] Saving all data to .csv file(s).
   - [ ] Calculate for each kernel at what problem size offloading the computation to the GPU becomes worthwhile.
     - i.e. the time taken on CPU becomes longer than on GPU
   - [ ] Add checksum to ensure each library implementation is getting the same answer.
 - [x] Create Makefile with options for:
   - [x] Selecting the compiler + compiler specific flags.
   - [x] Selecting the CPU library target (ArmPL, oneMKL, OpenBLAS) + relevant flags.
   - [x] Selecting the GPU library target (cuBLAS, oneMKL) + relevant flags.
 - [ ] Add naive implementations of kernels for Default CPU + Default GPU
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
   - [x] GEMM 
   - [ ] GEMV 
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