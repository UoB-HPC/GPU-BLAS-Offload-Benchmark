# GPU-BLOB: GPU BLas Offload Benchmark
GPU-BLOB is a benchmark tool which can be used to determine at what point (i.e. problem size) it is worthwhile to offload select BLAS computations to the GPU on a heterogeneous system.
Not only can this aid to help programmers understand the characteristics of the hardware they are optimising for, but also whether or not it would be useful for them to utilise the GPU at all for their specific application.

For each supported BLAS kernel (listed below) GPU-BLOB will run `n` iterations of each kernel on CPU and GPU, gradually increasing the problem size up to a user-defined maximum limit. The resulting large amount of performance data collected is output in multiple `csv` files in the `CSV_Results` directory; one for each BLAS kernel and probelm-type pair.

Each BLAS kernel is tested with a range of different problem size designs in an attempt to capture the performance differences that can occur between different problem sets when utilising the same underlying kernel. For each BLAS kernel and problem type pair, a table will be displayed which outlines the problem size at which offloading to the GPU became worthwhile. If for the number of iterations and maximum problem dimension this offload threshold cannot be found, the table will show `0` for each problem dimension and `N/A` for the attained GFLOP/s.

All computations performed by each BLAS library are assumed to be functionally correct. However, a simple checksum is calculated after each CPU and GPU run for each problem size to ensure all utilised libraries are computing the same result.\
Only when an error occurs will any checksum be displayed to the user.

GFLOP/s are calculated using the following Total FLOPs formulas. The compute time excludes any initialisation, but does include any data movement / prefetching to/from the GPU device:
 - **GEMM** : FLOPs = Alpha * (2 * M * N * K) + Beta * (M * N)

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
make COMPILER=ARM CPU_LIB=ARMPL
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
make COMPILER=ARM CPU_LIB=ARMPL GPU_LIB=CUBLAS
```
The supported Libraries are as follows:
 - NVIDIA cuBLAS : `CUBLAS`
 <!-- - Intel OneMKL : `ONEMKL` -->
 <!-- - AMD rocBLAS : `ROCBLAS` -->

If no library is selected then no GPU BLAS kernels will be executed.

### <u>Additional Flags</u>
Some combinations of BLAS Libraries and compilers will require additional flags. Many of these have been pre-configured in the Makefile, but some require the inclusion of additional shared objects etc. These can be passed to the Makefile using `CXXFLAGS=`:
```bash
make COMPILER=ARM CPU_LIB=ARMPL GPU_LIB=CUBLAS CXXFLAGS="-I/my/include/dir -g"
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
 2. Prefix the run command with `numactl -Na -ma` where `a` is the NUMA node the device is connected to.
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
   - [x] Running each BLAS kernel for all input types & shapes on GPU.
     - [x] Increase each dimension by 1 each run until reached upper-limit (user defined?).
     - [x] Each for `n` iterations (user defined?).
       - [x] Offload data once at start, once at end.
       - [x] Offload data each iteration.
       - [x] Unified memory solution.
   - [x] Calculate GLFOPs achieved for each BLAS kernel run.
   - [x] Saving all data to .csv file(s).
   - [x] Calculate for each kernel at what problem size offloading the computation to the GPU becomes worthwhile.
     - i.e. the time taken on CPU becomes longer than on GPU
   - [x] Add checksum to ensure each library implementation is getting the same answer.
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