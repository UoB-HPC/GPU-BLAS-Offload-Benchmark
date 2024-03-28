#pragma once

// Define CPU related macros
#if defined CPU_ARMPL
#include <armpl.h>
#define CPU_LIB_NAME "Arm Performance Libraries"
#define CPU_FP16 __fp16
#elif defined CPU_ONEMKL
#include <mkl.h>
#define CPU_LIB_NAME "Intel OneMKL"
#define CPU_FP16 MKL_F16
#elif defined CPU_OPENBLAS
#define CPU_LIB_NAME "OpenBLAS"
#elif defined CPU_AOCL
#define CPU_LIB_NAME "AMD Optimized CPU Libraries"
#else
#define CPU_DEFAULT
#define CPU_LIB_NAME "None"
#define CPU_ENABLED false
#endif

#ifndef CPU_ENABLED
#define CPU_ENABLED true
#endif

// Define GPU related macros
#if defined GPU_CUBLAS
#include <cublas_v2.h>
#define GPU_LIB_NAME "NVIDIA cuBLAS"
#define GPU_FP16 __half
#elif defined GPU_ONEMKL
#include <sycl/sycl.hpp>
#define GPU_LIB_NAME "Intel OneMKL"
#define GPU_FP16 sycl::half
#elif defined GPU_ROCBLAS
#define GPU_LIB_NAME "AMD rocBLAS"
#else
#define GPU_DEFAULT
#define GPU_LIB_NAME "None"
#define GPU_ENABLED false
#endif

#ifndef GPU_ENABLED
#define GPU_ENABLED true
#endif

// Ensure CPU and GPU FP16 types are defined in some capacity
#ifndef CPU_FP16
#define CPU_FP16 char
#endif
#ifndef GPU_FP16
#define GPU_FP16 char
#endif

// Define macros for alpha and beta
#define ALPHA 1
#define BETA 0

// Define output directory for csv files.
#define CSV_DIR "CSV_Results"

// Define seed for random number generation - use seeded srand() to ensure
// inputs across libraries are consistent & comparable
const unsigned int SEED = 19123005;

// Define enum class for GPU offload type
enum class gpuOffloadType : uint8_t {
  always = 0,
  once,
  unified,
};

// Define struct which contains a runtime, checksum value, and gflop/s value
struct time_checksum_gflop {
  double runtime = 0.0;
  double checksum = 0.0;
  double gflops = 0.0;
};

// External consume function used to ensure naive code is performed and not
// optimised away, and that all iterations of any library BLAS call are
// performed.
extern "C" {
int consume(void* a, void* b, void* c);
}