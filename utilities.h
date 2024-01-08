#pragma once

#include <math.h>
#include <omp.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>

// Define CPU related macros
#if defined CPU_ARMPL
#define CPU_LIB_NAME "Arm Performance Libraries"
#elif defined CPU_ONEMKL
#define CPU_LIB_NAME "Intel OneMKL"
#elif defined CPU_OPENBLAS
#define CPU_LIB_NAME "OpenBLAS"
#elif defined CPU_AOCL
#define CPU_LIB_NAME "AMD Optimized CPU Libraries"
#else
#define CPU_DEFAULT
#define CPU_LIB_NAME "None"
#endif

// Define GPU related macros
#if defined GPU_CUBLAS
#define GPU_LIB_NAME "NVIDIA cuBLAS"
#elif defined GPU_ONEMKL
#define GPU_LIB_NAME "Intel OneMKL"
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

// Define general macros
#ifndef ITERATIONS
#define ITERATIONS 10
#endif

#ifndef UPPER_LIMIT
#define UPPER_LIMIT 1000
#endif

#define ALPHA 1
#define BETA 0

// Define MIN and MAX Macros
#define MIN(X, Y) (((X) < (Y)) ? (X) : (Y))
#define MAX(X, Y) (((X) > (Y)) ? (X) : (Y))

// Define function to calculate GFLOPs
double calcGflops(const uint64_t flops, const double seconds) {
  return (flops / seconds) * 1e-9;
}

// Define data type enums
typedef enum { _fp32_, _fp64_ } dataTypes;