#pragma once

#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <omp.h>
#include <string.h>
#include <math.h>

// Define CPU related macros
#if !defined CPU_ARMPL && !defined CPU_ONEMKL && !defined CPU_AOCL &&          \
    !defined CPU_OPENBLAS
#define CPU_DEFAULT
#endif

#if defined CPU_DEFAULT
#define CPU_LIB_NAME "None"
#elif defined CPU_ARMPL
#define CPU_LIB_NAME "Arm Performance Libraries"
#include <armpl.h>
#elif defined CPU_ONEMKL
#define CPU_LIB_NAME "Intel OneMKL"
#elif defined CPU_OPENBLAS
#define CPU_LIB_NAME "OpenBLAS"
#elif defined CPU_AOCL
#define CPU_LIB_NAME "AMD Optimized CPU Libraries"
#endif

// Define GPU related macros
#if !defined GPU_CUBLAS && !defined GPU_ONEMKL && !defined GPU_ROCBLAS
#define GPU_DEFAULT
#endif

#if defined GPU_DEFAULT
#define GPU_LIB_NAME "None"
#elif defined GPU_CUBLAS
#define GPU_LIB_NAME "NVIDIA cuBLAS"
#elif defined GPU_ONEMKL
#define GPU_LIB_NAME "Intel OneMKL"
#elif defined GPU_ROCBLAS
#define GPU_LIB_NAME "AMD rocBLAS"
#endif

#ifndef GPU_DEFAULT
#define GPU_ENABLED true
#else
#define GPU_ENABLED false
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
typedef enum {_fp32_, _fp64_ } dataTypes;