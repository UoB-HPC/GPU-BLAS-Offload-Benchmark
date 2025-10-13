#pragma once

#if defined GPU_CUBLAS

#include <cusparse.h>
#include <cuda_runtime.h>
#include <cstdio>

/** Macro function to check if error occurred when calling cuBLAS. */
/** Macro function to check if error occurred when calling CUDA. */
#define cudaCheckError(f)                                                 \
  do {                                                                    \
    if (cudaError_t e = (f); e != cudaSuccess) {                          \
      std::cout << "CUDA error: " << __FILE__ << ":" << __LINE__ << ": "; \
      std::cout << cudaGetErrorString(e) << std::endl;                    \
      exit(1);                                                            \
    }                                                                     \
  } while (false)

/** Macro function to check if error occurred when calling cuBLAS. */
#define cublasCheckError(f)                                                   \
  do {                                                                        \
    cublasStatus_t status = (f);                                              \
    if (status != CUBLAS_STATUS_SUCCESS) {                                    \
      std::cout << "CUBLAS error: " << __FILE__ << ":" << __LINE__ << ": ";   \
      std::cout << cublasGetStatusName(status) << " - ";                      \
      std::cout << cublasGetStatusString(status) << std::endl;                \
      exit(1);                                                                \
    }                                                                         \
  } while (false)

/** Macro function to check if error occurred when calling cuSPARSE. */
#define cusparseCheckError(f)                                                 \
  do {                                                                        \
    cusparseStatus_t status = (f);                                            \
    if (status != CUSPARSE_STATUS_SUCCESS) {                                  \
      std::cout << "CUSPARSE error: " << __FILE__ << ":" << __LINE__ << ": "; \
      std::cout << cusparseGetErrorName(status) << " - ";                     \
      std::cout << cusparseGetErrorString(status) << std::endl;               \
      exit(1);                                                                \
    }                                                                         \
  } while (false)                                                             \

#endif


