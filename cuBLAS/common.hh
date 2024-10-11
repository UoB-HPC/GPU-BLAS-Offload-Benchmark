#pragma once

#if defined GPU_CUBLAS

#include "cusparse.h"

/** Macro function to check if error occurred when calling cuBLAS. */
/** Macro function to check if error occurred when calling CUDA. */
#define cudaCheckError(f)                                                \
  do {                                                                   \
    if (cudaError_t e = (f); e != cudaSuccess) {                         \
      std::cout << "CUDA error: " << __FILE__ << ":" << __LINE__ << ": " \
                << cudaGetErrorString(e) << std::endl;                   \
      exit(1);                                                           \
    }                                                                    \
  } while (false)

/** Macro function to check if error occurred when calling cuBLAS. */
#define cublasCheckError(f)                                                \
  do {                                                                     \
    if (cublasStatus_t e = (f); e != CUBLAS_STATUS_SUCCESS) {              \
      std::cout << "CUBLAS error: " << __FILE__ << ":" << __LINE__ << ": " \
                << cublasGetStatusString(e) << std::endl;                  \
      exit(1);                                                             \
    }                                                                      \
  } while (false)

#define cusparseCheckError(f)                                                 \
  do {                                                                        \
    cusparseStatus_t status = (f);                                            \
    if (status != CUSPARSE_STATUS_SUCCESS) {                                  \
      std::cout << "CUSPARSE error: " << __FILE__ << ":" << __LINE__ << ": "  \
      << cusparseGetErrorString(status) << std::endl;                         \
      exit(1);                                                                \
    }                                                                         \
  } while (false)                                                             \

#endif