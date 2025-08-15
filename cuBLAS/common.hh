#pragma once

#if defined GPU_CUBLAS

#include <cusparse_v2.h>
#include <cuda_runtime.h>
#include <cstdio>

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
#define cublasCheckError(f)                                              \
  do {                                                                   \
    switch (f) {                                                         \
        case CUBLAS_STATUS_SUCCESS:                                      \
          break;                                                         \
        case CUBLAS_STATUS_NOT_INITIALIZED:                              \
          std::cout << "CUBLAS error: " << __FILE__ << ": " << __LINE__  \
          << ": CUBLAS_STATUS_NOT_INITIALIZED" << std::endl;             \
          exit(1);                                                       \
        case CUBLAS_STATUS_ALLOC_FAILED:                                 \
          std::cout << "CUBLAS error: " << __FILE__ << ": " << __LINE__  \
          << ": CUBLAS_STATUS_ALLOC_FAILED" << std::endl;                \
          exit(1);                                                       \
        case CUBLAS_STATUS_INVALID_VALUE:                                \
          std::cout << "CUBLAS error: " << __FILE__ << ": " << __LINE__  \
          << ": CUBLAS_STATUS_INVALID_VALUE" << std::endl;               \
          exit(1);                                                       \
        case CUBLAS_STATUS_ARCH_MISMATCH:                                \
          std::cout << "CUBLAS error: " << __FILE__ << ": " << __LINE__  \
          << ": CUBLAS_STATUS_ARCH_MISMATCH" << std::endl;               \
          exit(1);                                                       \
        case CUBLAS_STATUS_MAPPING_ERROR:                                \
          std::cout << "CUBLAS error: " << __FILE__ << ": " << __LINE__  \
          << ": CUBLAS_STATUS_MAPPING_ERROR" << std::endl;               \
          exit(1);                                                       \
        case CUBLAS_STATUS_EXECUTION_FAILED:                             \
          std::cout << "CUBLAS error: " << __FILE__ << ": " << __LINE__  \
          << ": CUBLAS_STATUS_EXECUTION_FAILED" << std::endl;            \
          exit(1);                                                       \
        case CUBLAS_STATUS_INTERNAL_ERROR:                               \
          std::cout << "CUBLAS error: " << __FILE__ << ": " << __LINE__  \
          << ": CUBLAS_STATUS_INTERNAL_ERROR" << std::endl;              \
          exit(1);                                                       \
        case CUBLAS_STATUS_NOT_SUPPORTED:                                \
          std::cout << "CUBLAS error: " << __FILE__ << ": " << __LINE__  \
          << ": CUBLAS_STATUS_NOT_SUPPORTED" << std::endl;               \
          exit(1);                                                       \
        case CUBLAS_STATUS_LICENSE_ERROR:                                \
          std::cout << "CUBLAS error: " << __FILE__ << ": " << __LINE__  \
          << ": CUBLAS_STATUS_LICENSE_ERROR" << std::endl;               \
          exit(1);                                                       \
        default:                                                         \
          std::cout << "CUBLAS error: " << __FILE__ << ": " << __LINE__  \
          << ": other error not in switch statement" << std::endl;       \
          exit(1);                                                       \
    }                                                                    \
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

inline cudaError_t safeCudaMemPrefetchAsync(
    const void* devPtr,
    size_t count,
    int dstDevice,
    cudaStream_t stream)
{
    if (devPtr == nullptr || count == 0) {
        // Nothing to prefetch
        return cudaSuccess;
    }

    cudaPointerAttributes attr;
    cudaError_t err = cudaPointerGetAttributes(&attr, devPtr);

    if (err != cudaSuccess) {
        // Could not get attributes — treat as non-managed
        cudaGetLastError(); // clear error state
        return cudaSuccess;
    }

    // Check for unified (managed) memory
#if CUDART_VERSION >= 10000
    bool isManaged = (attr.type == cudaMemoryTypeManaged);
#else
    bool isManaged = (attr.memoryType == cudaMemoryTypeManaged);
#endif

    if (!isManaged) {
        // Not managed memory — skip prefetch
        // printf("Skipping prefetch: pointer is not managed memory\n");
        return cudaSuccess;
    }

    // Perform prefetch
    return cudaMemPrefetchAsync(devPtr, count, dstDevice, stream);
}

#endif


