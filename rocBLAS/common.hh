#pragma once

#if defined GPU_ROCBLAS

/** Macro function to check if error occurred when calling cuBLAS. */
#define hipCheckError(f)                                                \
  do {                                                                  \
    if (hipError_t e = (f); e != hipSuccess) {                          \
      std::cout << "HIP error: " << __FILE__ << ":" << __LINE__ << ": " \
                << hipGetErrorString(e) << std::endl;                   \
      exit(1);                                                          \
    }                                                                   \
  } while (false)

#endif