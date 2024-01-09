#include "../gpuKernels.h"
#include "commonHeaders.h"

double gemm_gpu(const dataTypes dType, const int iters, const int m,
                const int n, const int k, const bool offloadOnce) {
  // If no GPU BLAS library has been selected at compile time then no GPU
  // kernels will be run. To maintain functionality, Infinity is returned as
  // time taken.
  return INFINITY;
}