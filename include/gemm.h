#include "flags.h"

/** A Macro for calculating FLOPs performed by a GEMM. */
#define gemmFlops(M, N, K) (M * N * K * 2)

/** Performs GEMM operations of type `dType` on host CPU for `iters` iterations.
 * Returns the time taken to perform the operation in seconds. */
double gemm_cpu(const dataTypes dType, const uint64_t iters, const uint64_t m,
                const uint64_t n, const uint64_t k) {
  clock_t t = clock();
  t = clock() - t;
  return ((double)t) / CLOCKS_PER_SEC;
}

/** Performs GEMM operations of type `dType` on host GPU for `iters` iterations.
 * Returns the time taken to perform the operation in seconds.
 *  - `offloadOnce` refers to whether the matrix data should be offloaded to the
 *    device once before computation and then copied back at the end of all
 *    iterations, or if the matrcies should be offloaded to/from the device
 *    every iteration. */
double gemm_gpu(const dataTypes dType, const uint64_t iters, const uint64_t m,
                const uint64_t n, const uint64_t k, const bool offloadOnce) {
  // Conditionally execute the kernel
  if (!GPU_ENABLED) {
    return 0.0;
  }

  clock_t t = clock();
  t = clock() - t;
  return ((double)t) / CLOCKS_PER_SEC;
}