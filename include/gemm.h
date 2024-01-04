#include "./flags.h"

/** Performs GEMM operations of type `dType` on host CPU for `iters` iterations.
 * Returns the total number of floating point operations (FLOPs) performed. */
uint64_t gemm_cpu(const dataTypes dType, const uint64_t iters, const uint64_t m,
                  const uint64_t k, const uint64_t n) {
  return 0;
}

/** Performs GEMM operations of type `dType` on host GPU for `iters` iterations.
 * `offloadOnce` refers to whether the matrix data should be offloaded to the
 * device once before computation and then copied back at the end of all
 * iterations, or if the matrcies should be offloaded to/from the device every
 * iteration.
 * Returns the total number of floating point operations (FLOPs)
 * performed. */
uint64_t gemm_gpu(const dataTypes dType, const uint64_t iters, const uint64_t m,
                  const uint64_t k, const uint64_t n, const bool offloadOnce) {
  return 0;
}