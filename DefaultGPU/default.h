#pragma once

/** As there are no default implementations of the GPU BLAS kernels, these
 * functions are here to maintain functionality inside main.c.
 * As such, all functions should return `inf` to represent the kernel not being
 * run. */

double gemm_gpu(const dataTypes dType, const uint64_t iters, const uint64_t m,
                const uint64_t n, const uint64_t k, const bool offloadOnce) {
  return INFINITY;
}