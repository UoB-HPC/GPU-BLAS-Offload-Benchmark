#pragma once

#if defined GPU_DEFAULT

#include <cmath>

#include "../include/kernels/GPU/sp_gemm.hh"
#include "../include/utilities.hh"

namespace gpu {
/** A class for GEMM GPU BLAS kernels. */
template <typename T>
class sp_gemm_gpu : public sp_gemm<T> {
 public:
  using sp_gemm<T>::sp_gemm;

  /** Call the BLAS kernel n times, with 1 warmup run.
   * Returns the time elapsed for n BLAS calls in seconds. */
  time_checksum_gflop compute() {
    // Override function in base `kernel` class as DefaultGPU should do nothing.
    return {INFINITY, INFINITY, 0.0};
  }

  /** Initialise the required data structures. */
  void initialise(gpuOffloadType offload, int m, int n, int k) override {
    // Default GPU implementation - do nothing.
  }

 private:
  /** Make a call to the BLAS Library Kernel. */
  void callGemm() override {
    // Default GPU implementation - do nothing.
  }

  /** Perform any required steps before calling the GEMM kernel that should
   * be timed. */
  void preLoopRequirements() override {
    // Default GPU implementation - do nothing.
  }

  /** Perform any required steps after calling the GEMM kernel that should
   * be timed. */
  void postLoopRequirements() override {
    // Default GPU implementation - do nothing.
  }

  /** Do any necessary cleanup (free pointers, close library handles, etc.)
   * after Kernel has been called. */
  void postCallKernelCleanup() override {
    // Default GPU implementation - do nothing.
  }
};
}  // namespace gpu
#endif