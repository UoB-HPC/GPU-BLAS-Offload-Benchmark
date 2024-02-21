#pragma once

#if defined GPU_DEFAULT

#include <cmath>

#include "../include/kernels/GPU/gemm.hh"
#include "../include/utilities.hh"

namespace gpu {
/** A class for GEMM GPU BLAS kernels. */
template <typename T>
class gemm_gpu : public gemm<T> {
 public:
  using gemm<T>::gemm;

  /** Call the BLAS kernel n times, with 1 warmup run.
   * Returns the time elapsed for n BLAS calls in seconds. */
  double compute() {
    // Override function in base `kernel` class as DefaultGPU should do nothing.
    return INFINITY;
  }

  /** Initialise the required data structures. */
  virtual void initialise(gpuOffloadType offload, int m, int n,
                          int k) override {
    // Default GPU implementation - do nothing.
  }

 private:
  /** Make a class to the BLAS Library Kernel. */
  virtual void callGemm() override {
    // Default GPU implementation - do nothing.
  }

  /** Perform any required steps before the calling the GEMM kernel that should
   * be timed. */
  virtual void preLoopRequirements() override {
    // Default GPU implementation - do nothing.
  }

  /** Perform any required steps after the calling the GEMM kernel that should
   * be timed. */
  virtual void postLoopRequirements() override {
    // Default GPU implementation - do nothing.
  }

  /** Do any necessary cleanup (free pointers, close library handles, etc.)
   * after Kernel has been called. */
  virtual void postCallKernelCleanup() override {
    // Default GPU implementation - do nothing.
  }
};
}  // namespace gpu
#endif