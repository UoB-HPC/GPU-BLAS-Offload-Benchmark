#pragma once

#if defined GPU_DEFAULT
#include <time.h>

#include <cmath>
#include <vector>

#include "../include/GPU/gemm.hh"
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
  virtual void callKernel(const int iterations) override {
    // Default GPU implementation - do nothing.
  }

  /** Call the extern consume() function. */
  void callConsume() override {
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