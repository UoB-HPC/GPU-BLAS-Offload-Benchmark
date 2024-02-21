#pragma once

#include <chrono>

/** A generic abstract class defining the operation of timing a GEMM BLAS
 * kernel for n iterations. */
template <typename T>
class gemm {
 public:
  gemm(const int iters) : iterations_(iters) {}

  /** Call the BLAS kernel n times.
   * Returns the time elapsed for n BLAS calls in seconds. */
  double compute() {
    // Start timer
    std::chrono::time_point<std::chrono::high_resolution_clock> startTime =
        std::chrono::high_resolution_clock::now();

    // Perform all GEMM calls
    preLoopRequirements();
    for (int i = 0; i < iterations_; i++) {
      callGemm();
    }
    postLoopRequirements();

    // Stop Timer
    std::chrono::time_point<std::chrono::high_resolution_clock> endTime =
        std::chrono::high_resolution_clock::now();
    // Get time elapsed in seconds
    std::chrono::duration<double> time_s = endTime - startTime;

    postCallKernelCleanup();

    return time_s.count();
  }

 private:
  /** Perform any required steps before the calling the GEMM kernel that should
   * be timed. */
  virtual void preLoopRequirements() = 0;

  /** Perform the GEMM kernel. */
  virtual void callGemm() = 0;

  /** Perform any required steps after the calling the GEMM kernel that should
   * be timed. */
  virtual void postLoopRequirements() = 0;

  /** Do any necessary cleanup (free pointers, close library handles, etc.)
   * after Kernel has been called. */
  virtual void postCallKernelCleanup() = 0;

 protected:
  /** Call the extern consume() function. */
  void callConsume() { consume((void*)A_, (void*)B_, (void*)C_); }

  /** The number of iterations to perform per problem size. */
  const int iterations_;

  /** Matrix dimension M. */
  int m_ = 0;

  /** Matrix dimension N. */
  int n_ = 0;

  /** Matrix dimension K. */
  int k_ = 0;

  /** Input matrix A. */
  T* A_;

  /** Input matrix B. */
  T* B_;

  /** Input matrix C. */
  T* C_;
};
