#pragma once

#include <algorithm>
#include <chrono>
#include <cmath>
#include <limits>

#include "../utilities.hh"

/** A generic abstract class defining the operation of timing a GEMM BLAS
 * kernel for n iterations. */
template <typename T>
class gemm {
 public:
  gemm(const int iters) : iterations_(iters) {}

  /** Call the BLAS kernel n times.
   * Returns the time elapsed for n BLAS calls in seconds. */
  time_checksum compute() {
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

    uint64_t checksum = calcChecksum();

    postCallKernelCleanup();

    return {time_s.count(), checksum};
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

  uint64_t calcChecksum() {
    // Checksum for GEMM calculated by summing all four corners of A, B and C
    // together
    double retVal =
        A_[0] + A_[m_ - 1] + A_[k_ - 1] + A_[((m_ - 1) * k_) + (k_ - 1)];
    retVal += B_[0] + B_[k_ - 1] + B_[n_ - 1] + B_[((k_ - 1) * n_) + (n_ - 1)];
    retVal += C_[0] + C_[m_ - 1] + C_[n_ - 1] + C_[((m_ - 1) * n_) + (n_ - 1)];
    retVal = fabs(retVal);

    return std::min(static_cast<uint64_t>(std::floor(retVal)),
                    std::numeric_limits<uint64_t>::max());
  }

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
