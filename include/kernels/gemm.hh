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
  time_checksum_gflop compute() {
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

    double checksum = calcChecksum();

    postCallKernelCleanup();

    return {time_s.count(), checksum, 0.0};
  }

 private:
  /** Perform any required steps before calling the GEMM kernel that should
   * be timed. */
  virtual void preLoopRequirements() = 0;

  /** Perform the GEMM kernel. */
  virtual void callGemm() = 0;

  /** Perform any required steps after calling the GEMM kernel that should
   * be timed. */
  virtual void postLoopRequirements() = 0;

  /** Do any necessary cleanup (free pointers, close library handles, etc.)
   * after Kernel has been called. */
  virtual void postCallKernelCleanup() = 0;

  /** Calculate a checksum from the result matrix C. */
  constexpr double calcChecksum() {
    // Checksum for GEMM calculated by summing all four corners of C together
    return ((double)C_[0] + (double)C_[m_ - 1] + (double)C_[(m_ * (n_ - 1))] +
            (double)C_[m_ * n_ - 1]);
  }

 protected:
  /** Initialise the input matrices. */
  void initInputMatrices() {
    // Seed the random number generator
    srand(SEED);
    for (int y = 0; y < m_; y++) {
      for (int x = 0; x < k_; x++) {
        A_[y * k_ + x] = (T)((double)(rand() % 100) / 7.0);
      }
    }
    for (int y = 0; y < k_; y++) {
      for (int x = 0; x < n_; x++) {
        B_[y * n_ + x] = (T)((double)(rand() % 100) / 3.0);
      }
    }
    for (int y = 0; y < m_; y++) {
      for (int x = 0; x < n_; x++) {
        C_[y * n_ + x] = (T)0.0;
      }
    }
  }

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
