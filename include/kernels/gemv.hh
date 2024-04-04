#pragma once

#include <algorithm>
#include <chrono>
#include <cmath>
#include <limits>

#include "../utilities.hh"

/** A generic abstract class defining the operation of timing a GEMM BLAS
 * kernel for n iterations. */
template <typename T>
class gemv {
 public:
  gemv(const int iters) : iterations_(iters) {}

  /** Call the BLAS kernel n times.
   * Returns the time elapsed for n BLAS calls in seconds. */
  time_checksum_gflop compute() {
    // Start timer
    std::chrono::time_point<std::chrono::high_resolution_clock> startTime =
        std::chrono::high_resolution_clock::now();

    // Perform all GEMM calls
    preLoopRequirements();
    for (int i = 0; i < iterations_; i++) {
      callGemv();
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
  /** Perform any required steps before calling the GEMV kernel that should
   * be timed. */
  virtual void preLoopRequirements() = 0;

  /** Perform the GEMV kernel. */
  virtual void callGemv() = 0;

  /** Perform any required steps after calling the GEMV kernel that should
   * be timed. */
  virtual void postLoopRequirements() = 0;

  /** Do any necessary cleanup (free pointers, close library handles, etc.)
   * after Kernel has been called. */
  virtual void postCallKernelCleanup() = 0;

  /** Calculate a checksum from the result vector y. */
  constexpr double calcChecksum() {
    // Checksum for GEMV calculated by summing max and min element of output
    // vector
    return ((double)y_[0] + (double)y_[n_ - 1]);
  }

 protected:
  /** Initialise the input data structures. */
  void initInputMatrixVector() {
    // Seed the random number generator
    srand(SEED);
    for (int y = 0; y < m_; y++) {
      for (int x = 0; x < n_; x++) {
        A_[y * n_ + x] = (T)((double)(rand() % 100) / 7.0);
      }
    }
    for (int y = 0; y < n_; y++) {
      x_[y] = (T)((double)(rand() % 100) / 3.0);
    }
    for (int y = 0; y < m_; y++) {
      c_[y] = (T)0.0;
    }
  }

  /** Call the extern consume() function. */
  void callConsume() { consume((void*)A_, (void*)x_, (void*)y_); }

  /** The number of iterations to perform per problem size. */
  const int iterations_;

  /** Matrix dimension M. */
  int m_ = 0;

  /** Matrix / vector dimension N. */
  int n_ = 0;

  /** Input matrix A. */
  T* A_;

  /** Input vector x. */
  T* x_;

  /** Input vector y. */
  T* y_;

  /** The distance between two vector elements. */
  const int vecIncrement_ = 1;
};
