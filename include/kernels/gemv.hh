#pragma once

#include <algorithm>
#include <chrono>
#include <cmath>
#include <limits>
#include <random>

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
    return ((double)y_[0] + (double)y_[m_ - 1]);
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
      y_[y] = (T)0.0;
    }
  }

  void initInputMatrixVectorSparse() {
    // Initialise sparse matrix
    for (int i = 0; i < (n_ * n_); i++) {
      A_[i] = 0.0;
    }

    // Random number generator objects for use in descent
    std::default_random_engine gen;
    gen.seed(std::chrono::system_clock::now()
                     .time_since_epoch().count());
    std::uniform_real_distribution<double> dist(0.0, 1.0);

    uint64_t edges = 1 + (uint64_t)((double)n_ * (double)n_ * (1.0 -
            sparsity_));

    // Using a=0.45 and b=c=0.22 as default probabilities
    for (uint64_t i = 0; i < edges; i++) {
      while (!rMat(A_, n_, 0, n_ - 1, 0, n_ - 1, 0.45, 0.22, 0.22, &gen, dist,
                   false)) {}
    }

    // Initialise the input and output vectors
    for (int y = 0; y < n_; y++) {
      x_[y] = (T)((double)(rand() % 100) / 3.0);
    }
    for (int y = 0; y < m_; y++) {
      y_[y] = (T)0.0;
    }
  }

  /** Recursive function to populate sparse matrices */
  bool rMat(T* M, int n, int x1, int x2, int y1, int y2, float a, float b,
            float c, std::default_random_engine* gen,
            std::uniform_real_distribution<double> dist, bool bin) {
    // If a 1x1 submatrix, then add an edge and return out
    if (x1 >= x2 && y1 >= y2) {
      // Needed to avoid overfloe segfaults with large problem sizes
      uint64_t index = (((uint64_t)y1 * (uint64_t)n) + (uint64_t)x1);
      if (abs(M[index]) > 0.1) {
        return false;
      } else {
        // Add 1.0 if this is a binary graph, and a random real number otherwise
        M[index] = (bin) ? 1.0 : (((rand() % 10000) / 100.0) - 50.0);
        return true;
      }
    } else {
      // Divide up the matrix
      int xMidPoint = x1 + floor((x2 - x1) / 2);
      int yMidPoint = y1 + floor((y2 - y1) / 2);

      // ToDo -- add some noise to these values between iterations
      float newA = a;
      float newB = b;
      float newC = c;

      // Work out which quarter to recurse into
      // There are some ugly ternary operators here to avoid going out of bounds in the edge case
      // that we are already at 1 width or 1 height
      float randomNum = dist(*gen);
      if (randomNum < a) {
        return rMat(M, n, x1, xMidPoint, y1, yMidPoint,
                    newA, newB, newC, gen, dist, bin);
      } else if (randomNum < (a + b)) {
        return rMat(M, n, ((xMidPoint < x2) ? xMidPoint + 1 : xMidPoint), x2, y1, yMidPoint,
                    newA, newB, newC, gen, dist, bin);
      } else if (randomNum < (a + b + c)) {
        return rMat(M, n, x1, xMidPoint, ((yMidPoint < y2) ? yMidPoint + 1 : yMidPoint), y2,
                    newA, newB, newC, gen, dist, bin);
      } else {
        return rMat(M, n, ((xMidPoint < x2) ? xMidPoint + 1 : xMidPoint), x2,
                    ((yMidPoint < y2) ? yMidPoint + 1 : yMidPoint), y2, newA, newB, newC,
                    gen, dist, bin);
      }
    }
    return true;
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

  double sparsity_ = 0.0;
};
