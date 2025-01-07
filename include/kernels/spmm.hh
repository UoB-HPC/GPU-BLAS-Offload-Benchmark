#pragma one

#include <algorithm>
#include <chrono>
#include <cmath>
#include <limits>
#include <random>
#include <iostream>

#include "../utilities.hh"

/** A generic abstract class defining the operation of timing a SPMM BLAS
 * kernel for n iterations */
template <typename T>
class spmm {
public:
    spmm(const int iters) : iterations_(iters) {}

    /** Call the kernel n times.  Returns the time elapsed for all n calls
     * in seconds */
    time_checksum_gflop compute() {
      // Start the timer
      std::chrono::time_point<std::chrono::high_resolution_clock> startTime =
              std::chrono::high_resolution_clock::now();

      // perform tje SPMM calls
      preLoopRequirements();
      for (int i = 0; i < iterations_; i++) {
        callSpmm();
      }
      postLoopRequirements();

      // Stop the timer
      std::chrono::time_point<std::chrono::high_resolution_clock> endTime =
              std::chrono::high_resolution_clock::now();
      std::chrono::duration<double> time_s = endTime - startTime;

      double checksum = calcChecksum();

      postCallKernelCleanup();

      return {time_s.count(), checksum, 0.0};
    }

    int64_t nnzA_ = 0;
    int64_t nnzB_ = 0;
    int64_t nnzC_ = 0;

private:
    /** Performs the steps required before calling the SPMM kernel that
     * should be timed */
    virtual void preLoopRequirements() = 0;

    /** Perform the SPMM kernel. */
    virtual void callSpmm() = 0;

    /** Perform any steps required after calling the SPMM kernel that should
     * be timed */
    virtual void postLoopRequirements() = 0;

    /** Do the necessary cleanup after the kernel has been finished that
     * should not be timed */
    virtual void postCallKernelCleanup() = 0;

    /** Calculate a checksum from the result matrix C. */
    constexpr double calcChecksum() {
      // Todo -- think about how this can sensibly be done for SPMM
      return 0.0;
    }

protected:
    /** Set up the starting matrices */
    void initInputMatrices() {
      for (size_t i = 0; i < (m_ * k_); i++) {
        A_[i] = 0.0;
      }
      for (size_t i = 0; i < (k_ * n_); i++) {
        B_[i] = 0.0;
      }

      // Random number generator objects for use in descent
      std::default_random_engine gen;
      gen.seed(std::chrono::system_clock::now()
                       .time_since_epoch().count());
      std::uniform_real_distribution<double> dist(0.0, 1.0);

      // Using a=0.45 and b=c=0.22 as default probabilities
      for (size_t i = 0; i < nnzA_; i++) {
        while (!rMat(A_, k_, 0, k_ - 1, 0, m_ - 1, 0.45, 0.22, 0.22, &gen, dist,
                     false)) {}
      }
      for (size_t i = 0; i < nnzB_; i++) {
        while (!rMat(B_, n_, 0, n_ - 1, 0, k_ - 1, 0.45, 0.22, 0.22, &gen, dist,
                     false)) {}
      }

      toSparseFormat()
    }

    /** Move matrices into the sparse representation of for the given library */
    virtual void toSparseFormat() = 0;

    /** Call the external consume() function on the matrices */
    void callConsume() { consume((void*)A_, (void*)B_, (void*)C_); }/** Recursive function to populate sparse matrices */

    // On first iteration, n should be x2 + 1
    bool rMat(T* M, int n, int x1, int x2, int y1, int y2, float a, float b,
              float c, std::default_random_engine* gen,
              std::uniform_real_distribution<double> dist, bool bin) {
      // If a 1x1 submatrix, then add an edge and return out
      if (x1 >= x2 && y1 >= y2) {
        // Needed to avoid overflow segfaults with large problem sizes
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

        // Work out which quarter to recurse into
        // There are some ugly ternary operators here to avoid going out of bounds in the edge case
        // that we are already at 1 width or 1 height
        float randomNum = dist(*gen);
        if (randomNum < a) {
          return rMat(M, n, x1, xMidPoint, y1, yMidPoint,
                      a, b, c, gen, dist, bin);
        } else if (randomNum < (a + b)) {
          return rMat(M, n, ((xMidPoint < x2) ? xMidPoint + 1 : xMidPoint), x2, y1, yMidPoint,
                      a, b, c, gen, dist, bin);
        } else if (randomNum < (a + b + c)) {
          return rMat(M, n, x1, xMidPoint, ((yMidPoint < y2) ? yMidPoint + 1 : yMidPoint), y2,
                      a, b, c, gen, dist, bin);
        } else {
          return rMat(M, n, ((xMidPoint < x2) ? xMidPoint + 1 : xMidPoint), x2,
                      ((yMidPoint < y2) ? yMidPoint + 1 : yMidPoint), y2, a,
                      b, c, gen, dist, bin);
        }
      }
    }

    /** The number of iterations to perform per problem size. */
    const int iterations_;

    /** Matrix dimension M. */
    int m_ = 0;

    /** Matrix dimension N. */
    int n_ = 0;

    /** Matrix dimension K. */
    int k_ = 0;

    /** Dense representation of input matrix A. */
    T* A_;

    /** Dense representation of input matrix B. */
    T* B_;

    /** Dense representation of output matrix C. */
    T* C_;

};