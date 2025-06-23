
#pragma once

#include <algorithm>
#include <chrono>
#include <cmath>
#include <limits>
#include <random>

#include "../utilities.hh"

/** A generic abstract class defining the operation of timing an SPGEMM BLAS
 * kernel for n iterations. */
template <typename T>
class spgemv {
public:
    spgemv(const int iters) : iterations_(iters) {}

    /** Call the BLAS kernel n times.
     * Returns the time elapsed for n BLAS calls in seconds. */
    time_checksum_gflop compute() {
      // Start timer
      std::chrono::time_point<std::chrono::high_resolution_clock> startTime =
              std::chrono::high_resolution_clock::now();

      // Perform all SPGEMM calls
      preLoopRequirements();
      for (int i = 0; i < iterations_; i++) {
        callSpgemv();
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

    int64_t nnz_ = 0;

private:
    /** Perform any required steps before calling the SPGEMV kernel that should
     * be timed. */
    virtual void preLoopRequirements() = 0;

    /** Perform the SPGEMV kernel. */
    virtual void callSpgemv() = 0;

    /** Perform any required steps after calling the SPGEMV kernel that should
     * be timed. */
    virtual void postLoopRequirements() = 0;

    /** Do any necessary cleanup (free pointers, close library handles, etc.)
     * after Kernel has been called. */
    virtual void postCallKernelCleanup() = 0;

    /** Calculate a checksum from the result vector y. */
    // Todo -- work out how to sensibly do this for sparse
    constexpr double calcChecksum() {
      // Checksum for GEMV calculated by summing max and min element of output
      // vector
      return ((double)y_[0] + (double)y_[m_ - 1]);
    }

protected:
    void initInputMatrixVector() {
      // Initialise matric to
      for (int i = 0; i < (n_ * m_); i++) {
        A_[i] = 0.0;
      }

      // Random number generator objects for use in descent
      std::default_random_engine gen;
      gen.seed(std::chrono::system_clock::now().time_since_epoch().count());
      std::uniform_real_distribution<double> dist(0.0, 1.0);

      if (print_) std::cout << "DEBUG: Generating sparse matrix with R-MAT" << std::endl;
      int successful_inserts = 0;
      int failed_attempts = 0;
      const int max_attempts_per_element = 100;

      for (int i = 0; i < nnz_; i++) {
        int attempts = 0;
        bool inserted = false;

        while (!inserted && attempts < max_attempts_per_element) {
          inserted = rMat(A_, n_, 0, n_ - 1, 0, m_ - 1, 0.45, 0.22, 0.22,
                          &gen, dist, false);
          attempts++;
        }

        if (inserted) {
          successful_inserts++;
        } else {
          failed_attempts++;
          if (print_) std::cout << "WARNING: Failed to insert element " << i << " after " << attempts << " attempts" << std::endl;
        }

        // Progress update
        if ((i + 1) % 1000 == 0) {
          if (print_) std::cout << "  Generated " << (i + 1) << "/" << nnz_ << " non-zeros" << std::endl;
        }
      }

      if (print_) std::cout << "DEBUG: R-MAT generation complete. Successful: " << successful_inserts << ", Failed: " << failed_attempts << std::endl;


      // Initialise the input and output vectors
      for (int y = 0; y < n_; y++) {
        x_[y] = (T)((double)(rand() % 100) / 3.0);
      }
      for (int y = 0; y < m_; y++) {
        y_[y] = (T)0.0;
      }

      toSparseFormat();
    }

    bool print_ = false;

    /** Move starting matrix into the sparse representation of for the given
     * library */
    virtual void toSparseFormat() = 0;

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
