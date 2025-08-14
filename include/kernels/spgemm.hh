#pragma once

#include <algorithm>
#include <chrono>
#include <cmath>
#include <limits>
#include <random>
#include <iostream>

#include "../utilities.hh"

/**
* A generic abstract class defining the operation of timing a sparse GEMM
 * BLAS kernel for n iterations
*/
template <typename T>
class spgemm {
public:
    spgemm(const int iters) : iterations_(iters) {}

    /** Call the kernel n times.  Returns the time elapsed for all n calls
     * in seconds */
    time_checksum_gflop compute() {
      // Start the timer
      std::chrono::time_point<std::chrono::high_resolution_clock> startTime =
              std::chrono::high_resolution_clock::now();

      // perform the SPMM calls
      if (print_) std::cout << "\t\tPre-loop requirements" << std::endl;
      preLoopRequirements();
      for (int i = 0; i < iterations_; i++) {
        if (print_) std::cout << "\t\tcallSpgemm" << std::endl;
        callSpgemm();
      }
      if (print_) std::cout << "\t\tPost-loop requirements" << std::endl;
      postLoopRequirements();

      // Stop the timer
      std::chrono::time_point<std::chrono::high_resolution_clock> endTime =
              std::chrono::high_resolution_clock::now();
      std::chrono::duration<double> time_s = endTime - startTime;

      double checksum = calcChecksum();

      postCallKernelCleanup();

      return {time_s.count(), checksum, 0.0};
    }

    int64_t nnz_ = 0;

private:
    /** Performs the steps required before calling the SPMM kernel that
     * should be timed */
    virtual void preLoopRequirements() = 0;

    /** Perform the sparse GEMM kernel. */
    virtual void callSpgemm() = 0;

    /** Perform any steps required after calling the SPMM kernel that should
     * be timed */
    virtual void postLoopRequirements() = 0;

    /** Do the necessary cleanup after the kernel has been finished that
     * should not be timed */
    virtual void postCallKernelCleanup() = 0;

    /** Calculate a checksum from the result matrix C. */
    constexpr double calcChecksum() {
      // Checksum for GEMM calculated by summing all four corners of C together
      return ((double)C_[0] + (double)C_[m_ - 1] + (double)C_[(m_ * (n_ - 1))] +
              (double)C_[m_ * n_ - 1]);
    }

protected:
    /** Set up the starting matrices */
    void initInputMatrices() {
      if (print_) std::cout << "DEBUG: initInputMatrices - Start" << std::endl;
      if (print_) std::cout << "  m_=" << m_ << ", n_=" << n_ << ", k_=" << k_ << std::endl;
      if (print_) std::cout << "  nnz_=" << nnz_ << ", sparsity_=" << sparsity_ << std::endl;

      // Initialize A to zero
      if (print_) std::cout << "DEBUG: Zeroing matrix A (size=" << (m_ * k_) << ")" << std::endl;
      for (int i = 0; i < (m_ * k_); i++) {
        if (print_) std::cout << "A_[" << i << "] = 0.0;" << std::endl;
        A_[i] = 0.0;
      }

      // Initialize B with random values
      if (print_) std::cout << "DEBUG: Initializing matrix B" << std::endl;
      srand(SEED);
      for (int i = 0; i < (k_ * n_); i++) {
        B_[i] = (T)((double)(rand() % 100) / 7.0);
      }

      // Initialize C to zero
      if (print_) std::cout << "DEBUG: Initializing matrix C" << std::endl;
      for (int i = 0; i < (m_ * n_); i++) {
        C_[i] = (T)0.0;
      }

      // Generate sparse matrix using R-MAT
      if (print_) std::cout << "DEBUG: Generating sparse matrix with R-MAT" << std::endl;
      rMat(A_, m_, k_, nnz_);

      if (print_) std::cout << "DEBUG: R-MAT generation complete. Successful: " << successful_inserts << ", Failed: " << failed_attempts << std::endl;

      // Count actual non-zeros
      int actual_nnz = 0;
      for (int i = 0; i < (m_ * k_); i++) {
        if (std::abs(A_[i]) > 1e-10) {
          actual_nnz++;
        }
      }
      if (print_) std::cout << "DEBUG: Actual non-zeros in A: " << actual_nnz << std::endl;

      if (print_) std::cout << "DEBUG: Calling toSparseFormat()" << std::endl;
      toSparseFormat();
      if (print_) std::cout << "DEBUG: initInputMatrices - Complete" << std::endl;
    }

    bool print_ = false;

    /** Move matrices into the sparse representation of for the given library */
    virtual void toSparseFormat() = 0;

    /** Call the external consume() function on the matrices */
    void callConsume() { consume((void*)A_, (void*)B_, (void*)C_); }/** Recursive function to populate sparse matrices */

    /** The number of iterations to perform per problem size. */
    const int iterations_;

    /** Matrix dimension M. */
    int64_t m_ = 0;

    /** Matrix dimension N. */
    int64_t n_ = 0;

    /** Matrix dimension K. */
    int64_t k_ = 0;

    /** Dense representation of input matrix A. */
    T* A_;

    /** Dense representation of input matrix B. */
    T* B_;

    /** Dense representation of output matrix C. */
    T* C_;

    double sparsity_;
};