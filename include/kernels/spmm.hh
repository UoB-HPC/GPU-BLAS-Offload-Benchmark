#pragma once

#include <algorithm>
#include <chrono>
#include <cmath>
#include <limits>
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

      // perform the SPMM calls
//      std::cout << ".. pre";
      preLoopRequirements();
      for (int i = 0; i < iterations_; i++) {
//        std::cout << ".. SPMM";
        callSpmm();
      }
//      std::cout << ".. post";
      postLoopRequirements();

      // Stop the timer
      std::chrono::time_point<std::chrono::high_resolution_clock> endTime =
              std::chrono::high_resolution_clock::now();
      std::chrono::duration<double> time_s = endTime - startTime;

      double checksum = calcChecksum();

//      std::cout << ".. cleanup";
      postCallKernelCleanup();

//      std::cout << ".. DONE";
      return {time_s.count(), checksum, 0.0};
    }

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
      if (C_nnz_ == 0) {
        return (double)0.0; // No non-zeros, return zero checksum
      } else if (C_nnz_ == 1) {
        return (double)C_vals_[0]; // Single non-zero, return its value
      } else {
        return (double)C_vals_[0] + (double)C_vals_[C_nnz_ - 1];
      }
    }

protected:
    /** Set up the starting matrices */
    void initInputMatrices() {
      bool valid_C = false;
      while (!valid_C) {
        for (int i = 0; i < (m_ * k_); i++) {
          A_[i] = 0.0;
        }
        for (int i = 0; i < (k_ * n_); i++) {
          B_[i] = 0.0;
        }

        rMat(A_, m_, k_, A_nnz_);
        rMat(B_, k_, n_, B_nnz_);

        // Count actual non-zeros
        int actual_nnz = 0;
        for (int i = 0; i < (m_ * k_); i++) {
          if (std::abs(A_[i]) > 1e-10) {
            actual_nnz++;
          }
        }
        if (actual_nnz == 0) {
          A_[0] = 7.8;
          actual_nnz = 1;
        }
        A_nnz_ = actual_nnz;
        actual_nnz = 0;
        for (int i = 0; i < (k_ * n_); i++) {
          if (std::abs(B_[i]) > 1e-10) {
            actual_nnz++;
          }
        }
        if (actual_nnz == 0) {
          B_[0] = 7.8;
          actual_nnz = 1;
        }
        B_nnz_ = actual_nnz;

        for (int Ar = 0; Ar < m_; Ar++) {
          for (int Bc = 0; Bc < n_; Bc++) {
            for (int i = 0; i < k_; i++) {
              int Aind = (Ar * k_) + i;
              int Bind = (i * n_) + Bc;
              if (std::abs(A_[Aind] * B_[Bind]) > 1e-10) {
                valid_C = true; 
                break;
              }
            }
            if (valid_C) break;
          }
          if (valid_C) break;
        }
      }
      toSparseFormat();
    }

    /** Move matrices into the sparse representation of for the given library */
    virtual void toSparseFormat() = 0;

    /** Call the external consume() function on the matrices */
    void callConsume() { consume((void*)A_, (void*)B_, (void*)C_); }/** Recursive function to populate sparse matrices */

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

    /** CSR representation of output matrix C. */
    int64_t C_nnz_;
    int64_t* C_rows_;
    int64_t* C_cols_;
    T* C_vals_;

    int64_t A_nnz_ = 0;
    int64_t B_nnz_ = 0;

    double sparsity_;

    bool print_ = false;

};