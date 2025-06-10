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
      std::cout << ".. pre";
      preLoopRequirements();
      for (int i = 0; i < iterations_; i++) {
        std::cout << ".. SPMM";
        callSpmm();
      }
      std::cout << ".. post";
      postLoopRequirements();

      // Stop the timer
      std::chrono::time_point<std::chrono::high_resolution_clock> endTime =
              std::chrono::high_resolution_clock::now();
      std::chrono::duration<double> time_s = endTime - startTime;

      double checksum = calcChecksum();

      std::cout << ".. cleanup";
      postCallKernelCleanup();

      std::cout << ".. DONE";
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
      std::cout << "Zeroing A.. ";
      for (int i = 0; i < (m_ * k_); i++) {
        A_[i] = 0.0;
      }
      std::cout << "Zeroing B.. ";
      for (int i = 0; i < (k_ * n_); i++) {
        B_[i] = 0.0;
      }
      std::cout << "Zeroing C.. ";
      for (int i = 0; i < (m_ * n_); i++) {
        C_[i] = 0.0;
      }

      // Random number generator objects for use in descent
      std::default_random_engine gen;
      gen.seed(std::chrono::system_clock::now()
                       .time_since_epoch().count());
      std::uniform_real_distribution<double> dist(0.0, 1.0);

      // Using a=0.45 and b=c=0.22 as default probabilities
      std::cout << std::endl << "RMAT for A (nnz = " << nnzA_ << "): ";
      for (int i = 0; i < nnzA_; i++) {
        while (!rMat(A_, k_, 0, k_ - 1, 0, m_ - 1, 0.45, 0.22, 0.22, &gen, dist,
                     false)) {std::cout << "fail,  ";}
        std::cout << "success " << i << ", ";
      }
      std::cout << std::endl << "RMAT for B (nnz = " << nnzB_ << "): ";
      for (int i = 0; i < nnzB_; i++) {
        while (!rMat(B_, n_, 0, n_ - 1, 0, k_ - 1, 0.45, 0.22, 0.22, &gen, dist,
                     false)) {std::cout << "fail,  ";}
        std::cout << "success " << i << ", ";
      }

      std::cout << std::endl << "To Sparse!";

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

    double sparsity_;

};