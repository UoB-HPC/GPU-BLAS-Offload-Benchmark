#pragma once

#include "../spmspm.hh"

#include <random>
#include <memory>
#include <iostream>

namespace cpu {

/** An abstract class for sparse matrix-sparse matrix BLAS kernels. */
template <typename T>
class spmspm : public ::spmspm<T> {
public:
  using ::spmspm<T>::spmspm;
  using ::spmspm<T>::initInputMatrices;
  using ::spmspm<T>::iterations_;
  using ::spmspm<T>::A_nnz_;
  using ::spmspm<T>::B_nnz_;
  using ::spmspm<T>::sparsity_;
  using ::spmspm<T>::type_;
  using ::spmspm<T>::m_;
  using ::spmspm<T>::n_;
  using ::spmspm<T>::k_;
  using ::spmspm<T>::C_rows_;
  using ::spmspm<T>::C_cols_;
  using ::spmspm<T>::C_vals_;
  using ::spmspm<T>::C_nnz_;

public:
  /** Initialise the required data structures. */
  void initialise(int n, int m, int k, double sparsity,
                  matrixType type, bool binary = false) {
    n_ = n;
    m_ = m;
    k_ = k;

    sparsity_ = sparsity;
    type_ = type;

    /** Determine the number of nnz elements in A and B */
    A_nnz_ = 1 + (uint64_t)((double)m_ * (double)k_ * (1.0 - sparsity_));
    B_nnz_ = 1 + (uint64_t)((double)k_ * (double)n_ * (1.0 - sparsity_));

    initInputMatrices();
  }

private:
    /** Do any necessary cleanup (free pointers, close library handles, etc.)
     * after Kernel has been called. */
  void postCallKernelCleanup() {}
};
}  // namespace cpu
