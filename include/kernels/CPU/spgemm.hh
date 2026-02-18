#pragma once

#include "../spgemm.hh"

#include <random>
#include <memory>
#include <iostream>

namespace cpu {

/** An abstract class for sparse matrix-sparse matrix BLAS kernels. */
template <typename T>
class spgemm : public ::spgemm<T> {
public:
  using ::spgemm<T>::spgemm;
  using ::spgemm<T>::initInputMatrices;
  using ::spgemm<T>::iterations_;
  using ::spgemm<T>::A_nnz_;
  using ::spgemm<T>::B_nnz_;
  using ::spgemm<T>::sparsity_;
  using ::spgemm<T>::type_;
  using ::spgemm<T>::m_;
  using ::spgemm<T>::n_;
  using ::spgemm<T>::k_;
  using ::spgemm<T>::C_rows_;
  using ::spgemm<T>::C_cols_;
  using ::spgemm<T>::C_vals_;
  using ::spgemm<T>::C_nnz_;

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
