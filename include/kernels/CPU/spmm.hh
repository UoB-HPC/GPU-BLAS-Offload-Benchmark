#pragma once

#include "../spmm.hh"

#include <random>
#include <memory>
#include <iostream>

namespace cpu {

/** An abstract class for sparse matrix-sparse matrix BLAS kernels. */
template <typename T>
class spmm : public ::spmm<T> {
public:
  using ::spmm<T>::spmm;
  using ::spmm<T>::initInputMatrices;
  using ::spmm<T>::iterations_;
  using ::spmm<T>::A_nnz_;
  using ::spmm<T>::B_nnz_;
  using ::spmm<T>::sparsity_;
  using ::spmm<T>::type_;
  using ::spmm<T>::m_;
  using ::spmm<T>::n_;
  using ::spmm<T>::k_;
  using ::spmm<T>::C_rows_;
  using ::spmm<T>::C_cols_;
  using ::spmm<T>::C_vals_;
  using ::spmm<T>::C_nnz_;

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
  void postCallKernelCleanup() {
  }
};
}  // namespace cpu
