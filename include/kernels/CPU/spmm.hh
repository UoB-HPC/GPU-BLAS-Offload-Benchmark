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
  using ::spmm<T>::nnzA_;
  using ::spmm<T>::nnzB_;
  using ::spmm<T>::sparsity_;
  using ::spmm<T>::m_;
  using ::spmm<T>::n_;
  using ::spmm<T>::k_;
  using ::spmm<T>::A_;
  using ::spmm<T>::B_;
  using ::spmm<T>::C_;

public:
  /** Initialise the required data structures. */
  void initialise(int n, int m, int k, double sparsity,
                  bool binary = false) {
    n_ = n;
    m_ = m;
    k_ = k;

    sparsity_ = sparsity;

    /** Determine the number of nnz elements in A and B */
    nnzA_ = 1 + (uint64_t)((double)m_ * (double)k_ * (1.0 - sparsity_));
    nnzB_ = 1 + (uint64_t)((double)k_ * (double)n_ * (1.0 - sparsity_));

    A_ = (T*)malloc(sizeof(T) * m_ * k_);
    B_ = (T*)malloc(sizeof(T) * k_ * n_);
    C_ = (T*)calloc(sizeof(T) * m_ * n_);

    initInputMatrices();
  }

private:
    /** Do any necessary cleanup (free pointers, close library handles, etc.)
     * after Kernel has been called. */
  void postCallKernelCleanup() {
    free(A_);
    free(B_);
    free(C_);
  }
};
}  // namespace cpu
