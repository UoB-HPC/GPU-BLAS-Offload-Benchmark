#pragma once

#include "../spmv.hh"

#include <random>
#include <memory>

namespace cpu {

/** An abstract class for spmv BLAS kernels. */
template <typename T>
class spmv : public ::spmv<T> {
public:
  using ::spmv<T>::spmv;
  using ::spmv<T>::initInputMatrixVector;
  using ::spmv<T>::m_;
  using ::spmv<T>::n_;
  using ::spmv<T>::x_;
  using ::spmv<T>::y_;
  using ::spmv<T>::sparsity_;
  using ::spmv<T>::nnz_;
  using ::spmv<T>::type_; 

public:
  /** Initialise the required data structures. */
  void initialise(int m, int n, double sparsity, matrixType type) {
    m_ = m;
    n_ = n;
    sparsity_ = sparsity;
    type_ = type;

    // Note that the below should be the same as the edges calculation
    // used in the initInputMatricesSparse function.  If changed here,
    // change there
    nnz_ = 1 + (uint64_t)((double)m_ * (double)n_ * (1.0 - sparsity_));

    x_ = (T*)malloc(sizeof(T) * n_);
    y_ = (T*)malloc(sizeof(T) * m_);

    // Initialise the matrix and vectors
    initInputMatrixVector();
  }

private:
  /** Do any necessary cleanup (free pointers, close library handles, etc.)
    * after Kernel has been called. */
  void postCallKernelCleanup() {
    free(x_);
    free(y_);
  }
};
}  // namespace cpu