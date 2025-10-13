#pragma once

#include "../spmdnv.hh"

#include <random>
#include <memory>

namespace cpu {

/** An abstract class for SpMDnV BLAS kernels. */
template <typename T>
class spmdnv : public ::spmdnv<T> {
public:
  using ::spmdnv<T>::spmdnv;
  using ::spmdnv<T>::initInputMatrixVector;
  using ::spmdnv<T>::m_;
  using ::spmdnv<T>::n_;
  using ::spmdnv<T>::x_;
  using ::spmdnv<T>::y_;
  using ::spmdnv<T>::sparsity_;
  using ::spmdnv<T>::nnz_;
  using ::spmdnv<T>::type_; 

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