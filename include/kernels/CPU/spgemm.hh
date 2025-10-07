#pragma once

#include "../spgemm.hh"

namespace cpu {

/**
 * An abstract class for sparse matrix-dense matrix BLAS kernels
 */
template <typename T>
class spgemm : public :: spgemm<T> {
public:
  using ::spgemm<T>::spgemm;
  using ::spgemm<T>::initInputMatrices;
  using ::spgemm<T>::iterations_;
  using ::spgemm<T>::nnz_;
  using ::spgemm<T>::sparsity_;
  using ::spgemm<T>::type_;
  using ::spgemm<T>::m_;
  using ::spgemm<T>::n_;
  using ::spgemm<T>::k_;
  using ::spgemm<T>::B_;
  using ::spgemm<T>::C_;

public:
  /**
    * Initialise the required data structures.
    */
  void initialise(int m, int n, int k, double sparsity,
                  matrixType type, bool binary = false) {
    m_ = m;
    n_ = n;
    k_ = k;
    sparsity_ = sparsity;
    type_ = type;

    nnz_ = 1 + (uint64_t)((double)m_ * (double)k_ * (1.0 - sparsity_));

    // Allocate memory for dense matrices
    B_ = (T*)calloc(k_ * n_, sizeof(T));
    C_ = (T*)calloc(m_ * n_, sizeof(T));

    // Check for allocation failures
    if (!B_ || !C_) {
      std::cerr << "ERROR: Memory allocation failed in spgemm initialization" << std::endl;
      exit(1);
    }

    initInputMatrices();
  }

private:
    /** Do any necessary cleanup (free pointers, close library handles, etc.)
     * after Kernel has been called. */
    void postCallKernelCleanup() {
      free(B_);
      free(C_);
    }
};

}