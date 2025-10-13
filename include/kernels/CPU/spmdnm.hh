#pragma once

#include "../spmdnm.hh"

namespace cpu {

/**
 * An abstract class for sparse matrix-dense matrix BLAS kernels
 */
template <typename T>
class spmdnm : public :: spmdnm<T> {
public:
  using ::spmdnm<T>::spmdnm;
  using ::spmdnm<T>::initInputMatrices;
  using ::spmdnm<T>::iterations_;
  using ::spmdnm<T>::nnz_;
  using ::spmdnm<T>::sparsity_;
  using ::spmdnm<T>::type_;
  using ::spmdnm<T>::m_;
  using ::spmdnm<T>::n_;
  using ::spmdnm<T>::k_;
  using ::spmdnm<T>::B_;
  using ::spmdnm<T>::C_;

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
      std::cerr << "ERROR: Memory allocation failed in spmdnm initialization" << std::endl;
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