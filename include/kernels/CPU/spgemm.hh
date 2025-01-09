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
    using ::spgemm<T>::m_;
    using ::spgemm<T>::n_;
    using ::spgemm<T>::k_;
    using ::spgemm<T>::A_;
    using ::spgemm<T>::B_;
    using ::spgemm<T>::C_;

public:
    /**
     * Initialise the required data structures.
     */
    void initialise(int n, int m, int k, double sparsity,
                    bool binary = false) {
      n_ = n;
      m_ = m;
      k_ = k;

      sparsity_ = sparsity;

      nnz_ = 1 + (uint64_t)((double)m_ * (double)k_ * (1.0 - sparsity_));

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

}