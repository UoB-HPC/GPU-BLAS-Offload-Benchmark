#pragma once

#include "../spgemv.hh"

#include <random>
#include <memory>

namespace cpu {

/** An abstract class for GEMV BLAS kernels. */
    template <typename T>
    class spgemv : public ::spgemv<T> {
    public:
        using ::spgemv<T>::spgemv;
        using ::spgemv<T>::initInputMatrixVector;
        using ::spgemv<T>::m_;
        using ::spgemv<T>::n_;
        using ::spgemv<T>::x_;
        using ::spgemv<T>::y_;
        using ::spgemv<T>::sparsity_;
        using ::spgemv<T>::nnz_;
        using ::spgemv<T>::type_; 

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