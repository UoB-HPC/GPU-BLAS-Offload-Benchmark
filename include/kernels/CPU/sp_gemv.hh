#pragma once

#include "../gemv.hh"

#include <random>
#include <memory>

namespace cpu {

/** An abstract class for GEMV BLAS kernels. */
    template <typename T>
    class sp_gemv : public ::gemv<T> {
    public:
        using ::gemv<T>::gemv;
        using ::gemv<T>::initInputMatrixVectorSparse;
        using ::gemv<T>::m_;
        using ::gemv<T>::n_;
        using ::gemv<T>::A_;
        using ::gemv<T>::x_;
        using ::gemv<T>::y_;
        using ::gemv<T>::sparsity_;

    public:
        /** Initialise the required data structures. */
        void initialise(int n, double sparsity) {
          m_ = n;
          n_ = n;
          sparsity_ = sparsity;

          A_ = (T*)malloc(sizeof(T) * m_ * n_);
          x_ = (T*)malloc(sizeof(T) * n_);
          y_ = (T*)malloc(sizeof(T) * m_);

          // Initialise the matrix and vectors
          initInputMatrixVectorSparse();
        }

    private:
        /** Do any necessary cleanup (free pointers, close library handles, etc.)
         * after Kernel has been called. */
        void postCallKernelCleanup() override {
          free(A_);
          free(x_);
          free(y_);
        }
    };
}  // namespace cpu