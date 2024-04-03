#pragma once

#include "../gemv.hh"

namespace cpu {

/** An abstract class for GEMV BLAS kernels. */
template <typename T>
class gemv : public ::gemv<T> {
 public:
  using ::gemv<T>::gemv;
  using ::gemv<T>::initInputMatrices;
  using ::gemv<T>::m_;
  using ::gemv<T>::n_;
  using ::gemv<T>::A_;
  using ::gemv<T>::x_;
  using ::gemv<T>::y_;

 public:
  /** Initialise the required data structures. */
  void initialise(int m, int n) {
    m_ = m;
    n_ = n;

    A_ = (T*)malloc(sizeof(T) * m_ * n_);
    x_ = (T*)malloc(sizeof(T) * n_);
    y_ = (T*)malloc(sizeof(T) * m_);

    // Initialise the matrix and vector
    initInputMatrixVector();
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