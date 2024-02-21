#pragma once

#include "../gemm.hh"

namespace cpu {

/** An abstract class for GEMM BLAS kernels. */
template <typename T>
class gemm : public ::gemm<T> {
 public:
  using ::gemm<T>::gemm;
  using ::gemm<T>::m_;
  using ::gemm<T>::n_;
  using ::gemm<T>::k_;
  using ::gemm<T>::A_;
  using ::gemm<T>::B_;
  using ::gemm<T>::C_;

 public:
  /** Initialise the required data structures. */
  void initialise(int m, int n, int k) {
    m_ = m;
    n_ = n;
    k_ = k;

    A_ = (T*)malloc(sizeof(T) * m_ * k_);
    B_ = (T*)malloc(sizeof(T) * k_ * n_);
    C_ = (T*)malloc(sizeof(T) * m_ * n_);

    // Initialise the matricies
    srand(SEED);
    for (int y = 0; y < m; y++) {
      for (int x = 0; x < k; x++) {
        A_[y * k + x] = (((T)(rand() % 10000) / 100.0) - 30.0);
      }
    }
    for (int y = 0; y < k; y++) {
      for (int x = 0; x < n; x++) {
        B_[y * n + x] = (((T)(rand() % 10000) / 100.0) - 30.0);
      }
    }
  }

 private:
  /** Do any necessary cleanup (free pointers, close library handles, etc.)
   * after Kernel has been called. */
  void postCallKernelCleanup() override {
    free(A_);
    free(B_);
    free(C_);
  }
};
}  // namespace cpu