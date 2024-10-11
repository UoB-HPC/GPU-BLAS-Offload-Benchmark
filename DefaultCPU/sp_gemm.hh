#pragma once

#if defined CPU_DEFAULT

#include "../include/kernels/CPU/sp_gemm.hh"
#include "../include/utilities.hh"

namespace cpu {
/** A class for GEMM CPU BLAS kernels. */
template <typename T>
class sp_gemm_cpu : public sp_gemm<T> {
 public:
  using sp_gemm<T>::sp_gemm;
  using sp_gemm<T>::callConsume;
  using sp_gemm<T>::m_;
  using sp_gemm<T>::n_;
  using sp_gemm<T>::k_;
  using sp_gemm<T>::A_;
  using sp_gemm<T>::B_;
  using sp_gemm<T>::C_;

 private:
  /** Perform the GEMM kernel. */
  void callGemm() override {
    /** A naive implementation of a column-major GEMM. Alpha and Beta are always
     * 1 and 0 respectively.
     * Operation takes the form of C[M,N] = A[M,K] * B[K,N].
     * callConsume() is required to ensure that the compiler does not optimise
     * away this function. */
    int x, y, z;
    T acc;
    for (x = 0; x < m_; x++) {
      for (y = 0; y < n_; y++) {
        acc = 0.0;
        for (z = 0; z < k_; z++) {
          acc += A_[z * m_ + x] * B_[y * k_ + z];
        }
        C_[y * m_ + x] = acc;
      }
    }
    // Ensure compiler doesn't optimise away the work being done
    callConsume();
  }

  /** Perform any required steps before calling the GEMM kernel that should
   * be timed. */
  void preLoopRequirements() override {}

  /** Perform any required steps after calling the GEMM kernel that should
   * be timed. */
  void postLoopRequirements() override {}
};

}  // namespace cpu
#endif
