#pragma once

#if defined CPU_DEFAULT

#include "../include/kernels/CPU/gemm.hh"
#include "../include/utilities.hh"

namespace cpu {
/** A class for GEMM CPU BLAS kernels. */
template <typename T>
class gemm_cpu : public gemm<T> {
 public:
  using gemm<T>::gemm;
  using gemm<T>::callConsume;
  using gemm<T>::m_;
  using gemm<T>::n_;
  using gemm<T>::k_;
  using gemm<T>::A_;
  using gemm<T>::B_;
  using gemm<T>::C_;

 private:
  /** Perform the GEMM kernel `iterations_` times. */
  virtual void callGemm() override {
    /** A naive implementation of a GEMM. Alpha and Beta are always 1 and 0
     * respectively.
     * Operation takes the form of C[M,N] = A[M,K] * B[K,N].
     * A return value is required to ensure that the compiler does not
     * optimise away this function. */
    int x, y, z;
    T acc;
    for (x = 0; x < m_; x++) {
      for (y = 0; y < n_; y++) {
        acc = 0.0;
        for (z = 0; z < k_; z++) {
          acc += A_[x * k_ + z] * B_[z * n_ + y];
        }
        C_[x * n_ + y] = acc;
      }
    }
    // Ensure compiler doesn't optimise away the work being done
    callConsume();
  }

  /** Perform any required steps before the calling the GEMM kernel that should
   * be timed. */
  virtual void preLoopRequirements() override {}

  /** Perform any required steps after the calling the GEMM kernel that should
   * be timed. */
  virtual void postLoopRequirements() override {}
};

}  // namespace cpu
#endif