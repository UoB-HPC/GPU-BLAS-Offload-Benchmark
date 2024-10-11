#pragma once

#ifdef CPU_AOCL
#include <blis.h>

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
  /** Make call to the GEMM kernel. */
  void callGemm() override {
    if constexpr (std::is_same_v<T, float>) {
      bli_sgemm(BLIS_NO_TRANSPOSE, BLIS_NO_TRANSPOSE, m_, n_, k_, &alpha, A_,
                rowStride, std::max(1, m_), B_, rowStride, std::max(1, k_),
                &beta, C_, rowStride, std::max(1, m_));
    } else if constexpr (std::is_same_v<T, double>) {
      bli_dgemm(BLIS_NO_TRANSPOSE, BLIS_NO_TRANSPOSE, m_, n_, k_, &alpha, A_,
                rowStride, std::max(1, m_), B_, rowStride, std::max(1, k_),
                &beta, C_, rowStride, std::max(1, m_));
    } else {
      // Un-specialised class will not do any work - print error and exit.
      std::cout << "ERROR - Datatype for AOCL CPU GEMM kernel not supported."
                << std::endl;
      exit(1);
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

  /** The constant value Alpha. */
  T alpha = ALPHA;

  /** The constant value Beta. */
  T beta = BETA;

  /** The distance in elements to the next column. */
  const int rowStride = 1;
};
}  // namespace cpu
#endif