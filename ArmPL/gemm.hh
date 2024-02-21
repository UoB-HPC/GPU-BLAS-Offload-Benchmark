#pragma once

#ifdef CPU_ARMPL
#include <armpl.h>
#include <omp.h>

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
  void callGemm() override {
    if constexpr (std::is_same_v<T, float>) {
      cblas_sgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, m_, n_, k_, ALPHA,
                  A_, MAX(1, m_), B_, MAX(1, k_), BETA, C_, MAX(1, m_));
    } else if constexpr (std::is_same_v<T, double>) {
      cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, m_, n_, k_, ALPHA,
                  A_, MAX(1, m_), B_, MAX(1, k_), BETA, C_, MAX(1, m_));
    } else {
      // Un-specialised class will not do any work - print error and exit.
      std::cout << "ERROR - Datatype for ArmPL CPU GEMM kernel not supported."
                << std::endl;
      exit(1);
    }
    // Ensure compiler doesn't optimise away the work being done
    callConsume();
  }

  /** Perform any required steps before the calling the GEMM kernel that should
   * be timed. */
  void preLoopRequirements() override {}

  /** Perform any required steps after the calling the GEMM kernel that should
   * be timed. */
  void postLoopRequirements() override {}
};
}  // namespace cpu
#endif