#pragma once

#ifdef CPU_ONEMKL
#include <mkl.h>

#include <algorithm>

#include "../../include/kernels/CPU/gemv.hh"
#include "../../include/utilities.hh"

namespace cpu {
/** A class for GEMV CPU BLAS kernels. */
template <typename T>
class gemv_cpu : public gemv<T> {
 public:
  using gemv<T>::gemv;
  using gemv<T>::initInputMatrixVector;
  using gemv<T>::callConsume;
  using gemv<T>::m_;
  using gemv<T>::n_;
  using gemv<T>::A_;
  using gemv<T>::x_;
  using gemv<T>::y_;
  using gemv<T>::vecIncrement_;

  /** Initialise the required data structures. */
  void initialise(int m, int n) {
    m_ = m;
    n_ = n;

    A_ = (T*)mkl_malloc(sizeof(T) * m_ * n_, 64);
    x_ = (T*)mkl_malloc(sizeof(T) * n_, 64);
    y_ = (T*)mkl_malloc(sizeof(T) * m_, 64);

    // Initialise the input matrix and vectors
    initInputMatrixVector();
  }

 private:
  /** Make call to the GEMV kernel. */
  void callGemv() override {
    if constexpr (std::is_same_v<T, float>) {
      cblas_sgemv(CblasColMajor, CblasNoTrans, m_, n_, alpha, A_,
                  std::max(1, m_), x_, vecIncrement_, beta, y_, vecIncrement_);
    } else if constexpr (std::is_same_v<T, double>) {
      cblas_dgemv(CblasColMajor, CblasNoTrans, m_, n_, alpha, A_,
                  std::max(1, m_), x_, vecIncrement_, beta, y_, vecIncrement_);
    } else {
      // Un-specialised class will not do any work - print error and exit.
      std::cout << "ERROR - Datatype for OneMKL CPU GEMV kernel not supported."
                << std::endl;
      exit(1);
    }
    // Ensure compiler doesn't optimise away the work being done
    callConsume();
  }

  /** Perform any required steps before calling the GEMV kernel that should
   * be timed. */
  void preLoopRequirements() override {}

  /** Perform any required steps after calling the GEMV kernel that should
   * be timed. */
  void postLoopRequirements() override {}

  /** Do any necessary cleanup (free pointers, close library handles, etc.)
   * after Kernel has been called. */
  void postCallKernelCleanup() override {
    mkl_free(A_);
    mkl_free(x_);
    mkl_free(y_);
  }

  /** The constant value Alpha. */
  const T alpha = ALPHA;

  /** The constant value Beta. */
  const T beta = BETA;
};
}  // namespace cpu
#endif