#pragma once

#include <time.h>

#include <vector>

#ifdef CPU_ARMPL
#include <armpl.h>
#include <omp.h>
#endif

#include "../include/CPU/gemm.hh"
#include "../include/utilities.hh"

namespace cpu {

#if defined CPU_ARMPL
/** A class for GEMM CPU BLAS kernels. */
template <typename T>
class gemm_cpu : public gemm<T> {
 public:
  using gemm<T>::gemm;
  using gemm<T>::m_;
  using gemm<T>::n_;
  using gemm<T>::k_;

  /** Initialise the required data structures. */
  virtual void initialise(int m, int n, int k) override {
    m_ = m;
    n_ = n;
    k_ = k;

    A_.reserve(m * k);
    B_.reserve(k * n);
    C_.reserve(m * n);

    // Initialise the matricies
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
  /** Make a class to the BLAS Library Kernel. */
  virtual void callKernel() override {
    if constexpr (std::is_same_v<T, float>) {
      cblas_sgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, m_, n_, k_, ALPHA,
                  A_.data(), MAX(1, m_), B_.data(), MAX(1, k_), BETA, C_.data(),
                  MAX(1, m_));
    } else if constexpr (std::is_same_v<T, double>) {
      cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, m_, n_, k_, ALPHA,
                  A_.data(), MAX(1, m_), B_.data(), MAX(1, k_), BETA, C_.data(),
                  MAX(1, m_));
    } else {
      // Un-specialised class will not do any work - print error and exit.
      std::cout << "ERROR - Datatype for ArmPL CPU GEMM kernel not supported."
                << std::endl;
      exit(1);
    }
  }

  /** Call the extern consume() function. */
  virtual void callConsume() override {
    consume((void*)A_.data(), (void*)B_.data(), (void*)C_.data());
  }

  /** Input matrix A. */
  std::vector<T> A_;

  /** Input matrix B. */
  std::vector<T> B_;

  /** Input matrix C. */
  std::vector<T> C_;
};
#endif
}  // namespace cpu