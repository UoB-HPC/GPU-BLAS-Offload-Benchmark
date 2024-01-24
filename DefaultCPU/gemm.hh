#pragma once

#include <time.h>

#include <vector>

#include "../include/CPU/gemm.hh"
#include "../include/utilities.hh"

namespace cpu {

#if defined CPU_DEFAULT
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
    /** A naive implementation of a GEMM. Alpha and Beta are always 1 and 0
     * respectively.
     * Operation takes the form of C[M,N] = A[M,K] * B[K,N].
     * A return value is required to ensure that the compiler does not optimise
     * away this function. */
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