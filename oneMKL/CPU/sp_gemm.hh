#pragma once

#ifdef CPU_ONEMKL
#include <mkl.h>

#include <algorithm>

#include "../../include/kernels/CPU/sp_gemm.hh"
#include "../../include/utilities.hh"

namespace cpu {
/** A class for GEMM CPU Sparse BLAS kernels. */
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
  using sp_gemm<T>::nnz_;
  using sp_gemm<T>::A_vals_;
  using sp_gemm<T>::B_vals_;
  using sp_gemm<T>::C_vals_;

 private:
  void callGemm() override {
    º
  }
};

}
#endif
