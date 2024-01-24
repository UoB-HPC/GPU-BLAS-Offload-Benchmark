#pragma once

#include "../kernel.hh"

namespace cpu {

/** An abstract class for GEMM BLAS kernels. */
template <typename T>
class gemm : public ::kernel<T> {
 public:
  using kernel<T>::kernel;

  /** Initialise the required data structures. */
  virtual void initialise(int m, int n, int k) = 0;

 protected:
  /** Matrix dimension M. */
  int m_ = 0;

  /** Matrix dimension N. */
  int n_ = 0;

  /** Matrix dimension K. */
  int k_ = 0;
};
}  // namespace cpu