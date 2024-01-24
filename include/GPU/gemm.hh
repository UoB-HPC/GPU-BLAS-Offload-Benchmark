#pragma once

#include "kernel.hh"

namespace gpu {

/** An abstract class for GEMM BLAS kernels. */
template <typename T>
class gemm : public kernel<T> {
 public:
  using kernel<T>::kernel;

  /** Initialise the required data structures.
   * `offloadOnce` refers to whether the data should be offloaded to/from the
   * GPU every iteration, or offloaded once before all iterations and collected
   * after all iterations.
   *  - TRUE = offload before all iterations, collect after all iterations
   *  - FALSE = offload to/from each iteration */
  virtual void initialise(bool offloadOnce, int m, int n, int k) = 0;

 protected:
  /** Whether data should be offloaded to/from the GPU each iteration, or just
   * before & after. */
  bool offloadOnce_ = false;

  /** Matrix dimension M. */
  int m_ = 0;

  /** Matrix dimension N. */
  int n_ = 0;

  /** Matrix dimension K. */
  int k_ = 0;
};
}  // namespace gpu