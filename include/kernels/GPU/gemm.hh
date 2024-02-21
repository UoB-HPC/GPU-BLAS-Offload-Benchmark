#pragma once

#include "../gemm.hh"

namespace gpu {

/** An abstract class for GEMM BLAS kernels. */
template <typename T>
class gemm : public ::gemm<T> {
 public:
  using ::gemm<T>::gemm;

  /** Initialise the required data structures.
   * `offload` refers to the data offload type:
   *  - Once:    Move data from host to device before all iterations & move from
   *             device to host after all iterations
   *  - Always:  Move data from host to device and device to host each iteration
   *  - Unified: Initialise data as unified memory; no data movement semantics
   *             required */
  virtual void initialise(gpuOffloadType offload, int m, int n, int k) = 0;

 protected:
  /** Whether data should be offloaded to/from the GPU each iteration, or just
   * before & after. */
  gpuOffloadType offload_ = gpuOffloadType::always;
};
}  // namespace gpu