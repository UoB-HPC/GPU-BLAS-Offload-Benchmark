#pragma once

#ifdef CPU_NVPL

#include "../include/kernels/CPU/spmm.hh"
#include "../include/utilities.hh"

namespace cpu {
template <typename T>
class spmm_cpu : public spmm<T> {
public:

  void initialise(int m, int n, int k, double sparsity,
                  matrixType type, bool binary = false) {}

protected:
  void toSparseFormat() override {}

private:
  void preLoopRequirements() override {}

  void callSpmm() override {}

  void postLoopRequirements() override {}

  void postCallKernelCleanup() override {}
};
}

#endif
