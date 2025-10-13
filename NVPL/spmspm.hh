#pragma once

#ifdef CPU_NVPL

#include "../include/kernels/CPU/spmspm.hh"
#include "../include/utilities.hh"

namespace cpu {
template <typename T>
class spmspm_cpu : public spmspm<T> {
public:

  void initialise(int m, int n, int k, double sparsity,
                  matrixType type, bool binary = false) {}

protected:
  void toSparseFormat() override {}

private:
  void preLoopRequirements() override {}

  void callSpmspm() override {}

  void postLoopRequirements() override {}

  void postCallKernelCleanup() override {}
};
}

#endif
