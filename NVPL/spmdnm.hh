#pragma once

#ifdef CPU_NVPL

#include "../include/kernels/CPU/spmdnm.hh"
#include "../include/utilities.hh"

namespace cpu {
template <typename T>
class spmdnm_cpu : public spmdnm<T> {
public:
  using spmdnm<T>::spmdnm;
  using spmdnm<T>::callConsume;
  using spmdnm<T>::initInputMatrices;
  using spmdnm<T>::m_;
  using spmdnm<T>::n_;
  using spmdnm<T>::k_;
  using spmdnm<T>::B_;
  using spmdnm<T>::C_;
  using spmdnm<T>::sparsity_;
  using spmdnm<T>::type_;
  using spmdnm<T>::nnz_;
  using spmdnm<T>::iterations_;

  void initialise(int m, int n, int k, double sparsity,
                  matrixType type, bool binary = false) {}

protected:
  void toSparseFormat() override {}

private:
  void preLoopRequirements() override {}

  void callSpmdnm() override {}

  void postLoopRequirements() override {}

  void postCallKernelCleanup() override {}
};
}


#endif
