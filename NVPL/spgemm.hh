#pragma once

#ifdef CPU_NVPL

#include "../include/kernels/CPU/spgemm.hh"
#include "../include/utilities.hh"

namespace cpu {
template <typename T>
class spgemm_cpu : public spgemm<T> {
public:
  using spgemm<T>::spgemm;
  using spgemm<T>::callConsume;
  using spgemm<T>::initInputMatrices;
  using spgemm<T>::m_;
  using spgemm<T>::n_;
  using spgemm<T>::k_;
  using spgemm<T>::B_;
  using spgemm<T>::C_;
  using spgemm<T>::sparsity_;
  using spgemm<T>::type_;
  using spgemm<T>::nnz_;
  using spgemm<T>::iterations_;

  void initialise(int m, int n, int k, double sparsity,
                  matrixType type, bool binary = false) {}

protected:
  void toSparseFormat() override {}

private:
  void preLoopRequirements() override {}

  void callSpgemm() override {}

  void postLoopRequirements() override {}

  void postCallKernelCleanup() override {}
};
}


#endif
