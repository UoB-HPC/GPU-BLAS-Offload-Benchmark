#pragma once

#ifdef CPU_ARMPL

#include "../include/kernels/CPU/spmm.hh"
#include "../include/utilities.hh"

namespace cpu {
template <typename T>
class spmm_cpu : public spmm<T> {
public:
  using spmm<T>::spmm;
  using spmm<T>::callConsume;
  using spmm<T>::initInputMatrices;
  using spmm<T>::m_;
  using spmm<T>::n_;
  using spmm<T>::k_;
  using spmm<T>::B_;
  using spmm<T>::C_;
  using spmm<T>::sparsity_;
  using spmm<T>::type_;
  using spmm<T>::nnz_;
  using spmm<T>::iterations_;

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
