#pragma once

#include "kernel.hh"

namespace cpu {

/** An abstract class for GEMM BLAS kernels. */
    template <typename T>
    class spGemm : public kernel<T> {
    public:
        using kernel<T>::kernel;

        /** Initialise the required data structures. */
        virtual void initialise(int n, float sparsity) = 0;

    protected:
        /** Matrix size -- matrix will be nxn */
        int n_ = 0;

}  // namespace cpu