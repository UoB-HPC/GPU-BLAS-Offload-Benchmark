#pragma once

#ifdef CPU_AOCL

#include "aoclsparse.h"
#include <algorithm>

#include "./common.hh"
#include "../include/kernels/CPU/spmv.hh"
#include "../include/utilities.hh"

namespace cpu {
template <typename T>
class spmv_cpu : public spmv<T> {
public:
    using spmv<T>::spmv;
    using spmv<T>::callConsume;
    using spmv<T>::initInputMatrixVector;
    using spmv<T>::m_;
    using spmv<T>::n_;
    using spmv<T>::x_;
    using spmv<T>::y_;
    using spmv<T>::sparsity_;
    using spmv<T>::type_;
    using spmv<T>::nnz_;
    using spmv<T>::iterations_;

    void initialise(int m, int n, double sparsity, matrixType type, 
                    bool binary = false) {
      base_ = aoclsparse_index_base_zero;
      operation_ = aoclsparse_operation_none;

      m_aocl_ = m_ = m;
      n_aocl_ = n_ = n;
      sparsity_ = sparsity;
      type_ = type;

      nnz_ = 1 + (uint64_t)((double)m_ * (double)n_ * (1.0 - sparsity_));
      nnz_aocl_ = nnz_;

      x_ = (T*)calloc(n_, sizeof(T));
      y_ = (T*)calloc(m_, sizeof(T));

      initInputMatrixVector();

      aoclCheckError(aoclsparse_create_mat_descr(&A_description_));
    }

protected:
    void toSparseFormat() override {
      A_vals_ = (T*)calloc(nnz_aocl_, sizeof(T));
      A_cols_ = (aoclsparse_int*)calloc(nnz_aocl_, sizeof(aoclsparse_int));
      A_rows_ = (aoclsparse_int*)calloc(m_ + 1, sizeof(aoclsparse_int));
      if (type_ == matrixType::rmat) {
        rMatCSR<T, aoclsparse_int>(A_vals_, A_cols_, A_rows_, m_, n_, nnz_);
      } else if (type_ == matrixType::random) {
        randomCSR<T, aoclsparse_int>(A_vals_, A_cols_, A_rows_, m_, n_, nnz_);
      } else if (type_ == matrixType::bandedDiagonal) {
        bandedDiagonalCSR<T, aoclsparse_int>(A_vals_, A_cols_, A_rows_, m_, n_, nnz_);
      } else {
        std::cerr << "Matrix type not supported" << std::endl;
        exit(1);
      }
      

      // Move into the AOCL CSR matrix handle
      if constexpr (std::is_same_v<T, float>) {
        aoclCheckError(aoclsparse_create_scsr(&A_aocl_, 
                                              base_, 
                                              m_aocl_, 
                                              n_aocl_, 
                                              nnz_aocl_, 
                                              A_rows_, 
                                              A_cols_, 
                                              A_vals_));
      } else if constexpr (std::is_same_v<T, double>) {
        aoclCheckError(aoclsparse_create_dcsr(&A_aocl_, 
                                              base_, 
                                              m_aocl_, 
                                              n_aocl_, 
                                              nnz_aocl_, 
                                              A_rows_, 
                                              A_cols_, 
                                              A_vals_));
      }
    }

private:
    void preLoopRequirements() override {
      aoclCheckError(aoclsparse_set_mv_hint(A_aocl_, 
                                            operation_, 
                                            A_description_,
                                            10)); // Currently hard coded iternation count

      aoclCheckError(aoclsparse_optimize(A_aocl_));
    }

    void callSpmv() override {
      if constexpr (std::is_same_v<T, float>) {
        aoclCheckError(aoclsparse_smv(operation_, 
                                      &alpha, 
                                      A_aocl_, 
                                      A_description_, 
                                      x_, 
                                      &beta, 
                                      y_));
      } else if constexpr (std::is_same_v<T, double>) {
        aoclCheckError(aoclsparse_dmv(operation_, 
                                      &alpha, 
                                      A_aocl_, 
                                      A_description_, 
                                      x_, 
                                      &beta, 
                                      y_));
      }
    }

    void postLoopRequirements() override {}

    void postCallKernelCleanup() override {
      aoclCheckError(aoclsparse_destroy_mat_descr(A_description_));
      aoclCheckError(aoclsparse_destroy(&A_aocl_));

      delete[] A_vals_;
      delete[] A_cols_;
      delete[] A_rows_;
      delete[] x_;
      delete[] y_;
    }

    aoclsparse_status status_;

    aoclsparse_operation operation_;
    aoclsparse_index_base base_;

    aoclsparse_matrix A_aocl_;
    aoclsparse_int* A_rows_;
    aoclsparse_int* A_cols_;
    T* A_vals_;
    aoclsparse_int m_aocl_;
    aoclsparse_int n_aocl_;
    aoclsparse_int nnz_aocl_;

    aoclsparse_mat_descr A_description_;

    const T alpha = ALPHA;
    const T beta = BETA;
};
}


#endif
