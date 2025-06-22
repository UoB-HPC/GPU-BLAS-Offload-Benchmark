#pragma once

#ifdef CPU_AOCL
#include "aoclsparse.h"

#include <algorithm>

#include "../include/kernels/CPU/spgemv.hh"
#include "../include/utilities.hh"

namespace cpu {
template <typename T>
class spgemv_cpu : public spgemv<T> {
public:
    using spgemv<T>::spgemv;
    using spgemv<T>::callConsume;
    using spgemv<T>::initInputMatrixVector;
    using spgemv<T>::m_;
    using spgemv<T>::n_;
    using spgemv<T>::A_;
    using spgemv<T>::x_;
    using spgemv<T>::y_;
    using spgemv<T>::sparsity_;
    using spgemv<T>::nnz_;
    using spgemv<T>::iterations_;

    void initialise(int m, int n, double sparsity, bool binary = false) {
      base_ = aoclsparse_index_base_zero;
      operation_ = aoclsparse_operation_none;

      m_aocl_ = m_ = m;
      n_aocl_ = n_ = n;
      sparsity_ = sparsity;

      nnz_ = 1 + (uint64_t)((double)m_ * (double)n_ * (1.0 - sparsity_));
      nnz_aocl_ = nnz_;

      A_rows_ = (aoclsparse_int*)malloc(sizeof(aoclsparse_int) * (m_ + 1));
      A_cols_ = (aoclsparse_int*)malloc(sizeof(aoclsparse_int) * nnz_);
      A_vals_ = (T*)malloc(sizeof(T) * nnz_);


      initInputMatrixVector();
    }

protected:
    void toSparseFormat() override {
      int nnz_encountered = 0;

      A_rows_[0] = 0;

      for (int row = 0; row < m_; row++) {
        A_rows_[row + 1] = nnz_encountered;
        for (int col = 0; col < n_; col++) {
          if (A_[(row * n_) + col] != 0.0) {
            A_cols_[nnz_encountered] = col;
            A_vals_[nnz_encountered] = static_cast<T>(A_[(row * n_) + col]);
            nnz_encountered++;
          }
        }
      }
      status_ = aoclsparse_create_mat_descr(&A_description_);

      if constexpr (std::is_same_v<T, float>) {
        status_ = aoclsparse_create_scsr(&A_aocl_,
                                         base_,
                                         m_aocl_,
                                         n_aocl_,
                                         nnz_aocl_,
                                         A_rows_,
                                         A_cols_,
                                         A_vals_);
      } else if constexpr (std::is_same_v<T, double>) {
        status_ = aoclsparse_create_dcsr(&A_aocl_,
                                         base_,
                                         m_aocl_,
                                         n_aocl_,
                                         nnz_aocl_,
                                         A_rows_,
                                         A_cols_,
                                         A_vals_);
      } else {
        // Un-specialised class will not do any work - print error and exit.
        std::cerr << "ERROR - Datatype for AOCL CPU SPGEMV kernel not supported."
                  << std::endl;
        exit(1);
      }
    }

private:
    void preLoopRequirements() override {
      status_ = aoclsparse_set_mv_hint(A_aocl_,
                                       operation_,
                                       A_description_,
                                       iterations_);
      status_ = aoclsparse_optimize(A_aocl_);
    }

    void callSpgemv() override {
      if constexpr (std::is_same_v<T, float>) {
        aoclsparse_smv(operation_,
                       &alpha,
                       A_aocl_,
                       A_description_,
                       x_,
                       &beta,
                       y_);
      } else if constexpr (std::is_same_v<T, double>) {
        aoclsparse_dmv(operation_,
                       &alpha,
                       A_aocl_,
                       A_description_,
                       x_,
                       &beta,
                       y_);
      } else {
        // Un-specialised class will not do any work - print error and exit.
        std::cerr << "ERROR - Datatype for AOCL CPU SPGEMV kernel not "
                     "supported." << std::endl;
        exit(1);
      }
      callConsume();
    }

    void postLoopRequirements() override {
    }

    void postCallKernelCleanup() override {
      status_ = aoclsparse_destroy_mat_descr(A_description_);
      status_ = aoclsparse_destroy(&A_aocl_);
      free(A_rows_);
      free(A_cols_);
      free(A_vals_);
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
