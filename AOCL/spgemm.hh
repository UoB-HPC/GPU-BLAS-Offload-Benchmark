#pragma once

#ifdef CPU_AOCL
#include "aoclsparse.h"

#include <algorithm>

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
    using spgemm<T>::A_;
    using spgemm<T>::B_;
    using spgemm<T>::C_;
    using spgemm<T>::sparsity_;
    using spgemm<T>::nnz_;
    using spgemm<T>::iterations_;

    void initialise(int m, int n, int k, double sparsity,
                    bool binary = false) {
      base_ = aoclsparse_index_base_zero;
      operation_ = aoclsparse_operation_none;
      order_ = aoclsparse_order_row;

      m_aocl_ = m_ = m;
      n_aocl_ = n_ = n;
      k_aocl_ = k_ = k;
      sparsity_ = sparsity;

      nnz_ = 1 + (uint64_t)((double)m_ * (double)n_ * (1.0 - sparsity_));
      nnz_aocl_ = nnz_;

      A_rows_ = (aoclsparse_int*)malloc(sizeof(aoclsparse_int) * (m_ + 1));
      A_cols_ = (aoclsparse_int*)malloc(sizeof(aoclsparse_int) * nnz_);
      A_vals_ = (T*)malloc(sizeof(T) * nnz_);

      initInputMatrices();
    }

protected:
    void toSparseFormat() override {
      int nnz_encountered = 0;

      A_rows_[0] = 0;

      for (int row = 0; row < m_; row++) {
        A_rows_[row + 1] = nnz_encountered;
        for (int col = 0; col < k_; col++) {
          if (A_[(row * k_) + col] != 0.0) {
            A_cols_[nnz_encountered] = col;
            A_vals_[nnz_encountered] = static_cast<T>(A_[(row * k_) + col]);
            nnz_encountered++;
          }
        }
      }
      status_ = aoclsparse_create_mat_descr(&A_description_);

      if constexpr (std::is_same_v<T, float>) {
        status_ = aoclsparse_create_scsr(&A_aocl_,
                                         base_,
                                         m_aocl_,
                                         k_aocl_,
                                         nnz_aocl_,
                                         A_rows_,
                                         A_cols_,
                                         A_vals_);
      } else if constexpr (std::is_same_v<T, double>) {
        status_ = aoclsparse_create_dcsr(&A_aocl_,
                                         base_,
                                         m_aocl_,
                                         k_aocl_,
                                         nnz_aocl_,
                                         A_rows_,
                                         A_cols_,
                                         A_vals_);
      } else {
      // Un-specialised class will not do any work - print error and exit.
      std::cout << "ERROR - Datatype for AOCL CPU SPGEMV kernel not supported."
                << std::endl;
      exit(1);
      }
      status_ = aoclsparse_set_mat_index_base(A_description_, base_);
    }

private:
    void preLoopRequirements() override {


    }

    void callSpgemm() override {
      if constexpr (std::is_same_v<T, float>) {
        // AOCL assumes column-major for B and C.  As they are just randomly
        // filled arrays, this doesn't actually matter here.
        aoclsparse_scsrmm(operation_,
                          alpha,
                          A_aocl_,
                          A_description_,
                          order_,
                          B_,
                          n_aocl_,
                          k_aocl_,
                          beta,
                          C_,
                          m_aocl_);
      } else if constexpr (std::is_same_v<T, double>) {
        aoclsparse_dcsrmm(operation_,
                          alpha,
                          A_aocl_,
                          A_description_,
                          order_,
                          B_,
                          n_aocl_,
                          k_aocl_,
                          beta,
                          C_,
                          m_aocl_);
      } else {
        // Un-specialised class will not do any work - print error and exit.
        std::cout << "ERROR - Datatype for AOCL CPU SPGEMV kernel not "
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
    aoclsparse_order order_;

    aoclsparse_operation operation_;
    aoclsparse_index_base base_;

    aoclsparse_matrix A_aocl_;
    aoclsparse_int* A_rows_;
    aoclsparse_int* A_cols_;
    T* A_vals_;
    aoclsparse_int m_aocl_;
    aoclsparse_int n_aocl_;
    aoclsparse_int k_aocl_;
    aoclsparse_int nnz_aocl_;

    aoclsparse_mat_descr A_description_;

    const T alpha = ALPHA;
    const T beta = BETA;
};
}


#endif
