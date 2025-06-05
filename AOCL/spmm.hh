#pragma once

#ifdef CPU_AOCL
#include "aoclsparse.h"

#include <algorithm>

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
    using spmm<T>::A_;
    using spmm<T>::B_;
    using spmm<T>::C_;
    using spmm<T>::sparsity_;
    using spmm<T>::nnzA_;
    using spmm<T>::nnzB_;
    using spmm<T>::iterations_;

    void initialise(int m, int n, int k, double sparsity,
                    bool binary = false) {
      base_ = aoclsparse_index_base_zero;
      operationA_ = aoclsparse_operation_none;
      operationB_ = aoclsparse_operation_none;

      m_aocl_ = m_ = m;
      n_aocl_ = n_ = n;
      k_aocl_ = k_ = k;
      sparsity_ = sparsity;

      nnzA_ = 1 + (uint64_t)((double)m_ * (double)n_ * (1.0 - sparsity_));
      nnzA_aocl_ = nnzA_;
      nnzB_ = 1 + (uint64_t)((double)m_ * (double)n_ * (1.0 - sparsity_));
      nnzB_aocl_ = nnzB_;

      A_rows_ = (aoclsparse_int*)malloc(sizeof(aoclsparse_int) * (m_ + 1));
      A_cols_ = (aoclsparse_int*)malloc(sizeof(aoclsparse_int) * nnzA_);
      A_vals_ = (T*)malloc(sizeof(T) * nnzA_);

      B_rows_ = (aoclsparse_int*)malloc(sizeof(aoclsparse_int) * (k_ + 1));
      B_cols_ = (aoclsparse_int*)malloc(sizeof(aoclsparse_int) * nnzB_);
      B_vals_ = (T*)malloc(sizeof(T) * nnzB_);

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
                                         nnzA_aocl_,
                                         A_rows_,
                                         A_cols_,
                                         A_vals_);
      } else if constexpr (std::is_same_v<T, double>) {
        status_ = aoclsparse_create_dcsr(&A_aocl_,
                                         base_,
                                         m_aocl_,
                                         k_aocl_,
                                         nnzA_aocl_,
                                         A_rows_,
                                         A_cols_,
                                         A_vals_);
      } else {
      // Un-specialised class will not do any work - print error and exit.
      std::cout << "ERROR - Datatype for AOCL CPU SPMM kernel not supported."
                << std::endl;
      exit(1);
      }



      nnz_encountered = 0;

      B_rows_[0] = 0;

      for (int row = 0; row < k_; row++) {
        B_rows_[row + 1] = nnz_encountered;
        for (int col = 0; col < n_; col++) {
          if (B_[(row * n_) + col] != 0.0) {
            B_cols_[nnz_encountered] = col;
            B_vals_[nnz_encountered] = static_cast<T>(B_[(row * n_) + col]);
            nnz_encountered++;
          }
        }
      }

      status_ = aoclsparse_create_mat_descr(&B_description_);

      if constexpr (std::is_same_v<T, float>) {
        status_ = aoclsparse_create_scsr(&B_aocl_,
                                         base_,
                                         k_aocl_,
                                         n_aocl_,
                                         nnzB_aocl_,
                                         B_rows_,
                                         B_cols_,
                                         B_vals_);
      } else if constexpr (std::is_same_v<T, double>) {
        status_ = aoclsparse_create_dcsr(&B_aocl_,
                                         base_,
                                         k_aocl_,
                                         n_aocl_,
                                         nnzB_aocl_,
                                         B_rows_,
                                         B_cols_,
                                         B_vals_);
      } else {
      // Un-specialised class will not do any work - print error and exit.
      std::cout << "ERROR - Datatype for AOCL CPU SPMM kernel not supported."
                << std::endl;
      exit(1);
      }


      status_ = aoclsparse_set_mat_index_base(A_description_, base_);
      status_ = aoclsparse_set_mat_index_base(B_description_, base_);
    }

private:
    void preLoopRequirements() override {


    }

    void callSpmm() override {
      /**
       * STEP 1 -- count NNZ values for C
       */
      request_ = aoclsparse_stage_nnz_count;
      if constexpr (std::is_same_v<T, float>) {
        aoclsparse_scsr2m(operationA_,
                          A_description_,
                          A_aocl_,
                          operationB_,
                          B_description_,
                          B_aocl_,
                          request_,
                          &C_aocl_);
      } else if constexpr (std::is_same_v<T, double>) {
        aoclsparse_dcsr2m(operationA_,
                          A_description_,
                          A_aocl_,
                          operationB_,
                          B_description_,
                          B_aocl_,
                          request_,
                          &C_aocl_);
      } else {
        // Un-specialised class will not do any work - print error and exit.
        std::cout << "ERROR - Datatype for AOCL CPU SPGEMV kernel not "
                     "supported." << std::endl;
        exit(1);
      }

      /**
       * Move values into CSR arrays
       */
      if constexpr (std::is_same_v<T, float>) {
        aoclsparse_export_scsr(C_aocl_,
                               &base_,
                               &m_aocl_,
                               &n_aocl_,
                               &nnzC_aocl_,
                               &C_cols_,
                               &C_rows_,
                               &C_vals_);
      } else if constexpr (std::is_same_v<T, double>) {
        aoclsparse_export_dcsr(C_aocl_,
                               &base_,
                               &m_aocl_,
                               &n_aocl_,
                               &nnzC_aocl_,
                               &C_cols_,
                               &C_rows_,
                               &C_vals_);
      }

      /**
       * Step 2 -- finalise the values in C
       */
      request_ = aoclsparse_stage_finalize;
      if constexpr (std::is_same_v<T, float>) {
        aoclsparse_scsr2m(operationA_,
                          A_description_,
                          A_aocl_,
                          operationB_,
                          B_description_,
                          B_aocl_,
                          request_,
                          &C_aocl_);
      } else if constexpr (std::is_same_v<T, double>) {
        aoclsparse_dcsr2m(operationA_,
                          A_description_,
                          A_aocl_,
                          operationB_,
                          B_description_,
                          B_aocl_,
                          request_,
                          &C_aocl_);
      } else {
        // Un-specialised class will not do any work - print error and exit.
        std::cout << "ERROR - Datatype for AOCL CPU SPGEMV kernel not "
                     "supported." << std::endl;
        exit(1);
      }

      /**
       * Move values into CSR arrays
       */
      if constexpr (std::is_same_v<T, float>) {
        aoclsparse_export_scsr(C_aocl_,
                               &base_,
                               &m_aocl_,
                               &n_aocl_,
                               &nnzC_aocl_,
                               &C_cols_,
                               &C_rows_,
                               &C_vals_);
      } else if constexpr (std::is_same_v<T, double>) {
        aoclsparse_export_dcsr(C_aocl_,
                               &base_,
                               &m_aocl_,
                               &n_aocl_,
                               &nnzC_aocl_,
                               &C_cols_,
                               &C_rows_,
                               &C_vals_);
      }



      callConsume();
    }

    void postLoopRequirements() override {
    }

    void postCallKernelCleanup() override {
      status_ = aoclsparse_destroy_mat_descr(A_description_);
      status_ = aoclsparse_destroy_mat_descr(B_description_);
      status_ = aoclsparse_destroy(&A_aocl_);
      status_ = aoclsparse_destroy(&B_aocl_);
      status_ = aoclsparse_destroy(&C_aocl_);
      free(A_rows_);
      free(A_cols_);
      free(A_vals_);
      free(B_rows_);
      free(B_cols_);
      free(B_vals_);
      free(C_rows_);
      free(C_cols_);
      free(C_vals_);
    }

    aoclsparse_status status_;

    aoclsparse_operation operationA_;
    aoclsparse_operation operationB_;
    aoclsparse_index_base base_;
    aoclsparse_request request_;

    aoclsparse_matrix A_aocl_;
    aoclsparse_int* A_rows_;
    aoclsparse_int* A_cols_;
    T* A_vals_;

    aoclsparse_matrix B_aocl_;
    aoclsparse_int* B_rows_;
    aoclsparse_int* B_cols_;
    T* B_vals_;

    aoclsparse_matrix C_aocl_;
    aoclsparse_int* C_rows_;
    aoclsparse_int* C_cols_;
    T* C_vals_;
    aoclsparse_int C_M;
    aoclsparse_int C_N;

    aoclsparse_int m_aocl_;
    aoclsparse_int n_aocl_;
    aoclsparse_int k_aocl_;
    aoclsparse_int nnzA_aocl_;
    aoclsparse_int nnzB_aocl_;
    aoclsparse_int nnzC_aocl_;

    aoclsparse_mat_descr A_description_;
    aoclsparse_mat_descr B_description_;
    aoclsparse_mat_descr C_description_;

    const T alpha = ALPHA;
    const T beta = BETA;
};
}


#endif
