#pragma once

#ifdef CPU_ONEMKL
#include <mkl.h>

#include <algorithm>
#include <iostream>

#include "../../include/kernels/CPU/spgemm.hh"
#include "../../include/utilities.hh"

namespace cpu {
/** A class for sparse matrix-dense matrix BLAS kernels. */
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
    using spgemm<T>::nnz_;

    void initialise(int m, int n, int k, double sparsity,
                    bool binary = false) {

      m_ = m;
      n_ = n;
      k_ = k;

      m_mkl_ = m;
      n_mkl_ = n;
      k_mkl_ = k;

      sparsity_ = sparsity;

      /** Determine the number of nnz elements in A and B */
      nnz_ = 1 + (uint64_t)((double)m_ * (double)k_ * (1.0 - sparsity_));
      B_ = (T*)mkl_malloc(sizeof(T) * k_ * n_, 64);
      C_ = (T*)mkl_malloc(sizeof(T) * m_ * n_, 64);

      initInputMatrices();
    }

protected:
    void toSparseFormat() override {
      A_vals_ = new T[nnz_];
      A_cols_ = new MKL_INT[nnz_];
      A_rowsb_ = new MKL_INT[m_ + 1];
      A_rowse_ = new MKL_INT[m_ + 1];

      rMatCSR<T, MKL_INT>(A_vals_, A_cols_, A_rowsb_, m_, k_, nnz_);

      for (uint64_t i = 0; i < m_; i++) {
        A_rowse_[i] = A_rowsb_[i + 1] - 1;
      }
      A_rowse_[m_] = A_rowsb_[m_ + 1];
    }

private:
    void callSpgemm() override {
      /**
       * Using:
       * sparse_status_t mkl_sparse_s_mm (
       *    const sparse_operation_t operation,
       *    const float alpha,
       *    const sparse_matrix_t A,
       *    const struct matrix_descr descr,
       *    const sparse_layout_t layout,
       *    const float *B,
       *    const MKL_INT columns,
       *    const MKL_INT ldb,
       *    const float beta,
       *    float *C,
       *    const MKL_INT ldc);
       */
      if constexpr (std::is_same_v<T, float>) {
        status_ = mkl_sparse_s_mm(operation_, alpha, A_csr_, description_,
                                  layout_, B_, n_mkl_, k_mkl_, beta, C_,
                                  m_mkl_);
      } else if constexpr (std::is_same_v<T, double>) {
        status_ = mkl_sparse_d_mm(operation_, alpha, A_csr_, description_,
                                  layout_, B_, n_mkl_, k_mkl_, beta, C_,
                                  m_mkl_);
      } else {
        // Un-specialised class will not do any work - print error and exit.
        std::cout << "ERROR - Datatype for OneMKL CPU SpGEMV kernel not "
                     "supported." << std::endl;
        exit(1);
      }

      callConsume();
    }

    void preLoopRequirements() override {
      if constexpr (std::is_same_v<T, float>) {
        status_ = mkl_sparse_s_create_csr(&A_csr_,
                                          indexing_,
                                          m_,
                                          k_,
                                          A_rowsb_,
                                          A_rowse_,
                                          A_cols_,
                                          A_vals_);
        if (status_ != SPARSE_STATUS_SUCCESS) {
          std::cout << "ERROR " << status_ << std::endl;
          exit(1);
        }
      } else if constexpr (std::is_same_v<T, double>) {
        status_ = mkl_sparse_d_create_csr(&A_csr_,
                                          indexing_,
                                          m_,
                                          k_,
                                          A_rowsb_,
                                          A_rowse_,
                                          A_cols_,
                                          A_vals_);
        if (status_ != SPARSE_STATUS_SUCCESS) {
          std::cout << "ERROR " << status_ << std::endl;
          exit(1);
        }
      }
    }

    void postLoopRequirements() override {
      status_ = mkl_sparse_destroy(A_csr_);
      if (status_ != SPARSE_STATUS_SUCCESS) {
        std::cout << "ERROR " << status_ << std::endl;
        exit(1);
      }
    }

    void postCallKernelCleanup() override {
      mkl_free(A_rowsb_);
      mkl_free(A_rowse_);
      mkl_free(A_cols_);
      mkl_free(A_vals_);
      mkl_free(B_);
      mkl_free(C_);
    }

    sparse_status_t status_;

    sparse_index_base_t indexing_ = SPARSE_INDEX_BASE_ZERO;
    sparse_operation_t operation_ = SPARSE_OPERATION_NON_TRANSPOSE;
    // Todo -- investigate if other options for description_ improve performance
    matrix_descr description_ = {SPARSE_MATRIX_TYPE_GENERAL,
                                 SPARSE_FILL_MODE_LOWER,
                                 SPARSE_DIAG_NON_UNIT};
    sparse_layout_t layout_ = SPARSE_LAYOUT_COLUMN_MAJOR;

    MKL_INT m_mkl_;
    MKL_INT n_mkl_;
    MKL_INT k_mkl_;

    T* A_vals_;
    MKL_INT* A_cols_;
    MKL_INT* A_rowsb_;
    MKL_INT* A_rowse_;

    sparse_matrix_t A_csr_;

    const T alpha = ALPHA;
    const T beta = BETA;
};
}


#endif