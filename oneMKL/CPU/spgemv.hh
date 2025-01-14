#pragma once

#ifdef CPU_ONEMKL
#include <mkl.h>

#include <algorithm>

#include "../../include/kernels/CPU/spgemv.hh"
#include "../../include/utilities.hh"

namespace cpu {
template <typename T>
class spgemv_cpu : public spgemv<T> {
public:
    using spgemv<T>::spgemv;
    using spgemv<T>::callConsume;
    using spgemv<T>::initInputMatrices;
    using spgemv<T>::m_;
    using spgemv<T>::n_;
    using spgemv<T>::A_;
    using spgemv<T>::x_;
    using spgemv<T>::y_;
    using spgemv<T>::sparsity_;
    using spgemv<T>::nnz_;

    void initialise(int m, int n, double sparsity) {
      m_ = m;
      n_ = n;
      sparsity_ = sparsity;

      nnz_ = 1 + (uint64_t)((double)m_ * (double)n_ * (1.0 - sparsity_));

      A_ = (T*)mkl_malloc(sizeof(T) * m_ * n_, 64);
      x_ = (T*)mkl_malloc(sizeof(T) * n_, 64);
      y_ = (T*)mkl_malloc(sizeof(T) * m_, 64);

      initInputMatrices();
    }

protected:
    void toSparseFormat() override {
      A_vals_ = new T[nnz_];
      A_cols_ = new MKL_INT[nnz_];
      A_rowsb_ = new MKL_INT[m_ + 1];
      A_rowse_ = new MKL_INT[m_ + 1];

      int nnz_encountered = 0;

      A_rowsb_[0] = 0;
      A_rowse_[0] = 0;

      for (int row = 0; row < m_; row++) {
        A_rowsb_[row + 1] = nnz_encountered;
        for (int col = 0; col < n_; col++) {
          if (A_[(row * n_) + col] != 0.0) {
            A_cols_[nnz_encountered] = col;
            A_vals_[nnz_encountered] = static_cast<T>(A_[(row * n_) + col]);
            nnz_encountered++;
          }
        }
        A_rowse_[row + 1] = nnz_encountered;
      }
    }

private:

    void callGemv() override {
      /**
       * sparse_status_t mkl_sparse_s_mv (
       *    const sparse_operation_t operation,
       *    const float alpha,
       *    const sparse_matrix_t A,
       *    const struct matrix_descr descr,
       *    const float *x,
       *    const float beta,
       *    float *y);
       */
      if constexpr (std::is_same_v<T, float>) {
        status_ = mkl_sparse_s_mv(operation_, alpha, A_csr_, description_, x_,
                                  beta, y_);
      } else if constexpr (std::is_same_v<T, float>) {
        status_ = mkl_sparse_s_mv(operation_, alpha, A_csr_, description_, x_,
                                  beta, y_);
      }
      if (status_ != SPARSE_STATUS_SUCCESS) {
        std::cout << "ERROR " << status_ << std::endl;
        exit(1);
      }
      callConsume();
    }

    void preLoopRequirements() override {
      if constexpr (std::is_same_v<T, float>) {
        status_ = mkl_sparse_s_create_csr(&A_csr_,
                                          indexing_,
                                          m_,
                                          n_,
                                          A_rowsb_,
                                          A_rowse_,
                                          A_cols_,
                                          A_vals_);
      } else if constexpr (std::is_same_v<T, double>) {
        status_ = mkl_sparse_d_create_csr(&A_csr_,
                                          indexing_,
                                          m_,
                                          n_,
                                          A_rowsb_,
                                          A_rowse_,
                                          A_cols_,
                                          A_vals_);
      }
      if (status_ != SPARSE_STATUS_SUCCESS) {
        std::cout << "ERROR " << status_ << std::endl;
        exit(1);
      }

    }

    void postLoopRequirements() override {
      status_ = mkl_sparse_destroy(A_csr_);
      if (status_ != SPARSE_STATUS_SUCCESS) {
        std::cout << "ERROR " << status_ << std::endl;
        exit(1);
      }
    }

    void postKernelCleanup() override {
      mkl_free(A_);
      mkl_free(x_);
      mkl_free(y_);
    }

    sparse_status_t status_;

    sparse_index_base_t indexing_ = SPARSE_INDEX_BASE_ZERO;
    sparse_operation_t operation_ = SPARSE_OPERATION_NON_TRANSPOSE;
    sparse_matrix_type_t description_ = SPARSE_MATRIX_TYPE_GENERAL;

    MKL_INT m_mkl_;
    MKL_INT n_mkl_;

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
