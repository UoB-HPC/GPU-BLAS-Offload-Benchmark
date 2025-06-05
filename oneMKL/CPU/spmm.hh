#pragma once

#ifdef CPU_ONEMKL
#include <mkl.h>
#include <mkl_spblas.h>

#include <algorithm>

#include "../../include/kernels/CPU/spmm.hh"
#include "../../include/utilities.hh"

namespace cpu {
/** A class for sparse matrix-sparse matrix CPU BLAS kernels. */
template <typename T>
class spmm_cpu : public spmm<T> {
public:
    using spmm<T>::spmm;
    using spmm<T>::initInputMatrices;
    using spmm<T>::callConsume;
    using spmm<T>::m_;
    using spmm<T>::n_;
    using spmm<T>::k_;
    using spmm<T>::A_;
    using spmm<T>::B_;
    using spmm<T>::C_;
    using spmm<T>::sparsity_;
    using spmm<T>::nnzA_;
    using spmm<T>::nnzB_;

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
      nnzA_ = 1 + (uint64_t)((double)m_ * (double)k_ * (1.0 - sparsity_));
      nnzB_ = 1 + (uint64_t)((double)k_ * (double)n_ * (1.0 - sparsity_));

      A_ = (T*)mkl_malloc(sizeof(T) * m_ * k_, 64);
      B_ = (T*)mkl_malloc(sizeof(T) * k_ * n_, 64);
      C_ = (T*)mkl_malloc(sizeof(T) * m_ * n_, 64);

      initInputMatrices();
    }

protected:
    void toSparseFormat() override {
      A_vals_ = new T[nnzA_];
      A_cols_ = new MKL_INT[nnzA_];
      A_rowsb_ = new MKL_INT[m_ + 1];
      A_rowse_ = new MKL_INT[m_ + 1];

      int nnz_encountered = 0;

      A_rowsb_[0] = 0;
      A_rowse_[0] = 0;

      for (int row = 0; row < m_; row++) {
        A_rowsb_[row + 1] = nnz_encountered;
        for (int col = 0; col < k_; col++) {
          if (A_[(row * k_) + col] != 0.0) {
            A_cols_[nnz_encountered] = col;
            A_vals_[nnz_encountered] = static_cast<T>(A_[(row * k_) + col]);
            nnz_encountered++;
          }
        }
        A_rowse_[row + 1] = nnz_encountered;
      }


      B_vals_ = new T[nnzB_];
      B_cols_ = new MKL_INT[nnzB_];
      B_rowsb_ = new MKL_INT[k_ + 1];
      B_rowse_ = new MKL_INT[k_ + 1];

      nnz_encountered = 0;

      B_rowsb_[0] = 0;
      B_rowse_[0] = 0;

      for (int row = 0; row < k_; row++) {
        B_rowsb_[row + 1] = nnz_encountered;
        for (int col = 0; col < n_; col++) {
          if (B_[(row * n_) + col] != 0.0) {
            B_cols_[nnz_encountered] = col;
            B_vals_[nnz_encountered] = static_cast<T>(B_[(row * n_) + col]);
            nnz_encountered++;
          }
        }
        B_rowse_[row + 1] = nnz_encountered;
      }
    }

private:
    void callSpmm() override {
      /**
       * sparse_status_t mkl_sparse_spmm (
       *  const sparse_operation_t operation,
       *  const sparse_matrix_t A,
       *  const sparse_matrix_t B,
       *  sparse_matrix_t *C);
       */
       status_ = mkl_sparse_spmm(operation_, A_csr_, B_csr_, &C_csr_);
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
                                          k_,
                                          A_rowsb_,
                                          A_rowse_,
                                          A_cols_,
                                          A_vals_);
        if (status_ != SPARSE_STATUS_SUCCESS) {
          std::cout << "ERROR " << status_ << std::endl;
          exit(1);
        }

        status_ = mkl_sparse_s_create_csr(&B_csr_,
                                          indexing_,
                                          k_,
                                          n_,
                                          B_rowsb_,
                                          B_rowse_,
                                          B_cols_,
                                          B_vals_);
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


        status_ = mkl_sparse_d_create_csr(&B_csr_,
                                          indexing_,
                                          k_,
                                          n_,
                                          B_rowsb_,
                                          B_rowse_,
                                          B_cols_,
                                          B_vals_);
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
      status_ = mkl_sparse_destroy(B_csr_);
      if (status_ != SPARSE_STATUS_SUCCESS) {
        std::cout << "ERROR " << status_ << std::endl;
        exit(1);
      }
      status_ = mkl_sparse_destroy(C_csr_);
      if (status_ != SPARSE_STATUS_SUCCESS) {
        std::cout << "ERROR " << status_ << std::endl;
        exit(1);
      }
    }
    void postCallKernelCleanup() override {
      mkl_free(A_);
      mkl_free(B_);
      mkl_free(C_);
    }

    sparse_status_t status_;

    sparse_index_base_t indexing_ = SPARSE_INDEX_BASE_ZERO;
    sparse_operation_t operation_ = SPARSE_OPERATION_NON_TRANSPOSE;

    MKL_INT m_mkl_;
    MKL_INT n_mkl_;
    MKL_INT k_mkl_;

    T* A_vals_;
    MKL_INT* A_cols_;
    MKL_INT* A_rowsb_;
    MKL_INT* A_rowse_;

    T* B_vals_;
    MKL_INT* B_cols_;
    MKL_INT* B_rowsb_;
    MKL_INT* B_rowse_;

    T* C_vals_;
    MKL_INT* C_cols_;
    MKL_INT* C_rowsb_;
    MKL_INT* C_rowse_;

    sparse_matrix_t A_csr_;
    sparse_matrix_t B_csr_;
    sparse_matrix_t C_csr_;


    const T alpha = ALPHA;
    const T beta = BETA;
};
}


#endif