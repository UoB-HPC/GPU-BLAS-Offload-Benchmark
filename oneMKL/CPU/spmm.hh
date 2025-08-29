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
    using spmm<T>::sparsity_;
    using spmm<T>::A_nnz_;
    using spmm<T>::B_nnz_;
    using spmm<T>::C_nnz_;
    using spmm<T>::C_rows_;
    using spmm<T>::C_cols_;
    using spmm<T>::C_vals_;

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
      A_nnz_ = 1 + (uint64_t)((double)m_ * (double)k_ * (1.0 - sparsity_));
      B_nnz_ = 1 + (uint64_t)((double)k_ * (double)n_ * (1.0 - sparsity_));
      initInputMatrices();
    }

protected:
    void toSparseFormat() override {
      A_vals_ = new T[A_nnz_];
      A_cols_ = new MKL_INT[A_nnz_];
      A_rowsb_ = new MKL_INT[m_ + 1];
      A_rowse_ = new MKL_INT[m_ + 1];

      rMatCSR<T, MKL_INT>(A_vals_, A_cols_, A_rowsb_, m_, k_, nnz_);

      for (uint64_t i = 0; i < m_; i++) {
        A_rowse_[i] = A_rowsb_[i + 1] - 1;
      }
      A_rowse_[m_] = A_rowsb_[m_ + 1];


      B_vals_ = new T[B_nnz_];
      B_cols_ = new MKL_INT[B_nnz_];
      B_rowsb_ = new MKL_INT[k_ + 1];
      B_rowse_ = new MKL_INT[k_ + 1];


      rMatCSR<T, MKL_INT>(B_vals_, B_cols_, B_rowsb_, k_, n_, nnz_);

      for (uint64_t i = 0; i < k_; i++) {
        B_rowse_[i] = B_rowsb_[i + 1] - 1;
      }
      B_rowse_[k_] = B_rowsb_[k_ + 1];
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
      delete[] A_vals_;
      delete[] A_cols_;
      delete[] A_rowsb_;
      delete[] A_rowse_;

      delete[] B_vals_;
      delete[] B_cols_;
      delete[] B_rowsb_;
      delete[] B_rowse_;

      delete[] C_cols_;
      delete[] C_rowsb_;
      delete[] C_rowse_;
      delete[] C_vals_;
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