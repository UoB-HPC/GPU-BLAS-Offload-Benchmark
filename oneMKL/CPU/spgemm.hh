#pragma once

#ifdef CPU_ONEMKL
#include <mkl.h>
#include <mkl_spblas.h>

#include <algorithm>

#include "../../include/kernels/CPU/spgemm.hh"
#include "../../include/utilities.hh"

namespace cpu {
/** A class for sparse matrix-sparse matrix CPU BLAS kernels. */
template <typename T>
class spgemm_cpu : public spgemm<T> {
public:
    using spgemm<T>::spgemm;
    using spgemm<T>::initInputMatrices;
    using spgemm<T>::callConsume;
    using spgemm<T>::m_;
    using spgemm<T>::n_;
    using spgemm<T>::k_;
    using spgemm<T>::sparsity_;
    using spgemm<T>::type_;
    using spgemm<T>::A_nnz_;
    using spgemm<T>::B_nnz_;
    using spgemm<T>::C_nnz_;
    using spgemm<T>::C_vals_;

    void initialise(int m, int n, int k, double sparsity,
                    matrixType type, bool binary = false) {
      m_ = m;
      n_ = n;
      k_ = k;

      m_mkl_ = m;
      n_mkl_ = n;
      k_mkl_ = k;

      sparsity_ = sparsity;
      type_ = type;

      /** Determine the number of nnz elements in A and B */
      A_nnz_ = 1 + (uint64_t)((double)m_ * (double)k_ * (1.0 - sparsity_));
      B_nnz_ = 1 + (uint64_t)((double)k_ * (double)n_ * (1.0 - sparsity_));
      initInputMatrices();
    }

protected:
    void toSparseFormat() override {
      A_vals_ = (T*)mkl_malloc(sizeof(T) * A_nnz_, 64);
      A_cols_ = (MKL_INT*)mkl_malloc(sizeof(MKL_INT) * A_nnz_, 64);
      MKL_INT* A_rows = (MKL_INT*)mkl_malloc(sizeof(MKL_INT) * (m_ + 1), 64);
      A_rowsb_ = (MKL_INT*)mkl_malloc(sizeof(MKL_INT) * m_, 64);
      A_rowse_ = (MKL_INT*)mkl_malloc(sizeof(MKL_INT) * m_, 64);
      B_vals_ = (T*)mkl_malloc(sizeof(T) * B_nnz_, 64);
      B_cols_ = (MKL_INT*)mkl_malloc(sizeof(MKL_INT) * B_nnz_, 64);
      MKL_INT* B_rows = (MKL_INT*)mkl_malloc(sizeof(MKL_INT) * (k_ + 1), 64);
      B_rowsb_ = (MKL_INT*)mkl_malloc(sizeof(MKL_INT) * k_, 64);
      B_rowse_ = (MKL_INT*)mkl_malloc(sizeof(MKL_INT) * k_, 64);

      int seedOffset = 0;
      do {
        if (type_ == matrixType::rmat) {
          rMatCSR<T, MKL_INT>(A_vals_, A_cols_, A_rows, m_, k_, A_nnz_, SEED + seedOffset++);
          rMatCSR<T, MKL_INT>(B_vals_, B_cols_, B_rows, k_, n_, B_nnz_, SEED + seedOffset++);
        } else if (type_ == matrixType::random) {
          randomCSR<T, MKL_INT>(A_vals_, A_cols_, A_rows, m_, k_, A_nnz_, SEED + seedOffset++);
          randomCSR<T, MKL_INT>(B_vals_, B_cols_, B_rows, k_, n_, B_nnz_, SEED + seedOffset++);
        } else if (type_ == matrixType::finiteElements) {
          finiteElementCSR<T, MKL_INT>(A_vals_, A_cols_, A_rows, m_, k_, A_nnz_, SEED + seedOffset++);
          finiteElementCSR<T, MKL_INT>(B_vals_, B_cols_, B_rows, k_, n_, B_nnz_, SEED + seedOffset++);
        } else {
          std::cerr << "Unknown matrix type" << std::endl;
          exit(1);
        }
      } while (calcCNNZ<MKL_INT>(m_, A_nnz_, A_rows, A_cols_, k_, B_nnz_, B_rows, B_cols_) == 0);

      for (uint64_t i = 0; i < m_; i++) {
        A_rowsb_[i] = A_rows[i];
        A_rowse_[i] = A_rows[i + 1];
      }

      mkl_free(A_rows);

      for (uint64_t i = 0; i < k_; i++) {
        B_rowsb_[i] = B_rows[i];
        B_rowse_[i] = B_rows[i + 1];
      }
      mkl_free(B_rows);
    }

private:

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
          std::cerr << "ERROR " << status_ << std::endl;
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
          std::cerr << "ERROR " << status_ << std::endl;
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
          std::cerr << "ERROR " << status_ << std::endl;
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
          std::cerr << "ERROR " << status_ << std::endl;
          exit(1);
        }
      }
    }

    void callSpgemm() override {
      status_ = mkl_sparse_spmm(operation_, A_csr_, B_csr_, &C_csr_);
      if (status_ != SPARSE_STATUS_SUCCESS) {
        std::cerr << "ERROR " << status_ << std::endl;
        exit(1);
      }

      status_ = mkl_sparse_order(C_csr_);
      if (status_ != SPARSE_STATUS_SUCCESS) {
        std::cerr << "ERROR " << status_ << std::endl;
        exit(1);
      }
      callConsume();
    }

    void postLoopRequirements() override {
      if constexpr(std::is_same_v<T, float>) {
        status_ = mkl_sparse_s_export_csr(C_csr_,
                                          &indexing_,
                                          &m_mkl_,
                                          &n_mkl_,
                                          &C_rowsb_,
                                          &C_rowse_,
                                          &C_cols_,
                                          &C_vals_);
      } else if constexpr (std::is_same_v<T, double>) {
        status_ = mkl_sparse_d_export_csr(C_csr_,
                                          &indexing_,
                                          &m_mkl_,
                                          &n_mkl_,
                                          &C_rowsb_,
                                          &C_rowse_,
                                          &C_cols_,
                                          &C_vals_);
      }
      if (status_ != SPARSE_STATUS_SUCCESS) {
        std::cerr << "ERROR " << status_ << std::endl;
        exit(1);
      }

      C_nnz_ = C_rowse_[m_ - 1];
    }

    void postCallKernelCleanup() override {
      status_ = mkl_sparse_destroy(A_csr_);
      if (status_ != SPARSE_STATUS_SUCCESS) {
        std::cerr << "ERROR " << status_ << std::endl;
        exit(1);
      }
      status_ = mkl_sparse_destroy(B_csr_);
      if (status_ != SPARSE_STATUS_SUCCESS) {
        std::cerr << "ERROR " << status_ << std::endl;
        exit(1);
      }
      status_ = mkl_sparse_destroy(C_csr_);
      if (status_ != SPARSE_STATUS_SUCCESS) {
        std::cerr << "ERROR " << status_ << std::endl;
        exit(1);
      }

      mkl_free(A_vals_);
      mkl_free(A_cols_);
      mkl_free(A_rowsb_);
      mkl_free(A_rowse_);

      mkl_free(B_vals_);
      mkl_free(B_cols_);
      mkl_free(B_rowsb_);
      mkl_free(B_rowse_);
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