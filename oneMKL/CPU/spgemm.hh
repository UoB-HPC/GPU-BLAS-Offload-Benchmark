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
    using spgemm<T>::type_;
    using spgemm<T>::nnz_;

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
      nnz_ = 1 + (uint64_t)((double)m_ * (double)k_ * (1.0 - sparsity_));
      B_ = (T*)mkl_malloc(sizeof(T) * k_ * n_, 64);
      C_ = (T*)mkl_malloc(sizeof(T) * m_ * n_, 64);

      initInputMatrices();
    }

protected:
    void toSparseFormat() override {
      A_vals_ = (T*)mkl_malloc(sizeof(T) * nnz_, 64);
      A_cols_ = (MKL_INT*)mkl_malloc(sizeof(MKL_INT) * nnz_, 64);
      // Make a temporary rows array of the ususal CSR type, to then turn into the two-array MKL version
      MKL_INT* A_rows_ = (MKL_INT*)mkl_malloc(sizeof(MKL_INT) * (m_ + 1), 64);
      A_rowsb_ = (MKL_INT*)mkl_malloc(sizeof(MKL_INT) * m_, 64);
      A_rowse_ = (MKL_INT*)mkl_malloc(sizeof(MKL_INT) * m_, 64);

      if (type_ == matrixType::rmat) {
        rMatCSR<T, MKL_INT>(A_vals_, A_cols_, A_rows_, m_, k_, nnz_);
      } else if (type_ == matrixType::random) {
        randomCSR<T, MKL_INT>(A_vals_, A_cols_, A_rows_, m_, k_, nnz_);
      } else {
        std::cerr << "Unknown matrix type" << std::endl;
        exit(1);
      }

      for (uint64_t i = 0; i < m_; i++) {
        A_rowsb_[i] = A_rows_[i];
        A_rowse_[i] = A_rows_[i + 1];
      }
      // Clean up the temporary array
      mkl_free(A_rows_);
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
    
    void callSpgemm() override {
      if constexpr (std::is_same_v<T, float>) {
        status_ = mkl_sparse_s_mm(operation_, alpha, A_csr_, description_,
                                  layout_, B_, n_mkl_, n_mkl_, beta, C_,
                                  n_mkl_);
      } else if constexpr (std::is_same_v<T, double>) {
        status_ = mkl_sparse_d_mm(operation_, alpha, A_csr_, description_,
                                  layout_, B_, n_mkl_, n_mkl_, beta, C_,
                                  n_mkl_);
      } else {
        // Un-specialised class will not do any work - print error and exit.
        std::cerr << "ERROR - Datatype for OneMKL CPU SpGEMV kernel not "
                     "supported." << std::endl;
        exit(1);
      }

      callConsume();
    }

    void postLoopRequirements() override {
      status_ = mkl_sparse_destroy(A_csr_);
      if (status_ != SPARSE_STATUS_SUCCESS) {
        std::cerr << "ERROR " << status_ << std::endl;
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


    void printInputMatrices() {
      std::cout << "Matrix A (CSR format):" << std::endl;
      std::cout << "RowsB: [";
      for (int i = 0; i < m_; i++) {
        std::cout << A_rowsb_[i];
        if (i < (m_ - 1)) std::cout << ", ";
      }
      std::cout << "]" << std::endl;
      std::cout << "RowsE: [";
      for (int i = 0; i < m_; i++) {
        std::cout << A_rowse_[i];
        if (i < (m_ - 1)) std::cout << ", ";
      }
      std::cout << "]" << std::endl;
      std::cout << "Cols: [";
      for (int i = 0; i < nnz_; i++) {
        std::cout << A_cols_[i];
        if (i < nnz_ - 1) std::cout << ", ";
      }
      std::cout << "]" << std::endl;
      std::cout << "Vals: [";
      for (int i = 0; i < nnz_; i++) {
        std::cout << A_vals_[i];
        if (i < nnz_ - 1) std::cout << ", ";
      }
      std::cout << "]" << std::endl;
      std::cout << "B: [";
      for (int i = 0; i < k_ * n_; i++) {
        std::cout << B_[i];
        if (i == (m_ * n_) - 1) std::cout << "]" << std::endl;
        else if (i % n_ == n_ - 1) std::cout << std::endl;
        else if (i < k_ * n_ - 1) std::cout << ", ";
      }
    }

    void printOutputMatrix() {
      std::cout << "Matrix C: [";
      for (int i = 0; i < m_ * n_; i++) {
        std::cout << C_[i];
        if (i == (m_ * n_) - 1) std::cout << "]" << std::endl;
        else if (i % n_ == n_ - 1) std::cout << std::endl;
        else if (i < m_ * n_ - 1) std::cout << ", ";
      }
    }


    sparse_status_t status_;

    sparse_index_base_t indexing_ = SPARSE_INDEX_BASE_ZERO;
    sparse_operation_t operation_ = SPARSE_OPERATION_NON_TRANSPOSE;
    // Todo -- investigate if other options for description_ improve performance
    matrix_descr description_ = {SPARSE_MATRIX_TYPE_GENERAL,
                                 SPARSE_FILL_MODE_LOWER,
                                 SPARSE_DIAG_NON_UNIT};
    sparse_layout_t layout_ = SPARSE_LAYOUT_ROW_MAJOR;

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