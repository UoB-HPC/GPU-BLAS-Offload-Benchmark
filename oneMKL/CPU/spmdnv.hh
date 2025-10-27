#pragma once

#ifdef CPU_ONEMKL
#include <mkl.h>

#include <algorithm>

#include "../../include/kernels/CPU/spmdnv.hh"
#include "../../include/utilities.hh"

namespace cpu {
template <typename T>
class spmdnv_cpu : public spmdnv<T> {
public:
    using spmdnv<T>::spmdnv;
    using spmdnv<T>::callConsume;
    using spmdnv<T>::initInputMatrixVector;
    using spmdnv<T>::m_;
    using spmdnv<T>::n_;
    using spmdnv<T>::x_;
    using spmdnv<T>::y_;
    using spmdnv<T>::sparsity_;
    using spmdnv<T>::type_;
    using spmdnv<T>::nnz_;

    void initialise(int m, int n, double sparsity, matrixType type, 
                    bool binary = false) {
      m_ = m;
      n_ = n;
      sparsity_ = sparsity;
      type_ = type;

      nnz_ = 1 + (uint64_t)((double)m_ * (double)n_ * (1.0 - sparsity_));

      x_ = (T*)mkl_malloc(sizeof(T) * n_, 64);
      y_ = (T*)mkl_malloc(sizeof(T) * m_, 64);

      initInputMatrixVector();
    }

protected:
    void toSparseFormat() override {
      A_vals_ = (T*)mkl_malloc(sizeof(T) * nnz_, 64);
      A_cols_ = (MKL_INT*)mkl_malloc(sizeof(MKL_INT) * nnz_, 64);
      MKL_INT* A_rows_ = (MKL_INT*)mkl_malloc(sizeof(MKL_INT) * (m_ + 1), 64);
      A_rowsb_ = (MKL_INT*)mkl_malloc(sizeof(MKL_INT) * m_, 64);
      A_rowse_ = (MKL_INT*)mkl_malloc(sizeof(MKL_INT) * m_, 64);

      if (type_ == matrixType::rmat) {
        rMatCSR<T, MKL_INT>(A_vals_, A_cols_, A_rows_, m_, n_, nnz_);
      } else if (type_ == matrixType::random) {
        randomCSR<T, MKL_INT>(A_vals_, A_cols_, A_rows_, m_, n_, nnz_);
      } else if (type_ == matrixType::finiteElements) {
        finiteElementCSR<T, MKL_INT>(A_vals_, A_cols_, A_rows_, m_, n_, nnz_);
      } else {
        std::cerr << "Unknown matrix type" << std::endl;
        exit(1);
      }
      
      for (uint64_t i = 0; i < m_; i++) {
        A_rowsb_[i] = A_rows_[i];
        A_rowse_[i] = A_rows_[i + 1];
      }

      mkl_free(A_rows_);
    }

private:
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

    void callSpMDnV() override {
      if constexpr (std::is_same_v<T, float>) {
        status_ = mkl_sparse_s_mv(operation_, alpha, A_csr_, description_, x_,
                                  beta, y_);
      } else if constexpr (std::is_same_v<T, double>) {
        status_ = mkl_sparse_d_mv(operation_, alpha, A_csr_, description_, x_,
                                  beta, y_);
      }
      if (status_ != SPARSE_STATUS_SUCCESS) {
        std::cerr << "ERROR " << status_ << std::endl;
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
      mkl_free(x_);
      mkl_free(y_);
    }

    sparse_status_t status_;

    sparse_index_base_t indexing_ = SPARSE_INDEX_BASE_ZERO;
    sparse_operation_t operation_ = SPARSE_OPERATION_NON_TRANSPOSE;
    matrix_descr description_ = {SPARSE_MATRIX_TYPE_GENERAL,
                                 SPARSE_FILL_MODE_LOWER,
                                 SPARSE_DIAG_NON_UNIT};

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
