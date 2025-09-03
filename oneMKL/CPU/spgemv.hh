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
    using spgemv<T>::initInputMatrixVector;
    using spgemv<T>::m_;
    using spgemv<T>::n_;
    using spgemv<T>::x_;
    using spgemv<T>::y_;
    using spgemv<T>::sparsity_;
    using spgemv<T>::type_;
    using spgemv<T>::nnz_;

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
      A_vals_ = new T[nnz_];
      A_cols_ = new MKL_INT[nnz_];
      A_rowsb_ = new MKL_INT[m_ + 1];
      A_rowse_ = new MKL_INT[m_ + 1];

      if (type_ == matrixType::rmat) {
        rMatCSR<T, MKL_INT>(A_vals_, A_cols_, A_rowsb_, m_, n_, nnz_);
      } else if (type_ == matrixType::random) {
        randomCSR<T, MKL_INT>(A_vals_, A_cols_, A_rowsb_, m_, n_, nnz_);
      } else {
        std::cerr << "Unknown matrix type" << std::endl;
        exit(1);
      }
      
      for (uint64_t i = 0; i < m_; i++) {
        A_rowse_[i] = A_rowsb_[i + 1] - 1;
      }
      A_rowse_[m_] = A_rowsb_[m_ + 1];

      if (print_) {
        std::cout << "=============================================" << std::endl;
        std::cout << "==================== CPU ====================" << std::endl;
        std::cout << "=============================================" << std::endl;
        std::cout << "                    INPUT"  << std::endl;
        std::cout << "_____________________________________________" << std::endl;
        std::cout << "x:" << std::endl;
        std::cout << "[";
        for (int i = 0; i < n_; i++) {
          std::cout << x_[i];
          if (i == (n_ - 1)) std::cout << "]" << std::endl;
          else std::cout << ", ";
        }
        std::cout << "A_rowsb_:" << std::endl;
        std::cout << "[";
        for (int i = 0; i < (m_ + 1); i++) {
          std::cout << A_rowsb_[i];
          if (i == (m_)) std::cout << "]" << std::endl;
          else std::cout << ", ";
        }
        std::cout << "A_rowse_:" << std::endl;
        std::cout << "[";
        for (int i = 0; i < (m_ + 1); i++) {
          std::cout << A_rowse_[i];
          if (i == (m_)) std::cout << "]" << std::endl;
          else std::cout << ", ";
        }
        std::cout << "A_cols_:" << std::endl;
        std::cout << "[";
        for (int i = 0; i < (nnz_); i++) {
          std::cout << A_cols_[i];
          if (i == (nnz_ - 1)) std::cout << "]" << std::endl;
          else std::cout << ", ";
        }
        std::cout << "A_vals_:" << std::endl;
        std::cout << "[";
        for (int i = 0; i < (nnz_); i++) {
          std::cout << A_vals_[i];
          if (i == (nnz_ - 1)) std::cout << "]" << std::endl;
          else std::cout << ", ";
        }
        std::cout << "_____________________________________________" << std::endl;
      }
    }

private:

    void callSpgemv() override {
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
      } else if constexpr (std::is_same_v<T, double>) {
        status_ = mkl_sparse_d_mv(operation_, alpha, A_csr_, description_, x_,
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

    void postCallKernelCleanup() override {
      mkl_free(A_rowsb_);
      mkl_free(A_rowse_);
      mkl_free(A_cols_);
      mkl_free(A_vals_);
      mkl_free(x_);
      mkl_free(y_);
    }

    bool print_ = false;

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
