#pragma once

#ifdef CPU_NVPL
#include <nvpl_blas_cblas.h>

#include "../include/kernels/CPU/spgemv.hh"
#include "../include/utilities.hh"

namespace cpu {
/** A class for GEMV CPU BLAS kernels. */
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
  using spgemv<T>::iterations_;

  void initialise (int m, int n, double sparsity, matrixType type, 
                   bool binary = false) {
    m_ = m;
    n_ = n;
    sparsity_ = sparsity;
    type_ = type;

    nnz_ = 1 + (uint64_t)((double)m_ * (double)n_ * (1.0 - sparsity_));

    if constexpr (std::is_same_v<T, float>) {
      dataType_ = NVPL_SPARSE_R_32F;
    } else if constexpr (Std::is_same_v<T, double>) {
      dataType_ = NVPL_SPARSE_R_64F;
    } else {
      throw std::runtime_error("Only float and double are supported for NVPL.");
    }

    x_ = (T*)calloc(n_, sizeof(T));
    y_ = (T*)calloc(m_, sizeof(T));
    z_ = (T*)calloc(m_, sizeof(T));

    initInputMatrixVector();
  }

protected:
  void toSparseFormat() override {
    A_vals_ = (T*)calloc(nnz_, sizeof(T));
    A_cols_ = (int64_t*)calloc(nnz_, sizeof(int64_t));
    A_rows_ = (int64_t*)calloc(m_ + 1, sizeof(int64_t));

    // Fill the CSR arrays
    if (type_ == matrixType::rmat) {
      rMatCSR<T, int64_t>(A_vals_, A_cols_, A_rows_, m_, n_, nnz_);
    } else if (type_ == matrixType::random) {
      randomCSR<T, int64_t>(A_vals_, A_cols_, A_rows_, m_, n_, nnz_);
    } else {
      std::cerr << "Matrix type not supported" << std::endl;
      exit(1);
    }

    // Make the NVPL descriptors
    status_ = nvpl_sparse_create_const_csr(&A_descr_,
                                           m_,
                                           n_,
                                           nnz_,
                                           A_rows_,
                                           A_cols_,
                                           A_vals_,
                                           indexType_,
                                           indexType_,
                                           base_,
                                           dataType_);
    if (status_ != NVPL_SPARSE_STATUS_SUCCESS) {
      std::cerr << "nvpl_sparse_create_csr failed with error: " << status_ << std::endl;
      exit(1);
    }

    status_ = nvpl_sparse_create_const_dn_vec(X_descr_,
                                              n_,
                                              x_,
                                              dataType_);
    if (status_ != NVPL_SPARSE_STATUS_SUCCESS) {
      std::cerr << "nvpl_sparse_create_const_dn_vec failed with error: " << status_ << std::endl;
      exit(1);
    }

    status_ = nvpl_sparse_create_dn_vec(Y_descr_,
                                        m_,
                                        y_,
                                        dataType_);
    if (status_ != NVPL_SPARSE_STATUS_SUCCESS) {
      std::cerr << "nvpl_sparse_create_dn_vec failed with error: " << status_ << std::endl;
      exit(1);
    }
    status_ = nvpl_sparse_create_dn_vec(Z_descr_,
                                        m_,
                                        z_,
                                        dataType_);
    if (status_ != NVPL_SPARSE_STATUS_SUCCESS) {
      std::cerr << "nvpl_sparse_create_dn_vec failed with error: " << status_ << std::endl;
      exit(1);
    }
  }

private:
  void preLoopRequirements() override {}

  void callSpgemv() override {
    size_t bufferSize;
    status_ = nvpl_sparse_spmv_buffer_size(handle_,
                                           operation_,
                                           &alpha,
                                           A_descr_,
                                           X_descr_,
                                           &beta,
                                           Z_descr_,
                                           Y_descr_,
                                           dataType_,
                                           algorithm_,
                                           description_,
                                           &bufferSize);
    if (status_ != NVPL_SPARSE_STATUS_SUCCESS) {
      std::cer << "nvpl_sparse_spmv_buffer_size failed with error: " << status_ << std::endl;
      exit(1);
    }

    void* externalBuffer = malloc(bufferSize);
    status_ = nvpl_sparse_spmv_analysis(handle_,
                                        operation_,
                                        &alpha,
                                        A_descr_,
                                        X_descr_,
                                        &beta,
                                        Z_descr_,
                                        Y_descr_,
                                        dataType_,
                                        algorithm_,
                                        description_,
                                        externalBuffer);
    if (status_ != NVPL_SPARSE_STATUS_SUCCESS) {
      std::cerr << "nvpl_sparse_spmv_analysis failed with error: " << status_ << std::endl;
      exit(1);
    }

    status_ = nvpl_sparse_spmv(handle_,
                               operation_,
                               &alpha_,
                               A_descr_,
                               X_descr_,
                               &beta,
                               Z_descr_,
                               Y_descr_,
                               dataType_,
                               algorithm_,
                               description_);
    if (status_ != NVPL_SPARSE_STATUS_SUCCESS) {
      std::cerr << "nvpl_sparse_spmv failed with error: " << status_ << std::endl;
      exit(1);
    }

    free(externalBuffer);
  }

  void postLoopRequirements() override {}

  void postCallKernelCleanup() override {
    free(x_);
    free(y_);
    free(z_);
    free(A_rows_);
    free(A_cols_);
    free(A_vals_);
  }

  nvpl_sparse_status_t status_;
  nvpl_sparse_handle_t handle_;
  nvpl_Sparse_spmv_descr_t description_;

  nvpl_sparse_spmv_alg_t algorithm_ = NVPL_SPARSE_SPMV_CSR_ALG1;
  nvpl_sparse_operation_t operation_ = NVPL_SPARSE_OPERATION_NON_TRANSPOSE;
  nvpl_sparse_data_type_t dataType_;
  nvpl_sparse_index_type_t indexType_ = NVPL_SPARSE_INDEX_64I;
  nvpl_sparse_index_base_t base_ = NVPL_SPARSE_INDEX_BASE_ZERO;

  // Being a bit weird with the naming here.
  // For consistency with the other libraries which don't have a
  // seperate addition vector, I'm keeping Y as the output 
  // vector.  Even though the NVPL documentation uses Y for
  // the addition vector and Z for the output vector
  nvpl_sparse_const_sp_mat_descr_t A_descr_;
  nvpl_sparse_const_dn_vec_descr_t X_descr_;
  nvpl_sparse_dn_vec_descr_t Z_descr_;
  nvpl_sparse_dn_vec_descr_t Y_descr_;

  // Arrays for Matrix A and unused addition vector Z
  int64_t* A_vals_;
  int64_t* A_cols_;
  T* A_vals_;
  T* z_;
  

  const T alpha = ALPHA;
  const T beta = BETA;
};
}  // namespace cpu
#endif