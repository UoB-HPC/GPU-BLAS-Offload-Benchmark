/**
 * ToDo -- This is all currently written for GEMM, but NVPL does not support
 * GEMM, so this needs to be adjusted to spmv -- which is supported
 */





#pragma once

#ifdef CPU_NVPL
#include <nvpl_sparse.h>

#include "../include/kernels/CPU/gemm.hh"
#include "../include/utilities.hh"

namespace cpu {
/** A class for GEMM CPU BLAS kernels. */
template <typename T>
class sp_gemm_cpu : public sp_gemm<T> {
 public:
  using sp_gemm<T>::gemm;
  using sp_gemm<T>::callConsume;
  using sp_gemm<T>::m_;
  using sp_gemm<T>::n_;
  using sp_gemm<T>::k_;
  using sp_gemm<T>::A_;
  using sp_gemm<T>::B_;
  using sp_gemm<T>::C_;

 private:
  /** Make call to the GEMM kernel. */
  void callGemm() override {

    // Ensure compiler doesn't optimise away the work being done
    callConsume();
  }

  /** Perform any required steps before calling the GEMM kernel that should
   * be timed. */
  void preLoopRequirements() override {
    // Set type enum
    if constexpr (std::is_same_v<T, float>) {
      type_ = NVPL_SPARSE_R_32F;
    } else if constexpr (std::is_same_v<T, double>) {
      type_ = NVPL_SPARSE_R_64F;
    } else {
      // Un-specialised class will not do any work - print error and exit.
      std::cout << "ERROR - Datatype for NVPL sparse GEMM kernel not supported."
                << std::endl;
      exit(1);
    }
    status_ = nvpl_sparse_create(&handle_);
    // Todo -- error check

    // Todo -- Make const?
    status_ = nvpl_sparse_create_csr(A_nvpl_, n_, n_, nnz_, A_row_ptr_nvpl_,
                                     A_col_index_nvpl_, A_vals_nvpl_,
                                     index_type_, index_type_, base_, type_);

    status_ = nvpl_sparse_create_csr(B_nvpl_, n_, n_, nnz_, B_row_ptr_nvpl_,
                                     B_col_index_nvpl_, B_vals_nvpl_,
                                     index_type_, index_type_, base_, type_);
    // Todo -- error check


  }

  /** Perform any required steps after calling the GEMM kernel that should
   * be timed. */
  void postLoopRequirements() override {
    status_ = nvpl_sparse_destroy(handle_);
    // Todo -- error check
    status_ = nvpl_sparse_destroy_sp_mat(A_nvpl_);
    status_ = nvpl_sparse_destroy_sp_mat(B_nvpl_);
    status_ = nvpl_sparse_destroy_sp_mat(C_nvpl_);
  }

  /** The constant value Alpha. */
  T alpha = ALPHA;

  /** The constant value Beta. */
  T beta = BETA;

  /**
   * Sparse metadata
  */
  nvpl_sparse_status_t status_;
  nvpl_sparse_handle_t handle_;
  nvpl_sparse_data_type_t type_;

  nvpl_sparse_operation_t op_ = NVPL_SPARSE_OPERATION_NON_TRANSPOSE;
  nvpl_sparse_index_base_t base_ = NVPL_SPARSE_INDEX_BASE_ZERO;
  nvpl_sparse_format_t format_ = NVPL_SPARSE_FORMAT_CSR;
  nvpl_sparse_order_t order_ = NVPL_SPARSE_ORDER_COL;
  nvpl_sparse_index_type_t index_type_ = NVPL_SPARSE_INDEX_64I;

  /**
   * Sparse matrix descriptors
  */
  nvpl_sparse_sp_mat_descr_t* A_nvpl_;
  nvpl_sparse_sp_mat_descr_t* B_nvpl_;
  nvpl_sparse_sp_mat_descr_t* C_nvpl_;

  void* A_row_ptr_nvpl_;
  void* B_row_ptr_nvpl_;
  void* C_row_ptr_nvpl_;
  void* A_col_idnex_nvpl_;
  void* B_col_idnex_nvpl_;
  void* C_col_idnex_nvpl_;
  void* A_vals_nvpl_;
  void* B_vals_nvpl_;
  void* C_vals_nvpl_;
};
}  // namespace cpu
#endif