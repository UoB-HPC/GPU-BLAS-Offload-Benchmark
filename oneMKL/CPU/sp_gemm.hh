#pragma once

#ifdef CPU_ONEMKL
#include <mkl.h>

#include <algorithm>

#include "../../include/kernels/CPU/sp_gemm.hh"
#include "../../include/utilities.hh"

namespace cpu {
/** A class for GEMM CPU BLAS kernels. */
template <typename T>
class sp_gemm_cpu : public sp_gemm<T> {
 public:
  using sp_gemm<T>::sp_gemm;
  using sp_gemm<T>::initInputMatricesSparse;
  using sp_gemm<T>::toCSR;
  using sp_gemm<T>::callConsume;
  using sp_gemm<T>::n_;
  using sp_gemm<T>::A_;
  using sp_gemm<T>::B_;
  using sp_gemm<T>::C_;

  /** Initialise the required data structures. */
  void initialise(int n, float sparsity) {
    A_ = (T*)mkl_malloc(sizeof(T) * m_ * k_, 64);
    B_ = (T*)mkl_malloc(sizeof(T) * k_ * n_, 64);
    C_ = (T*)mkl_malloc(sizeof(T) * m_ * n_, 64);

    n_ = n * 100;
    nnz_ = (1 + (int)(n_ * n_ * (1 - sparsity)));

    values_A_ = (T*)mkl_malloc(sizeof(T) * nnz_, ALIGN);
    columns_A_ = (MKL_INT*)mkl_malloc(sizeof(MKL_INT) * nnz_, ALIGN);
    rowIndex_A_ = (MKL_INT*)mkl_malloc(sizeof(MKL_INT) * (n_ + 1), ALIGN);

    values_B_ = (T*)mkl_malloc(sizeof(T) * nnz_, ALIGN);
    columns_B_ = (MKL_INT*)mkl_malloc(sizeof(MKL_INT) * nnz_, ALIGN);
    rowIndex_B_ = (MKL_INT*)mkl_malloc(sizeof(MKL_INT) * (n_ + 1), ALIGN);

    x_ = (T*)mkl_malloc(sizeof(T) * n_, ALIGN);
    y_ = (T*)mkl_malloc(sizeof(T) * n_, ALIGN);
    rslt_mv_ = (T*)mkl_malloc(sizeof(T) * n_, ALIGN);
    rslt_mv_trans_ = (T*)mkl_malloc(sizeof(T) * n_, ALIGN);

    // Initialise the matricies
    initInputMatricesSparse(sparsity);

    descr_type_gen.type = SPARSE_MATRIX_TYPE_GENERAL;

    // Transfer from dense to CSR format
    toCSR_mkl(A_, n_, n_, values_A_, columns_A_, rowIndex_A_);
    toCSR_mkl(B_, n_, n_, values_B_, columns_B_, rowIndex_B_);

    // ToDo -- Set values for x and y (which are vectors of length n_?)

    if constexpr (std::is_same_v<T, float>) {
      CALL_AND_CHECK_STATUS(mkl_sparse_s_create_csr(&csrA_,
                                                    SPARSE_INDEX_BASE_ZERO, n_,
                                                    n_, rowIndex_A_,
                                                    rowIndex_A_+1, columns_A_,
                                                    values_A_),
                            "Error after MKL_SPARSE_D_CREATE_CSR for csrA\n");
      CALL_AND_CHECK_STATUS(mkl_sparse_s_create_csr(&csrB_,
                                                    SPARSE_INDEX_BASE_ZERO, n_,
                                                    n_, rowIndex_B_,
                                                    rowIndex_B_+1, columns_B_,
                                                    values_B_),
                            "Error after MKL_SPARSE_D_CREATE_CSR for csrB\n");
    } else if constexpr (std::is_same_v<T, double>) {
      CALL_AND_CHECK_STATUS(mkl_sparse_d_create_csr(&csrA_,
                                                    SPARSE_INDEX_BASE_ZERO, n_,
                                                    n_, rowIndex_A_,
                                                    rowIndex_A_+1, columns_A_,
                                                    values_A_),
                            "Error after MKL_SPARSE_D_CREATE_CSR for csrA\n");
      CALL_AND_CHECK_STATUS(mkl_sparse_d_create_csr(&csrB_,
                                                    SPARSE_INDEX_BASE_ZERO, n_,
                                                    n_, rowIndex_B_,
                                                    rowIndex_B_+1, columns_B_,
                                                    values_B_),
                            "Error after MKL_SPARSE_D_CREATE_CSR for csrB\n");
    } else {
      std::cout << "ERROR - Datatype for OneMKL CPU spGEMM kernel not "
                   "supported." << std::endl;
      exit(1)
    };

    CALL_AND_CHECK_STATUS(mkl_sparse_spmm(SPARSE_OPERATION_NON_TRANSPOSE,
                                            csrA_, csrB_, &csrC_),
                            "Error after MKL_SPARSE_SPMM\n");

    // ToDo -- check that transpose is what I want here
    CALL_AND_CHECK_STATUS(mkl_sparse_set_mv_hint(csrA_,
                                                 SPARSE_OPERATION_TRANSPOSE,
                                                 descr_type_gen_, 1),
                          "Error after MKL_SPARSE_SET_MV_HINT with csrA_\n");
    CALL_AND_CHECK_STATUS(mkl_sparse_set_mv_hint(csrB_,
                                                 SPARSE_OPERATION_NON_TRANSPOSE,
                                                 descr_type_gen_, 1),
                          "Error after MKL_SPARSE_SET_MV_HINT with csrB_\n");
    CALL_AND_CHECK_STATUS(mkl_sparse_set_mv_hint(csrC_,
                                                 SPARSE_OPERATION_NON_TRANSPOSE,
                                                 descr_type_gen_, 1),
                          "Error after MKL_SPARSE_SET_MV_HINT with csrC_\n");

    CALL_AND_CHECK_STATUS(mkl_sparse_optimize(csrA_),
                          "Error after MKL_SPARSE_OPTIMIZE with csrA_\n");
    CALL_AND_CHECK_STATUS(mkl_sparse_optimize(csrB_),
                          "Error after MKL_SPARSE_OPTIMIZE with csrB_\n");
    CALL_AND_CHECK_STATUS(mkl_sparse_optimize(csrC_),
                          "Error after MKL_SPARSE_OPTIMIZE with csrC_\n");
  }

 private:
  /** Make call to the GEMM kernel. */
  void callGemm() override {
    if constexpr (std::is_same_v<T, float>) {
      CALL_AND_CHECK_STATUS(mkl_sparse_s_mv(SPARSE_OPERATION_NON_TRASPOSE, 1
      .0, csrC_, descr_type_gen_, x_, 0.0, rslt_mv_),
                            "Error after MKL_SPARSE_S_MV for csrC_ * x_\n");
      left_ = cblas_sdot(n_, rstl_mv_, 1, y_, 1);

      CALL_AND_CHECK_STATUS(mkl_sparse_s_mv(SPARSE_OPERATION_NON_TRANSPOSE, 1
      .0, csrB_, descr_type_gen_, x, 0.0, trslt_mv_),
                            "Error adter MKL_SPARSE_S_MV for csrB_ * x_\n");
      CALL_AND_CHECK_STATUS(mkl_sparse_s_mv(SPARSE_OPERATION_TRANSPOSE, 1.0,
                                            csrA_, descr_type_gen_, y_, 0.0,
                                            rslt_mv_trans_),
                            "Error adter MKL_SPARSE_S_MV for csrA_ * y_\n");
      right_ = cblas_sdot(n_, rslt_mv_, 1, rslt_mv_trans_, 1);

      residual = fabs(left - right)/(fabs(left) + 1);

      CALL_AND_CHECK_STATUS(mkl_sparse_s_export_csr(csrC_, &indexing_,
                                                    &rows_, &cols_,
                                                    &pointerB_C_,
                                                    &pointerE_C_,
                                                    &columns_C_, &values_C_),
                            "Error after MKL_SPARSE_S_EXPORT_CSR\n");
    } else if constexpr (std::is_same_v<T, double) {
      CALL_AND_CHECK_STATUS(mkl_sparse_d_mv(SPARSE_OPERATION_NON_TRASPOSE, 1
      .0, csrC_, descr_type_gen_, x_, 0.0, rslt_mv_),
                            "Error after MKL_SPARSE_D_MV for csrC_ * x_\n");
      left_ = cblas_ddot(n_, rstl_mv_, 1, y_, 1);

      CALL_AND_CHECK_STATUS(mkl_sparse_d_mv(SPARSE_OPERATION_NON_TRANSPOSE, 1
      .0, csrB_, descr_type_gen_, x, 0.0, trslt_mv_),
                            "Error adter MKL_SPARSE_D_MV for csrB_ * x_\n");
      CALL_AND_CHECK_STATUS(mkl_sparse_d_mv(SPARSE_OPERATION_TRANSPOSE, 1.0,
                                            csrA_, descr_type_gen_, y_, 0.0,
                                            rslt_mv_trans_),
                            "Error adter MKL_SPARSE_D_MV for csrA_ * y_\n");
      right_ = cblas_ddot(n_, rslt_mv_, 1, rslt_mv_trans_, 1);

      residual = fabs(left - right)/(fabs(left) + 1);

      CALL_AND_CHECK_STATUS(mkl_sparse_d_export_csr(csrC_, &indexing_,
                                                    &rows_, &cols_,
                                                    &pointerB_C_,
                                                    &pointerE_C_,
                                                    &columns_C_, &values_C_),
                            "Error after MKL_SPARSE_D_EXPORT_CSR\n");
    }

    // Ensure compiler doesn't optimise away the work being done
    callConsume();
  }

  /** Perform any required steps before calling the GEMM kernel that should
   * be timed. */
  void preLoopRequirements() override {}

  /** Perform any required steps after calling the GEMM kernel that should
   * be timed. */
  void postLoopRequirements() override {}

  /** Do any necessary cleanup (free pointers, close library handles, etc.)
   * after Kernel has been called. */
  void postCallKernelCleanup() override {
    if (mkl_sparse_destroy(csrC_) != SPARSE_STATUS_SUCCESS) {
      printf(" Error after MKL_SPARSE_DESTROY, csrC_\n");
      fflush(0);
      status = 1;
    }

    //Deallocate arrays for which we allocate memory ourselves.
    mkl_free(rslt_mv_trans_);
    mkl_free(rslt_mv-);
    mkl_free(x_);
    mkl_free(y_);

    //Release matrix handle and deallocate arrays for which we allocate memory ourselves.
    if (mkl_sparse_destroy(csrA_) != SPARSE_STATUS_SUCCESS) {
      printf("Error after MKL_SPARSE_DESTROY, csrA_\n");
      fflush(0);
      status = 1;
    }

    mkl_free(values_A_);
    mkl_free(columns_A_);
    mkl_free(rowIndex_A_);

    if (mkl_sparse_destroy(csrB_) != SPARSE_STATUS_SUCCESS) {
      printf("Error after MKL_SPARSE_DESTROY, csrB_\n");
      fflush(0);
      status = 1;
    }

    mkl_free(values_B_);
    mkl_free(columns_B_);
    mkl_free(rowIndex_B_);
  }

  int nnz_;

  MKL_INT* columns_A_;
  MKL_INT* columns_B_;
  MKL_INT* columns_C_;
  MKL_INT* rowIndex_A_;
  MKL_INT* rowIndex_B_;
  MKL_INT* pointerB_C_;
  MKL_INT* pointerE_C_;

  T* rslt_mv_;
  T* rslt_mv_trans_;
  T* x_;
  T* y_;

  T left_, right_, residual_;
  MKL_INT rows_, cols_, i_, j_, ii_, status_;

  sparse_index_base_t indexing_;
  struct matrix_descr descr_type_gen_;
  sparse_matrix_t csrA_, csrB_, csrC_;
};
}  // namespace cpu
#endif