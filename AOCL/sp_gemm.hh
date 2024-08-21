#pragma once

#ifdef CPU_AOCL
#include <blis.h>

#include "../include/kernels/CPU/gemm.hh"
#include "../include/utilities.hh"

namespace cpu {
/** A class for GEMM CPU BLAS kernels. */
template <typename T>
class gemm_cpu : public gemm<T> {
 public:
  using gemm<T>::gemm;
  using gemm<T>::callConsume;
  using gemm<T>::m_;
  using gemm<T>::n_;
  using gemm<T>::k_;
  using gemm<T>::A_;
  using gemm<T>::B_;
  using gemm<T>::C_;

 private:
  /** Make call to the GEMM kernel. */
  void callGemm() override {
    if constexpr (std::is_same_v<T, float>) {
      bli_sgemm(BLIS_NO_TRANSPOSE, BLIS_NO_TRANSPOSE, m_, n_, k_, &alpha, A_,
                rowStride, std::max(1, m_), B_, rowStride, std::max(1, k_),
                &beta, C_, rowStride, std::max(1, m_));
    } else if constexpr (std::is_same_v<T, double>) {
      // Todo -- base?
      aoclsparse_create_dscr(&A_csr_, base, n_, n_, nnz_, cst_row_ptr_A_.data
      (), csr_col_ind_A_.data(), csr_val_A_.data());
      aoclsparse_create_dscr(&B_csr_, base, n_, n_, nnz_, cst_row_ptr_B_.data
      (), csr_col_ind_B_.data(), csr_val_B_.data());

      aoclsparse_spmm(aoclsparse_operation_none, A_csr_, B_csr_, &C_csr_);
      aoclsparse_export_dcsr(C_csr_, &base, &C_M_, &C_N_, &nnz_C_,
                             &csr_row_ptr_C_, &csr_col_ind_C_, (void**)
                             &csr_val_C_);
    } else {
      // Un-specialised class will not do any work - print error and exit.
      std::cout << "ERROR - Datatype for AOCL CPU GEMM kernel not supported."
                << std::endl;
      exit(1);
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

  /** The constant value Alpha. */
  T alpha = ALPHA;

  /** The constant value Beta. */
  T beta = BETA;

  /** The distance in elements to the next column. */
  const int rowStride = 1;

  aoclsparse_matrix A_csr_;
  aoclsparse_int* csr_row_ptr_A_;
  aoclsparse_int* csr_col_ind_A_;
  T* csr_val_A_;

  aoclsparse_matrix B_csr_;
  aoclsparse_int* csr_row_ptr_B_;
  aoclsparse_int* csr_col_ind_B_;
  T* csr_val_B_;

  aoclsparse_matrix C_csr_;
  aoclsparse_int* csr_row_ptr_C_;
  aoclsparse_int* csr_col_ind_C_;
  T* csr_val_C_;
  aoclsparse_int C_M_;
  aoclsparse_int C_N_;

  aoclsparse_status status;
};
}  // namespace cpu
#endif