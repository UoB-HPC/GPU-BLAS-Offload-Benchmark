#pragma once

#ifdef CPU_ARMPL
#include <stdio.h>
#include <stdlib.h>
#include <armpl.h>
#include <omp.h>

#include <algorithm>

#include "../include/kernels/CPU/sp_gemm.hh"
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

    /**
     * Flow of ARMPL Sparse LA:
     *
     * 1. Create sparse matrix objects: armpl_spmat_create_csr[sdcz]()
     *
     * 2. Supply hints on usage: armpl_spmat_hint()
     *
     * 3. Optimise for SpMV: armpl_spmv_optimize()
     *
     * 4. Solve SpMV case: armpl_spmv_exec_[sdcz]()
     *
     * 5. Destroy sparse matrix object: armpl_spmat_destroy()
     *
     * In addiion, users can choose to update a set of non-zero values using
     * armpl_spmat_update_[sdcz]()
     */

    // Todo -- See if using armpl_spmat_hint can improve performance here.
    //  If so, follow with optimisation functions




    if (std::is_same_v<T, float>) {
      status_ = armpl_spmm_exec_s(transA,
                                  transB,
                                  alpha,
                                  A_armpl_,
                                  B_armpl,
                                  beta,
                                  C_armpl_);
    } else if constexpr (std::is_same_v<T, double>) {
      status_ = armpl_spmm_exec_d(transA,
                                  transB,
                                  alpha,
                                  A_armpl_,
                                  B_armpl,
                                  beta,
                                  C_armpl_);
    } else {
      // Un-specialised class will not do any work - print error and exit.
      std::cout << "ERROR - Datatype for ArmPL CPU GEMM kernel not supported."
                << std::endl;
      exit(1);
    }

    if (status_ != ARMPL_STATUS_SUCCESS) {
      std::cout << "ERROR " << status_ << std::endl;
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
  void postLoopRequirements() override {
    status_ = armpl_spmat_destroy(A_armpl_);
    if (status_ != ARMPL_STATUS_SUCCESS) {
      std::cout << "ERROR " << status_ << std::endl;
      exit(1);
    }
    status_ = armpl_spmat_destroy(B_armpl_);
    if (status_ != ARMPL_STATUS_SUCCESS) {
      std::cout << "ERROR " << status_ << std::endl;
      exit(1);
    }
    status_ = armpl_spmat_destroy(C_armpl_);
    if (status_ != ARMPL_STATUS_SUCCESS) {
      std::cout << "ERROR " << status_ << std::endl;
      exit(1);
    }
  }

  /** The constant value Alpha. */
  const T alpha = ALPHA;

  /** The constant value Beta. */
  const T beta = BETA;

  armpl_status_t status_;

  armpl_spmat_t armpl_A, armpl_B, armpl_C;

  @override
  void toCSR() {
    n_armpl_ = n_;
    // ToDo -- check whether flags_ is correct!
    flags_ = 0;

    // Move A to CSR
    A_armpl_row_ptr_ = new armpl_int_t[n_ + 1];
    A_armpl_col_index_ = new armpl_int_t[nnz_];
    A_vals_ = new T[nnz_];
    int nnz_encountered = 0;
    for (int row = 0; row < n_; row++) {
      A_armpl_row_ptr_[row] = nnz_encountered;
      for (int col = 0; col < n_; col++) {
        if (A_[(row * n_) + col] != 0.0) {
          A_armpl_col_index_[nnz_encountered] = col;
          A_vals_[nnz_encountered] = A_[(row * n_) + col];
          nnz_encountered++;
        }
      }
    }

    // Move B to CSR
    B_armpl_row_ptr_ = new armpl_int_t[n_ + 1];
    B_armpl_col_index_ = new armpl_int_t[nnz_];
    B_vals_ = new T[nnz_];
    nnz_encountered = 0;
    for (int row = 0; row < n_; row++) {
      B_armpl_row_ptr_[row] = nnz_encountered;
      for (int col = 0; col < n_; col++) {
        if (B_[(row * n_) + col] != 0.0) {
          B_armpl_col_index_[nnz_encountered] = col;
          B_vals_[nnz_encountered] = B_[(row * n_) + col];
          nnz_encountered++;
        }
      }
    }

    if (std::is_sam_v<T, float>) {
      status_ = armpl_spmat_create_csr_s(A_armpl_,
                                         n_armpl_,
                                         n_armpl_,
                                         A_armpl_row_ptr_,
                                         A_armpl_col_index_,
                                         A_vals_,
                                         flags);
      if (status_ != ARMPL_STATUS_SUCCESS) {
        std::cout << "ERROR " << status_ << std::endl;
        exit(1);
      }

      status_ = armpl_spmat_create_csr_s(B_armpl_,
                                         n_armpl_,
                                         n_armpl_,
                                         B_armpl_row_ptr_,
                                         B_armpl_col_index_,
                                         B_vals_,
                                         flags);
      if (status_ != ARMPL_STATUS_SUCCESS) {
        std::cout << "ERROR " << status_ << std::endl;
        exit(1);
      }
    } else if (std::is_same_v<T, double>) {
      status_ = armpl_spmat_create_csr_d(A_armpl_,
                                         n_armpl_,
                                         n_armpl_,
                                         A_armpl_row_ptr_,
                                         A_armpl_col_index_,
                                         A_vals_,
                                         flags);
      if (status_ != ARMPL_STATUS_SUCCESS) {
        std::cout << "ERROR " << status_ << std::endl;
        exit(1);
      }

      status_ = armpl_spmat_create_csr_d(B_armpl_,
                                         n_armpl_,
                                         n_armpl_,
                                         B_armpl_row_ptr_,
                                         B_armpl_col_index_,
                                         B_vals_,
                                         flags);
      if (status_ != ARMPL_STATUS_SUCCESS) {
        std::cout << "ERROR " << status_ << std::endl;
        exit(1);
      }
    }


  }

  armpl_int_t flags_;

  armpl_int_t n_armpl_;

  armpl_int_t* A_armpl_row_ptr_;
  armpl_int_t* A_armpl_col_index_;
  armpl_int_t* B_armpl_row_ptr_;
  armpl_int_t* B_armpl_col_index_;
  armpl_int_t* C_armpl_row_ptr_;
  armpl_int_t* C_armpl_col_index_;

  armpl_spmat_t* A_armpl_;
  armpl_spmat_t* B_armpl_;
  armpl_spmat_t* C_armpl_;

  sparse_hint_value transA = ARMPL_SPARSE_OPERATION_NOTRANS;
  sparse_hint_value transB = ARMPL_SPARSE_OPERATION_NOTRANS;

};
}  // namespace cpu
#endif