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
  using sp_gemm<T>::sp_gemm;
  using sp_gemm<T>::callConsume;
  using sp_gemm<T>::m_;
  using sp_gemm<T>::n_;
  using sp_gemm<T>::k_;
  using sp_gemm<T>::A_;
  using sp_gemm<T>::B_;
  using sp_gemm<T>::C_;
  using sp_gemm<T>::nnz_;

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




    if constexpr (std::is_same_v<T, float>) {
      status_ = armpl_spmm_exec_s(transA_,
                                  transB_,
                                  alpha,
                                  *A_armpl_,
                                  *B_armpl_,
                                  beta,
                                  *B_armpl_);
    } else if constexpr (std::is_same_v<T, double>) {
      std::cout << "About to execute dgemm" << std::endl;
      status_ = armpl_spmm_exec_d(transA_,
                                  transB_,
                                  alpha,
                                  *A_armpl_,
                                  *B_armpl_,
                                  beta,
                                  *B_armpl_);
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
  void preLoopRequirements() override {
    // Need to put A_ and B_ into A_armpl_ and B_armpl_
    // ToDo -- Error catching
    toCSR_armpl();
//    std::cout << "toCSR_armpl() wrapped up without a problem" << std::endl;
  }

  /** Perform any required steps after calling the GEMM kernel that should
   * be timed. */
  void postLoopRequirements() override {
    status_ = armpl_spmat_destroy(*A_armpl_);
    if (status_ != ARMPL_STATUS_SUCCESS) {
      std::cout << "ERROR " << status_ << std::endl;
      exit(1);
    }
    status_ = armpl_spmat_destroy(*B_armpl_);
    if (status_ != ARMPL_STATUS_SUCCESS) {
      std::cout << "ERROR " << status_ << std::endl;
      exit(1);
    }
//    status_ = armpl_spmat_destroy(*C_armpl_);
//    if (status_ != ARMPL_STATUS_SUCCESS) {
//      std::cout << "ERROR " << status_ << std::endl;
//      exit(1);
//    }

//    delete [] A_armpl_row_ptr_;
//    delete [] A_armpl_col_index_;
//    delete [] A_vals_;
//    delete [] B_armpl_row_ptr_;
//    delete [] B_armpl_col_index_;
//    delete [] B_vals_;
//    delete [] C_armpl_row_ptr_;
//    delete [] C_armpl_col_index_;
//    delete [] C_vals_;

  }

  /** The constant value Alpha. */
  const T alpha = ALPHA;

  /** The constant value Beta. */
  const T beta = BETA;

  armpl_status_t status_;

  armpl_spmat_t armpl_A, armpl_B, armpl_C;

  void toCSR_armpl() {
    n_armpl_ = n_;
    // ToDo -- check whether flags_ is correct!
    flags_ = 0;

    // Move A to CSR
    A_armpl_row_ptr_ = new armpl_int_t[n_ + 1];
    A_armpl_col_index_ = new armpl_int_t[nnz_];
    A_vals_ = new T[nnz_];
    A_armpl_row_ptr_[0] = 0;

    int nnz_encountered = 0;
//    std::cout << "About to load A into csr" << std::endl;
    for (int row = 0; row < n_; row++) {
//      std::cout << "\tRow " << (row + 1) << " = " << nnz_encountered << std::endl;
      A_armpl_row_ptr_[row + 1] = nnz_encountered;
      for (int col = 0; col < n_; col++) {
        if (A_[(row * n_) + col] != 0.0) {
//          std::cout << "\t\tCol " << col << " = " << A_[(row * n_) + col] <<
//          std::endl;
          A_armpl_col_index_[nnz_encountered] = col;
          A_vals_[nnz_encountered] = static_cast<T>(A_[(row * n_) + col]);
          nnz_encountered++;
//          std::cout << "\tnnz_encountered = " << nnz_encountered << std::endl;
        }
      }
    }

//    std::cout << "___A =" << std::endl << "\t\t[";
//    for (int i = 0; i < (n_ + 1); i++) {
//      if (i != 0) {
//        std::cout << ", ";
//      }
//      std::cout << A_armpl_row_ptr_[i];
//    }
//    std::cout << "]" << std::endl << "\t\t[";
//    for (int i = 0; i < nnz_; i++) {
//      if (i != 0) {
//        std::cout << ", ";
//      }
//      std::cout << A_armpl_col_index_[i];
//    }
//    std::cout << "]" << std::endl << "\t\t[";
//    for (int i = 0; i < nnz_; i++) {
//      if (i != 0) {
//        std::cout << ", ";
//      }
//      std::cout << A_vals_[i];
//    }
//    std::cout << "]" << std::endl;


//    std::cout << "About to load B into csr" << std::endl;

    // Move B to CSR
    B_armpl_row_ptr_ = new armpl_int_t[n_ + 1];
    B_armpl_col_index_ = new armpl_int_t[nnz_];
    B_vals_ = new T[nnz_];
    B_armpl_row_ptr_[0] = 0;

    nnz_encountered = 0;
    for (int row = 0; row < n_; row++) {
//      std::cout << "\tRow " << (row + 1) << " = " << nnz_encountered <<
//      std::endl;
      B_armpl_row_ptr_[row + 1] = nnz_encountered;
      for (int col = 0; col < n_; col++) {
        if (B_[(row * n_) + col] != 0.0) {
//          std::cout << "\t\tCol " << col << " = " << B_[(row * n_) + col] << std::endl;
          B_armpl_col_index_[nnz_encountered] = col;
          B_vals_[nnz_encountered] = static_cast<T>(B_[(row * n_) + col]);
          nnz_encountered++;
//          std::cout << "\tnnz_encountered = " << nnz_encountered << std::endl;
        }
      }
    }
//    std::cout << "___B =" << std::endl << "\t\t[";
//    for (int i = 0; i < (n_ + 1); i++) {
//      if (i != 0) {
//        std::cout << ", ";
//      }
//      std::cout << B_armpl_row_ptr_[i];
//    }
//    std::cout << "]" << std::endl << "\t\t[";
//    for (int i = 0; i < nnz_; i++) {
//      if (i != 0) {
//        std::cout << ", ";
//      }
//      std::cout << B_armpl_col_index_[i];
//    }
//    std::cout << "]" << std::endl << "\t\t[";
//    for (int i = 0; i < nnz_; i++) {
//      if (i != 0) {
//        std::cout << ", ";
//      }
//      std::cout << B_vals_[i];
//    }
//    std::cout << "]" << std::endl;


//    // Move B to CSR
//    C_armpl_row_ptr_ = new armpl_int_t[n_ + 1];
//    C_armpl_col_index_ = new armpl_int_t[nnz_];
//    C_vals_ = new T[nnz_];
//    C_armpl_row_ptr_[0] = 0;
//
//    nnz_encountered = 0;
////    std::cout << "About to load C into csr" << std::endl;
//    for (int row = 0; row < n_; row++) {
////      std::cout << "\tRow " << (row + 1) << " = " << nnz_encountered << std::endl;
//      C_armpl_row_ptr_[row + 1] = nnz_encountered;
//      for (int col = 0; col < n_; col++) {
//        if (A_[(row * n_) + col] != 0.0) {
//          C_armpl_col_index_[nnz_encountered] = col;
//          C_vals_[nnz_encountered] = A_[(row * n_) + col];
//          nnz_encountered++;
////          std::cout << "\t\tCol " << col << " = " << C_vals_[nnz_encountered] <<
////          std::endl;
////          std::cout << "\tnnz_encountered = " << nnz_encountered << std::endl;
//        }
//      }
//    }

//    std::cout << "___C =" << std::endl << "\t\t[";
//    for (int i = 0; i < (n_ + 1); i++) {
//      if (i != 0) {
//        std::cout << ", ";
//      }
//      std::cout << C_armpl_row_ptr_[i];
//    }
//    std::cout << "]" << std::endl << "\t\t[";
//    for (int i = 0; i < nnz_; i++) {
//      if (i != 0) {
//        std::cout << ", ";
//      }
//      std::cout << C_armpl_col_index_[i];
//    }
//    std::cout << "]" << std::endl << "\t\t[";
//    for (int i = 0; i < nnz_; i++) {
//      if (i != 0) {
//        std::cout << ", ";
//      }
//      std::cout << C_vals_[i];
//    }
//    std::cout << "]" << std::endl;



//    std::cout << "Loading csr A into armpl storage formats" << std::endl;
    if constexpr (std::is_same_v<T, float>) {
      std::cout << "\tn_armpl_ = " << n_armpl_ << std::endl;
      std::cout << "\tA_armpl_row_ptr_ (size = " << sizeof
      (A_armpl_row_ptr_[0]) << ") = [" << A_armpl_row_ptr_[0];
      for (int i = 1; i < (n_ + 1); i++) {
        std::cout << ", " << A_armpl_row_ptr_[i];
      }
      std::cout << "]" << std::endl << "\tA_armpl_col_index_ (size = " <<
      sizeof(A_armpl_col_index_[0]) << ") = [" <<
      A_armpl_col_index_[0];
      for (int i = 1; i < nnz_; i++) {
        std::cout << ", " << A_armpl_col_index_[i];
      }
      std::cout << "]" << std::endl << "\tA_vals_ (size = " << sizeof
      (A_vals_[0]) << ") = [" << A_vals_[0];
      for (int i = 1; i < nnz_; i++) {
        std::cout << ", " << A_vals_[i];
      }
      std::cout << "]" << std::endl << "flags: " << flags_ << std::endl;

      status_ = armpl_spmat_create_csr_s(A_armpl_,
                                         n_armpl_,
                                         n_armpl_,
                                         A_armpl_row_ptr_,
                                         A_armpl_col_index_,
                                         A_vals_,
                                         flags_);
      if (status_ != ARMPL_STATUS_SUCCESS) {
        std::cout << "ERROR " << status_ << std::endl;
        exit(1);
      }

//      std::cout << "Loading csr C into armpl storage formats" << std::endl;
//      status_ = armpl_spmat_create_csr_s(C_armpl_,
//                                         n_armpl_,
//                                         n_armpl_,
//                                         C_armpl_row_ptr_,
//                                         C_armpl_col_index_,
//                                         C_vals_,
//                                         flags_);
//      if (status_ != ARMPL_STATUS_SUCCESS) {
//        std::cout << "ERROR " << status_ << std::endl;
//        exit(1);
//      }

//      std::cout << "Loading csr B into armpl storage formats" << std::endl;
      status_ = armpl_spmat_create_csr_s(B_armpl_,
                                         n_armpl_,
                                         n_armpl_,
                                         B_armpl_row_ptr_,
                                         B_armpl_col_index_,
                                         B_vals_,
                                         flags_);
      if (status_ != ARMPL_STATUS_SUCCESS) {
        std::cout << "ERROR " << status_ << std::endl;
        exit(1);
      }
    } else if constexpr (std::is_same_v<T, double>) {
      std::cout << "\tn_armpl_ = " << n_armpl_ << std::endl;
      std::cout << "\tA_armpl_row_ptr_ (size = " << sizeof
      (A_armpl_row_ptr_[0]) << ") = [" << A_armpl_row_ptr_[0];
      for (int i = 1; i < (n_ + 1); i++) {
        std::cout << ", " << A_armpl_row_ptr_[i];
      }
      std::cout << "]" << std::endl << "\tA_armpl_col_index_ (size = " <<
      sizeof(A_armpl_col_index_[0]) << ") = [" <<
      A_armpl_col_index_[0];
      for (int i = 1; i < nnz_; i++) {
        std::cout << ", " << A_armpl_col_index_[i];
      }
      std::cout << "]" << std::endl << "\tA_vals_ (size = " << sizeof
      (A_vals_[0]) << ") = [" << A_vals_[0];
      for (int i = 1; i < nnz_; i++) {
        std::cout << ", " << A_vals_[i];
      }
      std::cout << "]" << std::endl << "flags: " << flags_ << std::endl;


      std::cout << "About to create CSR A (double)" << std::endl;
      status_ = armpl_spmat_create_csr_d(A_armpl_,
                                         n_armpl_,
                                         n_armpl_,
                                         A_armpl_row_ptr_,
                                         A_armpl_col_index_,
                                         A_vals_,
                                         flags_);
      if (status_ != ARMPL_STATUS_SUCCESS) {
        std::cout << "ERROR " << status_ << std::endl;
        exit(1);
      }

//      std::cout << "Loading csr C into armpl storage formats" << std::endl;
//      status_ = armpl_spmat_create_csr_d(C_armpl_,
//                                         n_armpl_,
//                                         n_armpl_,
//                                         C_armpl_row_ptr_,
//                                         C_armpl_col_index_,
//                                         C_vals_,
//                                         flags_);
//      if (status_ != ARMPL_STATUS_SUCCESS) {
//        std::cout << "ERROR " << status_ << std::endl;
//        exit(1);
//      }

//      std::cout << "Loading csr B into armpl storage formats" << std::endl;
      std::cout << "About to create CSR B (double)" << std::endl;
      status_ = armpl_spmat_create_csr_d(B_armpl_,
                                         n_armpl_,
                                         n_armpl_,
                                         B_armpl_row_ptr_,
                                         B_armpl_col_index_,
                                         B_vals_,
                                         flags_);
      if (status_ != ARMPL_STATUS_SUCCESS) {
        std::cout << "ERROR " << status_ << std::endl;
        exit(1);
      }
    }

//    std::cout << "Okay, all matrices made!!" << std::endl;
  }

  armpl_int_t flags_;

  armpl_int_t n_armpl_;

  armpl_int_t* A_armpl_row_ptr_;
  armpl_int_t* A_armpl_col_index_;
  armpl_int_t* B_armpl_row_ptr_;
  armpl_int_t* B_armpl_col_index_;
  armpl_int_t* C_armpl_row_ptr_;
  armpl_int_t* C_armpl_col_index_;

  T* A_vals_;
  T* B_vals_;
  T* C_vals_;

  armpl_spmat_t* A_armpl_;
  armpl_spmat_t* B_armpl_;
  armpl_spmat_t* C_armpl_;

  armpl_sparse_hint_value transA_ = ARMPL_SPARSE_OPERATION_NOTRANS;
  armpl_sparse_hint_value transB_ = ARMPL_SPARSE_OPERATION_NOTRANS;

};
}  // namespace cpu
#endif