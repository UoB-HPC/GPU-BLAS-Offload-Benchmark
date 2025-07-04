#pragma once

#ifdef CPU_ARMPL
#include <stdio.h>
#include <stdlib.h>
#include "armpl.h"
#include <omp.h>

#include <algorithm>

#include "../include/kernels/CPU/spgemv.hh"
#include "../include/utilities.hh"

namespace cpu {
/** A class for GEMM CPU BLAS kernels. */
template <typename T>
class spgemv_cpu : public spgemv<T> {
 public:
  using spgemv<T>::spgemv;
  using spgemv<T>::callConsume;
  using spgemv<T>::m_;
  using spgemv<T>::n_;
  using spgemv<T>::A_;
  using spgemv<T>::x_;
  using spgemv<T>::y_;
  using spgemv<T>::nnz_;
    /** Initialise the required data structures. */
    void initialise(int m, int n, double sparsity) override {
      m_ = m;
      n_ = n;
      sparsity_ = sparsity;

      // Note that the below should be the same as the edges calculation
      // used in the initInputMatricesSparse function.  If changed here,
      // change there
      nnz_ = 1 + (uint64_t)((double)m_ * (double)n_ * (1.0 - sparsity_));

      A_ = (T*)malloc(sizeof(T) * m_ * n_);
      x_ = (T*)malloc(sizeof(T) * n_);
      y_ = (T*)malloc(sizeof(T) * m_);

      // Initialise the matrix and vectors
      initInputMatrixVector();
    }

 private:
  /** Make call to the GEMM kernel. */
  void callGemv() override {

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

    if constexpr (std::is_same_v<T, float>) {
      status_ = armpl_spmv_exec_s(trans_,
                                  alpha,
                                  A_armpl_,
                                  x_,
                                  beta,
                                  y_);
    } else if constexpr (std::is_same_v<T, double>) {
      status_ = armpl_spmv_exec_d(trans_,
                                  alpha,
                                  A_armpl_,
                                  x_,
                                  beta,
                                  y_);
    } else {
      // Un-specialised class will not do any work - print error and exit.
      std::cout << "ERROR - Datatype for ArmPL CPU SPGEMM kernel not supported."
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

    /** providing hints to ARMPL and optimizing the matrix datastructures */
    // TODO -- is noallocs best here?
    status_ = armpl_spmat_hint(A_armpl_, ARMPL_SPARSE_HINT_MEMORY,
                               ARMPL_SPARSE_MEMORY_NOALLOCS);
    if (status_ != ARMPL_STATUS_SUCCESS) {
      std::cout << "ERROR " << status_ << std::endl;
      exit(1);
    }

    status_ = armpl_spmat_hint(A_armpl_, ARMPL_SPARSE_HINT_STRUCTURE,
                               ARMPL_SPARSE_STRUCTURE_UNSTRUCTURED);
    if (status_ != ARMPL_STATUS_SUCCESS) {
      std::cout << "ERROR " << status_ << std::endl;
      exit(1);
    }

    // TODO -- will this be FEW?
    status_ = armpl_spmat_hint(A_armpl_, ARMPL_SPARSE_HINT_SPMM_INVOCATIONS,
                               ARMPL_SPARSE_INVOCATIONS_MANY);
    if (status_ != ARMPL_STATUS_SUCCESS) {
      std::cout << "ERROR " << status_ << std::endl;
      exit(1);
    }

    status_ = armpl_spmat_hint(A_armpl_, ARMPL_SPARSE_HINT_SPMM_OPERATION,
                               ARMPL_SPARSE_OPERATION_NOTRANS);
    if (status_ != ARMPL_STATUS_SUCCESS) {
      std::cout << "ERROR " << status_ << std::endl;
      exit(1);
    }

    // TODO -- investigate whch is better here
    status_ = armpl_spmat_hint(A_armpl_, ARMPL_SPARSE_HINT_SPMM_STRATEGY,
                               ARMPL_SPARSE_SPMM_STRAT_OPT_PART_STRUCT);
    if (status_ != ARMPL_STATUS_SUCCESS) {
      std::cout << "ERROR " << status_ << std::endl;
      exit(1);
    }

//  TODO -- this is thorwing an error -- couldn't immediately fix so come
//   back to

//    /** provide hints for the optimisation of the spmm execution */
//    status_ = armpl_spmm_optimize(ARMPL_SPARSE_OPERATION_NOTRANS,
//                                  ARMPL_SPARSE_OPERATION_NOTRANS,
//                                  ARMPL_SPARSE_SCALAR_ONE,
//                                  A_armpl_, B_armpl_,
//                                  ARMPL_SPARSE_SCALAR_ZERO,
//                                  C_armpl_);
//    if (status_ != ARMPL_STATUS_SUCCESS) {
//      std::cout << "ERROR " << status_ << std::endl;
//      exit(1);
//    }
  }

  /** Perform any required steps after calling the GEMM kernel that should
   * be timed. */
  void postLoopRequirements() override {
    status_ = armpl_spmat_destroy(A_armpl_);
    if (status_ != ARMPL_STATUS_SUCCESS) {
      std::cout << "ERROR " << status_ << std::endl;
      exit(1);
    }

    delete [] A_armpl_row_ptr_;
    delete [] A_armpl_col_index_;
    delete [] A_vals_;
  }

  /** The constant value Alpha. */
  const T alpha = ALPHA;

  /** The constant value Beta. */
  const T beta = BETA;

  void toSparseFormat() {
    n_armpl_ = n_;
    // ToDo -- check whether flags_ is correct!
    flags_ = 0;

    // Move A to CSR
    A_armpl_row_ptr_ = new armpl_int_t[m_ + 1];
    A_armpl_col_index_ = new armpl_int_t[nnz_];
    A_vals_ = new T[nnz_];
    A_armpl_row_ptr_[0] = 0;
    int nnz_encountered = 0;

    for (int row = 0; row < m_; row++) {
      A_armpl_row_ptr_[row + 1] = nnz_encountered;
      for (int col = 0; col < n_; col++) {
        if (A_[(row * n_) + col] != 0.0) {
          A_armpl_col_index_[nnz_encountered] = col;
          A_vals_[nnz_encountered] = static_cast<T>(A_[(row * n_) + col]);
          nnz_encountered++;
        }
      }
    }

    if constexpr (std::is_same_v<T, float>) {
//      printCSR(n_armpl_, A_armpl_row_ptr_, A_armpl_col_index_, A_vals_,
//                nnz_, flags_);
      status_ = armpl_spmat_create_csr_s(&A_armpl_,
                                         m_armpl_,
                                         n_armpl_,
                                         A_armpl_row_ptr_,
                                         A_armpl_col_index_,
                                         A_vals_,
                                         flags_);
      if (status_ != ARMPL_STATUS_SUCCESS) {
        std::cout << "ERROR " << status_ << std::endl;
        exit(1);
      }
    } else if constexpr (std::is_same_v<T, double>) {
//      printCSR(n_armpl_, A_armpl_row_ptr_, A_armpl_col_index_, A_vals_,
//                nnz_, flags_
      status_ = armpl_spmat_create_csr_d(&A_armpl_,
                                         m_armpl_,
                                         n_armpl_,
                                         A_armpl_row_ptr_,
                                         A_armpl_col_index_,
                                         A_vals_,
                                         flags_);
      if (status_ != ARMPL_STATUS_SUCCESS) {
        std::cout << "ERROR " << status_ << std::endl;
        exit(1);
      }

//      std::cout << "Okay, all matrices made!!" << std::endl;
    }

  }

  void printCSR(armpl_int_t n, armpl_int_t* rp, armpl_int_t* ci, T* v,
                armpl_int_t nz, armpl_int_t f) {
    std::cout << "\tn = " << n << std::endl;
    std::cout << "\trow ptr (size = " << sizeof(rp[0]) << ") = [" << rp[0];
    for (int i = 1; i < (n + 1); i++) {
      std::cout << ", " << rp[i];
    }
    std::cout << "]" << std::endl << "\tcol ind (size = " << sizeof(ci[0]) <<
    ") = [" << ci[0];
    for (int i = 1; i < nz; i++) {
      std::cout << ", " << ci[i];
    }
    std::cout << "]" << std::endl << "\tvals (size = " << sizeof(v[0]) <<
    ") = [" << v[0];
    for (int i = 1; i < nz; i++) {
      std::cout << ", " << v[i];
    }
    std::cout << "]" << std::endl << "\tflags = " << f << std::endl;
  }


  armpl_status_t status_;

  armpl_int_t flags_;

  armpl_int_t n_armpl_;
  armpl_int_t m_armpl;

  T* A_vals_;
  armpl_int_t* A_armpl_row_ptr_;
  armpl_int_t* A_armpl_col_index_;

  armpl_spmat_t A_armpl_;

  armpl_sparse_hint_value trans_ = ARMPL_SPARSE_OPERATION_NOTRANS;

};
}  // namespace cpu
#endif