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
  using spgemv<T>::initInputMatrixVector;
  using spgemv<T>::m_;
  using spgemv<T>::n_;
  using spgemv<T>::x_;
  using spgemv<T>::y_;
  using spgemv<T>::sparsity_;
  using spgemv<T>::type_;
  using spgemv<T>::nnz_;
  using spgemv<T>::iterations_;

  /** Initialise the required data structures. */
  void initialise(int m, int n, double sparsity, matrixType type, 
                    bool binary = false) {
    m_armpl_ = m_ = m;
    n_armpl_ = n_ = n;
    sparsity_ = sparsity;
    type_ = type;

    nnz_armpl_ = nnz_ = 1 + (uint64_t)((double)m_ * (double)n_ * (1.0 - sparsity_));

    x_ = (T*)calloc(n_, sizeof(T));
    y_ = (T*)calloc(m_, sizeof(T));

    // Initialise the matrix and vectors
    initInputMatrixVector();
  }

protected:
  void toSparseFormat() override {
    // Make arrays for A
    A_vals_ = (T*)calloc(nnz_armpl_, sizeof(T));
    A_cols_ = (armpl_int_t*)calloc(nnz_armpl_, sizeof(armpl_int_t));
    A_rows_ = (armpl_int_t*)calloc(m_ + 1, sizeof(armpl_int_t));

    // Fill the CSR arrays
    if (type_ == matrixType::rmat) {
      rMatCSR<T, armpl_int_t>(A_vals_, A_cols_, A_rows_, m_armpl_, n_armpl_, nnz_);
    } else if (type_ == matrixType::random) {
      randomCSR<T, armpl_int_t>(A_vals_, A_cols_, A_rows_, m_armpl_, n_armpl_, nnz_);
    } else if (type_ == matrixType::finiteElements) {
      finiteElementCSR<T, armpl_int_t>(A_vals_, A_cols_, A_rows_, m_armpl_, n_armpl_, nnz_);
    } else {
      std::cerr << "Matrix type not supported" << std::endl;
      exit(1);
    }

    // Create the armpl object for this sparse matrix
    if constexpr (std::is_same_v<T, float>) {
      status_ = armpl_spmat_create_csr_s(&A_armpl_, 
                                         m_armpl_, 
                                         n_armpl_, 
                                         A_rows_, 
                                         A_cols_,
                                         A_vals_,
                                         0);
    } else if constexpr (std::is_same_v<T, double>) {
      status_ = armpl_spmat_create_csr_d(&A_armpl_, 
                                         m_armpl_, 
                                         n_armpl_, 
                                         A_rows_, 
                                         A_cols_,
                                         A_vals_,
                                         0);
    } else {
      // Un-specialised class will not do any work - print error and exit.
      std::cerr << "ERROR - Datatype for ArmPL CPU SpGEMV kernel not supported." << std::endl;
      exit(1);
    }
    if (status_ != ARMPL_STATUS_SUCCESS) {
      std::cerr << "ERROR " << status_ << std::endl;
      exit(1);
    }
  }

private:/** Perform any required steps before calling the SpGEMV kernel that should
   * be timed. */
  void preLoopRequirements() override {
    // Give the library some hints so it can optimise the performance of the kernel
    status_ = armpl_spmat_hint(A_armpl_,
                               ARMPL_SPARSE_HINT_MEMORY,
                               ARMPL_SPARSE_MEMORY_NOALLOCS);
    if (status_ != ARMPL_STATUS_SUCCESS) {
      std::cerr << "ERROR " << status_ << std::endl;
      exit(1);
    }                  

    status_ = armpl_spmat_hint(A_armpl_,
                               ARMPL_SPARSE_HINT_STRUCTURE,
                               ARMPL_SPARSE_STRUCTURE_UNSTRUCTURED);
    if (status_ != ARMPL_STATUS_SUCCESS) {
      std::cerr << "ERROR " << status_ << std::endl;
      exit(1);
    }

    status_ = armpl_spmat_hint(A_armpl_,
                               ARMPL_SPARSE_HINT_SPMV_INVOCATIONS,
                               ARMPL_SPARSE_INVOCATIONS_FEW);
    if (status_ != ARMPL_STATUS_SUCCESS) {
      std::cerr << "ERROR " << status_ << std::endl;
      exit(1);
    }

    status_ = armpl_spmat_hint(A_armpl_,
                               ARMPL_SPARSE_HINT_SPMV_OPERATION,
                               ARMPL_SPARSE_OPERATION_NOTRANS);
    if (status_ != ARMPL_STATUS_SUCCESS) {
      std::cerr << "ERROR " << status_ << std::endl;
      exit(1);
    }

    // Now optimise the matrix for SpMV based on the hints given
    status_ = armpl_spmv_optimize(A_armpl_);
    if (status_ != ARMPL_STATUS_SUCCESS) {
      std::cerr << "ERROR " << status_ << std::endl;
      exit(1);
    }
  }

  /** Make call to the SpGEMV kernel. */
  void callSpgemv() override {
    if constexpr (std::is_same_v<T, float>) {
      status_ = armpl_spmv_exec_s(ARMPL_SPARSE_OPERATION_NOTRANS,
                                  alpha,
                                  A_armpl_,
                                  x_,
                                  beta,
                                  y_);
    } else if constexpr (std::is_same_v<T, double>) {
      status_ = armpl_spmv_exec_d(ARMPL_SPARSE_OPERATION_NOTRANS,
                                  alpha,
                                  A_armpl_,
                                  x_,
                                  beta,
                                  y_);
    } else {
      // Un-specialised class will not do any work - print error and exit.
      std::cerr << "ERROR - Datatype for ArmPL CPU GEMV kernel not supported." << std::endl;
      exit(1);
    }
    if (status_ != ARMPL_STATUS_SUCCESS) {
      std::cerr << "ERROR: " << status_ << std::endl;
      exit(1); 
    }

    // Ensure compiler doesn't optimise away the work being done
    callConsume();
  }

  

  /** Perform any required steps after calling the SpGEMV kernel that should
   * be timed. */
  void postLoopRequirements() override {}

  void postCallKernelCleanup() override {
    status_ = armpl_spmat_destroy(A_armpl_);
    if (status_ != ARMPL_STATUS_SUCCESS) {
      std::cerr << "ERROR " << status_ << std::endl;
      exit(1);
    }

    free(A_rows_);
    free(A_cols_);
    free(A_vals_);
    free(x_);
    free(y_);
  }

  armpl_status_t status_;

  armpl_int_t n_armpl_;
  armpl_int_t m_armpl_;
  armpl_int_t nnz_armpl_;

  T* A_vals_;
  armpl_int_t* A_rows_;
  armpl_int_t* A_cols_;

  armpl_spmat_t A_armpl_;

  const T alpha = ALPHA;
  const T beta = BETA;
};
}  // namespace cpu
#endif