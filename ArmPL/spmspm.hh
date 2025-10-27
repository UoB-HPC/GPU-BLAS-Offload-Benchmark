#pragma once

#ifdef CPU_ARMPL
#include <stdlib.h>
#include "armpl.h"
#include <omp.h>

#include <algorithm>
#include <iostream>
#include <vector>
#include <utility>

#include "../include/kernels/CPU/spmspm.hh"
#include "../include/utilities.hh"

namespace cpu {
/** A class for sparse matrix-sparse matrix CPU BLAS kernels. */
template <typename T>
class spmspm_cpu : public spmspm<T> {
public:
  using spmspm<T>::spmspm;
  using spmspm<T>::callConsume;
  using spmspm<T>::initInputMatrices;
  using spmspm<T>::m_;
  using spmspm<T>::n_;
  using spmspm<T>::k_;
  using spmspm<T>::sparsity_;
  using spmspm<T>::type_;
  using spmspm<T>::A_nnz_;
  using spmspm<T>::B_nnz_;
  using spmspm<T>::iterations_;
  using spmspm<T>::C_vals_;
  using spmspm<T>::C_nnz_;

  void initialise(int m, int n, int k, double sparsity, matrixType type, 
                  bool binary = false) {
    sparsity_ = sparsity;
    type_ = type;
    
    m_armpl_ = m_ = m;
    n_armpl_ = n_ = n;
    k_armpl_ = k_ = k;

  
    uint64_t total_elements_A = (uint64_t)m_ * (uint64_t)k_;
    uint64_t total_elements_B = (uint64_t)k_ * (uint64_t)n_;
    nnzA_armpl_ = A_nnz_ = 1 + (uint64_t)((double)total_elements_A * (1.0 - sparsity));
    nnzB_armpl_ = B_nnz_ = 1 + (uint64_t)((double)total_elements_B * (1.0 - sparsity));
    
    initInputMatrices();
  }

protected:
  void toSparseFormat() override {
    A_vals_ = (T*)calloc(nnzA_armpl_, sizeof(T));
    A_cols_ = (armpl_int_t*)calloc(nnzA_armpl_, sizeof(armpl_int_t));
    A_rows_ = (armpl_int_t*)calloc(m_armpl_ + 1, sizeof(armpl_int_t));

    B_vals_ = (T*)calloc(nnzB_armpl_, sizeof(T));
    B_cols_ = (armpl_int_t*)calloc(nnzB_armpl_, sizeof(armpl_int_t));
    B_rows_ = (armpl_int_t*)calloc(k_armpl_ + 1, sizeof(armpl_int_t)); 

    int seedOffset = 0;
    do {
      if (type_ == matrixType::rmat) {
        rMatCSR<T, armpl_int_t>(A_vals_, A_cols_, A_rows_, m_armpl_, k_armpl_, nnzA_armpl_, SEED + seedOffset++);
        rMatCSR<T, armpl_int_t>(B_vals_, B_cols_, B_rows_, k_armpl_, n_armpl_, nnzB_armpl_, SEED + seedOffset++);
      } else if (type_ == matrixType::random) {
        randomCSR<T, armpl_int_t>(A_vals_, A_cols_, A_rows_, m_armpl_, k_armpl_, nnzA_armpl_, SEED + seedOffset++);
        randomCSR<T, armpl_int_t>(B_vals_, B_cols_, B_rows_, k_armpl_, n_armpl_, nnzB_armpl_, SEED + seedOffset++);
      } else if (type_ == matrixType::finiteElements) {
        finiteElementCSR<T, armpl_int_t>(A_vals_, A_cols_, A_rows_, m_armpl_, k_armpl_, nnzA_armpl_, SEED + seedOffset++);
        finiteElementCSR<T, armpl_int_t>(B_vals_, B_cols_, B_rows_, k_armpl_, n_armpl_, nnzB_armpl_, SEED + seedOffset++);
      } else {
        std::cerr << "Matrix type not supported" << std::endl;
        exit(1);
      }
    } while (calcCNNZ<armpl_int_t>(m_, A_nnz_, A_rows_, A_cols_, k_, B_nnz_, B_rows_, B_cols_) == 0);

    // Now make the sparse matrix objects
    if constexpr (std::is_same_v<T, float>) {
      status_ = armpl_spmat_create_csr_s(&A_armpl_, 
                                         m_armpl_, 
                                         k_armpl_, 
                                         A_rows_, 
                                         A_cols_,
                                         A_vals_,
                                         0);
      if (status_ != ARMPL_STATUS_SUCCESS) {
        std::cout << "ERROR " << status_ << std::endl;
        exit(1); 
      }
      status_ = armpl_spmat_create_csr_s(&B_armpl_, 
                                         k_armpl_, 
                                         n_armpl_, 
                                         B_rows_, 
                                         B_cols_,
                                         B_vals_,
                                         0);
      if (status_ != ARMPL_STATUS_SUCCESS) {
        std::cout << "ERROR " << status_ << std::endl;
        exit(1); 
      }
    } else if constexpr (std::is_same_v<T, double>) {
      status_ = armpl_spmat_create_csr_d(&A_armpl_, 
                                         m_armpl_, 
                                         k_armpl_, 
                                         A_rows_, 
                                         A_cols_,
                                         A_vals_,
                                         0);
      if (status_ != ARMPL_STATUS_SUCCESS) {
        std::cout << "ERROR " << status_ << std::endl;
        exit(1); 
      }
      status_ = armpl_spmat_create_csr_d(&B_armpl_, 
                                         k_armpl_, 
                                         n_armpl_, 
                                         B_rows_, 
                                         B_cols_,
                                         B_vals_,
                                         0);
      if (status_ != ARMPL_STATUS_SUCCESS) {
        std::cout << "ERROR " << status_ << std::endl;
        exit(1); 
      }
    }
    C_armpl_ = armpl_spmat_create_null(m_armpl_, n_armpl_);
  }

private:
  void preLoopRequirements() override {
    // Populate A and B with hints
    status_ = armpl_spmat_hint(A_armpl_,
                               ARMPL_SPARSE_HINT_MEMORY,
                               ARMPL_SPARSE_MEMORY_NOALLOCS);
    if (status_ != ARMPL_STATUS_SUCCESS) {
      std::cout << "ERROR " << status_ << std::endl;
      exit(1);
    }                  
    status_ = armpl_spmat_hint(A_armpl_,
                               ARMPL_SPARSE_HINT_STRUCTURE,
                               ARMPL_SPARSE_STRUCTURE_UNSTRUCTURED);
    if (status_ != ARMPL_STATUS_SUCCESS) {
      std::cout << "ERROR " << status_ << std::endl;
      exit(1);
    }                  
    status_ = armpl_spmat_hint(A_armpl_,
                               ARMPL_SPARSE_HINT_SPMM_INVOCATIONS,
                               ARMPL_SPARSE_INVOCATIONS_FEW);
    if (status_ != ARMPL_STATUS_SUCCESS) {
      std::cout << "ERROR " << status_ << std::endl;
      exit(1);
    }
    status_ = armpl_spmat_hint(A_armpl_,
                               ARMPL_SPARSE_HINT_SPMM_OPERATION,
                               ARMPL_SPARSE_OPERATION_NOTRANS);
    if (status_ != ARMPL_STATUS_SUCCESS) {
      std::cout << "ERROR " << status_ << std::endl;
      exit(1);
    }
    status_ = armpl_spmat_hint(A_armpl_,
                               ARMPL_SPARSE_HINT_SPMM_STRATEGY,
                               ARMPL_SPARSE_SPMM_STRAT_OPT_FULL_STRUCT);
    if (status_ != ARMPL_STATUS_SUCCESS) {
      std::cout << "ERROR " << status_ << std::endl;
      exit(1);
    }

    status_ = armpl_spmat_hint(B_armpl_,
                               ARMPL_SPARSE_HINT_MEMORY,
                               ARMPL_SPARSE_MEMORY_NOALLOCS);
    if (status_ != ARMPL_STATUS_SUCCESS) {
      std::cout << "ERROR " << status_ << std::endl;
      exit(1);
    }                  
    status_ = armpl_spmat_hint(B_armpl_,
                               ARMPL_SPARSE_HINT_STRUCTURE,
                               ARMPL_SPARSE_STRUCTURE_UNSTRUCTURED);
    if (status_ != ARMPL_STATUS_SUCCESS) {
      std::cout << "ERROR " << status_ << std::endl;
      exit(1);
    }
    status_ = armpl_spmat_hint(B_armpl_,
                               ARMPL_SPARSE_HINT_SPMM_INVOCATIONS,
                               ARMPL_SPARSE_INVOCATIONS_FEW);
    if (status_ != ARMPL_STATUS_SUCCESS) {
      std::cout << "ERROR " << status_ << std::endl;
      exit(1);
    }
    status_ = armpl_spmat_hint(B_armpl_,
                               ARMPL_SPARSE_HINT_SPMM_OPERATION,
                               ARMPL_SPARSE_OPERATION_NOTRANS);
    if (status_ != ARMPL_STATUS_SUCCESS) {
      std::cout << "ERROR " << status_ << std::endl;
      exit(1);
    }
    status_ = armpl_spmat_hint(B_armpl_,
                               ARMPL_SPARSE_HINT_SPMM_STRATEGY,
                               ARMPL_SPARSE_SPMM_STRAT_OPT_FULL_STRUCT);
    if (status_ != ARMPL_STATUS_SUCCESS) {
      std::cout << "ERROR " << status_ << std::endl;
      exit(1);
    }
    status_ = armpl_spmat_hint(B_armpl_,
                               ARMPL_SPARSE_HINT_SPMM_STRATEGY,
                               ARMPL_SPARSE_SPMM_STRAT_OPT_FULL_STRUCT);
    if (status_ != ARMPL_STATUS_SUCCESS) {
      std::cout << "ERROR " << status_ << std::endl;
      exit(1);
    }

    // Call the optimise function to apply hints
    status_ = armpl_spmm_optimize(ARMPL_SPARSE_OPERATION_NOTRANS,
                                  ARMPL_SPARSE_OPERATION_NOTRANS,
                                  ARMPL_SPARSE_SCALAR_ONE,
                                  A_armpl_,
                                  B_armpl_,
                                  ARMPL_SPARSE_SCALAR_ZERO,
                                  C_armpl_);
  }

  void callSpmspm() override{
    if constexpr (std::is_same_v<T, float>) {
      status_ = armpl_spmm_exec_s(ARMPL_SPARSE_OPERATION_NOTRANS,
                                   ARMPL_SPARSE_OPERATION_NOTRANS,
                                   alpha,
                                   A_armpl_,
                                   B_armpl_,
                                   beta,
                                   C_armpl_);
    } else if constexpr (std::is_same_v<T, double>) {
      status_ = armpl_spmm_exec_d(ARMPL_SPARSE_OPERATION_NOTRANS,
                                   ARMPL_SPARSE_OPERATION_NOTRANS,
                                   alpha,
                                   A_armpl_,
                                   B_armpl_,
                                   beta,
                                   C_armpl_);
    } else {
      // Un-specialised class will not do any work - print error and exit.
      std::cout << "ERROR - Datatype for ArmPL CPU SpMSpM kernel not supported." << std::endl;
      exit(1);
    }
    if (status_ != ARMPL_STATUS_SUCCESS) {
      std::cerr << "ERROR: " << status_ << std::endl;
      exit(1); 
    }
  }

  void postLoopRequirements() override {
    // Export the C arrays from the structure
    if constexpr (std::is_same_v<T, float>) {
      status_ = armpl_spmat_export_csr_s(C_armpl_,
                                         0,
                                         &m_armpl_,
                                         &n_armpl_,
                                         &C_rows_,
                                         &C_cols_,
                                         &C_vals_);
    } else if constexpr (std::is_same_v<T, double>) {
      status_ = armpl_spmat_export_csr_d(C_armpl_,
                                         0,
                                         &m_armpl_,
                                         &n_armpl_,
                                         &C_rows_,
                                         &C_cols_,
                                         &C_vals_);
    } else {
      // Un-specialised class will not do any work - print error and exit.
      std::cout << "ERROR - Datatype for ArmPL CPU SpMSpM kernel not supported." << std::endl;
      exit(1);
    }
    if (status_ != ARMPL_STATUS_SUCCESS) {
      std::cerr << "ERROR: " << status_ << std::endl;
      exit(1);
    }
    C_nnz_ = nnzC_armpl_ = C_rows_[m_armpl_];

    // ARMPL does not seem to enforce ordered column indices in its 
    // output matrices.  Therefore, to allow the checksum to take place
    // We have to order the output matrix here.  
    for (int i = 0; i < m_; i++) {
      int start = C_rows_[i];
      int end = C_rows_[i + 1];
      int len = end - start;
      if (len > 1) {
        std::vector<std::pair<armpl_int_t, T>> row_entries(len);
        for (int j = 0; j < len; j++) {
          row_entries[j] = {C_cols_[start + j], C_vals_[start + j]};
        }

        std::sort(row_entries.begin(), row_entries.end(),
                  [](const auto &a, const auto &b) { return a.first < b.first; });

        for (int j = 0; j < len; j++) {
          C_cols_[start + j] = row_entries[j].first;
          C_vals_[start + j] = row_entries[j].second;
        }
      }
    }
  }

  void postCallKernelCleanup() override {
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

    free(A_rows_);
    free(A_cols_);
    free(A_vals_);
    free(B_rows_);
    free(B_cols_);
    free(B_vals_);
    free(C_rows_);
    free(C_cols_);
    free(C_vals_);
  }

  const T alpha = ALPHA;
  const T beta = BETA;


  armpl_status_t status_;

  armpl_int_t n_armpl_;
  armpl_int_t m_armpl_;
  armpl_int_t k_armpl_;
  armpl_int_t nnzA_armpl_;
  armpl_int_t nnzB_armpl_;
  armpl_int_t nnzC_armpl_;

  armpl_int_t* A_cols_;
  armpl_int_t* A_rows_;
  T* A_vals_;

  armpl_int_t* B_rows_;
  armpl_int_t* B_cols_;
  T* B_vals_;

  armpl_int_t* C_rows_;
  armpl_int_t* C_cols_;
  // No C_vals_ needed as inheriting from 
  // parent in order to allow result check to carry out

  armpl_spmat_t A_armpl_;
  armpl_spmat_t B_armpl_;
  armpl_spmat_t C_armpl_;

};
}  // namespace cpu
#endif