#pragma once

#ifdef CPU_AOCL
#include "aoclsparse.h"

#include <algorithm>
#include <thread>
#include <chrono>

#include "./common.hh"
#include "../include/kernels/CPU/spgemm.hh"
#include "../include/utilities.hh"

namespace cpu {
template <typename T>
class spgemm_cpu : public spgemm<T> {
public:
    using spgemm<T>::spgemm;
    using spgemm<T>::callConsume;
    using spgemm<T>::initInputMatrices;
    using spgemm<T>::m_;
    using spgemm<T>::n_;
    using spgemm<T>::k_;
    using spgemm<T>::sparsity_;
    using spgemm<T>::type_;
    using spgemm<T>::A_nnz_;
    using spgemm<T>::B_nnz_;
    using spgemm<T>::iterations_;
    using spgemm<T>::C_rows_;
    using spgemm<T>::C_cols_;
    using spgemm<T>::C_vals_;
    using spgemm<T>::C_nnz_;

    void initialise(int m, int n, int k, double sparsity, matrixType type, 
                    bool binary = false) {      
      sparsity_ = sparsity;
      type_ = type;
      
      m_aocl_ = m_ = m;
      n_aocl_ = n_ = n;
      k_aocl_ = k_ = k;

      uint64_t total_elements_A = (uint64_t)m_ * (uint64_t)k_;
      uint64_t total_elements_B = (uint64_t)k_ * (uint64_t)n_;
      nnzA_aocl_ = A_nnz_ = 1 + (uint64_t)((double)total_elements_A * (1.0 - sparsity));
      nnzB_aocl_ = B_nnz_ = 1 + (uint64_t)((double)total_elements_B * (1.0 - sparsity));
      C_allocated = false;
      
      base_ = aoclsparse_index_base_zero;
      operationA_ = aoclsparse_operation_none;
      operationB_ = aoclsparse_operation_none;

      aoclCheckError(aoclsparse_create_mat_descr(&A_description_));

      aoclCheckError(aoclsparse_create_mat_descr(&B_description_));
      
      initInputMatrices();
    }

protected:
    void toSparseFormat() override {
      A_rows_ = (aoclsparse_int*)calloc(m_ + 1, sizeof(aoclsparse_int));
      A_cols_ = (aoclsparse_int*)calloc(nnzA_aocl_, sizeof(aoclsparse_int));
      A_vals_ = (T*)calloc(nnzA_aocl_, sizeof(T));
      if (A_rows_ == nullptr || A_cols_ == nullptr || A_vals_ == nullptr) {
        std::cerr << "Failed to allocate memory for A CSR arrays with problem size of " << m_ << "x" << k_ << " . " << k_ << "x" << n_ << std::endl;
        exit(1);
      }

      // Initialise datastructures for the CSR format
      B_rows_ = (aoclsparse_int*)calloc(k_ + 1, sizeof(aoclsparse_int));
      B_cols_ = (aoclsparse_int*)calloc(nnzB_aocl_, sizeof(aoclsparse_int));
      B_vals_ = (T*)calloc(nnzB_aocl_, sizeof(T)); 
      if (B_rows_ == nullptr || B_cols_ == nullptr || B_vals_ == nullptr) {
        std::cerr << "Failed to allocate memory for B CSR arrays with problem size of " << m_ << "x" << k_ << " . " << k_ << "x" << n_ << std::endl;
        exit(1);
      }

      int seedOffset = 0;
      do {
        if (type_ == matrixType::rmat) {
          rMatCSR<T, aoclsparse_int>(A_vals_, A_cols_, A_rows_, m_, k_, A_nnz_, SEED + seedOffset++);
          rMatCSR<T, aoclsparse_int>(B_vals_, B_cols_, B_rows_, k_, n_, B_nnz_, SEED + seedOffset++);
        } else if (type_ == matrixType::random) {
          randomCSR<T, aoclsparse_int>(A_vals_, A_cols_, A_rows_, m_, k_, A_nnz_, SEED + seedOffset++);
          randomCSR<T, aoclsparse_int>(B_vals_, B_cols_, B_rows_, k_, n_, B_nnz_, SEED + seedOffset++);
        } else if (type_ == matrixType::bandedDiagonal) {
          bandedDiagonalCSR<T, aoclsparse_int>(A_vals_, A_cols_, A_rows_, m_, k_, A_nnz_, SEED + seedOffset++);
          bandedDiagonalCSR<T, aoclsparse_int>(B_vals_, B_cols_, B_rows_, k_, n_, B_nnz_, SEED + seedOffset++);
        } else {
          std::cerr << "Matrix type not supported" << std::endl;
          exit(1);
        } 
      } while (calcCNNZ<aoclsparse_int>(m_, A_nnz_, A_rows_, A_cols_, k_, B_nnz_, B_rows_, B_cols_) == 0);

      // Move into the AOCL CSR matrix handle
      if constexpr (std::is_same_v<T, float>) {
        aoclCheckError(aoclsparse_create_scsr(&A_aocl_, 
                                              base_, 
                                              m_aocl_, 
                                              k_aocl_, 
                                              nnzA_aocl_, 
                                              A_rows_, 
                                              A_cols_, 
                                              A_vals_));
      } else if constexpr (std::is_same_v<T, double>) {
        aoclCheckError(aoclsparse_create_dcsr(&A_aocl_, 
                                              base_, 
                                              m_aocl_, 
                                              k_aocl_, 
                                              nnzA_aocl_, 
                                              A_rows_, 
                                              A_cols_, 
                                              A_vals_));
      }

      // Now sort the matrix -- needed for this AOCL function
      aoclCheckError(aoclsparse_order_mat(A_aocl_);
      
      // Move into the AOCL CSR matrix handle
      if constexpr (std::is_same_v<T, float>) {
        aoclCheckError(aoclsparse_create_scsr(&B_aocl_, 
                                              base_, 
                                              k_aocl_, 
                                              n_aocl_, 
                                              nnzB_aocl_, 
                                              B_rows_, 
                                              B_cols_, 
                                              B_vals_));
      } else if constexpr (std::is_same_v<T, double>) {
        aoclCheckError(aoclsparse_create_dcsr(&B_aocl_, 
                                              base_, 
                                              k_aocl_, 
                                              n_aocl_, 
                                              nnzB_aocl_, 
                                              B_rows_, 
                                              B_cols_, 
                                              B_vals_));
      }

      // Now sort the matrix -- needed for this AOCL function
      aoclCheckError(aoclsparse_order_mat(B_aocl_));
    }

private:
    void preLoopRequirements() override {}

    void callSpgemm() override {
      if (C_allocated) {
        if (C_vals_ != nullptr) {
          free(C_vals_);
          C_vals_ = nullptr;
        }
        if (C_cols_aocl_ != nullptr) {
          free(C_cols_aocl_);
          C_cols_aocl_ = nullptr;
        }
        if (C_rows_aocl_ != nullptr) {
          free(C_rows_aocl_);
          C_rows_aocl_ = nullptr;
        }
        C_allocated = false;
      }

      request_ = aoclsparse_stage_nnz_count;
      aoclCheckError(aoclsparse_sp2m(operationA_, 
                                     A_description_, 
                                     A_aocl_, 
                                     operationB_, 
                                     B_description_, 
                                     B_aocl_, 
                                     request_, 
                                     &C_aocl_));

      request_ = aoclsparse_stage_finalize;
      aoclCheckError(aoclsparse_sp2m(operationA_, 
                                     A_description_, 
                                     A_aocl_, 
                                     operationB_, 
                                     B_description_, 
                                     B_aocl_, 
                                     request_, 
                                     &C_aocl_));

      if constexpr (std::is_same_v<T, float>) {
        aoclCheckError(aoclsparse_export_scsr(C_aocl_, 
                                              &base_,
                                              &C_M,
                                              &C_N,
                                              &nnzC_aocl_,
                                              &C_rows_aocl_,
                                              &C_cols_aocl_,
                                              &C_vals_));
      } else if constexpr (std::is_same_v<T, double>) {
        aoclCheckError(aoclsparse_export_dcsr(C_aocl_, 
                                              &base_,
                                              &C_M,
                                              &C_N,
                                              &nnzC_aocl_,
                                              &C_rows_aocl_,
                                              &C_cols_aocl_,
                                              &C_vals_));
      }
      C_allocated = true;
    }

    void postLoopRequirements() override {
      C_nnz_ = nnzC_aocl_; // Needed for checksum
    }

    void postCallKernelCleanup() override {
      aoclCheckError(aoclsparse_destroy_mat_descr(A_description_));
      aoclCheckError(aoclsparse_destroy_mat_descr(B_description_));

      aoclCheckError(aoclsparse_destroy(&A_aocl_));
      aoclCheckError(aoclsparse_destroy(&B_aocl_));
      aoclCheckError(aoclsparse_destroy(&C_aocl_));
      
      free(A_rows_);
      free(A_cols_);
      free(A_vals_);
      free(B_rows_);
      free(B_cols_);
      free(B_vals_);
    }


    aoclsparse_status status_;

    aoclsparse_operation operationA_;
    aoclsparse_operation operationB_;
    aoclsparse_index_base base_;
    aoclsparse_request request_;

    aoclsparse_matrix A_aocl_;
    aoclsparse_int* A_rows_ = nullptr;
    aoclsparse_int* A_cols_ = nullptr;
    T* A_vals_ = nullptr;

    aoclsparse_matrix B_aocl_;
    aoclsparse_int* B_rows_ = nullptr;
    aoclsparse_int* B_cols_ = nullptr;
    T* B_vals_ = nullptr;

    aoclsparse_matrix C_aocl_;
    aoclsparse_int* C_rows_aocl_ = nullptr;
    aoclsparse_int* C_cols_aocl_ = nullptr;
    bool C_allocated = false;

    aoclsparse_int m_aocl_;
    aoclsparse_int n_aocl_;
    aoclsparse_int k_aocl_;
    aoclsparse_int nnzA_aocl_;
    aoclsparse_int nnzB_aocl_;
    aoclsparse_int nnzC_aocl_;

    aoclsparse_int C_M, C_N;

    aoclsparse_mat_descr A_description_;
    aoclsparse_mat_descr B_description_;

    const T alpha = ALPHA;
    const T beta = BETA;
};
}


#endif