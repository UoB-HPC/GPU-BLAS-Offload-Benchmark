#pragma once

#ifdef CPU_AOCL
#include "aoclsparse.h"

#include <algorithm>

#include "../include/kernels/CPU/spmm.hh"
#include "../include/utilities.hh"

namespace cpu {
template <typename T>
class spmm_cpu : public spmm<T> {
public:
    using spmm<T>::spmm;
    using spmm<T>::callConsume;
    using spmm<T>::initInputMatrices;
    using spmm<T>::m_;
    using spmm<T>::n_;
    using spmm<T>::k_;
    using spmm<T>::A_;
    using spmm<T>::B_;
    using spmm<T>::C_;
    using spmm<T>::sparsity_;
    using spmm<T>::nnzA_;
    using spmm<T>::nnzB_;
    using spmm<T>::iterations_;

    void initialise(int m, int n, int k, double sparsity,
                    bool binary = false) {
      m_aocl_ = m_ = m;
      n_aocl_ = n_ = n;
      k_aocl_ = k_ = k;

      uint64_t nnz = 1 + (uint64_t)((double)m_ * (double)k_ * (1.0 - sparsity_));
      nnzA_aocl_ = nnzA_ = nnz;
      nnzB_aocl_ = nnzB_ = nnz;

      A_ = (T*)calloc(m_ * k_, sizeof(T));
      B_ = (T*)calloc(k_ * n_, sizeof(T));
      C_ = (T*)calloc(m_ * n_, sizeof(T));

      base_ = aoclsparse_index_base_zero;
      operationA_ = aoclsparse_operation_none;
      operationB_ = aoclsparse_operation_none;

      status_ = aoclsparse_create_mat_descr(&A_description_);
      if (status_ != aoclsparse_status_success) {
        std::cerr << "aoclsparse_create_mat_descr failing for A" << std::endl;
        printAOCLError(status_);
      }
      status_ = aoclsparse_create_mat_descr(&B_description_);
      if (status_ != aoclsparse_status_success) {
        std::cerr << "aoclsparse_create_mat_descr failing for B" << std::endl;
        printAOCLError(status_);
      }

      initInputMatrices();
    }

protected:
    void toSparseFormat() override {
      // ____ START WITH A ____
      aoclsparse_int actual_nnz = 0;
      for (int i = 0; i < m_ * k_; i++) {
        if (A_[i] != static_cast<T>(0)) {
          actual_nnz++;
        }
      }
    
      if (actual_nnz != nnzA_aocl_) {
        if (print_) std::cerr << "Warning: Actual nnzA (" << actual_nnz << ") differs from expected nnzA (" << nnzA_aocl_ << ")" << std::endl;
        nnzA_ = nnzA_aocl_ = actual_nnz; // Update nnz_aocl_ to reflect actual count
      }

      // Initialise datastructures for the CSR format
      A_rows_ = new aoclsparse_int[m_ + 1];
      A_cols_ = new aoclsparse_int[nnzA_aocl_];
      A_vals_ = new T[nnzA_aocl_];

      // Initialize row pointer with base (zero, as we're using C++)
      A_rows_[0] = 0;
      
      
      // First pass: count non-zeros per row to build row pointer
      for (aoclsparse_int i = 0; i < m_; ++i) {
        aoclsparse_int row_nnz = 0;
        for (aoclsparse_int j = 0; j < k_; ++j) {
          if (A_[i * k_ + j] != static_cast<T>(0)) {
            row_nnz++;
          }
        }
        A_rows_[i + 1] = A_rows_[i] + row_nnz;
      }
    
      
      // Second pass: populate column indices and values
      aoclsparse_int current_val = 0;
      for (aoclsparse_int i = 0; i < m_; ++i) {
        for (aoclsparse_int j = 0; j < k_; ++j) {
          T val = A_[i * k_ + j];
          if (val != static_cast<T>(0)) {
            A_cols_[current_val] = j;  // Adjust for base indexing
            A_vals_[current_val] = val;
            current_val++;
          }
        }
      }
      
      // Optional: Verify CSR format integrity (useful for debugging)
      if (print_) {
          std::cout << "CSR conversion of A complete. Total nnz: " << nnzA_aocl_ << std::endl;
          std::cout << "First few rows: ";
          for (int i = 0; i <= m_; ++i) {
              std::cout << A_rows_[i] << " ";
          }
          std::cout << std::endl;
      }

      // Move into the AOCL CSR matrix handle
      if constexpr (std::is_same_v<T, float>) {
        status_ = aoclsparse_create_scsr(&A_aocl_, 
                                         base_, 
                                         m_aocl_, 
                                         k_aocl_, 
                                         nnzA_aocl_, 
                                         A_rows_, 
                                         A_cols_, 
                                         A_vals_);
      } else if constexpr (std::is_same_v<T, double>) {
        status_ = aoclsparse_create_dcsr(&A_aocl_, 
                                         base_, 
                                         m_aocl_, 
                                         k_aocl_, 
                                         nnzA_aocl_, 
                                         A_rows_, 
                                         A_cols_, 
                                         A_vals_);
      }
      if (status_ != aoclsparse_status_success) {
        std::cerr << "aoclsparse_create_?csr for A is failing with problem size of " << m_ << "x" << k_ << " . " << k_ << "x" << n_ << std::endl;
        printAOCLError(status_);
      }

      // Now sort the matrix -- needed for this AOCL function
      status_ = aoclsparse_order_mat(A_aocl_);
      if (status_ != aoclsparse_status_success) {
        std::cerr << "aoclsparse_order_mat for A is failing with problem size of " << m_ << "x" << k_ << " . " << k_ << "x" << n_ << std::endl;
        printAOCLError(status_);
      }

      // ____ NOW TRANSLATE B ____
      actual_nnz = 0;
      for (int i = 0; i < k_ * n_; i++) {
        if (B_[i] != static_cast<T>(0)) {
          actual_nnz++;
        }
      }
    
      if (actual_nnz != nnzB_aocl_) {
        if (print_) std::cerr << "Warning: Actual nnzB (" << actual_nnz << ") differs from expected nnzB (" << nnzB_aocl_ << ")" << std::endl;
        nnzB_ = nnzB_aocl_ = actual_nnz; // Update nnz_aocl_ to reflect actual count
      }

      // Initialise datastructures for the CSR format
      B_rows_ = new aoclsparse_int[k_ + 1];
      B_cols_ = new aoclsparse_int[nnzB_aocl_];
      B_vals_ = new T[nnzB_aocl_];

      // Initialize row pointer with base (zero, as we're using C++)
      B_rows_[0] = 0;
      
      
      // First pass: count non-zeros per row to build row pointer
      for (aoclsparse_int i = 0; i < k_; ++i) {
        aoclsparse_int row_nnz = 0;
        for (aoclsparse_int j = 0; j < n_; ++j) {
          if (B_[i * n_ + j] != static_cast<T>(0)) {
            row_nnz++;
          }
        }
        B_rows_[i + 1] = B_rows_[i] + row_nnz;
      }
    
      
      // Second pass: populate column indices and values
      current_val = 0;
      for (aoclsparse_int i = 0; i < k_; ++i) {
        for (aoclsparse_int j = 0; j < n_; ++j) {
          T val = A_[i * n_ + j];
          if (val != static_cast<T>(0)) {
            B_cols_[current_val] = j; 
            B_vals_[current_val] = val;
            current_val++;
          }
        }
      }
      
      // Optional: Verify CSR format integrity (useful for debugging)
      if (print_) {
          std::cout << "CSR conversion of B complete. Total nnz: " << nnzB_aocl_ << std::endl;
          std::cout << "First few rows: ";
          for (int i = 0; i <= k_; ++i) {
              std::cout << B_rows_[i] << " ";
          }
          std::cout << std::endl;
      }

      // Move into the AOCL CSR matrix handle
      if constexpr (std::is_same_v<T, float>) {
        status_ = aoclsparse_create_scsr(&B_aocl_, 
                                         base_, 
                                         k_aocl_, 
                                         n_aocl_, 
                                         nnzB_aocl_, 
                                         B_rows_, 
                                         B_cols_, 
                                         B_vals_);
      } else if constexpr (std::is_same_v<T, double>) {
        status_ = aoclsparse_create_dcsr(&B_aocl_, 
                                         base_, 
                                         k_aocl_, 
                                         n_aocl_, 
                                         nnzB_aocl_, 
                                         B_rows_, 
                                         B_cols_, 
                                         B_vals_);
      }
      if (status_ != aoclsparse_status_success) {
        std::cerr << "aoclsparse_create_?csr for B is failing with problem size of " << m_ << "x" << k_ << " . " << k_ << "x" << n_ << std::endl;
        printAOCLError(status_);
      }

      // Now sort the matrix -- needed for this AOCL function
      status_ = aoclsparse_order_mat(B_aocl_);
      if (status_ != aoclsparse_status_success) {
        std::cerr << "aoclsparse_order_mat for B is failing with problem size of " << m_ << "x" << k_ << " . " << k_ << "x" << n_ << std::endl;
        printAOCLError(status_);
      }
    }

private:
    void preLoopRequirements() override {
    }

    void callSpmm() override {
      request_ = aoclsparse_stage_nnz_count;
      status_ = aoclsparse_sp2m(operationA_, 
                                A_description_, 
                                A_aocl_, 
                                operationB_, 
                                B_description_, 
                                B_aocl_, 
                                request_, 
                                &C_aocl_);
      if (status_ != aoclsparse_status_success) {
        std::cerr << "aoclsparse_sp2m failing with request = aoclsparse_stage_nnz_count" << std::endl;
        printAOCLError(status_);
      }

      request_ = aoclsparse_stage_finalize;
      status_ = aoclsparse_sp2m(operationA_, 
                                A_description_, 
                                A_aocl_, 
                                operationB_, 
                                B_description_, 
                                B_aocl_, 
                                request_, 
                                &C_aocl_);
      if (status_ != aoclsparse_status_success) {
        std::cerr << "aoclsparse_sp2m failing with request = aoclsparse_stage_finalize" << std::endl;
        printAOCLError(status_);
      }

      status_ = aoclsparse_export_zcsr(C_aocl_, 
                                       &base_,
                                       &C_M,
                                       &C_N,
                                       &nnzC_aocl_,
                                       &C_rows_,
                                       &C_cols_,
                                       &C_vals_);
      if (status_ != aoclsparse_status_success) {
        std::cerr << "aoclsparse_export_zcsr failing" << std::endl;
        printAOCLError(status_);
      }
    }

    void postLoopRequirements() override {
    }

    void postCallKernelCleanup() override {
      status_ = aoclsparse_destroy_mat_descr(A_description_);
      if (status_ != aoclsparse_status_success) {
        std::cerr << "aoclsparse_destroy_mat_descr failing for A_description_" << std::endl;
        printAOCLError(status_);
      }
      status_ = aoclsparse_destroy_mat_descr(B_description_);
      if (status_ != aoclsparse_status_success) {
        std::cerr << "aoclsparse_destroy_mat_descr failing for B_description_" << std::endl;
        printAOCLError(status_);
      }

      status_ = aoclsparse_destroy(&A_aocl_);
      if (status_ != aoclsparse_status_success) {
        std::cerr << "aoclsparse_destroy failing for A" << std::endl;
        printAOCLError(status_);
      }
      status_ = aoclsparse_destroy(&B_aocl_);
      if (status_ != aoclsparse_status_success) {
        std::cerr << "aoclsparse_destroy failing for B" << std::endl;
        printAOCLError(status_);
      }
      status_ = aoclsparse_destroy(&C_aocl_);
      if (status_ != aoclsparse_status_success) {
        std::cerr << "aoclsparse_destroy failing for C" << std::endl;
        printAOCLError(status_);
      }
    }

    void printAOCLError(aoclsparse_status stat) {
      switch (stat) {
        case aoclsparse_status_success:
          std::cerr << "SUCCESS - The operation completed successfully";
          break;
        case aoclsparse_status_not_implemented:
          std::cerr << "NOT_IMPLEMENTED - The requested functionality is not yet implemented in this version";
          break;
        case aoclsparse_status_invalid_pointer:
          std::cerr << "INVALID_POINTER - One or more pointer parameters are NULL or otherwise invalid";
          break;
        case aoclsparse_status_invalid_size:
          std::cerr << "INVALID_SIZE - One or more size parameters (m, n, nnz, etc.) contain an invalid value (e.g., negative or zero where positive required)";
          break;
        case aoclsparse_status_internal_error:
          std::cerr << "INTERNAL_ERROR - Internal library failure";
          break;
        case aoclsparse_status_invalid_value:
          std::cerr << "INVALID_VALUE - Input parameters contain an invalid value (e.g., invalid enum value, base index neither 0 nor 1)";
          break;
        case aoclsparse_status_invalid_index_value:
          std::cerr << "INVALID_INDEX_VALUE - At least one index value is invalid (e.g., negative or out of bounds)";
          break;
        case aoclsparse_status_maxit:
          std::cerr << "MAXIT - function stopped after reaching number of iteration limit";
          break;
        case aoclsparse_status_user_stop:
          std::cerr << "USER_STOP - user requested termination";
          break;
        case aoclsparse_status_wrong_type:
          std::cerr << "WRONG_TYPE - Data type mismatch (e.g., matrix datatypes don't match between operations)";
          break;
        case aoclsparse_status_memory_error:
          std::cerr << "MEMORY_ERROR - memory allocation failure";
          break;
        case aoclsparse_status_numerical_error:
          std::cerr << "NUMERICAL_ERROR - numerical error, e.g., matrix is not positive definite, devide-by-zero error";
          break;
        case aoclsparse_status_invalid_operation:
          std::cerr << "INVALID_OPERATION - cannot proceed with the request at this point";
          break;
        case aoclsparse_status_unsorted_input:
          std::cerr << "UNSORTED_INPUT - the input matrices are not sorted";
          break;
        case aoclsparse_status::aoclsparse_status_invalid_kid:
          std::cerr << "INVALID_KID - user requested kernel id was not available";
          break;
        default:
          std::cerr << "UNKNOWN_STATUS - Unrecognized status code (" + std::to_string(stat) + ")";
          break;
      }
      std::cerr << std::endl;
      exit(1);
    }

    bool print_ = false;

    aoclsparse_status status_;

    aoclsparse_operation operationA_;
    aoclsparse_operation operationB_;
    aoclsparse_index_base base_;
    aoclsparse_request request_;

    aoclsparse_matrix A_aocl_;
    aoclsparse_int* A_rows_;
    aoclsparse_int* A_cols_;
    T* A_vals_;

    aoclsparse_matrix B_aocl_;
    aoclsparse_int* B_rows_;
    aoclsparse_int* B_cols_;
    T* B_vals_;

    aoclsparse_matrix C_aocl_;
    aoclsparse_int* C_rows_;
    aoclsparse_int* C_cols_;
    T* C_vals_;

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
