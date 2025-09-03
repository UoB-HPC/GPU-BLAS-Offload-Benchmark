#pragma once

#ifdef CPU_AOCL
#include "aoclsparse.h"

#include <algorithm>
#include <thread>
#include <chrono>

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
    using spmm<T>::sparsity_;
    using spmm<T>::type_;
    using spmm<T>::A_nnz_;
    using spmm<T>::B_nnz_;
    using spmm<T>::iterations_;
    using spmm<T>::C_rows_;
    using spmm<T>::C_cols_;
    using spmm<T>::C_vals_;
    using spmm<T>::C_nnz_;

    void initialise(int m, int n, int k, double sparsity, matrixType type, 
                    bool binary = false) {
      if (print_) {
          std::cout << "===========  CPU   ===========" << std::endl;
      }

      if (print_) std::cout << "setting up metadata" << std::endl;
      
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
      
      if (print_) std::cout << "Allocating dense matrix arrays" << std::endl;

      base_ = aoclsparse_index_base_zero;
      operationA_ = aoclsparse_operation_none;
      operationB_ = aoclsparse_operation_none;

      if (print_) std::cout << "Creating AOCL description for A" << std::endl;
      status_ = aoclsparse_create_mat_descr(&A_description_);
      if (status_ != aoclsparse_status_success) {
        std::cerr << "aoclsparse_create_mat_descr failing for A" << std::endl;
        printAOCLError(status_);
      }
      if (print_) std::cout << "Creating AOCL description for B" << std::endl;
      status_ = aoclsparse_create_mat_descr(&B_description_);
      if (status_ != aoclsparse_status_success) {
        std::cerr << "aoclsparse_create_mat_descr failing for B" << std::endl;
        printAOCLError(status_);
      }
      if (print_) std::cout << "initialising matrices" << std::endl;
      initInputMatrices();

      if (print_) {
        std::cout << "===============Initialised=================" << std::endl;
        std::cout << "___________________________________________" << std::endl;
        std::cout << "A nnz = " << A_nnz_ << std::endl;
        std::cout << "A rows = [";
        for (int64_t i = 0; i < (m_ + 1); i++) {
          std::cout << A_rows_[i];
          if (i < m_) std::cout << ", ";
        }
        std::cout << "]" << std::endl;
        std::cout << "A cols = [";
        for (int64_t i = 0; i < A_nnz_; i++) {
          std::cout << A_cols_[i];
          if (i < (A_nnz_ - 1)) std::cout << ", ";
        }
        std::cout << "]" << std::endl;
        std::cout << "A vals = [";
        for (int64_t i = 0; i < A_nnz_; i++) {
          std::cout << A_vals_[i];
          if (i < (A_nnz_ - 1)) std::cout << ", ";
        }
        std::cout << "]" << std::endl;
        
        std::cout << "B nnz = " << B_nnz_ << std::endl;
        std::cout << "B rows = [";
        for (int64_t i = 0; i < (k_ + 1); i++) {
          std::cout << B_rows_[i];
          if (i < k_) std::cout << ", ";
        }
        std::cout << "]" << std::endl;
        std::cout << "B cols = [";
        for (int64_t i = 0; i < B_nnz_; i++) {
          std::cout << B_cols_[i];
          if (i < (B_nnz_ - 1)) std::cout << ", ";
        }
        std::cout << "]" << std::endl;
        std::cout << "B vals = [";
        for (int64_t i = 0; i < B_nnz_; i++) {
          std::cout << B_vals_[i];
          if (i < (B_nnz_ - 1)) std::cout << ", ";
        }
        std::cout << "]" << std::endl << std::endl;
        std::cout << "___________________________________________" << std::endl;
      }
    }

protected:
    void toSparseFormat() override {
      if (print_) std::cout << "Allocating CSR Arrays for A" << std::endl;
      A_rows_ = (aoclsparse_int*)calloc(m_ + 1, sizeof(aoclsparse_int));
      A_cols_ = (aoclsparse_int*)calloc(nnzA_aocl_, sizeof(aoclsparse_int));
      A_vals_ = (T*)calloc(nnzA_aocl_, sizeof(T));
      if (A_rows_ == nullptr || A_cols_ == nullptr || A_vals_ == nullptr) {
        std::cerr << "Failed to allocate memory for A CSR arrays with problem size of " << m_ << "x" << k_ << " . " << k_ << "x" << n_ << std::endl;
        exit(1);
      }

      if (print_) std::cout << "Generating R-MAT matrix for A" << std::endl;
      if (type_ == matrixType::rmat){
        rMatCSR<T, aoclsparse_int>(A_vals_, A_cols_, A_rows_, m_, k_, nnzA_aocl_);
      } else if (type_ == matrixType::random) {
        randomCSR<T, aoclsparse_int>(A_vals_, A_cols_, A_rows_, m_, k_, nnzA_aocl_);
      } else {
        std::cerr << "Matrix type not supported" << std::endl;
        exit(1);
      }

      // Move into the AOCL CSR matrix handle
      if (print_) std::cout << "Creating AOCL CSR matrix for A" << std::endl;
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
      if (print_) std::cout << "Sorting AOCL CSR matrix for A" << std::endl;
      status_ = aoclsparse_order_mat(A_aocl_);
      if (status_ != aoclsparse_status_success) {
        std::cerr << "aoclsparse_order_mat for A is failing with problem size of " << m_ << "x" << k_ << " . " << k_ << "x" << n_ << std::endl;
        printAOCLError(status_);
      }

      // Initialise datastructures for the CSR format
      if (print_) std::cout << "Allocating CSR Arrays for B" << std::endl;
      B_rows_ = (aoclsparse_int*)calloc(k_ + 1, sizeof(aoclsparse_int));
      B_cols_ = (aoclsparse_int*)calloc(nnzB_aocl_, sizeof(aoclsparse_int));
      B_vals_ = (T*)calloc(nnzB_aocl_, sizeof(T)); 
      if (B_rows_ == nullptr || B_cols_ == nullptr || B_vals_ == nullptr) {
        std::cerr << "Failed to allocate memory for B CSR arrays with problem size of " << m_ << "x" << k_ << " . " << k_ << "x" << n_ << std::endl;
        exit(1);
      }

      if (print_) std::cout << "Generating R-MAT matrix for B" << std::endl;
      if (type_ == matrixType::rmat){
        rMatCSR<T, aoclsparse_int>(B_vals_, B_cols_, B_rows_, k_, n_, nnzB_aocl_, true);
      } else if (type_ == matrixType::random) {
        randomCSR<T, aoclsparse_int>(B_vals_, B_cols_, B_rows_, k_, n_, nnzB_aocl_, true);
      } else {
        std::cerr << "Matrix type not supported" << std::endl;
        exit(1);
      }
      
      // Move into the AOCL CSR matrix handle
      if (print_) std::cout << "Creating AOCL CSR matrix for B" << std::endl;
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
      if (print_) std::cout << "Sorting AOCL CSR matrix for B" << std::endl;
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
      if (C_allocated) {
        if (print_) std::cout << "Freeing C arrays" << std::endl;
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
      if (print_) std::cout << "Calculating number of non-zeros in C" << std::endl;
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
      if (print_) std::cout << "SpMM compute" << std::endl;
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

      if (print_) std::cout << "Exporting CSR arrays for C" << std::endl;
      if constexpr (std::is_same_v<T, float>) {
        status_ = aoclsparse_export_scsr(C_aocl_, 
                                         &base_,
                                         &C_M,
                                         &C_N,
                                         &nnzC_aocl_,
                                         &C_rows_aocl_,
                                         &C_cols_aocl_,
                                         &C_vals_);
      } else if constexpr (std::is_same_v<T, double>) {
        status_ = aoclsparse_export_dcsr(C_aocl_, 
                                         &base_,
                                         &C_M,
                                         &C_N,
                                         &nnzC_aocl_,
                                         &C_rows_aocl_,
                                         &C_cols_aocl_,
                                         &C_vals_);
      }
      if (status_ != aoclsparse_status_success) {
        std::cerr << "aoclsparse_export_zcsr failing" << std::endl;
        printAOCLError(status_);
      }
      C_allocated = true;
    }

    void postLoopRequirements() override {
      C_nnz_ = nnzC_aocl_; // Needed for checksum
    }

    void postCallKernelCleanup() override {
      if (print_) std::cout << "Destroying A description" << std::endl;
      status_ = aoclsparse_destroy_mat_descr(A_description_);
      if (status_ != aoclsparse_status_success) {
        std::cerr << "aoclsparse_destroy_mat_descr failing for A_description_" << std::endl;
        printAOCLError(status_);
      }
      if (print_) std::cout << "Destroying B description" << std::endl;
      status_ = aoclsparse_destroy_mat_descr(B_description_);
      if (status_ != aoclsparse_status_success) {
        std::cerr << "aoclsparse_destroy_mat_descr failing for B_description_" << std::endl;
        printAOCLError(status_);
      }

      if (print_) std::cout << "Destroying A aocl matrix handle" << std::endl;
      status_ = aoclsparse_destroy(&A_aocl_);
      if (status_ != aoclsparse_status_success) {
        std::cerr << "aoclsparse_destroy failing for A" << std::endl;
        printAOCLError(status_);
      }
      if (print_) std::cout << "Destroying B aocl matrix handle" << std::endl;
      status_ = aoclsparse_destroy(&B_aocl_);
      if (status_ != aoclsparse_status_success) {
        std::cerr << "aoclsparse_destroy failing for B" << std::endl;
        printAOCLError(status_);
      }

      if (print_) std::cout << "Destroying C aocl matrix handle" << std::endl;
      status_ = aoclsparse_destroy(&C_aocl_);
      if (status_ != aoclsparse_status_success) {
        std::cerr << "aoclsparse_destroy failing for C" << std::endl;
        printAOCLError(status_);
      }
      if (print_) std::cout << "Freeing CSR arrays for A" << std::endl;
      free(A_rows_);
      free(A_cols_);
      free(A_vals_);
      if (print_) std::cout << "Freeing CSR arrays for B" << std::endl;
      free(B_rows_);
      free(B_cols_);
      free(B_vals_);
      if (print_) std::cout << "AOCL cleanup complete" << std::endl;
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
