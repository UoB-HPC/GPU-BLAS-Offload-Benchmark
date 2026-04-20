#pragma once

#ifdef CPU_AOCL
#include "aoclsparse.h"

#include <algorithm>
#include <vector>

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
  using spmm<T>::B_;
  using spmm<T>::C_;
  using spmm<T>::sparsity_;
  using spmm<T>::type_;
  using spmm<T>::nnz_;
  using spmm<T>::iterations_;

  void initialise(int m, int n, int k, double sparsity,
                  matrixType type, bool binary = false) {
    base_ = aoclsparse_index_base_zero;
    order_ = aoclsparse_order_row;

    status_ = aoclsparse_create_mat_descr(&A_description_);
    if (status_ != aoclsparse_status_success) {
      std::cerr << "aoclsparse_create_mat_descr is failing with problem size of " << m_ << "x" << k_ << " . " << k_ << "x" << n_ << std::endl;
      printAOCLError(status_);
    }

    status_ = aoclsparse_set_mat_index_base(A_description_, base_);
    if (status_ != aoclsparse_status_success) {
      std::cerr << "aoclsparse_set_mat_index_base is failing with problem size of " << m_ << "x" << k_ << " . " << k_ << "x" << n_ << std::endl;
      printAOCLError(status_);
    }
    

    m_aocl_ = m_ = m;
    n_aocl_ = n_ = n;
    k_aocl_ = k_ = k;
    sparsity_ = sparsity;
    type_ = type;

    nnz_ = 1 + (uint64_t)((double)m_ * (double)k_ * (1.0 - sparsity_));
    nnz_aocl_ = nnz_;

    B_ = (T*)calloc(k_ * n_, sizeof(T));
    C_ = (T*)calloc(m_ * n_, sizeof(T));

    initInputMatrices();
  }

protected:
  void toSparseFormat() override {
  
    // Initialise datastructures for the CSR format
    A_rows_ = new aoclsparse_int[m_ + 1];
    A_cols_ = new aoclsparse_int[nnz_aocl_];
    A_vals_ = new T[nnz_aocl_];

    if (type_ == matrixType::rmat) {
      rMatCSR<T, aoclsparse_int>(A_vals_, A_cols_, A_rows_, m_, k_, nnz_);
    } else if (type_ == matrixType::random) {
      randomCSR<T, aoclsparse_int>(A_vals_, A_cols_, A_rows_, m_, k_, nnz_);
    } else if (type_ == matrixType::bandedDiagonal) {
      bandedDiagonalCSR<T, aoclsparse_int>(A_vals_, A_cols_, A_rows_, m_, k_, nnz_);
    } else {
      std::cerr << "Matrix type not supported" << std::endl;
      exit(1);
    }
    
    // Move into the AOCL CSR matrix handle
    if constexpr (std::is_same_v<T, float>) {
      status_ = aoclsparse_create_scsr(&A_aocl_, 
                                       base_, 
                                       m_aocl_, 
                                       k_aocl_, 
                                       nnz_aocl_, 
                                       A_rows_, 
                                       A_cols_, 
                                       A_vals_);
    } else if constexpr (std::is_same_v<T, double>) {
      status_ = aoclsparse_create_dcsr(&A_aocl_, 
                                       base_, 
                                       m_aocl_, 
                                       k_aocl_, 
                                       nnz_aocl_, 
                                       A_rows_, 
                                       A_cols_, 
                                       A_vals_);
    }
    if (status_ != aoclsparse_status_success) {
      std::cerr << "aoclsparse_create_?csr is failing with problem size of " << m_ << "x" << k_ << " . " << k_ << "x" << n_ << std::endl;
      printAOCLError(status_);
    }
  }

private:
  void preLoopRequirements() override {}

  void callSpmm() override {
    operation_ = aoclsparse_operation_none; // Just saying no transposition happening first
    if constexpr (std::is_same_v<T, float>) {
      status_ = aoclsparse_scsrmm(operation_, 
                                  alpha, 
                                  A_aocl_, 
                                  A_description_, 
                                  order_, 
                                  B_, 
                                  n_aocl_, 
                                  n_aocl_, 
                                  beta, 
                                  C_, 
                                  n_aocl_);
    } else if constexpr(std::is_same_v<T, double>) {
      status_ = aoclsparse_dcsrmm(operation_, 
                                  alpha, 
                                  A_aocl_, 
                                  A_description_, 
                                  order_, 
                                  B_, 
                                  n_aocl_, 
                                  n_aocl_, 
                                  beta, 
                                  C_, 
                                  n_aocl_);
    }
    if (status_ != aoclsparse_status_success) {
      std::cerr << "aoclsparse_?csrmm is failing with problem size of " << m_ << "x" << k_ << " . " << k_ << "x" << n_ << std::endl;
      std::cerr << "\tm_aocl_=" << m_aocl_ << std::endl;
      std::cerr << "\tn_aocl_=" << n_aocl_ << std::endl;
      std::cerr << "\tk_aocl_=" << k_aocl_ << std::endl;
      std::cerr << "\tnnz_aocl_=" << nnz_aocl_ << std::endl;
      printAOCLError(status_);
    }
  }

  void postLoopRequirements() override {
  }

  void postCallKernelCleanup() override {
    status_ = aoclsparse_destroy_mat_descr(A_description_);
    if (status_ != aoclsparse_status_success) {
      std::cerr << "aoclsparse_destroy_mat_descr is failing with problem size of " << m_ << "x" << k_ << " . " << k_ << "x" << n_ << std::endl;
      printAOCLError(status_);
    }
    status_ = aoclsparse_destroy(&A_aocl_);
    if (status_ != aoclsparse_status_success) {
      std::cerr << "aoclsparse_destroy is failing with problem size of " << m_ << "x" << k_ << " . " << k_ << "x" << n_ << std::endl;
      printAOCLError(status_);
    }
    delete[] A_vals_;
    delete[] A_cols_;
    delete[] A_rows_;
    delete[] B_;
    delete[] C_;
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

  aoclsparse_status status_;
  aoclsparse_order order_;

  aoclsparse_operation operation_;
  aoclsparse_index_base base_;

  aoclsparse_mat_descr A_description_;
  aoclsparse_matrix A_aocl_;
  aoclsparse_int* A_rows_;
  aoclsparse_int* A_cols_;
  T* A_vals_;

  aoclsparse_int m_aocl_;
  aoclsparse_int n_aocl_;
  aoclsparse_int k_aocl_;
  aoclsparse_int nnz_aocl_;


  const T alpha = ALPHA;
  const T beta = BETA;
};
}


#endif
