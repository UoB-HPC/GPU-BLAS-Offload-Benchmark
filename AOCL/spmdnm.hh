#pragma once

#ifdef CPU_AOCL
#include "aoclsparse.h"

#include <algorithm>
#include <vector>

#include "../include/kernels/CPU/spmdnm.hh"
#include "../include/utilities.hh"

namespace cpu {
template <typename T>
class spmdnm_cpu : public spmdnm<T> {
public:
  using spmdnm<T>::spmdnm;
  using spmdnm<T>::callConsume;
  using spmdnm<T>::initInputMatrices;
  using spmdnm<T>::m_;
  using spmdnm<T>::n_;
  using spmdnm<T>::k_;
  using spmdnm<T>::B_;
  using spmdnm<T>::C_;
  using spmdnm<T>::sparsity_;
  using spmdnm<T>::type_;
  using spmdnm<T>::nnz_;
  using spmdnm<T>::iterations_;

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
    } else if (type_ == matrixType::finiteElements) {
      finiteElementCSR<T, aoclsparse_int>(A_vals_, A_cols_, A_rows_, m_, k_, nnz_);
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

  void callSpmdnm() override {
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

  void internalCheck(aoclsparse_int          maj_dim,
                     aoclsparse_int          min_dim,
                     aoclsparse_int          nnz,
                     const aoclsparse_int   *idx_ptr,
                     const aoclsparse_int   *indices,
                     const void             *val,
                     int                    shape,
                     int                    base) {            
    if (idx_ptr == nullptr) {
      std::cerr << "INVALID ROWS ARRAY" << std::endl;
      exit(1);
    }
    if (indices == nullptr){
      std::cerr << "INVALID COLS ARRAY" << std::endl;
      exit(1);
    }
    if (val == nullptr){
      std::cerr << "INVALID VALS ARRAY" << std::endl;
      exit(1);
    }

    if ((min_dim < 0) || (maj_dim < 0) || (nnz < 0)) {
      std::cerr << "Wrong min_dim/maj_dim/nnz" << std::endl;
      exit(1);
    }

    if ((idx_ptr[0] - base) != 0) {
      std::cerr << "Wrong csr_row_ptr[0] or csc.col_ptr[0]" << std::endl;
      exit(1);
    }

    if ((idx_ptr[maj_dim] - base) != nnz) {
      std::cerr << "Wrong csr_row_ptr[m]!=nnz or csc.col_ptr[n]!=nnz" << std::endl;
      exit(1);
    }
    for (aoclsparse_int i = 1; i <= maj_dim; i++) {
      if (idx_ptr[i - 1] > idx_ptr[i]) {
        std::cerr << "Wrong csr_row_ptr/csc.col_ptr - not nondecreasing" << std::endl;
        exit (1);
      }
    }

    // assume indices are fully sorted & fulldiag matrix unless proved otherwise
    int sort = 1;
    bool fulldiag = true;

    aoclsparse_int idxstart, idxend, j, jmin = 0, jmax = min_dim - 1;
    for (aoclsparse_int i = 0; i < maj_dim; i++) {
      idxend   = idx_ptr[i + 1] - base;
      idxstart = idx_ptr[i] - base;
      if (shape == 1) {
          jmin = 0;
          jmax = i;
      } else if (shape == 2) {
          jmin = i;
          jmax = min_dim - 1;
      }
      // check if visited D, U group within this row
      bool diagonal = false, upper = false;
      aoclsparse_int prev = -1; // holds previous col index, initially set to -1

      for (aoclsparse_int idx = idxstart; idx < idxend; idx++) {
        j = indices[idx] - base;
        if (j < jmin || j > jmax) {
          std::cerr << "Wrong index - out of bounds or triangle, @idx=" << idx << ": j=" << j << ", i=" << i << std::endl;
          exit(1);
        }
        // check for sorting pattern for each element in a row
        if (sort != 3) {
          if (prev > j) sort = 2; // unsorted col idx (duplicate elements are allowed)
          else prev = j; // update previous col index

          // check for group-order
          if ((j <= i && upper) || (j < i && diagonal)) sort = 3;
        }
        if (j > i) upper = true;
        else if(j == i) {
          if (diagonal) {
            std::cerr << "Wrong diag - duplicate diag for i=j=" << i << std::endl;
            exit(1);
          }
          // diagonal element visited
          diagonal = true;
        }
      }
      if (!diagonal && i < min_dim) fulldiag = false; // missing diagonal
    }
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
