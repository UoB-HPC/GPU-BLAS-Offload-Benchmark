#pragma once

#ifdef CPU_AOCL
#include "aoclsparse.h"

#include <algorithm>
#include <vector>

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
  using spgemm<T>::A_;
  using spgemm<T>::B_;
  using spgemm<T>::C_;
  using spgemm<T>::sparsity_;
  using spgemm<T>::nnz_;
  using spgemm<T>::iterations_;

  void initialise(int m, int n, int k, double sparsity,
                  bool binary = false) {
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

    nnz_ = 1 + (uint64_t)((double)m_ * (double)k_ * (1.0 - sparsity_));
    nnz_aocl_ = nnz_;

    A_ = (T*)calloc(m_ * k_, sizeof(T));
    B_ = (T*)calloc(k_ * n_, sizeof(T));
    C_ = (T*)calloc(m_ * n_, sizeof(T));

    initInputMatrices();
  }

protected:
  void toSparseFormat() override {
    aoclsparse_int actual_nnz = 0;
    for (int i = 0; i < m_ * k_; i++) {
      if (A_[i] != static_cast<T>(0)) {
        actual_nnz++;
      }
    }
  
    if (actual_nnz != nnz_aocl_) {
      if (print_) std::cerr << "Warning: Actual nnz (" << actual_nnz << ") differs from expected nnz (" << nnz_aocl_ << ")" << std::endl;
      nnz_ = nnz_aocl_ = actual_nnz; // Update nnz_aocl_ to reflect actual count
    }

    // Initialise datastructures for the CSR format
    A_rows_ = new aoclsparse_int[m_ + 1];
    A_cols_ = new aoclsparse_int[nnz_aocl_];
    A_vals_ = new T[nnz_aocl_];

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
        std::cout << "CSR conversion complete. Total nnz: " << nnz_aocl_ << std::endl;
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

  void callSpgemm() override {
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
    delete[] A_;
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
    if (print_) std::cout << "CHECKING POINTERS" << std::endl;               
    if (idx_ptr == nullptr) {
      if (print_) std::cout << "INVALID ROWS ARRAY" << std::endl;
      exit(1);
    }
    if (indices == nullptr){
      if (print_) std::cout << "INVALID COLS ARRAY" << std::endl;
      exit(1);
    }
    if (val == nullptr){
      if (print_) std::cout << "INVALID VALS ARRAY" << std::endl;
      exit(1);
    }
    if (print_) std::cout << "CHECKING POSITIVE SIZES" << std::endl;

    if ((min_dim < 0) || (maj_dim < 0) || (nnz < 0)) {
      if (print_) std::cout << "Wrong min_dim/maj_dim/nnz" << std::endl;
      exit(1);
    }
    if (print_) std::cout << "CHECKING ROW 0 = 0" << std::endl;

    if ((idx_ptr[0] - base) != 0) {
      if (print_) std::cout << "Wrong csr_row_ptr[0] or csc.col_ptr[0]" << std::endl;
      exit(1);
    }
    if (print_) std::cout << "CHECKING LAST ROW = NNZ" << std::endl;
    if ((idx_ptr[maj_dim] - base) != nnz) {
      if (print_) std::cout << "Wrong csr_row_ptr[m]!=nnz or csc.col_ptr[n]!=nnz" << std::endl;
      exit(1);
    }
    if (print_) std::cout << "CHECKING ROW POINTERS INCREASE" << std::endl;
    for (aoclsparse_int i = 1; i <= maj_dim; i++) {
      if (idx_ptr[i - 1] > idx_ptr[i]) {
        if (print_) std::cout << "Wrong csr_row_ptr/csc.col_ptr - not nondecreasing" << std::endl;
        exit (1);
      }
    }

    // assume indices are fully sorted & fulldiag matrix unless proved otherwise
    int sort = 1;
    bool fulldiag = true;

    if (print_) std::cout << "CHECKING DIAGONALITY" << std::endl;
    aoclsparse_int idxstart, idxend, j, jmin = 0, jmax = min_dim - 1;
    for (aoclsparse_int i = 0; i < maj_dim; i++) {
      if (print_) std::cout << "i = " << i;
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
        if (print_) std::cout << ", idx = " << idx << ", diag = " << ((diagonal) ? "true" : "false") << std::endl;
        j = indices[idx] - base;
        if (j < jmin || j > jmax) {
          if (print_) std::cout << "Wrong index - out of bounds or triangle, @idx=" << idx << ": j=" << j
                    << ", i=" << i << std::endl;
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
            if (print_) std::cout << "Wrong diag - duplicate diag for i=j=" << i << std::endl;
            exit(1);
          }
          // diagonal element visited
          diagonal = true;
        }
      }
      if (!diagonal && i < min_dim) fulldiag = false; // missing diagonal
    }
  }

  bool print_ = false;

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
