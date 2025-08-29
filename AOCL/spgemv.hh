#pragma once

#ifdef CPU_AOCL

#include "aoclsparse.h"
#include <algorithm>

#include "../include/kernels/CPU/spgemv.hh"
#include "../include/utilities.hh"

namespace cpu {
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
    using spgemv<T>::nnz_;
    using spgemv<T>::iterations_;

    void initialise(int m, int n, double sparsity, bool binary = false) {
      if (print_) std::cout << "=========== Matrix = " << m << "x" << n << " ===========" << std::endl;
      base_ = aoclsparse_index_base_zero;
      operation_ = aoclsparse_operation_none;

      m_aocl_ = m_ = m;
      n_aocl_ = n_ = n;
      sparsity_ = sparsity;

      nnz_ = 1 + (uint64_t)((double)m_ * (double)n_ * (1.0 - sparsity_));
      nnz_aocl_ = nnz_;

      x_ = (T*)calloc(n_, sizeof(T));
      y_ = (T*)calloc(m_, sizeof(T));

      if (print_) std::cout << "About to initialise matrices" << std::endl;
      initInputMatrixVector();

      status_ = aoclsparse_create_mat_descr(&A_description_);
      if (status_ != aoclsparse_status_success) {
        std::cerr << "aoclsparse_create_mat_descr failing for A" << std::endl;
        printAOCLError(status_);
      }
    }

protected:
    void toSparseFormat() override {
      A_vals_ = (T*)calloc(nnz_aocl_, sizeof(T));
      A_cols_ = (aoclsparse_int*)calloc(nnz_aocl_, sizeof(aoclsparse_int));
      A_rows_ = (aoclsparse_int*)calloc(m_ + 1, sizeof(aoclsparse_int));
      rMatCSR<T, aoclsparse_int>(A_vals_, A_cols_, A_rows_, m_, n_, nnz_);

      // Move into the AOCL CSR matrix handle
      if constexpr (std::is_same_v<T, float>) {
        status_ = aoclsparse_create_scsr(&A_aocl_, 
                                        base_, 
                                        m_aocl_, 
                                        n_aocl_, 
                                        nnz_aocl_, 
                                        A_rows_, 
                                        A_cols_, 
                                        A_vals_);
      } else if constexpr (std::is_same_v<T, double>) {
        status_ = aoclsparse_create_dcsr(&A_aocl_, 
                                        base_, 
                                        m_aocl_, 
                                        n_aocl_, 
                                        nnz_aocl_, 
                                        A_rows_, 
                                        A_cols_, 
                                        A_vals_);
      }
      if (status_ != aoclsparse_status_success) {
        std::cerr << "aoclsparse_create_?csr is failing with problem size of " << m_ << "x" << n_ << " . " << n_ << "x" << n_ << std::endl;
        printAOCLError(status_);
      } else if (print_) {
        std::cout << "aoclsparse_create_?csr success" << std::endl;
      }
    }

private:
    void preLoopRequirements() override {
      status_ = aoclsparse_set_mv_hint(A_aocl_, 
                                       operation_, 
                                       A_description_,
                                       5); // Currently hard coded iternation count
      if (status_ != aoclsparse_status_success) {
        std::cerr << "aoclsparse_set_mv_hint failing" << std::endl;
        printAOCLError(status_);
      } else if (print_) {
        std::cout << "aoclsparse_set_mv_hint success" << std::endl;
      }

      status_ = aoclsparse_optimize(A_aocl_);
      if (status_ != aoclsparse_status_success) {
        std::cerr << "aoclsparse_optimize failing" << std::endl;
        printAOCLError(status_);
      } else if (print_) {
        std::cout << "aoclsparse_optimize success" << std::endl;
      }                           
    }

    void callSpgemv() override {
      if constexpr (std::is_same_v<T, float>) {
        status_ = aoclsparse_smv(operation_, 
                                 &alpha, 
                                 A_aocl_, 
                                 A_description_, 
                                 x_, 
                                 &beta, 
                                 y_);
      } else if constexpr (std::is_same_v<T, double>) {
        status_ = aoclsparse_dmv(operation_, 
                                 &alpha, 
                                 A_aocl_, 
                                 A_description_, 
                                 x_, 
                                 &beta, 
                                 y_);
      }
      if (status_ != aoclsparse_status_success) {
        std::cerr << "aoclsparse_?mv failing" << std::endl;
        printAOCLError(status_);
      } else if (print_) {
        std::cout << "aoclsparse_?mv success" << std::endl;
      }                   
    }

    void postLoopRequirements() override {
      if (debug) {
        std::cout << "==========   CPU   ==========" << std::endl;
        std::cout << "___________________________________________" << std::endl;
        std::cout << "x =" << std::endl;
        std::cout << "[";
        for (int64_t i = 0; i < n_; i++) {
          std::cout << x_[i];
          if (i < (n_ - 1)) std::cout << ", ";
        }
        std::cout << "]" << std::endl;
        
        std::cout << "y =" << std::endl;
        std::cout << "[";
        for (int64_t i = 0; i < m_; i++) {
          std::cout << y_[i];
          if (i < (m_ - 1)) std::cout << ", ";
        }
        std::cout << "]" << std::endl;
        std::cout << "___________________________________________" << std::endl;
      }
    }

    void postCallKernelCleanup() override {
      status_ = aoclsparse_destroy_mat_descr(A_description_);
      if (status_ != aoclsparse_status_success) {
        std::cerr << "aoclsparse_destroy_mat_descr failing" << std::endl;
        printAOCLError(status_);
      } else if (print_) {
        std::cout << "aoclsparse_destroy_mat_descr success" << std::endl;
      }               

      status_ = aoclsparse_destroy(&A_aocl_);
      if (status_ != aoclsparse_status_success) {
        std::cerr << "aoclsparse_destroy failing" << std::endl;
        printAOCLError(status_);
      } else if (print_) {
        std::cout << "aoclsparse_destroy success" << std::endl;
      } 

      delete[] A_vals_;
      delete[] A_cols_;
      delete[] A_rows_;
      delete[] x_;
      delete[] y_;
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
    bool debug = false;

    aoclsparse_status status_;

    aoclsparse_operation operation_;
    aoclsparse_index_base base_;

    aoclsparse_matrix A_aocl_;
    aoclsparse_int* A_rows_;
    aoclsparse_int* A_cols_;
    T* A_vals_;
    aoclsparse_int m_aocl_;
    aoclsparse_int n_aocl_;
    aoclsparse_int nnz_aocl_;

    aoclsparse_mat_descr A_description_;

    const T alpha = ALPHA;
    const T beta = BETA;
};
}


#endif
