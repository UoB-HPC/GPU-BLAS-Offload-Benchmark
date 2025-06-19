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
      std::cout << ".. setting metadata " << m << "x" << k << " and " << k << "x" << n;
      base_ = aoclsparse_index_base_zero;
      operation_ = aoclsparse_operation_none;
      order_ = aoclsparse_order_row;

      m_aocl_ = m_ = m;
      n_aocl_ = n_ = n;
      k_aocl_ = k_ = k;
      sparsity_ = sparsity;

      nnz_ = 1 + (uint64_t)((double)m_ * (double)k_ * (1.0 - sparsity_));
      nnz_aocl_ = (aoclsparse_int)nnz_;


      A_rows_ = new aoclsparse_int[m_ + 1];
      A_cols_ = new aoclsparse_int[nnz_aocl_];
      A_vals_ = new T[nnz_aocl_];

      A_ = (T*)calloc(m_ * k_, sizeof(T));
      B_ = (T*)calloc(k_ * n_, sizeof(T));
      C_ = (T*)calloc(m_ * n_, sizeof(T));

      initInputMatrices();
    }

protected:
    void toSparseFormat() override {
      std::cout << "DEBUG: toSparseFormat - Start" << std::endl;
      std::cout << "DEBUG: matrix = ";
      for (int i = 0; i < m_ * k_; i++) {
        if (i % k_ == 0) std::cout << std::endl << "\t";
        std::cout << A_[i] << " ";
      }
      std::cout << std::endl;


      int nnz_encountered = 0;

      A_rows_[0] = (aoclsparse_int)0;
      for (int row = 0; row < m_; row++) {

        for (int col = 0; col < k_; col++) {
          int index = (row * k_) + col;

          if (A_[index] != 0.0) {
            if (nnz_encountered >= nnz_) {
                std::cerr << "ERROR: More non-zeros than allocated: "
                          << "encountered=" << nnz_encountered
                          << ", allocated=" << nnz_ << std::endl;
                exit(1);
            }

            A_cols_[nnz_encountered] = (aoclsparse_int)col;
            A_vals_[nnz_encountered] = static_cast<T>(A_[index]);
            nnz_encountered++;
          }
        }
        A_rows_[row + 1] = (aoclsparse_int)nnz_encountered;
      }

      std::cout << "csr is:" << std::endl;
      std::cout << "\tRow pointers - ";
      for (int i = 0; i < m_ + 1; i++) {
        std::cout << A_rows_[i] << " ";
      }
      std::cout << std::endl;
      std::cout << "\tCol indices - ";
      for (int i = 0; i < nnz_; i++) {
        std::cout << A_cols_[i] << " ";
      }
      std::cout << std::endl;
      std::cout << "\tVals - ";
      for (int i = 0; i < nnz_; i++) {
        std::cout << A_vals_[i] << " ";
      }
      std::cout << std::endl;
      std::cout << "DEBUG: toSparseFormat - Complete" << std::endl;
    }

private:
    void preLoopRequirements() override {
      // Set up the aocl matrix descriptor
      status_ = aoclsparse_create_mat_descr(&A_description_);
      if (status_ != aoclsparse_status_success) {
        std::cout << "aoclsparse_create_mat_descr failing with: ";
        printAOCLError(status_);
      }

      // Set up the description's base (base zero here, as we're using C++)
      status_ = aoclsparse_set_mat_index_base(A_description_, base_);
      if (status_ != aoclsparse_status_success) {
        std::cout << "aoclsparse_set_mat_index_base failing with: ";
        printAOCLError(status_);
      }

      // Create the aocl sparse matrix for A_
      if constexpr (std::is_same_v<T, float>) {
        status_ = aoclsparse_create_scsr(&A_aocl_, // aoclsparse_matrix*
                                         base_, // aoclsparse_index_base
                                         m_aocl_,
                                         k_aocl_,
                                         nnz_aocl_,
                                         A_rows_,
                                         A_cols_,
                                         A_vals_);
      } else if constexpr (std::is_same_v<T, double>) {
        status_ = aoclsparse_create_dcsr(&A_aocl_, base_, m_aocl_, k_aocl_,
                                        nnz_aocl_, A_rows_, A_cols_, A_vals_);
      }

//      if (status_ != aoclsparse_status_success) {
//        std::cout << std::endl << "aoclsparse_create_?csr failing with: ";
//        printAOCLError(status_);
//      }

    }

    void callSpgemm() override {
      if constexpr (std::is_same_v<T, float>) {
        // AOCL assumes column-major for B and C.  As they are just randomly
        // filled arrays, this doesn't actually matter here.
        aoclsparse_scsrmm(operation_,
                          alpha,
                          A_aocl_,
                          A_description_,
                          order_,
                          B_,
                          n_aocl_,
                          k_aocl_,
                          beta,
                          C_,
                          m_aocl_);
      } else if constexpr (std::is_same_v<T, double>) {
        aoclsparse_dcsrmm(operation_,
                          alpha,
                          A_aocl_,
                          A_description_,
                          order_,
                          B_,
                          n_aocl_,
                          k_aocl_,
                          beta,
                          C_,
                          m_aocl_);
      } else {
        printAOCLError(status_);
      }
      callConsume();
    }

    void postLoopRequirements() override {
      delete[] A_rows_;
      delete[] A_cols_;
      delete[] A_vals_;
    }

    void postCallKernelCleanup() override {
      status_ = aoclsparse_destroy_mat_descr(A_description_);
      status_ = aoclsparse_destroy(&A_aocl_);
      free(A_rows_);
      free(A_cols_);
      free(A_vals_);
    }

    void printAOCLError(aoclsparse_status stat) {
      switch (stat) {
        case aoclsparse_status_success:
          std::cout << "SUCCESS - The operation completed successfully";
          break;
        case aoclsparse_status_not_implemented:
          std::cout << "NOT_IMPLEMENTED - The requested functionality is not yet implemented in this version";
          break;
        case aoclsparse_status_invalid_pointer:
          std::cout << "INVALID_POINTER - One or more pointer parameters are NULL or otherwise invalid";
          break;
        case aoclsparse_status_invalid_size:
          std::cout << "INVALID_SIZE - One or more size parameters (m, n, nnz, etc.) contain an invalid value (e.g., negative or zero where positive required)";
          break;
        case aoclsparse_status_internal_error:
          std::cout << "INTERNAL_ERROR - Internal library failure";
          break;
        case aoclsparse_status_invalid_value:
          std::cout << "INVALID_VALUE - Input parameters contain an invalid value (e.g., invalid enum value, base index neither 0 nor 1)";
          break;
        case aoclsparse_status_invalid_index_value:
          std::cout << "INVALID_INDEX_VALUE - At least one index value is invalid (e.g., negative or out of bounds)";
          break;
        case aoclsparse_status_maxit:
          std::cout << "MAXIT - function stopped after reaching number of iteration limit";
          break;
        case aoclsparse_status_user_stop:
          std::cout << "USER_STOP - user requested termination";
          break;
        case aoclsparse_status_wrong_type:
          std::cout << "WRONG_TYPE - Data type mismatch (e.g., matrix datatypes don't match between operations)";
          break;
        case aoclsparse_status_memory_error:
          std::cout << "MEMORY_ERROR - memory allocation failure";
          break;
        case aoclsparse_status_numerical_error:
          std::cout << "NUMERICAL_ERROR - numerical error, e.g., matrix is not positive definite, devide-by-zero error";
          break;
        case aoclsparse_status_invalid_operation:
          std::cout << "INVALID_OPERATION - cannot proceed with the request at this point";
          break;
        case aoclsparse_status_unsorted_input:
          std::cout << "UNSORTED_INPUT - the input matrices are not sorted";
          break;
        case aoclsparse_status::aoclsparse_status_invalid_kid:
          std::cout << "INVALID_KID - user requested kernel id was not available";
          break;
        default:
          std::cout << "UNKNOWN_STATUS - Unrecognized status code (" + std::to_string(stat) + ")";
          break;
      }
      std::cout << std::endl;
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
