#pragma once

#ifdef CPU_AOCL
#include "aoclsparse.h"

#include <algorithm>
#include <vector>

#include "./common.hh"
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

    aoclCheckError(aoclsparse_create_mat_descr(&A_description_));
    aoclCheckError(aoclsparse_set_mat_index_base(A_description_, base_));
    
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
      aoclCheckError(aoclsparse_create_scsr(&A_aocl_, 
                                            base_, 
                                            m_aocl_, 
                                            k_aocl_, 
                                            nnz_aocl_, 
                                            A_rows_, 
                                            A_cols_, 
                                            A_vals_));
    } else if constexpr (std::is_same_v<T, double>) {
      aoclCheckError(aoclsparse_create_dcsr(&A_aocl_, 
                                            base_, 
                                            m_aocl_, 
                                            k_aocl_, 
                                            nnz_aocl_, 
                                            A_rows_, 
                                            A_cols_, 
                                            A_vals_));
    }
  }

private:
  void preLoopRequirements() override {}

  void callSpmm() override {
    operation_ = aoclsparse_operation_none; // Just saying no transposition happening first
    if constexpr (std::is_same_v<T, float>) {
      aoclCheckError(aoclsparse_scsrmm(operation_, 
                                       alpha, 
                                       A_aocl_, 
                                       A_description_, 
                                       order_, 
                                       B_, 
                                       n_aocl_, 
                                       n_aocl_, 
                                       beta, 
                                       C_, 
                                       n_aocl_));
    } else if constexpr(std::is_same_v<T, double>) {
      aoclCheckError(aoclsparse_dcsrmm(operation_, 
                                       alpha, 
                                       A_aocl_, 
                                       A_description_, 
                                       order_, 
                                       B_, 
                                       n_aocl_, 
                                       n_aocl_, 
                                       beta, 
                                       C_, 
                                       n_aocl_));
    }
  }

  void postLoopRequirements() override {}

  void postCallKernelCleanup() override {
    aoclCheckError(aoclsparse_destroy_mat_descr(A_description_));
    aoclCheckError(aoclsparse_destroy(&A_aocl_));
    
    delete[] A_vals_;
    delete[] A_cols_;
    delete[] A_rows_;
    delete[] B_;
    delete[] C_;
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
