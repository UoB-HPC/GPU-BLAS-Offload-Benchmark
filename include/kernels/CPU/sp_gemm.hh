#pragma once

#include "../gemm.hh"

#include <random>
#include <memory>
#include <iostream>

namespace cpu {

/** An abstract class for GEMM BLAS kernels. */
		template <typename T>
		class sp_gemm : public ::gemm<T> {
		public:
        using ::gemm<T>::gemm;
        using ::gemm<T>::initInputMatricesSparse;
        using ::gemm<T>::toCSR_int;
				using ::gemm<T>::iterations_;
        using ::gemm<T>::m_;
				using ::gemm<T>::n_;
				using ::gemm<T>::k_;
				using ::gemm<T>::A_;
				using ::gemm<T>::B_;
				using ::gemm<T>::C_;

		public:
			/** Initialise the required data structures. */
			virtual void initialise(int n, double sparsity, bool binary = false) {
				n_ = n;
        sparsity_ = sparsity;

        // Note that the below should be the same as the edges calculation
        // used in the initInputMatricesSparse function.  If changed here,
        // change there
        nnz_ = 1 + (int) ((double)n_ * (double)n_ * (1.0 - sparsity_));
//        std::cout << "nnz_ = " << nnz_ << std::endl;

				A_ = (T*)malloc(sizeof(T) * n_ * n_);
				B_ = (T*)malloc(sizeof(T) * n_ * n_);
				C_ = (T*)malloc(sizeof(T) * n_ * n_);

				initInputMatricesSparse(sparsity_);

        toCSR_int();
			}

      int nnz_;

    private:
				/** Do any necessary cleanup (free pointers, close library handles, etc.)
				 * after Kernel has been called. */
      void postCallKernelCleanup() {
        free(A_);
        free(B_);
        free(C_);
      }

      void toCSR_int() {
        // Move A to CSR
        A_row_ptr_ = new int[n_ + 1];
        A_col_index_ = new int[nnz_];
        A_vals_ = new T[nnz_];
        int nnz_encountered = 0;
        for (int row = 0; row < n_; row++) {
          A_row_ptr_[row] = nnz_encountered;
          for (int col = 0; col < n_; col++) {
            if (A_[(row * n_) + col] != 0.0) {
              A_col_index_[nnz_encountered] = col;
              A_vals_[nnz_encountered] = A_[(row * n_) + col];
              nnz_encountered++;
            }
          }
        }

        // Move B to CSR
        B_row_ptr_ = new int[n_ + 1];
        B_col_index_ = new int[nnz_];
        B_vals_ = new T[nnz_];
        nnz_encountered = 0;
        for (int row = 0; row < n_; row++) {
          B_row_ptr_[row] = nnz_encountered;
          for (int col = 0; col < n_; col++) {
            if (B_[(row * n_) + col] != 0.0) {
              B_col_index_[nnz_encountered] = col;
              B_vals_[nnz_encountered] = B_[(row * n_) + col];
              nnz_encountered++;
            }
          }
        }
      }

      double sparsity_;

      int* A_row_ptr_;
      int* A_col_index_;
      int* B_row_ptr_;
      int* B_col_index_;
      int* C_row_ptr_;
      int* C_col_index_;
      T* A_vals_;
      T* B_vals_;
      T* C_vals_;

		};
}  // namespace cpu
