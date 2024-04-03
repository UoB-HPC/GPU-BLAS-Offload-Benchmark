#pragma once

#include "../gemm.hh"

#include <random>

namespace cpu {

/** An abstract class for GEMM BLAS kernels. */
		template <typename T>
		class sp_gemm : public ::gemm<T> {
		public:
				using ::gemm<T>::gemm;
        using ::gemm<T>::initInputMatricesSparse;
        using ::gemm<T>::toCSR;
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

				A_ = (T*)malloc(sizeof(T) * n_ * n_);
				B_ = (T*)malloc(sizeof(T) * n_ * n_);
				C_ = (T*)malloc(sizeof(T) * n_ * n_);

				initInputMatricesSparse(sparsity);
			}

			private:
				/** Do any necessary cleanup (free pointers, close library handles, etc.)
				 * after Kernel has been called. */
				void postCallKernelCleanup() {
					free(A_);
					free(B_);
					free(C_);
				}
		};
}  // namespace cpu