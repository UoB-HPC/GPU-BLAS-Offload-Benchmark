#pragma once

#ifdef GPU_CUBLAS
#include "cusparse.h"
#include <cuda_runtime.h>
#include <type_traits>

#include "../include/kernels/GPU/gemm.hh"
#include "../include/utilities.hh"
#include "common.hh"

namespace gpu {
/** A class for GEMM GPU BLAS kernels. */
template <typename T>
class sp_gemm_gpu : public gemm<T> {
 public:
  using gemm<T>::gemm;
  using gemm<T>::n_;
  using gemm<T>::A_;
  using gemm<T>::B_;
  using gemm<T>::C_;
  using gemm<T>::offload_;

	// ToDo -- just unified implemented so far.  Fill in Always and Once later

  /** Initialise the required data structures.
   * `offload` refers to the data offload type:
   *  - Once:    Move data from host to device before all iterations & move from
   *             device to host after all iterations
   *  - Always:  Move data from host to device and device to host each iteration
   *  - Unified: Initialise data as unified memory; no data movement semantics
   *             required */
  void initialise(gpuOffloadType offload, int n, float sparsity) override {
    offload_ = offload;

		// Create a handle for cuSPARSE
    cusparseCreate(&handle_);

		cudaDataType_ = (std::is_same_v<T, float>) ? CUDA_R_32F :
						CUDA_R_64F;

    n_ = n;

		cusparseCreateMatDescr(&descrA);
		cusparseCreateMatDescr(&descrB);
		cusparseCreateMatDescr(&descrC);

		cusparseSetMatType(descrA, CUSPARSE_MATRIX_TYPE_GENERAL);
		cusparseSetMatType(descrB, CUSPARSE_MATRIX_TYPE_GENERAL);
		cusparseSetMatType(descrC, CUSPARSE_MATRIX_TYPE_GENERAL);

		cusparseSetMatIndexBase(descrA, CUSPARSE_INDEX_BASE_ZERO);
		cusparseSetMatIndexBase(descrB, CUSPARSE_INDEX_BASE_ZERO);
		cusparseSetMatIndexBase(descrC, CUSPARSE_INDEX_BASE_ZERO);

    // Get device identifier
    cudaCheckError(cudaGetDevice(&gpuDevice_));

    // Initialise 3 streams to asynchronously move data between host and device
    cudaCheckError(cudaStreamCreate(&s1_));
    cudaCheckError(cudaStreamCreate(&s2_));
    cudaCheckError(cudaStreamCreate(&s3_));


		// Work out number of edges needed to achieve target sparsity
		int edges = 1 + (int) (n_ * n_ * (1 - sparsity));
		A_nnz_ = B_nnz_ = edges

		// ToDo -- for all of this mallocing, bear in mind that row will probably
		//  have fewer than 'edges' values (thats the whole point).  May need to
		//  reorganise

    cudaCheckError(cudaMallocManaged(A_num_rows_, sizeof(int)));
		cudaCheckError(cudaMallocManaged(A_num_cols_, sizeof(int)));
		cudaCheckError(cudaMallocManaged(A_nnz_, sizeof(int)));
		cudaCheckError(cudaMallocManaged(&A_val_, sizeof(T) * edges));
		cudaCheckError(cudaMallocManaged(&A_col_, sizeof(int) * edges));
		cudaCheckError(cudaMallocManaged(&A_row_, sizeof(int) * (n_ + 1)));

		cudaCheckError(cudaMallocManaged(B_num_rows_, sizeof(int)));
		cudaCheckError(cudaMallocManaged(B_num_cols_, sizeof(int)));
		cudaCheckError(cudaMallocManaged(B_nnz_, sizeof(int)));
		cudaCheckError(cudaMallocManaged(&B_val_, sizeof(T) * edges));
		cudaCheckError(cudaMallocManaged(&B_col_, sizeof(int) * edges));
		cudaCheckError(cudaMallocManaged(&B_row_, sizeof(int) * (n_ + 1)));

		C_val_ = NULL;
		C_col_ = NULL;
		C_row_ = NULL;


		// Initialise the host matricies
		// cusparseSpGEMM() works on CSR format only.  This helpfully makes our
		// sparse matrix format decision for us!

		// Initialise the matrices
		// Set initial values to 0
		for (int i = 0; i < (n_ * n_); i++) {
			A_[i] = 0.0;
			B_[i] = 0.0;
		}
		// Using a=0.45 and b=c=0.22 as default probabilities
		for (int i = 0; i < edges; i++) {
			while (!rMat(A_, n, 0, n - 1, 0, n - 1,
			             0.45, 0.22, 0.22,
			             &gen, dist, false)) {}
			while (!rMat(B_, n, 0, n - 1, 0, n - 1,
			             0.45, 0.22, 0.22,
			             &gen, dist, false)) {}
		}

		toCSR(A_, n, n, edges, A_val_, A_col_, A_row_);
		toCSR(B_, n, n, edges, B_val_, B_col_, B_row_);

  }



 private:


  /** Perform any required steps before calling the GEMM kernel that should
   * be timed. */
  void preLoopRequirements() override {
    // Prefetch memory to device
		cudaCheckError(cudaMemPrefetchAsync(A_num_rows_, sizeof(int), gpuDevice_,
																				s1_));
		cudaCheckError(cudaMemPrefetchAsync(A_num_cols_, sizeof(int), gpuDevice_,
																				s1_));
		cudaCheckError(cudaMemPrefetchAsync(A_nnz_, sizeof(int), gpuDevice_,
																				s1_));
		cudaCheckError(cudaMemPrefetchAsync(&A_val_, sizeof(T) * edges, gpuDevice_,
																				s1_));
		cudaCheckError(cudaMemPrefetchAsync(&A_col_, sizeof(int) * edges,
																				gpuDevice_, s1_));
		cudaCheckError(cudaMemPrefetchAsync(&A_row_, sizeof(int) * (n_ + 1),
																				gpuDevice_, s1_));

		cudaCheckError(cudaMemPrefetchAsync(B_num_rows_, sizeof(int), gpuDevice_,
																				s2_));
		cudaCheckError(cudaMemPrefetchAsync(B_num_cols_, sizeof(int), gpuDevice_,
																				s2_));
		cudaCheckError(cudaMemPrefetchAsync(B_nnz_, sizeof(int), gpuDevice_,
																				s2_));
		cudaCheckError(cudaMemPrefetchAsync(&B_val_, sizeof(T) * edges, gpuDevice_,
																				s2_));
		cudaCheckError(cudaMemPrefetchAsync(&B_col_, sizeof(int) * edges,
																				gpuDevice_, s2_));
		cudaCheckError(cudaMemPrefetchAsync(&B_row_, sizeof(int) * (n_ + 1),
																				gpuDevice_, s2_));
//
//		cudaCheckError(cudaMemPrefetchAsync(C_num_rows_, sizeof(int), gpuDevice_,
//																				s3_));
//		cudaCheckError(cudaMemPrefetchAsync(C_num_cols_, sizeof(int), gpuDevice_,
//																				s3_));
//		cudaCheckError(cudaMemPrefetchAsync(C_nnz_, sizeof(int), gpuDevice_,
//																				s3_));
//		cudaCheckError(cudaMemPrefetchAsync(&C_val_, sizeof(T) * edges, gpuDevice_,
//																				s3_));
//		cudaCheckError(cudaMemPrefetchAsync(&C_col_, sizeof(int) * edges,
//																				gpuDevice_, s3_));
//		cudaCheckError(cudaMemPrefetchAsync(&C_row_, sizeof(int) * edges,
//																				gpuDevice_, s3_));

		// Create the CSR matrices on the device
		cusparseCreateCsr(descrA_, n_, n_, A_nnz_, A_row_, A_col_, A_val_,
											CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
											CUSPARSE_INDEX_BASE_ZERO, cudaDateType_);
		cusparseCreateCsr(descrB_, n_, n_, B_nnz_, B_row_, B_col_, B_val_,
											CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
											CUSPARSE_INDEX_BASE_ZERO, cudaDateType_);
		cusparseCreateCsr(descrC_, n_, n_, 0, NULL, NULL, NULL,
											CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
											CUSPARSE_INDEX_BASE_ZERO, cudaDataType_);

		cusparseSpGEMM_createDescr(&spgemmDesc_);
  }

  /** Make a call to the BLAS Library Kernel. */
  void callGemm() override {
    cusparseSpGEMM_workEstimation(handle_, CUSPARSE_OPERATION_NON_TRANSPOSE,
																	 CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha,
																	 descrA_, descrB_, &beta, descrC_,
																	 CUSPARSE_SPGEMM_DEFAULT, cudaDataType_,
																	 spgemmDesc_, buffer_size1_, NULL);
		cudaCheckError(cudaMallocManaged(&buffer1_, buffer_size1_));
    cusparseSpGEMM_workEstimation(handle_, CUSPARSE_OPERATION_NON_TRANSPOSE,
																	 CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha,
																	 descrA_, descrB_, &beta, descrC_,
																	 CUSPARSE_SPGEMM_DEFAULT, cudaDataType_,
																	 spgemmDesc_, buffer_size1_, buffer1_);
		cusparseSpGEMM_cmopute(handle_, CUSPARSE_OPERATION_NON_TRANSPOSE,
													 CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha, descrA_,
													 descrB_, &beta, descrC_, CUSPARSE_SPGEMM_DEFAULT,
													 cudaDataType_, spgemmDesc_, buffer_size2_, NULL);
		cudaCheckError(cudaMallocManaged(&buffer2_, buffer_size2));

		if (cusparseSpGEMM_cmopute(handle_, CUSPARSE_OPERATION_NON_TRANSPOSE,
													 CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha, descrA_,
													 descrB_, &beta, descrC_, CUSPARSE_SPGEMM_DEFAULT,
													 cudaDataType_, spgemmDesc_, buffer_size2_, buffer2_)
						== CUSPARSE_SATUS_INSUFFICIENT_RESOURCES) {
			std::cout << "Insufficient resources" << std::endl;
			exit(1);
		}

		int rows, cols, nnz;

		cusparseSpMatGetSize(descrC_, &rows, &cols, &nnz_);
		C_nnz_ = nnz;
		cudaCheckError(cudaMallocManaged(C_val_), sizeof(T) * nnz);
		cudaCheckError(cudaMallocManaged(C_col_), sizeof(int) * nnz);
		cudaCheckError(cudaMallocManaged(C_row_), sizeof(int) * (n_ + 1));

		cusparseCstSetPointers(descrC_, *C_row, *C_colind, *C_val);
		cusparseSpGEMM_copy(handle_, CUSPARSE_OPERATION_NON_TRANSPOSE,
												CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha, descrA_,
												descrB_, &beta, descrC_, CUDA_R_32F,
												CUSPARSE_SPGEMM_DEFAULT, spgemmDesc_);
	}

  /** Perform any required steps after calling the GEMM kernel that should
   * be timed. */
  void postLoopRequirements() override {
    // Ensure all data resides on host once work has completed
		cudaCheckError(cudaMemPrefetchAsync(A_num_rows_, sizeof(int),
																				cudaCpuDeviceId_, s1_));
		cudaCheckError(cudaMemPrefetchAsync(A_num_cols_, sizeof(int),
																				cudaCpuDeviceId_, s1_));
		cudaCheckError(cudaMemPrefetchAsync(A_nnz_, sizeof(int),
																				cudaCpuDeviceId_, s1_));
		cudaCheckError(cudaMemPrefetchAsync(&A_val_, sizeof(T) * edges,
																				cudaCpuDeviceId_, s1_));
		cudaCheckError(cudaMemPrefetchAsync(&A_col_, sizeof(int) * edges,
																				cudaCpuDeviceId_, s1_));
		cudaCheckError(cudaMemPrefetchAsync(&A_row_, sizeof(int) * (n_ + 1),
																				cudaCpuDeviceId_, s1_));

		cudaCheckError(cudaMemPrefetchAsync(B_num_rows_, sizeof(int),
																				cudaCpuDeviceId_, s2_));
		cudaCheckError(cudaMemPrefetchAsync(B_num_cols_, sizeof(int),
																				cudaCpuDeviceId_, s2_));
		cudaCheckError(cudaMemPrefetchAsync(B_nnz_, sizeof(int),
																				cudaCpuDeviceId_, s2_));
		cudaCheckError(cudaMemPrefetchAsync(&B_val_, sizeof(T) * edges,
																				cudaCpuDeviceId_, s2_));
		cudaCheckError(cudaMemPrefetchAsync(&B_col_, sizeof(int) * edges,
																				cudaCpuDeviceId_, s2_));
		cudaCheckError(cudaMemPrefetchAsync(&B_row_, sizeof(int) * (n_ + 1),
																				cudaCpuDeviceId_, s2_));

		cudaCheckError(cudaMemPrefetchAsync(C_num_rows_, sizeof(int),
																				cudaCpuDeviceId_, s3_));
		cudaCheckError(cudaMemPrefetchAsync(C_num_cols_, sizeof(int),
																				cudaCpuDeviceId_, s3_));
		cudaCheckError(cudaMemPrefetchAsync(C_nnz_, sizeof(int),
																				cudaCpuDeviceId_, s3_));
		cudaCheckError(cudaMemPrefetchAsync(&C_val_, sizeof(T) * C_nnz_,
																				cudaCpuDeviceId_, s3_));
		cudaCheckError(cudaMemPrefetchAsync(&C_col_, sizeof(int) * C_nnz_,
																				cudaCpuDeviceId_, s3_));
		cudaCheckError(cudaMemPrefetchAsync(&C_row_, sizeof(int) * (n_ + 1),
																				cudaCpuDeviceId_, s3_));
    // Ensure device has finished all work.
    cudaCheckError(cudaDeviceSynchronize());
  }

  /** Do any necessary cleanup (free pointers, close library handles, etc.)
   * after Kernel has been called. */
  void postCallKernelCleanup() override {
    // Destroy the handle
    cusparseDestroy(handle_);

    // Destroy streams after use
    cudaCheckError(cudaStreamDestroy(s1_));
    cudaCheckError(cudaStreamDestroy(s2_));
    cudaCheckError(cudaStreamDestroy(s3_));

    cudaFree(A_);
    cudaFree(B_);
    cudaFree(C_);
  }

	bool rMat(T* M, int n, int x1, int x2, int y1, int y2,
					        float a, float b, float c, std::default_random_engine* gen,
					        std::uniform_real_distribution<double> dist, bool bin) {
		// If a 1x1 submatrix, then add an edge and return out
		if (x1 >= x2 && y1 >= y2) {
			if (abs(M[(y1 * n) + x1]) > 0.1) {
				return false;
			} else {
				// Add 1.0 if this is a binary graph, and a random real number otherwise
				M[(int) (y1 * n) + x1] = (bin) ? 1.0 : (((rand() % 10000) /
								100.0) - 50.0);
				return true;
			}
		} else {
			// Divide up the matrix
			int xMidPoint = x1 + floor((x2 - x1) / 2);
			int yMidPoint = y1 + floor((y2 - y1) / 2);

			// ToDo -- add some noise to these values between iterations
			float newA = a;
			float newB = b;
			float newC = c;

			// Work out which quarter to recurse into
			// There are some ugly ternary operators here to avoid going out of bounds in the edge case
			// that we are already at 1 width or 1 height
			float randomNum = dist(*gen);
			if (randomNum < a) {
				return rMat(M, n, x1, xMidPoint, y1, yMidPoint,
				            newA, newB, newC, gen, dist, bin);
			} else if (randomNum < (a + b)) {
				return rMat(M, n, ((xMidPoint < x2) ? xMidPoint + 1 : xMidPoint), x2, y1, yMidPoint,
				            newA, newB, newC, gen, dist, bin);
			} else if (randomNum < (a + b + c)) {
				return rMat(M, n, x1, xMidPoint, ((yMidPoint < y2) ? yMidPoint + 1 : yMidPoint), y2,
				            newA, newB, newC, gen, dist, bin);
			} else {
				return rMat(M, n, ((xMidPoint < x2) ? xMidPoint + 1 : xMidPoint), x2,
				            ((yMidPoint < y2) ? yMidPoint + 1 : yMidPoint), y2, newA, newB, newC,
				            gen, dist, bin);
			}
		}
		return true;
	}

	void toCSR(T* dense, int n_col, int n_row, int nnz, T* vals, int* col_index,
						 int* row_ptr) {
		int nnz_encountered = 0;
		int prev_row_ptr = 0;
		for (int row = 0; row < n_row; row++) {
			if (nnz_encountered >= nnz) break;
			row_ptr[row] = prev_row_ptr;
			int nnz_row = 0;
			for (int col = 0; col < n_col; col++) {
				if (nnz_encountered >= nnz) break;
				if (dense[(row * n_col) + col] != 0.0) {
					nnz_row++;
					col_index[nnz_encountered] = col;
					vals[nnz_encountered] = dense[(row * n_col) + col];
					nnz_encountered++;
				}
			}
			prev_row_ptr += nnz_row;
		}
	}

  /** Handle used when calling cuBLAS. */
  cublasHandle_t handle_;

  /** CUDA Stream 1 - used to asynchronously move data between host and device.
   */
  cudaStream_t s1_;

  /** CUDA Stream 1 - used to asynchronously move data between host and device.
   */
  cudaStream_t s2_;

  /** CUDA Stream 1 - used to asynchronously move data between host and device.
   */
  cudaStream_t s3_;

  /** The ID of the target GPU Device. */
  int gpuDevice_;

	/** CSR format vectors for matrices A, B and C on the host */
	int A_nnz_, B_nnz_, C_nnz_;
	T* A_val_, B_val_, C_val_;
	int* A_col_, A_row_, B_col_, B_row_, C_col_, C_row_;

  /** CSR format vectors for matrices A, B and C on the device. */
	int A_num_rows_dev_, A_num_cols_dev_, A_nnz_dev_, B_num_rows_dev_,
	B_num_cols_dev_, B_nnz_dev_, C_num_rows_dev_, C_num_cols_dev_, C_nnz_dev_;
	T* A_val_dev_, B_val_dev_, C_val_dev_;
	int* A_col_dev_, A_row_dev_, B_col_dev_, B_row_dev_, C_col_dev_, C_row_dev_;

  /** The constant value Alpha. */
  const T alpha = ALPHA;

  /** The constant value Beta. */
  const T beta = BETA;


	// Create descriptors for matrices A->C
	cusparseMatDescr_t descrA_, descrB_, descrC_;

	// index type depends on kernel being run
	cusparseIndexType_t cudaDataType_;

	cusparceSpGEMMDescr_t spgemmDesc_;

	size_t buffer_size1_ = 0;
	size_t buffer_size2_ = 0;
  void* buffer1_ = NULL;
	void* buffer2_ = NULL;
};
}  // namespace gpu
#endif