#pragma once

#ifdef GPU_CUBLAS
#include "cusparse.h"
#include <cuda_runtime.h>

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

    n_ = n;

		// Create descriptors for matrices A->C
		cusparseMatDescr_t descrA, descrB, descrC;

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

    if (offload_ == gpuOffloadType::unified) {
      cudaCheckError(cudaMallocManaged(&A_, sizeof(T) * n_ * n_));
      cudaCheckError(cudaMallocManaged(&B_, sizeof(T) * n_ * n_));
      cudaCheckError(cudaMallocManaged(&C_, sizeof(T) * n_ * n_));

			cudaCheckError(cudaMallocManaged(&A_val_, sizeof(T) * edges));
			cudaCheckError(cudaMallocManaged(&A_col_, sizeof(int) * edges));
			cudaCheckError(cudaMallocManaged(&A_row_, sizeof(int) * edges));

			cudaCheckError(cudaMallocManaged(&B_val_, sizeof(T) * edges));
			cudaCheckError(cudaMallocManaged(&B_col_, sizeof(int) * edges));
			cudaCheckError(cudaMallocManaged(&B_row_, sizeof(int) * edges));

			cudaCheckError(cudaMallocManaged(&C_val_, sizeof(T) * edges));
			cudaCheckError(cudaMallocManaged(&C_col_, sizeof(int) * edges));
			cudaCheckError(cudaMallocManaged(&C_row_, sizeof(int) * edges));
//			cudaCheckError(cudaMallocManaged(&DANnzPerRow, sizeof(int) * n_));
    } else {
      // Allocate matrices on host
			A_ = (T*)malloc(sizeof(T) * n_ * n_);
			B_ = (T*)malloc(sizeof(T) * n_ * n_);
			C_ = (T*)malloc(sizeof(T) * n_ * n_);

      // Allocate matrices on device
      cudaCheckError(cudaMalloc((void**)&A_device_, sizeof(T) * n_ * n_));
      cudaCheckError(cudaMalloc((void**)&B_device_, sizeof(T) * n_ * n_));
      cudaCheckError(cudaMalloc((void**)&C_device_, sizeof(T) * n_ * n_));
			// Alloce non-zero vector for A
//			cudaCheckError(cudaMalloc((void**)&dANnzPerRow, sizeof(int) * n_));
    }

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

//		for (int i = 0; i < (n_ * n_); i++) {
//			C_[i] = 0.0;
//		}
  }

 private:


  /** Perform any required steps before calling the GEMM kernel that should
   * be timed. */
  void preLoopRequirements() override {
    switch (offload_) {
      case gpuOffloadType::always: {
        // Offload data each iteration - no requirements
        break;
      }
      case gpuOffloadType::once: {
        // Offload data from host to the device.
        cudaCheckError(cudaMemcpyAsync(A_device_, A_, sizeof(T) * m_ * k_,
                                       cudaMemcpyHostToDevice, s1_));
        cudaCheckError(cudaMemcpyAsync(B_device_, B_, sizeof(T) * k_ * n_,
                                       cudaMemcpyHostToDevice, s2_));
        cudaCheckError(cudaMemcpyAsync(C_device_, C_, sizeof(T) * m_ * n_,
                                       cudaMemcpyHostToDevice, s3_));
        break;
      }
      case gpuOffloadType::unified: {
        // Prefetch memory to device
        cudaCheckError(
            cudaMemPrefetchAsync(A_, sizeof(T) * m_ * k_, gpuDevice_, s1_));
        cudaCheckError(
            cudaMemPrefetchAsync(B_, sizeof(T) * k_ * n_, gpuDevice_, s2_));
        cudaCheckError(
            cudaMemPrefetchAsync(C_, sizeof(T) * m_ * n_, gpuDevice_, s3_));
        break;
      }
    }
  }

  /** Make a call to the BLAS Library Kernel. */
  void callGemm() override {
    switch (offload_) {
      case gpuOffloadType::always: {
        // Offload data from host to the device.
        cudaCheckError(cudaMemcpyAsync(A_device_, A_, sizeof(T) * m_ * k_,
                                       cudaMemcpyHostToDevice, s1_));
        cudaCheckError(cudaMemcpyAsync(B_device_, B_, sizeof(T) * k_ * n_,
                                       cudaMemcpyHostToDevice, s2_));
        cudaCheckError(cudaMemcpyAsync(C_device_, C_, sizeof(T) * m_ * n_,
                                       cudaMemcpyHostToDevice, s3_));


        break;
      }
      case gpuOffloadType::once: {
        // Call cuSPRASE SpGEMM kernel
				// ToDo -- implement

        break;
      }
      case gpuOffloadType::unified: {
        // Call cuSPARSE SpGEMM kernel
				// ToDo -- implement

        break;
      }
    }
  }

  /** Perform any required steps after calling the GEMM kernel that should
   * be timed. */
	// ToDo -- check that this all still works
  void postLoopRequirements() override {
    switch (offload_) {
      case gpuOffloadType::always: {
        // Offload data each iteration - no requirements
        break;
      }
      case gpuOffloadType::once: {
        // Offload data from device to host
        cudaCheckError(cudaMemcpyAsync(A_, A_device_, sizeof(T) * m_ * k_,
                                       cudaMemcpyDeviceToHost, s1_));
        cudaCheckError(cudaMemcpyAsync(B_, B_device_, sizeof(T) * k_ * n_,
                                       cudaMemcpyDeviceToHost, s2_));
        cudaCheckError(cudaMemcpyAsync(C_, C_device_, sizeof(T) * m_ * n_,
                                       cudaMemcpyDeviceToHost, s3_));
        // Ensure device has finished all work.
        cudaCheckError(cudaDeviceSynchronize());
        break;
      }
      case gpuOffloadType::unified: {
        // Ensure all data resides on host once work has completed
        cudaCheckError(cudaMemPrefetchAsync(A_, sizeof(T) * m_ * k_,
                                            cudaCpuDeviceId, s1_));
        cudaCheckError(cudaMemPrefetchAsync(B_, sizeof(T) * k_ * n_,
                                            cudaCpuDeviceId, s2_));
        cudaCheckError(cudaMemPrefetchAsync(C_, sizeof(T) * m_ * n_,
                                            cudaCpuDeviceId, s3_));
        // Ensure device has finished all work.
        cudaCheckError(cudaDeviceSynchronize());
        break;
      }
    }
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

    if (offload_ == gpuOffloadType::unified) {
      cudaFree(A_);
      cudaFree(B_);
      cudaFree(C_);
    } else {
      // Free the memory held on host and device
      free(A_);
      free(B_);
      free(C_);
      cudaFree(A_device_);
      cudaFree(B_device_);
      cudaFree(C_device_);
    }
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

  /** Input matrix A, held on the device. */
  T* A_device_;

  /** Input matrix B, held on the device. */
  T* B_device_;

  /** Input matrix C, held on the device. */
  T* C_device_;

	/** Vector for number non-zeros, held on the device */
//	int* dANnzPerRow;

	/** CSR format vectors for matrices A, B and C on the device */
	T* A_val_, B_val_, C_val_;
	int* A_col_, A_row_, B_col_, B_row_, C_col_, C_row_;

  /** The constant value Alpha. */
  const T alpha = ALPHA;

  /** The constant value Beta. */
  const T beta = BETA;
};
}  // namespace gpu
#endif