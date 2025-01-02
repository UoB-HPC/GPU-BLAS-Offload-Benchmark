#pragma once

#ifdef GPU_CUBLAS
#include <cusparse_v2.h>
#include <cuda_runtime_api.h>
#include <type_traits>
#include <random>
#include <iostream>

#include "../include/kernels/GPU/sp_gemv.hh"
#include "../include/utilities.hh"
#include "common.hh"

namespace gpu {
/** A class for sparse GEMM GPU BLAS kernels. */
template <typename T>
class sp_gemv_gpu : public sp_gemv<T> {
 public:
  using sp_gemv<T>::sp_gemv;
  using sp_gemv<T>::initInputMatrixVectorSparse;
//  using sp_gemv<T>::toCSR_int;
  using sp_gemv<T>::m_;
  using sp_gemv<T>::n_;
  using sp_gemv<T>::A_;
  using sp_gemv<T>::x_;
  using sp_gemv<T>::y_;
  using sp_gemv<T>::offload_;
  using sp_gemv<T>::sparsity_;

  ~sp_gemv_gpu() {
    // ToDo -- destroy the handle

    // Destroy streams after use
    cudaCheckError(cudaStreamDestroy(s1_));
    cudaCheckError(cudaStreamDestroy(s2_));
    cudaCheckError(cudaStreamDestroy(s3_));
  }

	// ToDo -- No checksum for sparse yet.  Need to do

  /** Initialise the required data structures.
   * `offload` refers to the data offload type:
   *  - Once:    Move data from host to device before all iterations & move from
   *             device to host after all iterations
   *  - Always:  Move data from host to device and device to host each iteration
   *  - Unified: Initialise data as unified memory; no data movement semantics
   *             required */
  void initialise(gpuOffloadType offload, int n, float sparsity) override {
    std::cout << std::endl << "##############################" << std::endl
              << "\tCUSPARSE GEMV\t\tInitialising n = " << n << "\tOffload"
              << " type = " <<
              (((offload == gpuOffloadType::unified) ? "Unified" : (offload
              == gpuOffloadType::always) ? "Always" : "Once"))
              << std::endl
              << "##############################" << std::endl;
    offload_ = offload;

    sparsity_ = sparsity;


    /**
     *
     * 	T* A_val_;
     * 	int *A_col_, *A_row_;
     * 	T* A_val_dev_;
     * 	int *A_col_dev_, *A_row_dev_;
     * 	uint64_t A_nnz_, vals_size_, cols_size_, rows_size_;
     *
     *
     * 	T * x_host_, *y_host_;
     * 	T *x_dev_, *y_dev_;
     * 	uint64_t x_size_, y_size_;
     *
     */

    // Create a handle for cuSPARSE
    cusparseCheckError(cusparseCreate(&handle_));
    cudaCheckError(cudaGetDevice(&gpuDevice_));

    if (std::is_same_v<T, float>) cudaDataType_ = CUDA_R_32F;
    else if (std::is_same_v<T, double>) cudaDataType_ = CUDA_R_64F;
    else {
      std::cout << "INVALID DATA TYPE PASSED TO cuSPARSE" << std::endl;
      exit(1);
    }
    n_ = n;

    // Initialise 3 streams to asynchronously move data between host and device
    cudaCheckError(cudaStreamCreate(&s1_));
    cudaCheckError(cudaStreamCreate(&s2_));
    cudaCheckError(cudaStreamCreate(&s3_));

    std::cout << "\tcuda streams created" << std::endl;


   // Work out the sizes of all the vectors
    A_nnz_ = 1 + (uint64_t)(n_ * n_ * (1 - sparsity));
    vals_size_ = sizeof(T) * A_nnz_;
    cols_size_ = sizeof(int) * A_nnz_;
    rows_size_ = sizeof(int) * (n_ + 1);
    x_size_ = sizeof(T) * n_;
    y_size_ = sizeof(T) * n_;

    if (offload_ == gpuOffloadType::unified) {
      // Get device identifier
      cudaCheckError(cudaMallocManaged(&A_val_, vals_size_));
      cudaCheckError(cudaMallocManaged(&A_col_, cols_size_));
      cudaCheckError(cudaMallocManaged(&A_row_, rows_size_));

      cudaCheckError(cudaMallocManaged(&x_, x_size_));

      cudaCheckError(cudaMallocManaged(&y_, y_size_));
    } else {
      A_val_ = (T*)malloc(vals_size_);
      A_col_ = (int*)malloc(cols_size_);
      A_row_ = (int*)malloc(rows_size_);

      std::cout << "\tA_ local csr arrays made" << std::endl;

      x_ = (T*)malloc(x_size_);
      y_ = (T*)malloc(y_size_);

      std::cout << "\tx_ and y_ local arrays made" << std::endl;

      cudaCheckError(cudaMalloc((void**)&A_val_dev_, vals_size_));
      cudaCheckError(cudaMalloc((void**)&A_col_dev_, cols_size_));
      cudaCheckError(cudaMalloc((void**)&A_row_dev_, rows_size_));

      std::cout << "\tA_ dev csr arrays made" << std::endl;

      cudaCheckError(cudaMalloc((void**)&x_dev_, x_size_));

      cudaCheckError(cudaMalloc((void**)&y_dev_, y_size_));

      std::cout << "\tx_ and y_ dev arrays made" << std::endl;
    }

    // Initialise the host matricies
    // cusparseSpGEMM() works on CSR format only.  This helpfully makes our
    // sparse matrix format decision for us!

    // Initialise the matrices
    // Set initial values to 0
    A_ = (T*)malloc(sizeof(T) * n_ * n_);

    std::cout << "\tA_ dense array made" << std::endl;

    initInputMatrixVectorSparse();git branc

    std::cout << "\tinputs made" << std::endl;

    toCSR_int(A_, n_, n_, A_val_, A_col_, A_row_);

    std::cout << "\tA_ moved to CSR" << std::endl;

//    std::cout << "_____Matrix A_____" << std::endl;
//    printDenseMatrix(A_, n_, n_);
//    std::cout << std::endl << std::endl;
//    printCSR(A_val_, A_col_, A_row_, nnz_, n_, n_);

    std::cout << "\tInitialising done!" << std::endl;
  }

 private:
  /** Perform any required steps before calling the GEMM kernel that should
   * be timed. */
  void preLoopRequirements() override {
    std::cout << std::endl << "##############################" << std::endl
              << "\tPreloop Requirements" << std::endl
              << "##############################" << std::endl;
    switch(offload_) {
      case gpuOffloadType::always: {
        // Make matrix descriptor
        cusparseCheckError(
                cusparseCreateCsr(&descrA_, n_, n_, A_nnz_, A_row_dev_,
                                  A_col_dev_, A_val_dev_, rType_, cType_,
                                  indType_, cudaDataType_));
        std::cout << "\tA_ description made" << std::endl;
        // Create vector descriptor
        cusparseCheckError(cusparseCreateDnVec(&descrx_, n_, x_dev_,
                                               cudaDataType_));
        std::cout << "\tx_ description made" << std::endl;
        cusparseCheckError(cusparseCreateDnVec(&descry_, n_, NULL,
                                               cudaDataType_));
        std::cout << "\ty_ description made" << std::endl;
        break;
      }
      case gpuOffloadType::once: {
        cudaCheckError(cudaMemcpy(A_val_dev_, A_val_, vals_size_,
                                  cudaMemcpyHostToDevice));
        cudaCheckError(cudaMemcpy(A_col_dev_, A_col_, cols_size_,
                                  cudaMemcpyHostToDevice));
        cudaCheckError(cudaMemcpy(A_row_dev_, A_row_, rows_size_,
                                  cudaMemcpyHostToDevice));
        std::cout << "\tA_ csr dev arrays sunc" << std::endl;

        cudaCheckError(cudaMemcpy(x_dev_, x_, x_size_,
                                       cudaMemcpyHostToDevice));
        std::cout << "\tx_ dev array sunc" << std::endl;

        cudaCheckError(cudaMemcpy(y_dev_, y_, y_size_,
                                       cudaMemcpyHostToDevice));
        std::cout << "\ty_ dev array sunc" << std::endl;

        // Create matrix descriptor
        cusparseCheckError(
                cusparseCreateCsr(&descrA_, n_, n_, A_nnz_, A_row_dev_,
                                  A_col_dev_, A_val_dev_, rType_, cType_,
                                  indType_, cudaDataType_));
        std::cout << "\tA_ description made" << std::endl;
        // Create vector descriptor
        cusparseCheckError(cusparseCreateDnVec(&descrx_, n_, x_dev_,
                                               cudaDataType_));
        std::cout << "\tx_ description made" << std::endl;
        cusparseCheckError(cusparseCreateDnVec(&descry_, n_, NULL,
                                               cudaDataType_));
        std::cout << "\ty_ description made" << std::endl;
        break;
      }
      case gpuOffloadType::unified: {
        // Prefetch memory to device
        cudaCheckError(cudaMemPrefetchAsync(A_val_, vals_size_, gpuDevice_,
                                            s1_));
        cudaCheckError(cudaMemPrefetchAsync(A_col_, cols_size_, gpuDevice_,
                                            s1_));
        cudaCheckError(cudaMemPrefetchAsync(A_row_, rows_size_, gpuDevice_,
                                            s1_));
        std::cout << "\tA_ csr dev arrays sunc" << std::endl;

        cudaCheckError(cudaMemPrefetchAsync(x_, x_size_, gpuDevice_, s2_));
        std::cout << "\tx_ dev array sunc" << std::endl;

        cudaCheckError(cudaMemPrefetchAsync(y_, y_size_, gpuDevice_, s3_));
        std::cout << "\ty_ dev array sunc" << std::endl;
        cudaCheckError(cudaDeviceSynchronize());
        break;
      }
    }
  }

  /** Make a call to the BLAS Library Kernel. */
  void callGemv() override {
    std::cout << std::endl << "##############################" << std::endl
              << "\tCalling GEMV" << std::endl
              << "##############################" << std::endl;
    switch(offload_) {
      case gpuOffloadType::always: {
        cudaCheckError(cudaMemcpy(A_val_dev_, A_val_, vals_size_,
                                  cudaMemcpyHostToDevice));
        cudaCheckError(cudaMemcpy(A_col_dev_, A_col_, cols_size_,
                                  cudaMemcpyHostToDevice));
        cudaCheckError(cudaMemcpy(A_row_dev_, A_row_, rows_size_,
                                  cudaMemcpyHostToDevice));
        std::cout << "\tA_ csr dev arrays sunc" << std::endl;

        cudaCheckError(cudaMemcpy(x_dev_, x_, x_size_, cudaMemcpyHostToDevice));
        std::cout << "\tx_ dev array sunc" << std::endl;

        cudaCheckError(cudaMemcpy(y_dev_, y_, y_size_, cudaMemcpyHostToDevice));
        std::cout << "\ty_ dev array sunc" << std::endl;

        /**
         * Workflow is :
         *    cusparseSpMV_bufferSize
         *    cisparseSpMV_preprocess
         *    cusparseSpMV
         */
        cusparseCheckError(cusparseSpMV_bufferSize(handle_,
                                                   opA_,
                                                   &alpha,
                                                   descrA_,
                                                   descrx_,
                                                   &beta,
                                                   descry_,
                                                   cudaDataType_,
                                                   alg_,
                                                   &buffer_size_));

        std::cout << "\tbufferSize run" << std::endl;
        cudaCheckError(cudaMalloc((void**)&buffer_, buffer_size_));
        std::cout << "\tbuffer allocated" << std::endl;

        cusparseCheckError(cusparseSpMV_preprocess(handle_,
                                                   opA_,
                                                   &alpha,
                                                   descrA_,
                                                   descrx_,
                                                   &beta,
                                                   descry_,
                                                   cudaDataType_,
                                                   alg_,
                                                   buffer_));
        std::cout << "\tpreProcess run" << std::endl;
        cusparseCheckError(cusparseSpMV(handle_,
                                        opA_,
                                        &alpha,
                                        descrA_,
                                        descrx_,
                                        &beta,
                                        descry_,
                                        cudaDataType_,
                                        alg_,
                                        buffer_));
        std::cout << "\tSpMV run" << std::endl;

        cudaCheckError(cudaMemcpy(A_val_, A_val_dev_, vals_size_,
                                  cudaMemcpyDeviceToHost));
        cudaCheckError(cudaMemcpy(A_col_, A_col_dev_, cols_size_,
                                  cudaMemcpyDeviceToHost));
        cudaCheckError(cudaMemcpy(A_row_, A_row_dev_, rows_size_,
                                  cudaMemcpyDeviceToHost));

        std::cout << "\tA_ csr host arrays sunc" << std::endl;

        cudaCheckError(cudaMemcpy(x_, x_dev_, x_size_, cudaMemcpyDeviceToHost));
        std::cout << "\tx_ host array sunc" << std::endl;

        cudaCheckError(cudaMemcpy(y_, y_dev_, y_size_, cudaMemcpyDeviceToHost));
        std::cout << "\ty_ host array sunc" << std::endl;


        // Freeing memory
        cudaCheckError(cudaFree(buffer_));
        std::cout << "\tBuffer 1 freed" << std::endl;
        buffer_size_ = 0;
        break;
      }
      case gpuOffloadType::once: {
        cusparseCheckError(
                cusparseSpMV_bufferSize(handle_,
                                        opA_,
                                        &alpha,
                                        descrA_,
                                        descrx_,
                                        &beta,
                                        descry_,
                                        cudaDataType_,
                                        alg_,
                                        &buffer_size_));
        std::cout << "\tbufferSize run" << std::endl;

        cudaCheckError(cudaMalloc(&buffer_, buffer_size_));
        std::cout << "\tbuffer allocated" << std::endl;

        // ToDo -- only preprocess once?
        cusparseCheckError(
                cusparseSpMV_preprocess(handle_,
                                        opA_,
                                        &alpha,
                                        descrA_,
                                        descrx_,
                                        &beta,
                                        descry_,
                                        cudaDataType_,
                                        alg_,
                                        buffer_));
        std::cout << "\tpreProcess run" << std::endl;
        cusparseCheckError(
                cusparseSpMV(handle_,
                             opA_,
                             &alpha,
                             descrA_,
                             descrx_,
                             &beta,
                             descry_,
                             cudaDataType_,
                             alg_,
                             buffer_));
        std::cout << "\tSpMV run" << std::endl;

        // Freeing memory
        cudaCheckError(cudaFree(buffer_));
        std::cout << "\tBuffer 1 freed" << std::endl;
        break;
      }
      case gpuOffloadType::unified: {
        cusparseCheckError(cusparseSpMV_bufferSize(handle_,
                                                   opA_,
                                                   &alpha,
                                                   descrA_,
                                                   descrx_,
                                                   &beta,
                                                   descry_,
                                                   cudaDataType_,
                                                   alg_,
                                                   &buffer_size_));
        std::cout << "\tbufferSize run" << std::endl;

        cudaCheckError(cudaMallocManaged((void**)&buffer_, buffer_size_));
        std::cout << "\tbuffer allocated" << std::endl;

        cusparseCheckError(cusparseSpMV_preprocess(handle_,
                                                   opA_,
                                                   &alpha,
                                                   descrA_,
                                                   descrx_,
                                                   &beta,
                                                   descry_,
                                                   cudaDataType_,
                                                   alg_,
                                                   buffer_));
        std::cout << "\tpreProcess run" << std::endl;

        cusparseCheckError(cusparseSpMV(handle_,
                                        opA_,
                                        &alpha,
                                        descrA_,
                                        descrx_,
                                        &beta,
                                        descry_,
                                        cudaDataType_,
                                        alg_,
                                        buffer_));
        std::cout << "\tSpMV run" << std::endl;

        // Freeing memory
        cudaCheckError(cudaFree(buffer_));
        buffer_size_ = 0;
        break;
      }
    }
	}

  /** Perform any required steps after calling the GEMM kernel that should
   * be timed. */
  void postLoopRequirements() override {
    std::cout << std::endl << "##############################" << std::endl
              << "\tpostloop Requirements" << std::endl
              << "##############################" << std::endl;
    switch(offload_) {
      case gpuOffloadType::always: {
        break;
      }
      case gpuOffloadType::once: {
        cudaCheckError(cudaMemcpy(A_val_, A_val_dev_, vals_size_,
                                  cudaMemcpyDeviceToHost));
        cudaCheckError(cudaMemcpy(A_col_, A_col_dev_, cols_size_,
                                  cudaMemcpyDeviceToHost));
        cudaCheckError(cudaMemcpy(A_row_, A_row_dev_, rows_size_,
                                  cudaMemcpyDeviceToHost));
        std::cout << "\tA_ csr host arrays sunc" << std::endl;

        cudaCheckError(cudaMemcpy(x_, x_dev_, x_size_, cudaMemcpyDeviceToHost));
        std::cout << "\tx_ host array sunc" << std::endl;

        cudaCheckError(cudaMemcpy(y_, y_dev_, y_size_,
                                       cudaMemcpyDeviceToHost));
        std::cout << "\ty_ host array sunc" << std::endl;

        cusparseCheckError(cusparseDestroySpMat(descrA_));
        cusparseCheckError(cusparseDestroyDnVec(descrx_));
        cusparseCheckError(cusparseDestroyDnVec(descry_));
        break;
      }
      case gpuOffloadType::unified: {
        // Ensure all data resides on host once work has completed
        cudaCheckError(cudaMemPrefetchAsync(A_val_, vals_size_,
                                            cudaCpuDeviceId, s1_));
        cudaCheckError(cudaMemPrefetchAsync(A_col_, cols_size_,
                                            cudaCpuDeviceId, s1_));
        cudaCheckError(cudaMemPrefetchAsync(A_row_, rows_size_,
                                            cudaCpuDeviceId, s1_));
        std::cout << "\tA_ csr arrays sunc" << std::endl;

        cudaCheckError(cudaMemPrefetchAsync(x_, x_size_, cudaCpuDeviceId, s2_));
        std::cout << "\tx_ array sunc" << std::endl;

        cudaCheckError(cudaMemPrefetchAsync(y_, y_size_, cudaCpuDeviceId, s3_));
        std::cout << "\ty_ array sunc" << std::endl;


        // Ensure device has finished all work.
        cudaCheckError(cudaDeviceSynchronize());
        std::cout << "\tdevice and host sunc" << std::endl;

        cusparseCheckError(cusparseDestroySpMat(descrA_));
        cusparseCheckError(cusparseDestroyDnVec(descrx_));
        cusparseCheckError(cusparseDestroyDnVec(descry_));
        break;
      }
    }
  }

  /** Do any necessary cleanup (free pointers, close library handles, etc.)
   * after Kernel has been called. */
  void postCallKernelCleanup() override {

    free(A_);
    if (offload_ == gpuOffloadType::unified) {
      cudaCheckError(cudaFree(A_val_));
      cudaCheckError(cudaFree(A_col_));
      cudaCheckError(cudaFree(A_row_));
    } else {
      free(A_val_);
      free(A_col_);
      free(A_row_);
      cudaCheckError(cudaFree(A_val_dev_));
      cudaCheckError(cudaFree(A_col_dev_));
      cudaCheckError(cudaFree(A_row_dev_));
    }

    // Destroy the handle
    cusparseCheckError(cusparseDestroy(handle_));

    // Destroy streams after use
    cudaCheckError(cudaStreamDestroy(s1_));
    cudaCheckError(cudaStreamDestroy(s2_));
    cudaCheckError(cudaStreamDestroy(s3_));
  }


    void toCSR_int(T* dense, int n_col, int n_row, T* vals, int* col_index,
             int* row_ptr) {
      int nnz_encountered = 0;
      for (int row = 0; row < n_row; row++) {
        row_ptr[row] = nnz_encountered;
        for (int col = 0; col < n_col; col++) {
          if (dense[(row * n_) + col] != 0.0) {
            col_index[nnz_encountered] = col;
            vals[nnz_encountered] = dense[(row * n_) + col];
            nnz_encountered++;
          }
        }
      }
		};

  // ToDo -- the two following functons are useful for debugging.  I'm
  //  keeping them in to that end, though they are not used by the benchmark
  //  itself
  void printDenseMatrix(T* M, int rows, int cols) {
    for (int row = 0; row < rows; row++) {
      std::cout << "| ";
      for (int col = 0; col < cols; col++) {
        std::cout << M[(row * cols) + col] << " | ";
      }
      std::cout << std::endl;
    }
  }

  void printCSR(T* values, int* col_indices, int* row_pointers, int nnz,
                int rows, int cols) {
    std::cout << "\tRow pointers__" << std::endl;
    for (int p = 0; p < (rows + 1); p++) {
      std::cout << row_pointers[p] << ", ";
    }
    std::cout << std::endl << "\tColumn Indices__" << std::endl;
    for (int i = 0; i < nnz; i++) {
      std::cout << col_indices[i] << ", ";
    }
    std::cout << std::endl << "\tValues__" << std::endl;
    for (int v = 0; v < nnz; v++) {
      std::cout << values[v] << ", ";
    }
    std::cout << std::endl;
  }

  /**
   * ################################
   *        CUSPARSE STUFF
   * ################################
   */
  /** Handle used when calling cuBLAS. */
  cusparseHandle_t handle_;

  /** CUDA Streams - used to asynchronously move data between host and device.
   */
  cudaStream_t s1_;
  cudaStream_t s2_;
  cudaStream_t s3_;

  /** The ID of the target GPU Device. */
  int gpuDevice_;

	// Create descriptors for matrices A->C
	cusparseSpMatDescr_t descrA_;
  cusparseDnVecDescr_t descrx_, descry_;

	// Data type depends on kernel being run
	cudaDataType_t cudaDataType_;

	size_t buffer_size_ = 0;
  void* buffer_ = NULL;

  cusparseOperation_t opA_ = CUSPARSE_OPERATION_NON_TRANSPOSE;
  cusparseOperation_t opB_ = CUSPARSE_OPERATION_NON_TRANSPOSE;
  cusparseSpMVAlg_t alg_ = CUSPARSE_SPMV_CSR_ALG2;
  cusparseIndexType_t rType_ = CUSPARSE_INDEX_32I;
  cusparseIndexType_t cType_ = CUSPARSE_INDEX_32I;
  cusparseIndexBase_t indType_ = CUSPARSE_INDEX_BASE_ZERO;

  /** The constant value Alpha. */
  const T alpha = ALPHA;

  /** The constant value Beta. */
  const T beta = BETA;

  /**
   * ################################
   *        Matrix A parameters
   * ################################
   */
	/** CSR format vectors on the host (also used for USM) */
	T* A_val_;
	int *A_col_, *A_row_;
  /** CSR format vectors on the device. */
	T* A_val_dev_;
	int *A_col_dev_, *A_row_dev_;
  /** Metadata */
  uint64_t A_nnz_, vals_size_, cols_size_, rows_size_;

  /**
   * ################################
   *    Vectors x and y parameters
   * ################################
   */
  /** Vectors on the host (also used for USM) */
  T * x_host_, *y_host_;
  /** Vectors on the device */
  T *x_dev_, *y_dev_;
  /** Metadata */
  uint64_t x_size_, y_size_;
};
}  // namespace gpu
#endif