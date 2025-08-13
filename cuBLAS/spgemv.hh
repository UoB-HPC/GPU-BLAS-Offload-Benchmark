#pragma once

#ifdef GPU_CUBLAS
#include <cusparse_v2.h>
#include <cuda_runtime_api.h>
#include <type_traits>
#include <random>
#include <iostream>

#include "../include/kernels/GPU/spgemv.hh"
#include "../include/utilities.hh"
#include "common.hh"

namespace gpu {
/** A class for sparse GEMM GPU BLAS kernels. */
template <typename T>
class spgemv_gpu : public spgemv<T> {
 public:
  using spgemv<T>::spgemv;
  using spgemv<T>::initInputMatrixVector;
  using spgemv<T>::nnz_;
  using spgemv<T>::m_;
  using spgemv<T>::n_;
  using spgemv<T>::A_;
  using spgemv<T>::x_;
  using spgemv<T>::y_;
  using spgemv<T>::offload_;
  using spgemv<T>::sparsity_;

  ~spgemv_gpu() {
    // ToDo -- destroy the handle

    // Destroy streams after use
    cudaCheckError(cudaStreamDestroy(s1_));
    cudaCheckError(cudaStreamDestroy(s2_));
    cudaCheckError(cudaStreamDestroy(s3_));
  }

  void initialise(gpuOffloadType offload, int m, int n, 
                  double sparsity) override {
    if (print_) {
      switch (offload) {
        case gpuOffloadType::always:
          std::cout << "========== ALWAYS  ==========" << std::endl;
          break;
        case gpuOffloadType::unified:
          std::cout << "========== UNIFIED ==========" << std::endl;
          break;
        case gpuOffloadType::once:
          std::cout << "==========  ONCE   ==========" << std::endl;
          break;
      }
      std::cout << "Initialise" << std::endl;
    }

    offload_ = offload;

    sparsity_ = sparsity;

    // Create a handle for cuSPARSE
    cusparseCheckError(cusparseCreate(&handle_));
    cudaCheckError(cudaGetDevice(&gpuDevice_));

    // Setting cusparse metadata
    if (std::is_same_v<T, float>) dataType_ = CUDA_R_32F;
    else if (std::is_same_v<T, double>) dataType_ = CUDA_R_64F;
    else {
      std::cout << "INVALID DATA TYPE PASSED TO cuSPARSE" << std::endl;
      exit(1);
    }
    opA_ = opB_ = CUSPARSE_OPERATION_NON_TRANSPOSE;
    alg_ = CUSPARSE_SPMV_CSR_ALG2;
    index_ = CUSPARSE_INDEX_64I;
    base_ = CUSPARSE_INDEX_BASE_ZERO;


    m_ = m;
    n_ = n;

    // Initialise 3 streams to asynchronously move data between host and device
    cudaCheckError(cudaStreamCreate(&s1_));
    cudaCheckError(cudaStreamCreate(&s2_));
    cudaCheckError(cudaStreamCreate(&s3_));

    std::cout << "\tcuda streams created" << std::endl;


    vals_size_ = sizeof(T) * nnz_;
    cols_size_ = sizeof(int64_t) * nnz_;
    rows_size_ = sizeof(int64_t) * (m_ + 1);
    x_size_ = sizeof(T) * n_;
    y_size_ = sizeof(T) * m_;

    if (offload_ == gpuOffloadType::unified) {
      if (print_) std::cout << "\tAllocating arrays in unified memory" << std::endl;
      cudaCheckError(cudaMallocManaged(&A_vals_, vals_size_));
      cudaCheckError(cudaMallocManaged(&A_cols_, cols_size_));
      cudaCheckError(cudaMallocManaged(&A_rows_, rows_size_));

      cudaCheckError(cudaMallocManaged(&x_, x_size_));

      cudaCheckError(cudaMallocManaged(&y_, y_size_));
    } else {
      if (print_) std::cout << "\tAllocating arrays in local memory" << std::endl;
      A_vals_ = (T*)malloc(vals_size_);
      A_cols_ = (int64_t*)malloc(cols_size_);
      A_rows_ = (int64_t*)malloc(rows_size_);
      x_ = (T*)malloc(x_size_);
      y_ = (T*)malloc(y_size_);

      if (print_) std::cout << "\tAllocating arrays in GPU memory" << std::endl;
      cudaCheckError(cudaMalloc((void**)&A_vals_dev_, vals_size_));
      cudaCheckError(cudaMalloc((void**)&A_cols_dev_, cols_size_));
      cudaCheckError(cudaMalloc((void**)&A_rows_dev_, rows_size_));
      cudaCheckError(cudaMalloc((void**)&x_dev_, x_size_));
      cudaCheckError(cudaMalloc((void**)&y_dev_, y_size_));
    }

    A_ = (T*)malloc(sizeof(T) * m_ * n_);

    if (print_) std::cout << "\tInitialising input matrix and vector" << std::endl;
    initInputMatrixVector();
  }

protected:

  void toSparseFormat() override {
    if (print_) std::cout << "\tConverting matrix to sparse format" << std::endl;
    int64_t nnz_encountered = 0;
    for (int64_t row = 0; row < m_; row++) {
      A_rows_[row] = nnz_encountered;
      for (int64_t col = 0; col < n_; col++) {
        if (A_[(row * n_) + col] != 0.0) {
          A_cols_[nnz_encountered] = col;
          A_vals_[nnz_encountered] = A_[(row * n_) + col];
          nnz_encountered++;
        }
      }
    }
  }

 private:
  void preLoopRequirements() override {
    if (print_) std::cout << "Pre-loop stuff" << std::endl;
    switch(offload_) {
      case gpuOffloadType::always: {
        break;
      }
      case gpuOffloadType::once: {
        if (print_) std::cout << "\tCopying data to device" << std::endl;
        cudaCheckError(cudaMemcpy(A_vals_dev_, A_vals_, vals_size_,
                                  cudaMemcpyHostToDevice));
        cudaCheckError(cudaMemcpy(A_cols_dev_, A_cols_, cols_size_,
                                  cudaMemcpyHostToDevice));
        cudaCheckError(cudaMemcpy(A_rows_dev_, A_rows_, rows_size_,
                                  cudaMemcpyHostToDevice));
        cudaCheckError(cudaMemcpy(x_dev_, x_, x_size_,
                                       cudaMemcpyHostToDevice));
        cudaCheckError(cudaMemcpy(y_dev_, y_, y_size_,
                                       cudaMemcpyHostToDevice));
        cudaCheckError(cudaDeviceSynchronize());
        break;
      }
      case gpuOffloadType::unified: {
        if (print_) std::cout << "\tPrefetching memory to device" << std::endl;
        // Prefetch memory to device
        cudaCheckError(cudaMemPrefetchAsync(A_vals_, vals_size_, gpuDevice_,
                                            s1_));
        cudaCheckError(cudaMemPrefetchAsync(A_cols_, cols_size_, gpuDevice_,
                                            s1_));
        cudaCheckError(cudaMemPrefetchAsync(A_rows_, rows_size_, gpuDevice_,
                                            s1_));
        cudaCheckError(cudaMemPrefetchAsync(x_, x_size_, gpuDevice_, s2_));
        cudaCheckError(cudaMemPrefetchAsync(y_, y_size_, gpuDevice_, s3_));
        cudaCheckError(cudaDeviceSynchronize());
        break;
      }
    }
  }

  /** Make a call to the BLAS Library Kernel. */
  void callSpgemv() override {
    if (print_) std::cout << "Calling SpMV kernel" << std::endl;
    switch(offload_) {
      case gpuOffloadType::always: {
        if (print_) std::cout << "\tCopying data to device" << std::endl;
        cudaCheckError(cudaMemcpy(A_vals_dev_, A_vals_, vals_size_,
                                  cudaMemcpyHostToDevice));
        cudaCheckError(cudaMemcpy(A_cols_dev_, A_cols_, cols_size_,
                                  cudaMemcpyHostToDevice));
        cudaCheckError(cudaMemcpy(A_rows_dev_, A_rows_, rows_size_,
                                  cudaMemcpyHostToDevice));
        cudaCheckError(cudaMemcpy(x_dev_, x_, x_size_, cudaMemcpyHostToDevice));
        cudaCheckError(cudaMemcpy(y_dev_, y_, y_size_, cudaMemcpyHostToDevice));

        if (print_) std::cout << "\tMaking descriptors" << std::endl;
        cusparseCheckError(cusparseCreateCsr(&A_descr_,
                                             n_,
                                             m_,
                                             nnz_,
                                             A_rows_dev_,
                                             A_cols_dev_,
                                             A_vals_dev_,
                                             index_,
                                             index_,
                                             base_,
                                             dataType_));
        cusparseCheckError(cusparseCreateDnVec(&x_descr_,
                                               n_,
                                               x_dev_,
                                               dataType_));
        cusparseCheckError(cusparseCreateDnVec(&y_descr_,
                                               m_,
                                               y_dev_,
                                               dataType_));
        /*
         * Workflow is :
         *    cusparseSpMV_bufferSize
         *    cusparseSpMV
         */
        if (print_) std::cout << "\tCalling bufferSize" << std::endl;
        size_t bufferSize;
        void* dBuffer;
        cusparseCheckError(cusparseSpMV_bufferSize(handle_,
                                                   opA_,
                                                   &alpha,
                                                   A_descr_,
                                                   x_descr_,
                                                   &beta,
                                                   y_descr_,
                                                   dataType_,
                                                   alg_,
                                                   &bufferSize));

        if (print_) std::cout << "\tAllocating buffer" << std::endl;
        cudaCheckError(cudaMalloc(&dBuffer, bufferSize));

        if (print_) std::cout << "\tCalling SpMV" << std::endl;
        cusparseCheckError(cusparseSpMV(handle_,
                                        opA_,
                                        &alpha,
                                        A_descr_,
                                        x_descr_,
                                        &beta,
                                        y_descr_,
                                        dataType_,
                                        alg_,
                                        dBuffer));

        if (print_) std::cout << "\tDestroying descriptors" << std::endl;
        cusparseCheckError(cusparseDestroySpMat(A_descr_));
        cusparseCheckError(cusparseDestroyDnVec(x_descr_));
        cusparseCheckError(cusparseDestroyDnVec(y_descr_));
        cudaCheckError(cudaFree(dBuffer));

        if (print_) std::cout << "\tCopying data back to host" << std::endl;
        cudaCheckError(cudaMemcpy(y_, y_dev_, y_size_, cudaMemcpyDeviceToHost));
        break;
      }
      case gpuOffloadType::once: {
        if (print_) std::cout << "\tMaking descriptors" << std::endl;
        cusparseCheckError(cusparseCreateCsr(&A_descr_,
                                             n_,
                                             m_,
                                             nnz_,
                                             A_rows_dev_,
                                             A_cols_dev_,
                                             A_vals_dev_,
                                             index_,
                                             index_,
                                             base_,
                                             dataType_));
        cusparseCheckError(cusparseCreateDnVec(&x_descr_,
                                               n_,
                                               x_dev_,
                                               dataType_));
        cusparseCheckError(cusparseCreateDnVec(&y_descr_,
                                               m_,
                                               y_dev_,
                                               dataType_));
        /*
         * Workflow is :
         *    cusparseSpMV_bufferSize
         *    cusparseSpMV
         */
        if (print_) std::cout << "\tCalling bufferSize" << std::endl;
        size_t bufferSize;
        void* dBuffer;
        cusparseCheckError(cusparseSpMV_bufferSize(handle_,
                                                   opA_,
                                                   &alpha,
                                                   A_descr_,
                                                   x_descr_,
                                                   &beta,
                                                   y_descr_,
                                                   dataType_,
                                                   alg_,
                                                   &bufferSize));

        if (print_) std::cout << "\tAllocating buffer" << std::endl;
        cudaCheckError(cudaMalloc(&dBuffer, bufferSize));

        if (print_) std::cout << "\tCalling SpMV" << std::endl;
        cusparseCheckError(cusparseSpMV(handle_,
                                        opA_,
                                        &alpha,
                                        A_descr_,
                                        x_descr_,
                                        &beta,
                                        y_descr_,
                                        dataType_,
                                        alg_,
                                        dBuffer));

        if (print_) std::cout << "\tDestroying descriptors" << std::endl;
        cusparseCheckError(cusparseDestroySpMat(A_descr_));
        cusparseCheckError(cusparseDestroyDnVec(x_descr_));
        cusparseCheckError(cusparseDestroyDnVec(y_descr_));
        cudaCheckError(cudaFree(dBuffer));
        break;
      }
      case gpuOffloadType::unified: {
        if (print_) std::cout << "\tMaking descriptors" << std::endl;
        cusparseCheckError(cusparseCreateCsr(&A_descr_,
                                             n_,
                                             m_,
                                             nnz_,
                                             A_rows_,
                                             A_cols_,
                                             A_vals_,
                                             index_,
                                             index_,
                                             base_,
                                             dataType_));
        cusparseCheckError(cusparseCreateDnVec(&x_descr_,
                                               n_,
                                               x_,
                                               dataType_));
        cusparseCheckError(cusparseCreateDnVec(&y_descr_,
                                               m_,
                                               y_,
                                               dataType_));
        /*
         * Workflow is :
         *    cusparseSpMV_bufferSize
         *    cusparseSpMV
         */
        if (print_) std::cout << "\tCalling bufferSize" << std::endl;
        size_t bufferSize;
        void* dBuffer;
        cusparseCheckError(cusparseSpMV_bufferSize(handle_,
                                                   opA_,
                                                   &alpha,
                                                   A_descr_,
                                                   x_descr_,
                                                   &beta,
                                                   y_descr_,
                                                   dataType_,
                                                   alg_,
                                                   &bufferSize));

        if (print_) std::cout << "\tAllocating buffer" << std::endl;
        cudaCheckError(cudaMalloc(&dBuffer, bufferSize));

        if (print_) std::cout << "\tCalling SpMV" << std::endl;
        cusparseCheckError(cusparseSpMV(handle_,
                                        opA_,
                                        &alpha,
                                        A_descr_,
                                        x_descr_,
                                        &beta,
                                        y_descr_,
                                        dataType_,
                                        alg_,
                                        dBuffer));

        if (print_) std::cout << "\tDestroying descriptors" << std::endl;
        cusparseCheckError(cusparseDestroySpMat(A_descr_));
        cusparseCheckError(cusparseDestroyDnVec(x_descr_));
        cusparseCheckError(cusparseDestroyDnVec(y_descr_));
        cudaCheckError(cudaFree(dBuffer));
        break;
      }
    }
	}

  /** Perform any required steps after calling the GEMM kernel that should
   * be timed. */
  void postLoopRequirements() override {
    switch(offload_) {
      case gpuOffloadType::always: {
        break;
      }
      case gpuOffloadType::once: {
        if (print_) std::cout << "\tCopying result back to CPU" << std::endl;
        cudaCheckError(cudaMemcpyAsync(y_, y_dev_, sizeof(T) * m_,
                                       cudaMemcpyDeviceToHost, s3_));
        break;
      }
      case gpuOffloadType::unified: {
        if (print_) std::cout << "\tPrefetching result back to CPU" << std::endl;
        cudaCheckError(cudaMemPrefetchAsync(y_, y_size_, cudaCpuDeviceId, s3_));
        break;
      }
    }
  }

  /** Do any necessary cleanup (free pointers, close library handles, etc.)
   * after Kernel has been called. */
  void postCallKernelCleanup() override {
    free(A_);
    if (offload_ == gpuOffloadType::unified) {
      cudaCheckError(cudaFree(A_vals_));
      cudaCheckError(cudaFree(A_cols_));
      cudaCheckError(cudaFree(A_rows_));
      cudaCheckError(cudaFree(x_));
      cudaCheckError(cudaFree(y_));
    } else {
      free(A_vals_);
      free(A_cols_);
      free(A_rows_);
      free(x_);
      free(y_);
      cudaCheckError(cudaFree(A_vals_dev_));
      cudaCheckError(cudaFree(A_cols_dev_));
      cudaCheckError(cudaFree(A_rows_dev_));
      cudaCheckError(cudaFree(x_dev_));
      cudaCheckError(cudaFree(y_dev_));
    }

    // Destroy the handle
    cusparseCheckError(cusparseDestroy(handle_));

    // Destroy streams after use
    cudaCheckError(cudaStreamDestroy(s1_));
    cudaCheckError(cudaStreamDestroy(s2_));
    cudaCheckError(cudaStreamDestroy(s3_));
  }

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

  void printCSR(T* values, int64_t* col_indices, int64_t* row_pointers, int nnz,
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

  bool print_ = false;

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
	cusparseSpMatDescr_t A_descr_;
  cusparseDnVecDescr_t x_descr_, y_descr_;

	// cusparse metadata variables
	cudaDataType_t dataType_;
  cusparseOperation_t opA_;
  cusparseOperation_t opB_;
  cusparseSpMVAlg_t alg_;
  cusparseIndexType_t index_;
  cusparseIndexBase_t base_;

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
	T* A_vals_;
	int64_t* A_cols_;
  int64_t* A_rows_;
  /** CSR format vectors on the device. */
	T* A_vals_dev_;
	int64_t* A_cols_dev_;
	int64_t* A_rows_dev_;
  /** Metadata */
  uint64_t vals_size_;
  uint64_t cols_size_;
  uint64_t rows_size_;

  /**
   * ################################
   *    Vectors x and y parameters
   * ################################
   */
  /** Vectors on the host (also used for USM) */
  T* x_host_;
  T* y_host_;
  /** Vectors on the device */
  T* x_dev_;
  T* y_dev_;
  /** Metadata */
  uint64_t x_size_;
  uint64_t y_size_;
};
}  // namespace gpu
#endif