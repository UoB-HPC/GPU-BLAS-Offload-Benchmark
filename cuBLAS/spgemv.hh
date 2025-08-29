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
  using spgemv<T>::x_;
  using spgemv<T>::y_;
  using spgemv<T>::offload_;
  using spgemv<T>::sparsity_;

  ~spgemv_gpu() {}

  void initialise(gpuOffloadType offload, int m, int n, 
                  double sparsity) override {
    if (print_ || debug) {
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
    }
    if (print_) std::cout << "Initialise " << m_ << "x" << n_ << " . " << n_ << std::endl;
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
    alg_ = CUSPARSE_SPMV_ALG_DEFAULT;
    index_ = CUSPARSE_INDEX_64I;
    base_ = CUSPARSE_INDEX_BASE_ZERO;

    m_ = m;
    n_ = n;
    nnz_ = 1 + (uint64_t)((double)m_ * (double)n_ * (1.0 - sparsity_));

    // Initialise 3 streams to asynchronously move data between host and device
    cudaCheckError(cudaStreamCreate(&s1_));
    cudaCheckError(cudaStreamCreate(&s2_));
    cudaCheckError(cudaStreamCreate(&s3_));

    if (print_) std::cout << "\tcuda streams created" << std::endl;

    // Allocate dense data structures
    if (offload_ == gpuOffloadType::unified) {
      if (print_) std::cout << "\tAllocating arrays in unified memory" << std::endl;
      cudaCheckError(cudaMallocManaged(&x_, n_ * sizeof(T)));
      cudaCheckError(cudaMallocManaged(&y_, m_ * sizeof(T)));
      cudaCheckError(cudaDeviceSynchronize());
    } else {
      if (print_) std::cout << "\tAllocating arrays in local memory" << std::endl;
      x_ = (T*)malloc(n_ * sizeof(T));
      y_ = (T*)malloc(m_ * sizeof(T));

      if (print_) std::cout << "\tAllocating arrays in GPU memory" << std::endl;
      cudaCheckError(cudaMalloc((void**)&x_dev_, n_ * sizeof(T)));
      cudaCheckError(cudaMalloc((void**)&y_dev_, m_ * sizeof(T)));
      cudaCheckError(cudaDeviceSynchronize());
    }

    if (print_) std::cout << "\tInitialising input matrix and vector" << std::endl;
    initInputMatrixVector();
    if (debug) {
      std::cout << "===============Initialised=================" << std::endl;
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
      std::cout << "===============Sparsified==================" << std::endl;
      std::cout << "___________________________________________" << std::endl;
      std::cout << "nnz = " << nnz_ << std::endl;
      std::cout << "A rows = [";
      for (int64_t i = 0; i < (m_ + 1); i++) {
        std::cout << A_rows_[i];
        if (i < m_) std::cout << ", ";
      }
      std::cout << "]" << std::endl;
      std::cout << "A cols = [";
      for (int64_t i = 0; i < nnz_; i++) {
        std::cout << A_cols_[i];
        if (i < (nnz_ - 1)) std::cout << ", ";
      }
      std::cout << "]" << std::endl;
      std::cout << "A vals = [";
      for (int64_t i = 0; i < nnz_; i++) {
        std::cout << A_vals_[i];
        if (i < (nnz_ - 1)) std::cout << ", ";
      }
      std::cout << "]" << std::endl;
      std::cout << "___________________________________________" << std::endl;
    }
  }

protected:

  void toSparseFormat() override {

    if (print_) std::cout << "\tAllocating sparse data structures" << std::endl;
    if (offload_ == gpuOffloadType::unified) {
      cudaCheckError(cudaMallocManaged(&A_vals_, nnz_ * sizeof(T)));
      cudaCheckError(cudaMallocManaged(&A_cols_, nnz_ * sizeof(int64_t)));
      cudaCheckError(cudaMallocManaged(&A_rows_, (m_ + 1) * sizeof(int64_t)));
    } else {      
      A_vals_ = (T*)malloc(nnz_ * sizeof(T));
      A_cols_ = (int64_t*)malloc(nnz_ * sizeof(int64_t));
      A_rows_ = (int64_t*)malloc((m_ + 1) * sizeof(int64_t));
      cudaCheckError(cudaMalloc((void**)&A_vals_dev_, nnz_ * sizeof(T)));
      cudaCheckError(cudaMalloc((void**)&A_cols_dev_, nnz_ * sizeof(int64_t)));
      cudaCheckError(cudaMalloc((void**)&A_rows_dev_, (m_ + 1) * sizeof(int64_t)));
    }
    cudaCheckError(cudaDeviceSynchronize());

    rMatCSR<T, int64_t>(A_vals_, A_cols_, A_rows_, m_, n_, nnz_);
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
        cudaCheckError(cudaMemcpy(A_vals_dev_, A_vals_, nnz_ * sizeof(T), cudaMemcpyHostToDevice));
        cudaCheckError(cudaMemcpy(A_cols_dev_, A_cols_, nnz_ * sizeof(int64_t), cudaMemcpyHostToDevice));
        cudaCheckError(cudaMemcpy(A_rows_dev_, A_rows_, (m_ + 1) * sizeof(int64_t), cudaMemcpyHostToDevice));
        cudaCheckError(cudaMemcpy(x_dev_, x_, n_ * sizeof(T), cudaMemcpyHostToDevice));
        cudaCheckError(cudaMemcpy(y_dev_, y_, m_ * sizeof(T), cudaMemcpyHostToDevice));
        cudaCheckError(cudaDeviceSynchronize());
        break;
      }
      case gpuOffloadType::unified: {
        if (print_) std::cout << "\tPrefetching memory to device" << std::endl;
        // Prefetch memory to device
        cudaCheckError(cudaMemPrefetchAsync(A_vals_, nnz_ * sizeof(T), gpuDevice_, s1_));
        cudaCheckError(cudaMemPrefetchAsync(A_cols_, nnz_ * sizeof(int64_t), gpuDevice_, s1_));
        cudaCheckError(cudaMemPrefetchAsync(A_rows_, (m_ + 1) * sizeof(int64_t), gpuDevice_, s1_));
        cudaCheckError(cudaMemPrefetchAsync(x_, n_ * sizeof(T), gpuDevice_, s2_));
        cudaCheckError(cudaMemPrefetchAsync(y_, m_ * sizeof(T), gpuDevice_, s3_));
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
        cudaCheckError(cudaMemcpy(A_vals_dev_, A_vals_, nnz_ * sizeof(T),
                                  cudaMemcpyHostToDevice));
        cudaCheckError(cudaMemcpy(A_cols_dev_, A_cols_, nnz_ * sizeof(int64_t),
                                  cudaMemcpyHostToDevice));
        cudaCheckError(cudaMemcpy(A_rows_dev_, A_rows_, (m_ + 1) * sizeof(int64_t),
                                  cudaMemcpyHostToDevice));
        cudaCheckError(cudaMemcpy(x_dev_, x_, n_ * sizeof(T), cudaMemcpyHostToDevice));
        cudaCheckError(cudaMemcpy(y_dev_, y_, m_ * sizeof(T), cudaMemcpyHostToDevice));

        if (print_) std::cout << "\tMaking descriptors" << std::endl;
        cusparseCheckError(cusparseCreateCsr(&A_descr_,
                                             m_,
                                             n_,
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
        cudaCheckError(cudaDeviceSynchronize());
        /*
         * Workflow is :
         *    cusparseSpMV_bufferSize
         *    cusparseSpMV
         */
        if (print_) std::cout << "\tCalling bufferSize" << std::endl;
        size_t bufferSize;
        void* dBuffer = nullptr;
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
        cudaCheckError(cudaDeviceSynchronize());

        if (print_) std::cout << "\tAllocating buffer of size " << bufferSize << std::endl;
        if (bufferSize > 0) cudaCheckError(cudaMalloc(&dBuffer, bufferSize));
        cudaCheckError(cudaDeviceSynchronize());

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
        cudaCheckError(cudaDeviceSynchronize());

        if (print_) std::cout << "\tDestroying descriptors" << std::endl;
        cusparseCheckError(cusparseDestroySpMat(A_descr_));
        cusparseCheckError(cusparseDestroyDnVec(x_descr_));
        cusparseCheckError(cusparseDestroyDnVec(y_descr_));

        cudaCheckError(cudaDeviceSynchronize());
        if (dBuffer != nullptr) cudaCheckError(cudaFree(dBuffer));

        if (print_) std::cout << "\tCopying data back to host" << std::endl;
        cudaCheckError(cudaMemcpy(y_, y_dev_, m_ * sizeof(T), cudaMemcpyDeviceToHost));
        cudaCheckError(cudaDeviceSynchronize());
        break;
      }
      case gpuOffloadType::once: {
        if (print_) std::cout << "\tMaking descriptors" << std::endl;
        cusparseCheckError(cusparseCreateCsr(&A_descr_,
                                             m_,
                                             n_,
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
        cudaCheckError(cudaDeviceSynchronize());
        /*
         * Workflow is :
         *    cusparseSpMV_bufferSize
         *    cusparseSpMV
         */
        if (print_) std::cout << "\tCalling bufferSize" << std::endl;
        size_t bufferSize;
        void* dBuffer = nullptr;
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
        cudaCheckError(cudaDeviceSynchronize());

        if (print_) std::cout << "\tAllocating buffer of size " << bufferSize << std::endl;
        if (bufferSize > 0) cudaCheckError(cudaMalloc(&dBuffer, bufferSize));
        cudaCheckError(cudaDeviceSynchronize());

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
        cudaCheckError(cudaDeviceSynchronize());

        if (print_) std::cout << "\tDestroying descriptors" << std::endl;
        cusparseCheckError(cusparseDestroySpMat(A_descr_));
        cusparseCheckError(cusparseDestroyDnVec(x_descr_));
        cusparseCheckError(cusparseDestroyDnVec(y_descr_));
        cudaCheckError(cudaDeviceSynchronize());
        if (dBuffer != nullptr) cudaCheckError(cudaFree(dBuffer));
        cudaCheckError(cudaDeviceSynchronize());
        break;
      }
      case gpuOffloadType::unified: {
        if (print_) std::cout << "\tMaking descriptors" << std::endl;
        cusparseCheckError(cusparseCreateCsr(&A_descr_,
                                             m_,
                                             n_,
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
        cudaCheckError(cudaDeviceSynchronize());
        /*
         * Workflow is :
         *    cusparseSpMV_bufferSize
         *    cusparseSpMV
         */
        if (print_) std::cout << "\tCalling bufferSize" << std::endl;
        size_t bufferSize;
        void* dBuffer = nullptr;
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
        cudaCheckError(cudaDeviceSynchronize());

        if (print_) std::cout << "\tAllocating buffer of size " << bufferSize << std::endl;
        if (bufferSize > 0) cudaCheckError(cudaMalloc(&dBuffer, bufferSize));
        cudaCheckError(cudaDeviceSynchronize());

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
        cudaCheckError(cudaDeviceSynchronize());
        if (dBuffer != nullptr) cudaCheckError(cudaFree(dBuffer));
        cudaCheckError(cudaDeviceSynchronize());
        break;
      }
    }
	}

  /** Perform any required steps after calling the GEMM kernel that should
   * be timed. */
  void postLoopRequirements() override {
    if (print_) std::cout << "Post-loop stuff" << std::endl;
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
        cudaCheckError(cudaMemPrefetchAsync(y_, m_ * sizeof(T), cudaCpuDeviceId, s3_));
        break;
      }
    }
    cudaCheckError(cudaDeviceSynchronize());
  
  }

  /** Do any necessary cleanup (free pointers, close library handles, etc.)
   * after Kernel has been called. */
  void postCallKernelCleanup() override {
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
  bool debug = false;


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
};
}  // namespace gpu
#endif