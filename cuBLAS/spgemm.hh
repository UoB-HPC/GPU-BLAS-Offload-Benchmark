#pragma once

#ifdef GPU_CUBLAS
#include <cusparse_v2.h>
#include <cuda_runtime.h>
#include <type_traits>
#include <random>
#include <iostream>

#include "../include/kernels/GPU/spgemm.hh"
#include "../include/utilities.hh"
#include "common.hh"

namespace gpu {
    /**
     * A class for sparse matrix-dense matrix BLAS
     */
template <typename T>
class spgemm_gpu : public spgemm<T> {
public:
  using spgemm<T>::spgemm;
  using spgemm<T>::initInputMatrices;
  using spgemm<T>::m_;
  using spgemm<T>::n_;
  using spgemm<T>::k_;
  using spgemm<T>::B_;
  using spgemm<T>::C_;
  using spgemm<T>::offload_;
  using spgemm<T>::nnz_;
  using spgemm<T>::sparsity_;
  using spgemm<T>::type_;

  ~spgemm_gpu() {
    if (alreadyInitialised_) {
      cusparseCheckError(cusparseDestroy(handle_));

      cudaCheckError(cudaStreamDestroy(stream1_));
      cudaCheckError(cudaStreamDestroy(stream2_));
      cudaCheckError(cudaStreamDestroy(stream3_));
      cudaCheckError(cudaStreamDestroy(stream4_));
      cudaCheckError(cudaStreamDestroy(stream5_));

      alreadyInitialised_ = false;
    }
  }

  void initialise(gpuOffloadType offload, int m, int n, int k,
                  double sparsity, matrixType type, bool binary = false) override {
    if (!alreadyInitialised_) {
      alreadyInitialised_ = true;
      cusparseCheckError(cusparseCreate(&handle_));
      
      cudaCheckError(cudaStreamCreate(&stream1_));
      cudaCheckError(cudaStreamCreate(&stream2_));
      cudaCheckError(cudaStreamCreate(&stream3_));
      cudaCheckError(cudaStreamCreate(&stream4_));
      cudaCheckError(cudaStreamCreate(&stream5_));
      
      cusparseCheckError(cusparseSetStream(handle_, stream1_));

      // Get device identifier
      cudaCheckError(cudaGetDevice(&gpuDevice_));

    }

    offload_ = offload;
    sparsity_ = sparsity;
    type_ = type;

    m_ = m;
    n_ = n;
    k_ = k;

    B_ = C_ = B_dev_ = C_dev_ = A_vals_ = A_vals_dev_ = nullptr;
    A_rows_ = A_cols_ = A_rows_dev_ = A_cols_dev_ = nullptr;
    /** Determine the number of nnz elements in A and B */
    nnz_ = 1 + (uint64_t)((double)m_ * (double)k_ * (1.0 - sparsity_));
    
    // Set up cuSPARSE metadata
    opA_ = CUSPARSE_OPERATION_NON_TRANSPOSE;
    opB_ = CUSPARSE_OPERATION_NON_TRANSPOSE;
    alg_ = CUSPARSE_SPMM_ALG_DEFAULT;
    index_ = CUSPARSE_INDEX_64I;
    base_ = CUSPARSE_INDEX_BASE_ZERO;
    B_order_ = CUSPARSE_ORDER_ROW;
    C_order_ = CUSPARSE_ORDER_ROW;
    if (std::is_same_v<T, float>) {
      dataType_ = CUDA_R_32F;
    } else if (std::is_same_v<T, double>) {
      dataType_ = CUDA_R_64F;
    } else {
      std::cerr << "INVALID DATA TYPE PASSED TO cuSPARSE" << std::endl;
      exit(1);
    }

    if (offload_ == gpuOffloadType::unified) {
      cudaCheckError(cudaMallocManaged(&B_, sizeof(T) * k_ * n_));
      cudaCheckError(cudaMallocManaged(&C_, sizeof(T) * m_ * n_));
    } else {
      B_ = (T*)malloc(sizeof(T) * k_ * n_);
      C_ = (T*) malloc(sizeof(T) * m_ * n_);

      cudaCheckError(cudaMalloc((void**)&B_dev_, sizeof(T) * k_ * n_));
      cudaCheckError(cudaMalloc((void**)&C_dev_, sizeof(T) * m_ * n_));
    }
    cudaCheckError(cudaDeviceSynchronize());

    initInputMatrices();
  }

protected:
  void toSparseFormat() override {
    // Allocate CSR arrays
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

    if (type_ == matrixType::rmat) {
      rMatCSR<T, int64_t>(A_vals_, A_cols_, A_rows_, m_, k_, nnz_);
    } else if (type_ == matrixType::random) {
      randomCSR<T, int64_t>(A_vals_, A_cols_, A_rows_, m_, k_, nnz_);
    } else {
      exit(1);
    }
  }

private:
  void preLoopRequirements() override {
    switch(offload_) {
      case gpuOffloadType::always: {
        break;
      }
      case gpuOffloadType::once: {
        cudaCheckError(cudaMemcpyAsync(A_vals_dev_, A_vals_, nnz_ * sizeof(T), cudaMemcpyHostToDevice, stream1_));
        cudaCheckError(cudaMemcpyAsync(A_cols_dev_, A_cols_, nnz_ * sizeof(int64_t), cudaMemcpyHostToDevice, stream2_));
        cudaCheckError(cudaMemcpyAsync(A_rows_dev_, A_rows_, (m_ + 1) * sizeof(int64_t), cudaMemcpyHostToDevice, stream3_));
        cudaCheckError(cudaMemcpyAsync(B_dev_, B_, (k_ * n_) * sizeof(T), cudaMemcpyHostToDevice, stream4_));
        cudaCheckError(cudaMemcpyAsync(C_dev_, C_, (m_ * n_) * sizeof(T), cudaMemcpyHostToDevice, stream5_));
        break;
      }
      case gpuOffloadType::unified: {
        cudaCheckError(cudaMemPrefetchAsync(A_vals_, nnz_ * sizeof(T), gpuDevice_, stream1_));
        cudaCheckError(cudaMemPrefetchAsync(A_cols_, nnz_ * sizeof(int64_t), gpuDevice_, stream2_));
        cudaCheckError(cudaMemPrefetchAsync(A_rows_, (m_ + 1) * sizeof(int64_t), gpuDevice_, stream3_));
        cudaCheckError(cudaMemPrefetchAsync(B_, (n_ * k_) * sizeof(T), gpuDevice_, stream4_));
        cudaCheckError(cudaMemPrefetchAsync(C_, (m_ * n_) * sizeof(T), gpuDevice_, stream5_));
        cudaCheckError(cudaDeviceSynchronize());
        break;
      }
    }
  }

  void callSpgemm() override {
    switch(offload_) {
      case gpuOffloadType::always: {
        // Move over data
        cudaCheckError(cudaMemcpyAsync(A_vals_dev_, A_vals_, nnz_ * sizeof(T), cudaMemcpyHostToDevice, stream1_));
        cudaCheckError(cudaMemcpyAsync(A_cols_dev_, A_cols_, nnz_ * sizeof(int64_t), cudaMemcpyHostToDevice, stream2_));
        cudaCheckError(cudaMemcpyAsync(A_rows_dev_, A_rows_, (m_ + 1) * sizeof(int64_t), cudaMemcpyHostToDevice, stream3_));
        cudaCheckError(cudaMemcpyAsync(B_dev_, B_, (k_ * n_) * sizeof(T), cudaMemcpyHostToDevice, stream4_));
        cudaCheckError(cudaMemcpyAsync(C_dev_, C_, (m_ * n_) * sizeof(T), cudaMemcpyHostToDevice, stream5_));

        // Set up descriptors
        cusparseCheckError(cusparseCreateCsr(&A_descr_, 
                                             m_, 
                                             k_, 
                                             nnz_, 
                                             A_rows_dev_, 
                                             A_cols_dev_,
                                             A_vals_dev_, 
                                             index_, 
                                             index_, 
                                             base_, 
                                             dataType_));
        cusparseCheckError(cusparseCreateDnMat(&B_descr_, 
                                               k_, 
                                               n_,
                                               n_, 
                                               B_dev_, 
                                               dataType_,
                                               B_order_));
        cusparseCheckError(cusparseCreateDnMat(&C_descr_, 
                                               m_, 
                                               n_,
                                               n_, 
                                               C_dev_, 
                                               dataType_,
                                               C_order_));

        // Set up temporary buffers
        void* dBuffer = nullptr;
        size_t bufferSize = 0;

        // Begin matrix-matrix multiplication
        cusparseCheckError(cusparseSpMM_bufferSize(handle_, 
                                                   opA_, 
                                                   opB_, 
                                                   &alpha, 
                                                   A_descr_,
                                                   B_descr_, 
                                                   &beta, 
                                                   C_descr_,
                                                   dataType_, 
                                                   alg_, 
                                                   &bufferSize));

        // Allocate the temporary buffer
        if (bufferSize > 0) cudaCheckError(cudaMalloc((void**)&dBuffer, bufferSize));

        cusparseCheckError(cusparseSpMM_preprocess(handle_, 
                                                   opA_, 
                                                   opB_, 
                                                   &alpha, 
                                                   A_descr_,
                                                   B_descr_, 
                                                   &beta, 
                                                   C_descr_,
                                                   dataType_, 
                                                   alg_, 
                                                   dBuffer));

        cusparseCheckError(cusparseSpMM(handle_, 
                                        opA_, 
                                        opB_, 
                                        &alpha, 
                                        A_descr_,
                                        B_descr_,
                                        &beta, 
                                        C_descr_,
                                        dataType_,
                                        alg_,
                                        dBuffer));
        cudaCheckError(cudaDeviceSynchronize());

        // Clean up descriptors
        cusparseCheckError(cusparseDestroySpMat(A_descr_));
        cusparseCheckError(cusparseDestroyDnMat(B_descr_));
        cusparseCheckError(cusparseDestroyDnMat(C_descr_));

        // Free up the temporary buffer
        if (bufferSize > 0) cudaCheckError(cudaFree(dBuffer));

        // Move result back to CPU
        cudaCheckError(cudaMemcpyAsync(C_, C_dev_, (sizeof(T) * m_ * n_),
                                       cudaMemcpyDeviceToHost, stream1_));
        cudaCheckError(cudaDeviceSynchronize());
        break;
      }
      case gpuOffloadType::once: {
        // Set up descriptors
        cusparseCheckError(cusparseCreateCsr(&A_descr_, 
                                             m_, 
                                             k_, 
                                             nnz_, 
                                             A_rows_dev_, 
                                             A_cols_dev_,
                                             A_vals_dev_, 
                                             index_, 
                                             index_, 
                                             base_, 
                                             dataType_));
        cusparseCheckError(cusparseCreateDnMat(&B_descr_, 
                                               k_, 
                                               n_,
                                               n_, 
                                               B_dev_, 
                                               dataType_,
                                               B_order_));
        cusparseCheckError(cusparseCreateDnMat(&C_descr_, 
                                               m_, 
                                               n_,
                                               n_, 
                                               C_dev_, 
                                               dataType_,
                                               C_order_));

        // Set up temporary buffers
        void* dBuffer = nullptr;
        size_t bufferSize = 0;

        // Begin matrix-matrix multiplication
        cusparseCheckError(cusparseSpMM_bufferSize(handle_, 
                                                   opA_, 
                                                   opB_, 
                                                   &alpha, 
                                                   A_descr_,
                                                   B_descr_, 
                                                   &beta, 
                                                   C_descr_,
                                                   dataType_, 
                                                   alg_, 
                                                   &bufferSize));

        // Allocate the temporary buffer
        if (bufferSize > 0) cudaCheckError(cudaMalloc((void**)&dBuffer, bufferSize));

        cusparseCheckError(cusparseSpMM_preprocess(handle_, 
                                                   opA_, 
                                                   opB_, 
                                                   &alpha, 
                                                   A_descr_,
                                                   B_descr_, 
                                                   &beta, 
                                                   C_descr_,
                                                   dataType_, 
                                                   alg_, 
                                                   dBuffer));

        cusparseCheckError(cusparseSpMM(handle_, 
                                        opA_, 
                                        opB_, 
                                        &alpha, 
                                        A_descr_,
                                        B_descr_,
                                        &beta, 
                                        C_descr_,
                                        dataType_, 
                                        alg_, 
                                        dBuffer));

        // Clean up descriptors
        cusparseCheckError(cusparseDestroySpMat(A_descr_));
        cusparseCheckError(cusparseDestroyDnMat(B_descr_));
        cusparseCheckError(cusparseDestroyDnMat(C_descr_));

        // Free up the temporary buffer
        if (bufferSize > 0) cudaCheckError(cudaFree(dBuffer));
      }
      case gpuOffloadType::unified: {
        // Create descriptors for the matrices
        cusparseCheckError(cusparseCreateCsr(&A_descr_, 
                                             m_, 
                                             k_, 
                                             nnz_, 
                                             A_rows_, 
                                             A_cols_,
                                             A_vals_, 
                                             index_, 
                                             index_, 
                                             base_, 
                                             dataType_));
        cusparseCheckError(cusparseCreateDnMat(&B_descr_, 
                                               k_, 
                                               n_,
                                               n_, 
                                               B_, 
                                               dataType_,
                                               B_order_));
        cusparseCheckError(cusparseCreateDnMat(&C_descr_, 
                                               m_, 
                                               n_,
                                               n_, 
                                               C_, 
                                               dataType_,
                                               C_order_));

        // Set up temporary buffers
        void* dBuffer = nullptr;
        size_t bufferSize = 0;

        // Begin matrix-matrix multiplication
        cusparseCheckError(cusparseSpMM_bufferSize(handle_, 
                                                   opA_, 
                                                   opB_, 
                                                   &alpha, 
                                                   A_descr_,
                                                   B_descr_, 
                                                   &beta, 
                                                   C_descr_,
                                                   dataType_, 
                                                   alg_, 
                                                   &bufferSize));

        // Allocate the temporary buffer
        if (bufferSize > 0) cudaCheckError(cudaMalloc((void**)&dBuffer, bufferSize));

        cusparseCheckError(cusparseSpMM_preprocess(handle_, 
                                                   opA_, 
                                                   opB_, 
                                                   &alpha, 
                                                   A_descr_,
                                                   B_descr_, 
                                                   &beta, 
                                                   C_descr_,
                                                   dataType_, 
                                                   alg_, 
                                                   dBuffer));
        cudaCheckError(cudaDeviceSynchronize());

        cusparseCheckError(cusparseSpMM(handle_, 
                                        opA_, 
                                        opB_, 
                                        &alpha, 
                                        A_descr_, 
                                        B_descr_,
                                        &beta, 
                                        C_descr_, 
                                        dataType_, 
                                        alg_, 
                                        dBuffer));
        cudaCheckError(cudaDeviceSynchronize());

        // Clean up descriptors
        cusparseCheckError(cusparseDestroySpMat(A_descr_));
        cusparseCheckError(cusparseDestroyDnMat(B_descr_));
        cusparseCheckError(cusparseDestroyDnMat(C_descr_));
        cudaCheckError(cudaDeviceSynchronize());

        // Free up the temporary buffer
        if (bufferSize > 0)cudaCheckError(cudaFree(dBuffer));
        break;
      }
    }
  }

  void postLoopRequirements() override {
    switch (offload_) {
      case gpuOffloadType::always: {
        break;
      }  
      case gpuOffloadType::once: {
        // Move result back to CPU
        cudaCheckError(cudaMemcpyAsync(C_, C_dev_, (sizeof(T) * m_ * n_),
                                       cudaMemcpyDeviceToHost, stream1_));
        cudaCheckError(cudaDeviceSynchronize());
        break;
      }
      case gpuOffloadType::unified: {
        // Move result back to CPU
        cudaCheckError(cudaMemPrefetchAsync(C_, sizeof(T) * m_ * n_, 
                                            cudaCpuDeviceId, stream1_));
        cudaCheckError(cudaDeviceSynchronize());
        break;
      }
    }
  }

  void postCallKernelCleanup() override {
    if (offload_ == gpuOffloadType::unified) {
      cudaCheckError(cudaFree(A_vals_));
      cudaCheckError(cudaFree(A_cols_));
      cudaCheckError(cudaFree(A_rows_));
      cudaCheckError(cudaFree(B_));
      cudaCheckError(cudaFree(C_));
    } else {
      free(A_vals_);
      free(A_cols_);
      free(A_rows_);
      free(B_);
      free(C_);
      cudaCheckError(cudaFree(A_vals_dev_));
      cudaCheckError(cudaFree(A_cols_dev_));
      cudaCheckError(cudaFree(A_rows_dev_));
      cudaCheckError(cudaFree(B_dev_));
      cudaCheckError(cudaFree(C_dev_));
    }
  }

  bool alreadyInitialised_ = false;

  /** Handle used when calling cuBLAS. */
  cusparseHandle_t handle_;

  /** CUDA Streams - used to asynchronously move data between host and device. */
  cudaStream_t stream1_;
  cudaStream_t stream2_;
  cudaStream_t stream3_;
  cudaStream_t stream4_;
  cudaStream_t stream5_;
  
  /** The ID of the target GPU Device. */
  int gpuDevice_;

  /** The constant value Alpha. */
  const T alpha = ALPHA;

  /** The constant value Beta. */
  const T beta = BETA;

  // cuSPARSE parameters
  cusparseOperation_t opA_;
  cusparseOperation_t opB_;
  cusparseSpMMAlg_t alg_;
	cusparseIndexType_t index_;
  cusparseIndexBase_t base_;
  cudaDataType_t dataType_;

  /**
   * ___________ Host data ______________
   */
	/** CSR format vectors for matrix A */
  cusparseSpMatDescr_t A_descr_;
	T* A_vals_;
	int64_t* A_cols_;
  int64_t* A_rows_;
  int64_t A_num_rows_;
  int64_t A_num_cols_;

  /** dense format values for matrices B and C */
  cusparseDnMatDescr_t B_descr_;
  int64_t B_num_rows_;
  int64_t B_num_cols_;
  int64_t B_leading_dim_;
  cusparseOrder_t B_order_;

  cusparseDnMatDescr_t C_descr_;
  int64_t C_num_rows_;
  int64_t C_num_cols_;
  int64_t C_leading_dim_;
  cusparseOrder_t C_order_;

  /**
   * _____________ Device data ________________
   */
  T* A_vals_dev_;
  int64_t* A_cols_dev_;
  int64_t* A_rows_dev_;

  T* B_dev_;

  T* C_dev_;
};
};


#endif
