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
  using spgemm<T>::A_;
  using spgemm<T>::B_;
  using spgemm<T>::C_;
  using spgemm<T>::offload_;
  using spgemm<T>::nnz_;
  using spgemm<T>::sparsity_;

  void initialise(gpuOffloadType offload, int m, int n, int k,
                  double sparsity, bool binary = false) override {
    if (print_) {
      switch (offload) {
        case gpuOffloadType::always: {
          std::cout << "===========  ALWAYS  ===========" << std::endl;
          break;
        }
        case gpuOffloadType::once: {
          std::cout << "===========   ONCE   ===========" << std::endl;
          break;
        }
        case gpuOffloadType::unified: {
          std::cout << "===========  UNIFIED ===========" << std::endl;
          break;
        }
      }
      std::cout << "Initialising " << m << "x" << k << " . " << k << "x" << n <<std::endl;
    }

    offload_ = offload;
    sparsity_ = sparsity;

    m_ = m;
    n_ = n;
    k_ = k;

    A_ = B_ = C_ = B_dev_ = C_dev_ = A_vals_ = A_vals_dev_ = nullptr;
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
    if (std::is_same_v<T, float>) dataType_ = CUDA_R_32F;
    else if (std::is_same_v<T, double>) dataType_ = CUDA_R_64F;
    else {
      std::cout << "INVALID DATA TYPE PASSED TO cuSPARSE" << std::endl;
      exit(1);
    }

    if (print_) std::cout << "\tSetting up cuda streams" << std::endl;
    // Get device identifier
    cudaCheckError(cudaGetDevice(&gpuDevice_));

    // Initialise 3 streams to asynchronously move data between host and device
    cudaCheckError(cudaStreamCreate(&s1_));
    cudaCheckError(cudaStreamCreate(&s2_));
    cudaCheckError(cudaStreamCreate(&s3_));

    if (A_ != nullptr) cudaFreeAdaptive(A_, "A_");
    cudaCheckError(cudaMallocHost(&A_, sizeof(T) * m_ * k_));
    if (offload_ == gpuOffloadType::unified) {
      if (print_) std::cout << "\tAllocating unified memory" << std::endl;
      if (B_ != nullptr) cudaFreeAdaptive(B_, "B_");
      cudaCheckError(cudaMallocManaged(&B_, sizeof(T) * k_ * n_));
      if (C_ != nullptr) cudaFreeAdaptive(C_, "C_");
      cudaCheckError(cudaMallocManaged(&C_, sizeof(T) * m_ * n_));
    } else {
      if (print_) std::cout << "\tAllocating host memory" << std::endl;
      if (B_ != nullptr) cudaFreeAdaptive(B_, "B_");
      cudaCheckError(cudaMallocHost(&B_, sizeof(T) * k_ * n_));
      if (C_ != nullptr) cudaFreeAdaptive(C_, "C_");
      cudaCheckError(cudaMallocHost(&C_, sizeof(T) * m_ * n_));

      if (print_) std::cout << "\tAllocating device memory" << std::endl;
      if (B_dev_ != nullptr) cudaFreeAdaptive(B_dev_, "B_dev_");
      cudaCheckError(cudaMalloc((void**)&B_dev_, sizeof(T) * k_ * n_));
      if (C_dev_ != nullptr) cudaFreeAdaptive(C_dev_, "C_dev_");
      cudaCheckError(cudaMalloc((void**)&C_dev_, sizeof(T) * m_ * n_));
    }
    cudaCheckError(cudaDeviceSynchronize());

    cusparseCheckError(cusparseCreate(&handle_));

    if (print_) std::cout << "\tInitialising input matrices" << std::endl;
    initInputMatrices();
  }

protected:
  void toSparseFormat() override {
    // Allocate CSR arrays
    if (offload_ == gpuOffloadType::unified) {
      if (A_vals_ != nullptr) cudaFreeAdaptive(A_vals_, "A_vals_");
      cudaCheckError(cudaMallocManaged(&A_vals_, sizeof(T) * nnz_));
      if (A_cols_ != nullptr) cudaFreeAdaptive(A_cols_, "A_cols_");
      cudaCheckError(cudaMallocManaged(&A_cols_, sizeof(int64_t) * nnz_));
      if (A_rows_ != nullptr) cudaFreeAdaptive(A_rows_, "A_rows_");
      cudaCheckError(cudaMallocManaged(&A_rows_, sizeof(int64_t) * (m_ + 1)));
    } else {
      if (A_vals_ != nullptr) cudaFreeAdaptive(A_vals_, "A_vals_");
      cudaCheckError(cudaMallocHost(&A_vals_, sizeof(T) * nnz_));
      if (A_cols_ != nullptr) cudaFreeAdaptive(A_cols_, "A_cols_");
      cudaCheckError(cudaMallocHost(&A_cols_, sizeof(int64_t) * nnz_));
      if (A_rows_ != nullptr) cudaFreeAdaptive(A_rows_, "A_rows_");
      cudaCheckError(cudaMallocHost(&A_rows_, sizeof(int64_t) * (m_ + 1)));
      if (A_vals_dev_ != nullptr) cudaFreeAdaptive(A_vals_dev_, "A_vals_dev_");
      cudaCheckError(cudaMalloc((void**)&A_vals_dev_, sizeof(T) * nnz_));
      if (A_cols_dev_ != nullptr) cudaFreeAdaptive(A_cols_dev_, "A_cols_dev_");
      cudaCheckError(cudaMalloc((void**)&A_cols_dev_, sizeof(int64_t) * nnz_));
      if (A_rows_dev_ != nullptr) cudaFreeAdaptive(A_rows_dev_, "A_rows_dev_");
      cudaCheckError(cudaMalloc((void**)&A_rows_dev_, sizeof(int64_t) * (m_ + 1)));
    }
    cudaCheckError(cudaDeviceSynchronize());

    // Load A into CSR
    int nnz_encountered = 0;
    for (int row = 0; row < m_; row++) {
      A_rows_[row] = nnz_encountered;
      for (int col = 0; col < k_; col++) {
        if (A_[(row * k_) + col] != 0.0) {
          A_cols_[nnz_encountered] = col;
          A_vals_[nnz_encountered] = A_[(row * k_) + col];
          nnz_encountered++;
        }
        if (nnz_encountered == nnz_) break;
      }
      if (nnz_encountered == nnz_) break;
    }
    A_rows_[m_] = nnz_encountered;
    if (nnz_ != nnz_encountered) {
      std::cout << "ERROR -- NOT ENOUGH NON-ZERO VALUES!" << std::endl;
    }
    cudaCheckError(cudaDeviceSynchronize());
  }

private:
  void preLoopRequirements() override {
    switch(offload_) {
      case gpuOffloadType::always: {
        break;
      }
      case gpuOffloadType::once: {
        if (print_) std::cout << "\tMoving data to GPU" << std::endl;
        cudaCheckError(cudaMemcpy(A_vals_dev_, A_vals_, nnz_ * sizeof(T), cudaMemcpyHostToDevice));
        cudaCheckError(cudaMemcpy(A_cols_dev_, A_cols_, nnz_ * sizeof(int64_t), cudaMemcpyHostToDevice));
        cudaCheckError(cudaMemcpy(A_rows_dev_, A_rows_, (m_ + 1) * sizeof(int64_t), cudaMemcpyHostToDevice));
        cudaCheckError(cudaMemcpy(B_dev_, B_, (k_ * n_) * sizeof(T), cudaMemcpyHostToDevice));
        cudaCheckError(cudaMemcpy(C_dev_, C_, (m_ * n_) * sizeof(T), cudaMemcpyHostToDevice));
        cudaCheckError(cudaDeviceSynchronize());
        break;
      }
      case gpuOffloadType::unified: {
        if (print_) std::cout << "\tMoving data to GPU" << std::endl;
        cudaCheckError(cudaMemPrefetchAsync(A_vals_, nnz_ * sizeof(T), gpuDevice_, s1_));
        cudaCheckError(cudaMemPrefetchAsync(A_cols_, nnz_ * sizeof(int64_t), gpuDevice_, s1_));
        cudaCheckError(cudaMemPrefetchAsync(A_rows_, (m_ + 1) * sizeof(int64_t), gpuDevice_, s1_));
        cudaCheckError(cudaMemPrefetchAsync(B_, (n_ * k_) * sizeof(T), gpuDevice_, s2_));
        cudaCheckError(cudaMemPrefetchAsync(C_, (m_ * n_) * sizeof(T), gpuDevice_, s3_));
        cudaCheckError(cudaDeviceSynchronize());
        break;
      }
    }
  }

  void callSpgemm() override {
    switch(offload_) {
      case gpuOffloadType::always: {
        // Move over data
        if (print_) std::cout << "\tMoving data to GPU" << std::endl;
        cudaCheckError(cudaMemcpyAsync(A_vals_dev_, A_vals_, (sizeof(T) * nnz_),
                                       cudaMemcpyHostToDevice, s1_));
        cudaCheckError(cudaMemcpyAsync(A_cols_dev_, A_cols_,
                                       (sizeof(int64_t) * nnz_),
                                       cudaMemcpyHostToDevice, s1_));
        cudaCheckError(cudaMemcpyAsync(A_rows_dev_, A_rows_,
                                       (sizeof(int64_t) * (m_ + 1)),
                                       cudaMemcpyHostToDevice, s1_));

        cudaCheckError(cudaMemcpyAsync(B_dev_, B_, (sizeof(T) * k_ * n_),
                                       cudaMemcpyHostToDevice, s2_));

        cudaCheckError(cudaMemcpyAsync(C_dev_, C_, (sizeof(T) * m_ * n_),
                                       cudaMemcpyHostToDevice, s3_));

        // Set up descriptors
        if (print_) std::cout << "\tCreating matrix descriptor for A" << std::endl;
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
        if (print_) std::cout << "\tCreating matrix descriptor for B" << std::endl;
        cusparseCheckError(cusparseCreateDnMat(&B_descr_, 
                                               k_, 
                                               n_,
                                               n_, 
                                               B_dev_, 
                                               dataType_,
                                               B_order_));
        if (print_) std::cout << "\tCreating matrix descriptor for C" << std::endl;
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
        if (print_) std::cout << "\tCalculating buffer size" << std::endl;
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

        if (print_) std::cout << "\tBuffer size: " << bufferSize << std::endl;
        // Allocate the temporary buffer
        if (bufferSize > 0) {
          if (dBuffer != nullptr) cudaFreeAdaptive(dBuffer, "dBuffer");
          cudaCheckError(cudaMalloc((void**)&dBuffer, bufferSize));
        }

        if (print_) std::cout << "\tPreprocessing" << std::endl;
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

        if (print_) std::cout << "\tPerforming SpGEMM" << std::endl;
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
        if (dBuffer != nullptr) {
          cudaFreeAdaptive(dBuffer, "dBuffer");
          dBuffer = nullptr;
        }

        // Move result back to CPU
        cudaCheckError(cudaMemcpyAsync(C_, C_dev_, (sizeof(T) * m_ * n_),
                                       cudaMemcpyDeviceToHost, s3_));
        break;
      }
      case gpuOffloadType::once: {
        // Set up descriptors
        if (print_) std::cout << "\tCreating matrix descriptor for A" << std::endl;
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
        if (print_) std::cout << "\tCreating matrix descriptor for B" << std::endl;
        cusparseCheckError(cusparseCreateDnMat(&B_descr_, 
                                               k_, 
                                               n_,
                                               n_, 
                                               B_dev_, 
                                               dataType_,
                                               B_order_));
        if (print_) std::cout << "\tCreating matrix descriptor for C" << std::endl;
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
        if (print_) std::cout << "\tCalculating buffer size" << std::endl;
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

        if (print_) std::cout << "\tBuffer size: " << bufferSize << std::endl;
        // Allocate the temporary buffer
        if (bufferSize > 0) {
          if (dBuffer != nullptr) cudaFreeAdaptive(dBuffer, "dBuffer");
          cudaCheckError(cudaMalloc((void**)&dBuffer, bufferSize));
        }

        if (print_) std::cout << "\tPreprocessing" << std::endl;
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

        if (print_) std::cout << "\tPerforming SpGEMM" << std::endl;
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
        if (dBuffer != nullptr) {
          cudaFreeAdaptive(dBuffer, "dBuffer");
          dBuffer = nullptr;
        }
        break;
      }
      case gpuOffloadType::unified: {
        // Create descriptors for the matrices
        if (print_) std::cout << "\tCreating matrix descriptor for A" << std::endl;
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
        if (print_) std::cout << "\tCreating matrix descriptor for B" << std::endl;
        cusparseCheckError(cusparseCreateDnMat(&B_descr_, 
                                               k_, 
                                               n_,
                                               n_, 
                                               B_, 
                                               dataType_,
                                               B_order_));
        if (print_) std::cout << "\tCreating matrix descriptor for C" << std::endl;
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
        if (print_) std::cout << "\tCalculating buffer size" << std::endl;
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

        if (print_) std::cout << "\tBuffer size: " << bufferSize << std::endl;
        // Allocate the temporary buffer
        if (bufferSize > 0) {
          if (dBuffer != nullptr) cudaFreeAdaptive(dBuffer, "dBuffer");
          cudaCheckError(cudaMalloc((void**)&dBuffer, bufferSize));
        }

        if (print_) std::cout << "\tPreprocessing" << std::endl;
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

        if (print_) std::cout << "\tPerforming SpGEMM" << std::endl;
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
        if (dBuffer != nullptr) {
          cudaFreeAdaptive(dBuffer, "dBuffer");
          dBuffer = nullptr;
        }
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
        if (print_) std::cout << "\tMoving data to CPU" << std::endl;
        // Move result back to CPU
        cudaCheckError(cudaMemcpyAsync(C_, C_dev_, (sizeof(T) * m_ * n_),
                                       cudaMemcpyDeviceToHost, s3_));
        break;
      }
      case gpuOffloadType::unified: {
        if (print_) std::cout << "\tMoving data to CPU" << std::endl;
        // Move result back to CPU
        cudaCheckError(cudaMemPrefetchAsync(C_, sizeof(T) * m_ * n_, 
                                            cudaCpuDeviceId, s3_));
        break;
      }
    }
  }

  void postCallKernelCleanup() override {
    if (A_ != nullptr) {
      if (print_) std::cout << "Freeing A_" << std::endl;
      cudaFreeAdaptive(A_, "A_");
      A_ = nullptr;
    }
    if (A_vals_ != nullptr) {
      if (print_) std::cout << "Freeing A_vals_" << std::endl;
      cudaFreeAdaptive(A_vals_, "A_vals_"); 
      A_vals_ = nullptr;
    }
    if (A_cols_ != nullptr) {
      if (print_) std::cout << "Freeing A_cols_" << std::endl;
      cudaFreeAdaptive(A_cols_, "A_cols_"); 
      A_cols_ = nullptr; 
    }
    if (A_rows_ != nullptr) {
      if (print_) std::cout << "Freeing A_rows_" << std::endl;
      cudaFreeAdaptive(A_rows_, "A_rows_"); 
      A_rows_ = nullptr;
    }
    if (B_ != nullptr) {
      if (print_) std::cout << "Freeing B_" << std::endl;
      cudaFreeAdaptive(B_, "B_"); 
      B_ = nullptr;
    }
    if (C_ != nullptr) {
      if (print_) std::cout << "Freeing C_" << std::endl;
      cudaFreeAdaptive(C_, "C_"); 
      C_ = nullptr;
    }
    if (A_vals_dev_ != nullptr) {
      if (print_) std::cout << "Freeing A_vals_dev_" << std::endl;
      cudaFreeAdaptive(A_vals_dev_, "A_vals_dev_");
      A_vals_dev_ = nullptr;
    }
    if (A_cols_dev_ != nullptr) {
      if (print_) std::cout << "Freeing A_cols_dev_" << std::endl;
      cudaFreeAdaptive(A_cols_dev_, "A_cols_dev_");
      A_cols_dev_ = nullptr;
    }
    if (A_rows_dev_ != nullptr) {
      if (print_) std::cout << "Freeing A_rows_dev_" << std::endl;
      cudaFreeAdaptive(A_rows_dev_, "A_rows_dev_");
      A_rows_dev_ = nullptr;
    }
    if (B_dev_ != nullptr) {
      if (print_) std::cout << "Freeing B_dev_" << std::endl;
      cudaFreeAdaptive(B_dev_, "B_dev_");
      B_dev_ = nullptr;
    }
    if (C_dev_ != nullptr) {
      if (print_) std::cout << "Freeing C_dev_" << std::endl;
      cudaFreeAdaptive(C_dev_, "C_dev_");
      C_dev_ = nullptr;
    }
  }

  inline void cudaFreeAdaptive(void* ptr, std::string name) {
    if (!ptr) return;
    cudaPointerAttributes attr;
    cudaError_t err = cudaPointerGetAttributes(&attr, ptr);
    if (err != cudaSuccess) {
      // Pointer not recognized by CUDA (e.g. malloc) → just free?
      // But since you've moved to cudaMallocHost, we should treat this as error.
      std::cerr << "cudaPointerGetAttributes failed: " << cudaGetErrorString(err) << std::endl;
      return;
    }
    switch (attr.type) {
      case cudaMemoryTypeDevice:
      case cudaMemoryTypeManaged:
        // cudaMallocManaged
        cudaCheckError(cudaFree(ptr));
        break;

      case cudaMemoryTypeHost:
        // cudaMallocHost
        cudaCheckError(cudaFreeHost(ptr));
        break;

      default:
        std::cerr << "Unknown CUDA pointer type in cudaFreeAdaptive for: " << name << std::endl;
        break;
    }
  }

  bool print_ = false;

  /** Handle used when calling cuBLAS. */
  cusparseHandle_t handle_;

  /** CUDA Streams - used to asynchronously move data between host and device.
   */
  cudaStream_t s1_;
  cudaStream_t s2_;
  cudaStream_t s3_;

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
