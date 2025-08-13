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
    }


    offload_ = offload;
    sparsity_ = sparsity;

    m_ = m;
    n_ = n;
    k_ = k;

    A_ = (T*)malloc(sizeof(T) * m_ * k_);
    B_ = (T*)malloc(sizeof(T) * k_ * n_);
    C_ = (T*)calloc(m_ * n_, sizeof(T));

    /** Determine the number of nnz elements in A and B */
    nnz_ = 1 + (uint64_t)((double)m_ * (double)k_ * (1.0 - sparsity_));

    // Set up cuSPARSE metadata
    if (std::is_same_v<T, float>) dataType_ = CUDA_R_32F;
    else if (std::is_same_v<T, double>) dataType_ = CUDA_R_64F;
    else {
      std::cout << "INVALID DATA TYPE PASSED TO cuSPARSE" << std::endl;
      exit(1);
    }

    // Get device identifier
    cudaCheckError(cudaGetDevice(&gpuDevice_));

    // Initialise 3 streams to asynchronously move data between host and device
    cudaCheckError(cudaStreamCreate(&s1_));
    cudaCheckError(cudaStreamCreate(&s2_));
    cudaCheckError(cudaStreamCreate(&s3_));

    if (offload_ == gpuOffloadType::unified) {
      cudaCheckError(cudaMallocManaged(&A_val_, sizeof(T) * nnz_));
      cudaCheckError(cudaMallocManaged(&A_col_, sizeof(int64_t) * nnz_));
      cudaCheckError(cudaMallocManaged(&A_row_, sizeof(int64_t) * (m_ + 1)));

      cudaCheckError(cudaMallocManaged(&B_, sizeof(T) * k_ * n_));

      cudaCheckError(cudaMallocManaged(&C_, sizeof(T) * m_ * n_));
    } else {
      A_val_ = (T*)malloc(sizeof(T) * nnz_);
      A_col_ = (int64_t*)malloc(sizeof(int64_t) * nnz_);
      A_row_ = (int64_t*)malloc(sizeof(int64_t) * (m_ + 1));

      B_ = (T*)malloc(sizeof(T) * k_ * n_);

      C_ = (T*)malloc(sizeof(T) * m_ * n_);

      cudaCheckError(cudaMalloc((void**)&A_val_dev_, sizeof(T) * nnz_));
      cudaCheckError(cudaMalloc((void**)&A_col_dev_, sizeof(T) * nnz_));
      cudaCheckError(cudaMalloc((void**)&A_row_dev_, sizeof(T) * (m_ + 1)));

      cudaCheckError(cudaMalloc((void**)&B_dev_, sizeof(T) * k_ * n_));

      cudaCheckError(cudaMalloc((void**)&C_dev_, sizeof(T) * m_ * n_));
    }

    cusparseCheckError(cusparseCreate(&handle_));

    initInputMatrices();
  }

protected:
  void toSparseFormat() override {
    // Load A into CSR
    int nnz_encountered = 0;
    for (int row = 0; row < m_; row++) {
      A_row_[row] = nnz_encountered;
      int nnz_row = 0;
      for (int col = 0; col < k_; col++) {
        if (B_[(row * k_) + col] != 0.0) {
          nnz_row++;
          A_col_[nnz_encountered] = col;
          A_val_[nnz_encountered] = A_[(row * k_) + col];
          nnz_encountered++;
        }
      }
    }
    A_row_[m_] = nnz_encountered;

    B_order_ = C_order_ = CUSPARSE_ORDER_ROW;
  }

private:
  void preLoopRequirements() override {
    switch(offload_) {
      case gpuOffloadType::always: {
        break;
      }
      case gpuOffloadType::once: {
        cudaCheckError(cudaMemcpyAsync(A_val_dev_, A_val_, (sizeof(T) * nnz_),
                                       cudaMemcpyHostToDevice, s1_));
        cudaCheckError(cudaMemcpyAsync(A_col_dev_, A_col_,
                                       (sizeof(int64_t) * nnz_),
                                       cudaMemcpyHostToDevice, s1_));
        cudaCheckError(cudaMemcpyAsync(A_row_dev_, A_row_,
                                       (sizeof(int64_t) * (m_ + 1)),
                                       cudaMemcpyHostToDevice, s1_));

        cudaCheckError(cudaMemcpyAsync(B_dev_, B_, (sizeof(T) * k_ * n_),
                                       cudaMemcpyHostToDevice, s2_));

        cudaCheckError(cudaMemcpyAsync(C_dev_, C_, (sizeof(T) * m_ * n_),
                                       cudaMemcpyHostToDevice, s3_));
        cudaCheckError(cudaDeviceSynchronize());
        break;
      }
      case gpuOffloadType::unified: {
        cudaCheckError(cudaMemPrefetchAsync(A_val_, sizeof(T) * nnz_,
                                            gpuDevice_, s1_));
        cudaCheckError(cudaMemPrefetchAsync(A_col_, sizeof(int64_t) * nnz_,
                                            gpuDevice_, s1_));
        cudaCheckError(cudaMemPrefetchAsync(A_row_, sizeof(int64_t) * (m_ + 1),
                                            gpuDevice_, s1_));

        cudaCheckError(cudaMemPrefetchAsync(B_, sizeof(T) * n_ * k_,
                                            gpuDevice_, s2_));

        cudaCheckError(cudaMemPrefetchAsync(C_, sizeof(T) * m_ * n_,
                                            gpuDevice_, s3_));
        cudaCheckError(cudaDeviceSynchronize());
        break;
      }
    }
  }

  void callSpgemm() override {
    switch(offload_) {
      case gpuOffloadType::always: {
        // Move over data
        cudaCheckError(cudaMemcpyAsync(A_val_dev_, A_val_, (sizeof(T) * nnz_),
                                       cudaMemcpyHostToDevice, s1_));
        cudaCheckError(cudaMemcpyAsync(A_col_dev_, A_col_,
                                       (sizeof(int64_t) * nnz_),
                                       cudaMemcpyHostToDevice, s1_));
        cudaCheckError(cudaMemcpyAsync(A_row_dev_, A_row_,
                                       (sizeof(int64_t) * (m_ + 1)),
                                       cudaMemcpyHostToDevice, s1_));

        cudaCheckError(cudaMemcpyAsync(B_dev_, B_, (sizeof(T) * k_ * n_),
                                       cudaMemcpyHostToDevice, s2_));

        cudaCheckError(cudaMemcpyAsync(C_dev_, C_, (sizeof(T) * m_ * n_),
                                       cudaMemcpyHostToDevice, s3_));

        // Set up descriptors
        if (print_) std::cout << "\tCreating matrix descriptor for A" << std::endl;
        cusparseCheckError(cusparseCreateCsr(&descrA_, 
                                             m_, 
                                             k_, 
                                             nnz_, 
                                             A_row_dev_, 
                                             A_col_dev_,
                                             A_val_dev_, 
                                             index_, 
                                             index_, 
                                             base_, 
                                             dataType_));
        if (print_) std::cout << "\tCreating matrix descriptor for B" << std::endl;
        cusparseCheckError(cusparseCreateDnMat(&descrB_, 
                                               B_num_rows_, 
                                               B_num_cols_,
                                               B_leading_dim_, 
                                               B_dev_, 
                                               dataType_,
                                               B_order_));
        if (print_) std::cout << "\tCreating matrix descriptor for C" << std::endl;
        cusparseCheckError(cusparseCreateDnMat(&descrC_, 
                                               C_num_rows_, 
                                               C_num_cols_,
                                               C_leading_dim_, 
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
                                                   descrA_,
                                                   descrB_, 
                                                   &beta, 
                                                   descrC_,
                                                   dataType_, 
                                                   alg_, 
                                                   &bufferSize));

        // Allocate the temporary buffer
        if (bufferSize > 0) {
          cudaCheckError(cudaMalloc((void**)&dBuffer, bufferSize));
        }
        cusparseCheckError(cusparseSpMM_preprocess(handle_, 
                                                   opA_, 
                                                   opB_, 
                                                   &alpha, 
                                                   descrA_,
                                                   descrB_, 
                                                   &beta, 
                                                   descrC_,
                                                   dataType_, 
                                                   alg_, 
                                                   dBuffer));
        cusparseCheckError(cusparseSpMM(handle_, 
                                        opA_, 
                                        opB_, 
                                        &alpha, 
                                        descrA_, 
                                        descrB_,
                                        &beta, 
                                        descrC_, 
                                        dataType_, 
                                        alg_, 
                                        dBuffer));

        // Clean up descriptors
        cusparseCheckError(cusparseDestroySpMat(descrA_));
        cusparseCheckError(cusparseDestroyDnMat(descrB_));
        cusparseCheckError(cusparseDestroyDnMat(descrC_));

        // Free up the temporary buffer
        cudaCheckError(cudaFree(dBuffer));

        // Move result back to CPU
        cudaCheckError(cudaMemcpyAsync(C_, C_dev_, (sizeof(T) * m_ * n_),
                                       cudaMemcpyDeviceToHost, s3_));
        break;
      }
      case gpuOffloadType::once: {
        // Create descriptors for the matrices
        if (print_) std::cout << "\tCreating matrix descriptor for A" << std::endl;
        cusparseCheckError(cusparseCreateCsr(&descrA_, 
                                             m_, 
                                             k_, 
                                             nnz_, 
                                             A_row_dev_, 
                                             A_col_dev_,
                                             A_val_dev_, 
                                             index_, 
                                             index_, 
                                             base_, 
                                             dataType_));
        if (print_) std::cout << "\tCreating matrix descriptor for B" << std::endl;
        cusparseCheckError(cusparseCreateDnMat(&descrB_, 
                                               B_num_rows_, 
                                               B_num_cols_,
                                               B_leading_dim_, 
                                               B_dev_, 
                                               dataType_,
                                               B_order_));
        if (print_) std::cout << "\tCreating matrix descriptor for C" << std::endl;
        cusparseCheckError(cusparseCreateDnMat(&descrC_, 
                                               C_num_rows_, 
                                               C_num_cols_,
                                               C_leading_dim_, 
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
                                                   descrA_,
                                                   descrB_, 
                                                   &beta, 
                                                   descrC_,
                                                   dataType_, 
                                                   alg_, 
                                                   &bufferSize));

        // Allocate the temporary buffer
        if (bufferSize > 0) {
          cudaCheckError(cudaMalloc((void**)&dBuffer, bufferSize));
        }
        cusparseCheckError(cusparseSpMM_preprocess(handle_, 
                                                   opA_, 
                                                   opB_, 
                                                   &alpha, 
                                                   descrA_,
                                                   descrB_, 
                                                   &beta, 
                                                   descrC_,
                                                   dataType_, 
                                                   alg_, 
                                                   dBuffer));
        cusparseCheckError(cusparseSpMM(handle_, 
                                        opA_, 
                                        opB_, 
                                        &alpha, 
                                        descrA_, 
                                        descrB_,
                                        &beta, 
                                        descrC_, 
                                        dataType_, 
                                        alg_, 
                                        dBuffer));

        // Clean up descriptors
        cusparseCheckError(cusparseDestroySpMat(descrA_));
        cusparseCheckError(cusparseDestroyDnMat(descrB_));
        cusparseCheckError(cusparseDestroyDnMat(descrC_));

        // Free up the temporary buffer
        cudaCheckError(cudaFree(dBuffer));
        break;
      }
      case gpuOffloadType::unified: {
        // Create descriptors for the matrices
        if (print_) std::cout << "\tCreating matrix descriptor for A" << std::endl;
        cusparseCheckError(cusparseCreateCsr(&descrA_, 
                                             m_, 
                                             k_, 
                                             nnz_, 
                                             A_row_dev_, 
                                             A_col_dev_,
                                             A_val_dev_, 
                                             index_, 
                                             index_, 
                                             base_, 
                                             dataType_));
        if (print_) std::cout << "\tCreating matrix descriptor for B" << std::endl;
        cusparseCheckError(cusparseCreateDnMat(&descrB_, 
                                               B_num_rows_, 
                                               B_num_cols_,
                                               B_leading_dim_, 
                                               B_dev_, 
                                               dataType_,
                                               B_order_));
        if (print_) std::cout << "\tCreating matrix descriptor for C" << std::endl;
        cusparseCheckError(cusparseCreateDnMat(&descrC_, 
                                               C_num_rows_, 
                                               C_num_cols_,
                                               C_leading_dim_, 
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
                                                   descrA_,
                                                   descrB_, 
                                                   &beta, 
                                                   descrC_,
                                                   dataType_, 
                                                   alg_, 
                                                   &bufferSize));

        // Allocate the temporary buffer
        if (bufferSize > 0) {
          cudaCheckError(cudaMallocManaged((void**)&dBuffer, bufferSize));
        }
        cusparseCheckError(cusparseSpMM_preprocess(handle_, 
                                                   opA_, 
                                                   opB_, 
                                                   &alpha, 
                                                   descrA_,
                                                   descrB_, 
                                                   &beta, 
                                                   descrC_,
                                                   dataType_, 
                                                   alg_, 
                                                   dBuffer));
        cusparseCheckError(cusparseSpMM(handle_, 
                                        opA_, 
                                        opB_, 
                                        &alpha, 
                                        descrA_, 
                                        descrB_,
                                        &beta, 
                                        descrC_, 
                                        dataType_, 
                                        alg_, 
                                        dBuffer));

        // Clean up descriptors
        cusparseCheckError(cusparseDestroySpMat(descrA_));
        cusparseCheckError(cusparseDestroyDnMat(descrB_));
        cusparseCheckError(cusparseDestroyDnMat(descrC_));

        // Free up the temporary buffer
        cudaCheckError(cudaFree(dBuffer));
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

  }

  bool print_ = true;

  /** Handle used when calling cuBLAS. */
  cusparseHandle_t handle_;

  /** CUDA Streams - used to asynchronously move data between host and device.
   */
  cudaStream_t s1_;
  cudaStream_t s2_;
  cudaStream_t s3_;

  /** The ID of the target GPU Device. */
  int gpuDevice_;

  bool C_mem_allocated_always_;
  bool C_mem_allocated_once_;
  bool C_mem_allocated_unified_;

  /** The constant value Alpha. */
  const T alpha = ALPHA;

  /** The constant value Beta. */
  const T beta = BETA;

  // cuSPARSE parameters
  cusparseOperation_t opA_ = CUSPARSE_OPERATION_NON_TRANSPOSE;
  cusparseOperation_t opB_ = CUSPARSE_OPERATION_NON_TRANSPOSE;
  cusparseSpMMAlg_t alg_ = CUSPARSE_SPMM_ALG_DEFAULT;
	cusparseIndexType_t index_ = CUSPARSE_INDEX_64I;
  cusparseIndexBase_t base_ = CUSPARSE_INDEX_BASE_ZERO;
  cudaDataType_t dataType_;


  /**
   * ___________ Host data ______________
   */
	/** CSR format vectors for matrix A */
  cusparseSpMatDescr_t descrA_;
	T* A_val_;
	int64_t* A_col_;
  int64_t* A_row_;
  int64_t A_num_rows_;
  int64_t A_num_cols_;

  /** dense format values for matrices B and C */
  cusparseDnMatDescr_t descrB_;
  int64_t B_num_rows_;
  int64_t B_num_cols_;
  int64_t B_leading_dim_;
  cusparseOrder_t B_order_;

  cusparseDnMatDescr_t descrC_;
  int64_t C_num_rows_;
  int64_t C_num_cols_;
  int64_t C_leading_dim_;
  cusparseOrder_t C_order_;

  /**
   * _____________ Device data ________________
   */
  T* A_val_dev_;
  int64_t* A_col_dev_;
  int64_t* A_row_dev_;

  T* B_dev_;

  T* C_dev_;



};

};


#endif
