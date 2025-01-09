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
  using spmm<T>::spmm;
  using spmm<T>::initInputMatrices;
  using spmm<T>::m_
  using spmm<T>::n_;
  using spmm<T>::k_
  using spmm<T>::A_;
  using spmm<T>::B_;
  using spmm<T>::C_;
  using spmm<T>::offload_;
  using spmm<T>::nnz_;

  void initialise(gpuOffloadType offload, int n, double sparsity) override {
    offload_ = offload;

    if (std::is_same_v<T, float>) cudaDataType_ = CUDA_R_32F;
    else if (std::is_same_v<T, double>) cudaDataType_ = CUDA_R_64F;
    else {
      std::cout << "INVALID DATA TYPE PASSED TO cuSPARSE" << std::endl;
      exit(1);
    }
    m_ = m;
    n_ = n;
    k_ = k;

    A_ = (T*)malloc(sizeof(T) * m_ * k_);
    B_ = (T*)malloc(sizeof(T) * k_ * n_);
    C_ = (T*)calloc(sizeof(T) * m_ * n_);

    /** Determine the number of nnz elements in A and B */
    nnz_ = 1 + (uint64_t)((double)m_ * (double)k_ * (1.0 - sparsity_));

    // Get device identifier
    cudaCheckError(cudaGetDevice(&gpuDevice_));

    // Initialise 3 streams to asynchronously move data between host and device
    cudaCheckError(cudaStreamCreate(&s1_));
    cudaCheckError(cudaStreamCreate(&s2_));
    cudaCheckError(cudaStreamCreate(&s3_));

    if (offload_ == gpuOffloadType::unified) {
      cudaCheckError(cudaMallocManaged(&A_val_, sizeof(T) * nnz_));
      cudaCheckError(cudaMallocManaged(&A_col_, sizeof(int) * nnz_));
      cudaCheckError(cudaMallocManaged(&A_row_, sizeof(int) * (m_ + 1)));

      cudaCheckError(cudaMallocManaged(&B_, sizeof(T) * k_ * n_));

      cudaCheckError(cudaMallocManaged(&C_, sizeof(T) * m_ * n_));
    } else {
      A_val_ = (T*)malloc(sizeof(T) * nnz_);
      A_col_ = (int*)malloc(sizeof(int) * nnz_);
      A_row_ = (int*)malloc(sizeof(int) * (m_ + 1));

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
    // Todo -- do I need a SPMM description here?
    switch(offload_) {
      case gpuOffloadType::always: {
        [[fallthorugh]];
      }
      case gpuOffloadType::once: {
        cudaCheckError(cudaMemcpyAsync(A_val_dev_, A_val_, (sizeof(T) * nnz_),
                                       cudaMemcpyHostToDevice, s1_));
        cudaCheckError(cudaMemcpyAsync(A_col_dev_, A_col_,
                                       (sizeof(int) * nnz_),
                                       cudaMemcpyHostToDevice, s1_));
        cudaCheckError(cudaMemcpyAsync(A_row_dev_, A_row_,
                                       (sizeof(int) * (m_ + 1)),
                                       cudaMemcpyHostToDevice, s1_));

        cudaCheckError(cudaMemcpyAsync(B_dev_, B_, (sizeof(T) * k_ * n_),
                                       cudaMemcpyHostToDevice, s2_));

        cudaCheckError(cudaMemcpyAsync(C_dev_, C_, (sizeof(T) * m_ * n_),
                                       cudaMemcpyHostToDevice, s3_));


        cusparseCreateCsr(&descrA_, m_, k_, nnz_, A_row_dev_,
                                  A_col_dev_, A_val_dev_, rType_, cType_,
                                  indType_, cudaDataType_));
        cusparseCheckError(
                cusparseCreateDnMat(&descrB_, B_num_rows_, B_num_cols_,
                                    B_leading_dim_, B_dev_, cudaDataType_,
                                    B_order_));
        cusparseCheckError(
                cusparseCreateDnMat(&descrC_, C_num_rows_, C_num_cols_,
                                    C_leading_dim_, C_dev_, cudaDataType_,
                                    C_order_));
        break;
      }
      case gpuOffloadType::unified: {
        cudaCheckError(cudaMemPrefetchAsync(A_val_, sizeof(T) * nnz_,
                                            gpuDevice_, s1_));
        cudaCheckError(cudaMemPrefetchAsync(A_col_, sizeof(int) * nnz_,
                                            gpuDevice_, s1_));
        cudaCheckError(cudaMemPrefetchAsync(A_row_, sizeof(int) * (m_ + 1),
                                            gpuDevice_, s1_));

        cudaCheckError(cudaMemPrefetchAsync(B_, sizeof(T) * n_ * k_,
                                            gpuDevice_, s2_));

        cudaCheckError(cudaMemPrefetchAsync(C_, sizeof(T) * m_ * n_,
                                            gpuDevice_, s3_));

        cudaCheckError(cudaDeviceSynchronize());


        cusparseCheckError(
                cusparseCreateCsr(&descrA_, m_, k_, nnzA_, A_row_, A_col_,
                                  A_val_, rType_, cType_, indType_,
                                  cudaDataType_));
        cusparseCheckError(
                cusparseCreateDnMat(&descrB_, B_num_rows_, B_num_cols_,
                                    B_leading_dim_, B_, cudaDataType_,
                                    B_order_));
        cusparseCheckError(
                cusparseCreateDnMat(&descrC_, C_num_rows_, C_num_cols_,
                                    C_leading_dim_, C_, cudaDataType_,
                                    C_order_));
        break;
      }
    }
  }

  void callGemm() override {
    switch(offload_) {
      case gpuOffloadType::always: {
        // Clean up old descriptors
        cusparseCheckError(cusparseDestroySpMat(descrA_));
        cuspraseCheckError(cusparseDestroyDnMat(descrB_));
        cuspraseCheckError(cusparseDestroyDnMat(descrC_));

        // Move over data
        cudaCheckError(cudaMemcpyAsync(A_val_dev_, A_val_, (sizeof(T) * nnz_),
                                       cudaMemcpyHostToDevice, s1_));
        cudaCheckError(cudaMemcpyAsync(A_col_dev_, A_col_,
                                       (sizeof(int) * nnz_),
                                       cudaMemcpyHostToDevice, s1_));
        cudaCheckError(cudaMemcpyAsync(A_row_dev_, A_row_,
                                       (sizeof(int) * (m_ + 1)),
                                       cudaMemcpyHostToDevice, s1_));

        cudaCheckError(cudaMemcpyAsync(B_dev_, B_, (sizeof(T) * k_ * n_),
                                       cudaMemcpyHostToDevice, s2_));

        cudaCheckError(cudaMemcpyAsync(C_dev_, C_, (sizeof(T) * m_ * n_),
                                       cudaMemcpyHostToDevice, s3_));

        cudaCheckError(cudaDeviceSynchronize());

        // Set up descriptors
        cusparseCreateCsr(&descrA_, m_, k_, nnz_, A_row_dev_,
                                  A_col_dev_, A_val_dev_, rType_, cType_,
                                  indType_, cudaDataType_));
        cusparseCheckError(
                cusparseCreateDnMat(&descrB_, B_num_rows_, B_num_cols_,
                                    B_leading_dim_, B_dev_, cudaDataType_,
                                    B_order_));
        cusparseCheckError(
                cusparseCreateDnMat(&descrC_, C_num_rows_, C_num_cols_,
                                    C_leading_dim_, C_dev_, cudaDataType_,
                                    C_order_));

        // Begin matrix-matrix multiplication
        cusparseCheckError(
                cusparseSpMM_bufferSize(handle_, opA_, opB_, &alpha, descrA_,
                                        descrB_, &beta, descrC_,
                                        cudaDataType_, alg_, &buffer_size_1_));

        cudaCheckError(cudaMalloc((void**)&buffer1_, buffer_size_1_));
        cusparseCheckError(
                cusparseSpMM_preprocess(handle_, opA_, opB_, &alpha, descrA_,
                                        descrB_, &beta, descrC_,
                                        cudaDataType_, alg_, buffer1_));
        cusparseCheckError(
                cusparseSpMM(handle_, opA_, opB_, &alpha, descrA_, descrB_,
                             &beta, descrC_, cudaDataType_, alg_, buffer1_));
      }
    }
  }

  /** Handle used when calling cuBLAS. */
  cusparseHandle_t handle_;

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

  bool C_mem_allocated_always_;
  bool C_mem_allocated_once_;
  bool C_mem_allocated_unified_;

  /** The constant value Alpha. */
  const T alpha = ALPHA;

  /** The constant value Beta. */
  const T beta = BETA;


	size_t buffer_size1_ = 0;
	size_t buffer_size2_ = 0;
  void* buffer1_ = NULL;
	void* buffer2_ = NULL;

  cusparseOperation_t opA_ = CUSPARSE_OPERATION_NON_TRANSPOSE;
  cusparseOperation_t opB_ = CUSPARSE_OPERATION_NON_TRANSPOSE;
  cusparseSpMMAlg_t alg_ = CUSPARSE_SPMM_ALG_DEFAULT;

	// Data type depends on kernel being run
	cudaDataType_t cudaDataType_;

  /**
   * ___________ Host data ______________
   */
	/** CSR format vectors for matrix A */
  cusparseSpMatDescr_t descrA_;
	T* A_val_;
	int* A_col_;
  int* A_row_;
  int64_t A_num_rows_;
  int64_t A_num_cols_;

  /** dense format values for matrices B and C */
  cusparseDnMatDescr_t descrB_;
  int B_num_rows_;
  int B_num_cols_;
  int B_leading_dim_;
  cusparseOrder_t B_order_;

  cusaprseDnMatDescr_t descrC_;
  int C_num_rows_;
  int C_num_cols_;
  int C_leading_dim_;
  cusparseOrder_t C_order_;

  /**
   * _____________ Device data ________________
   */
  T* A_val_dev_;
  int* A_col_dev_;
  int* A_row_dev_;

  T* B_dev_;

  T* C_dev_;



};

};


#endif
