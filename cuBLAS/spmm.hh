#pragma once

#ifdef GPU_CUBLAS
#include <cusparse_v2.h>
#include <cuda_runtime.h>
#include <type_traits>
#include <random>
#include <iostream>

#include "../include/kernels/GPU/spmm.hh"
#include "../include/utilities.hh"
#include "common.hh"

namespace gpu {
/** A class for sparse GEMM GPU BLAS kernels. */
template <typename T>
class spmm_gpu : public spmm<T> {
 public:
  using spmm<T>::spmm;
  using spmm<T>::initInputMatrices;
  using spmm<T>::A_nnz_;
  using spmm<T>::B_nnz_;
  using spmm<T>::m_
  using spmm<T>::n_;
  using spmm<T>::k_
  using spmm<T>::A_;
  using spmm<T>::B_;
  using spmm<T>::C_;
  using spmm<T>::offload_;
  using spmm<T>::sprasity_;
  using spmm<T>::C_nnz_;
  using spmm<T>::C_rows_;
  using spmm<T>::C_cols_;
  using spmm<T>::C_vals_;

	// ToDo -- No checksum for sparse yet.  Need to do

  /** Initialise the required data structures.
   * `offload` refers to the data offload type:
   *  - Once:    Move data from host to device before all iterations & move from
   *             device to host after all iterations
   *  - Always:  Move data from host to device and device to host each iteration
   *  - Unified: Initialise data as unified memory; no data movement semantics
   *             required */
  void initialise(gpuOffloadType offload, int n, int m, int k, double sparsity)
  override {
    offload_ = offload;
    sparsity_ = sparsity;

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
    A_nnz_ = 1 + (uint64_t)((double)m_ * (double)k_ * (1.0 - sparsity_));
    B_nnz_ = 1 + (uint64_t)((double)k_ * (double)n_ * (1.0 - sparsity_));

    // Get device identifier
    cudaCheckError(cudaGetDevice(&gpuDevice_));

    // Initialise 3 streams to asynchronously move data between host and device
    cudaCheckError(cudaStreamCreate(&s1_));
    cudaCheckError(cudaStreamCreate(&s2_));
    cudaCheckError(cudaStreamCreate(&s3_));

    if (offload_ == gpuOffloadType::unified) {
      cudaCheckError(cudaMallocManaged(&A_val_, sizeof(T) * A_nnz_));
      cudaCheckError(cudaMallocManaged(&A_col_, sizeof(int) * A_nnz_));
      cudaCheckError(cudaMallocManaged(&A_row_, sizeof(int) * (m_ + 1)));

      cudaCheckError(cudaMallocManaged(&B_val_, sizeof(T) * B_nnz_));
      cudaCheckError(cudaMallocManaged(&B_col_, sizeof(int) * B_nnz_));
      cudaCheckError(cudaMallocManaged(&B_row_, sizeof(int) * (k_ + 1)));

      cudaCheckError(cudaMallocManaged(&C_row_, sizeof(int) * (n_ + 1)));
      C_val_ = NULL;
      C_col_ = NULL;
    } else {
      A_val_ = (T*)malloc(sizeof(T) * A_nnz_);
      A_col_ = (int*)malloc(sizeof(int) * A_nnz_);
      A_row_ = (int*)malloc(sizeof(int) * (m_ + 1));

      B_val_ = (T*)malloc(sizeof(T) * B_nnz_);
      B_col_ = (int*)malloc(sizeof(int) * B_nnz_);
      B_row_ = (int*)malloc(sizeof(int) * (k_ + 1));

      C_row_ = (int*)malloc(sizeof(int) * (n_ + 1));


      cudaCheckError(cudaMalloc((void**)&A_val_dev_, sizeof(T) * A_nnz_));
      cudaCheckError(cudaMalloc((void**)&A_col_dev_, sizeof(int) * A_nnz_));
      cudaCheckError(cudaMalloc((void**)&A_row_dev_, sizeof(int) * (m_ + 1)));

      cudaCheckError(cudaMalloc((void**)&B_val_dev_, sizeof(T) * B_nnz_));
      cudaCheckError(cudaMalloc((void**)&B_col_dev_, sizeof(int) * B_nnz_));
      cudaCheckError(cudaMalloc((void**)&B_row_dev_, sizeof(int) * (k_ + 1)));

      cudaCheckError(cudaMalloc((void**)&C_row_dev_, sizeof(int) * (n_ + 1)));
    }

    C_mem_allocated_always_ = false;
    C_mem_allocated_once_ = false;
    C_mem_allocated_unified_ = false;

//    std::cout << "_____Matrix A_____" << std::endl;
//    printDenseMatrix(A_, n_, n_);
//    std::cout << std::endl << std::endl;
//    printCSR(A_val_, A_col_, A_row_, A_nnz_, n_, n_);
//
//
//    std::cout << "_____Matrix B_____" << std::endl;
//    printDenseMatrix(B_, n_, n_);
//    std::cout << std::endl << std::endl;
//    printCSR(B_val_, B_col_, B_row_, B_nnz_, n_, n_);

    // Create a handle for cuSPARSE
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

    // Load B into CSR
    int nnz_encountered = 0;
    for (int row = 0; row < k_; row++) {
      B_row_[row] = nnz_encountered;
      int nnz_row = 0;
      for (int col = 0; col < n_; col++) {
        if (B_[(row * n_) + col] != 0.0) {
          nnz_row++;
          B_col_[nnz_encountered] = col;
          B_val_[nnz_encountered] = B_[(row * n_) + col];
          nnz_encountered++;
        }
      }
    }
    B_row_[k_] = nnz_encountered;
  }

 private:
  /** Perform any required steps before calling the GEMM kernel that should
   * be timed. */
  void preLoopRequirements() override {
    cusparseCheckError(cusparseSpGEMM_createDescr(&spgemmDesc_));
    switch(offload_) {
      case gpuOffloadType::always: {
        // Make matrix descriptors
        cusparseCheckError(
                cusparseCreateCsr(&descrA_, m_, k_, A_nnz_, A_row_dev_,
                                  A_col_dev_, A_val_dev_, rType_, cType_,
                                  indType_, cudaDataType_));
        cusparseCheckError(
                cusparseCreateCsr(&descrB_, k_, n_, B_nnz_, B_row_dev_,
                                  B_col_dev_, B_val_dev_, rType_, cType_,
                                  indType_, cudaDataType_));
        cusparseCheckError(
                cusparseCreateCsr(&descrC_, n_, m_, 0, C_row_dev_, NULL, NULL,
                                  rType_, cType_, indType_, cudaDataType_));
        break;
      }
      case gpuOffloadType::once: {
        cudaCheckError(cudaMemcpyAsync(A_val_dev_, A_val_, sizeof(T) *
                                       A_nnz_, cudaMemcpyHostToDevice, s1_));
        cudaCheckError(cudaMemcpyAsync(A_col_dev_, A_col_, sizeof(int) *
                                       A_nnz_, cudaMemcpyHostToDevice, s1_));
        cudaCheckError(cudaMemcpyAsync(A_row_dev_, A_row_, sizeof(int) * (m_
                                       + 1), cudaMemcpyHostToDevice, s1_));

        cudaCheckError(cudaMemcpyAsync(B_val_dev_, B_val_, sizeof(T) *
                                       B_nnz_, cudaMemcpyHostToDevice, s2_));
        cudaCheckError(cudaMemcpyAsync(B_col_dev_, B_col_, sizeof(int) *
                                       B_nnz_, cudaMemcpyHostToDevice, s2_));
        cudaCheckError(cudaMemcpyAsync(B_row_dev_, B_row_, sizeof(int) * (k_
                                       + 1), cudaMemcpyHostToDevice, s2_));

        cudaCheckError(cudaMemcpyAsync(C_row_dev_, C_row_, sizeof(int) * (n_
        + 1), cudaMemcpyHostToDevice, s3_));

        // Create matrix descriptors
        cusparseCheckError(
                cusparseCreateCsr(&descrA_, m_, k_, A_nnz_, A_row_dev_,
                                  A_col_dev_, A_val_dev_, rType_, cType_,
                                  indType_, cudaDataType_));
        cusparseCheckError(
                cusparseCreateCsr(&descrB_, k_, n_, B_nnz_, B_row_dev_,
                                  B_col_dev_, B_val_dev_, rType_, cType_,
                                  indType_, cudaDataType_));
        cusparseCheckError(
                cusparseCreateCsr(&descrC_, n_, m_, 0, C_row_dev_, NULL, NULL,
                                  rType_, cType_, indType_, cudaDataType_));
        break;
      }
      case gpuOffloadType::unified: {
        // Prefetch memory to device
        cudaCheckError(cudaMemPrefetchAsync(A_val_, sizeof(T) * A_nnz_,
                                            gpuDevice_, s1_));
        cudaCheckError(cudaMemPrefetchAsync(A_col_, sizeof(int) * A_nnz_,
                                            gpuDevice_, s1_));
        cudaCheckError(cudaMemPrefetchAsync(A_row_, sizeof(int) * (m_ + 1),
                                            gpuDevice_, s1_));

        cudaCheckError(cudaMemPrefetchAsync(B_val_, sizeof(T) * B_nnz_,
                                            gpuDevice_, s2_));
        cudaCheckError(cudaMemPrefetchAsync(B_col_, sizeof(int) * B_nnz_,
                                            gpuDevice_, s2_));
        cudaCheckError(cudaMemPrefetchAsync(B_row_, sizeof(int) * (k_ + 1),
                                            gpuDevice_, s2_));

        // Make matrix descriptors
        cusparseCheckError(
                cusparseCreateCsr(&descrA_, m_, k_, A_nnz_, A_row_, A_col_,
                                  A_val_, rType_, cType_, indType_,
                                  cudaDataType_));
        cusparseCheckError(
                cusparseCreateCsr(&descrB_, k_, n_, B_nnz_, B_row_, B_col_,
                                  B_val_, rType_, cType_, indType_,
                                  cudaDataType_));
        cusparseCheckError(
                cusparseCreateCsr(&descrC_, n_, m_, 0, C_row_, NULL, NULL,
                                  rType_, cType_, indType_, cudaDataType_));
        break;
      }
    }
  }

  /** Make a call to the BLAS Library Kernel. */
  void callSpmm() override {
    switch(offload_) {
      case gpuOffloadType::always: {
        if (C_mem_allocated_always_) {
          cusparseCheckError(cusparseDestroySpMat(descrA_));
          cusparseCheckError(cusparseDestroySpMat(descrB_));
          cusparseCheckError(cusparseDestroySpMat(descrC_));
        }
        cudaCheckError(cudaMemcpyAsync(A_val_dev_, A_val_, sizeof(T) *
        A_nnz_, cudaMemcpyHostToDevice, s1_));
        cudaCheckError(cudaMemcpyAsync(A_col_dev_, A_col_, sizeof(int) *
        A_nnz_, cudaMemcpyHostToDevice, s1_));
        cudaCheckError(cudaMemcpyAsync(A_row_dev_, A_row_, sizeof(int) * (m_
                                       + 1), cudaMemcpyHostToDevice, s1_));

        cudaCheckError(cudaMemcpyAsync(B_val_dev_, B_val_, sizeof(T) *
        B_nnz_, cudaMemcpyHostToDevice, s2_));
        cudaCheckError(cudaMemcpyAsync(B_col_dev_, B_col_, sizeof(int) *
        B_nnz_, cudaMemcpyHostToDevice, s2_));
        cudaCheckError(cudaMemcpyAsync(B_row_dev_, B_row_, sizeof(int) * (k_
                                       + 1), cudaMemcpyHostToDevice, s2_));

        cudaCheckError(cudaMemcpyAsync(C_row_dev_, C_row_, sizeof(int) * (n_
        + 1), cudaMemcpyHostToDevice, s3_));
        cudaCheckError(cudaDeviceSynchronize());

        // Make matrix descriptors
        cusparseCheckError(
                cusparseCreateCsr(&descrA_, m_, k_, A_nnz_, A_row_dev_,
                                  A_col_dev_, A_val_dev_, rType_, cType_,
                                  indType_, cudaDataType_));
        cusparseCheckError(
                cusparseCreateCsr(&descrB_, k_, n_, B_nnz_, B_row_dev_,
                                  B_col_dev_, B_val_dev_, rType_, cType_,
                                  indType_, cudaDataType_));
        cusparseCheckError(
                cusparseCreateCsr(&descrC_, n_, m_, 0, C_row_dev_, NULL, NULL,
                                  rType_, cType_, indType_, cudaDataType_));

        cusparseCheckError(
                cusparseSpGEMM_workEstimation(handle_, opA_, opB_, &alpha,
                                              descrA_, descrB_, &beta,
                                              descrC_, cudaDataType_, alg_,
                                              spgemmDesc_, &buffer_size1_,
                                              NULL));
        cudaCheckError(cudaMalloc((void**)&buffer1_, buffer_size1_));
        cusparseCheckError(
                cusparseSpGEMM_workEstimation(handle_, opA_, opB_, &alpha,
                                              descrA_, descrB_, &beta,
                                              descrC_, cudaDataType_, alg_,
                                              spgemmDesc_, &buffer_size1_,
                                              buffer1_));
        cusparseCheckError(
                cusparseSpGEMM_compute(handle_, opA_, opB_, &alpha, descrA_,
                                       descrB_, &beta, descrC_, cudaDataType_,
                                       alg_, spgemmDesc_, &buffer_size2_,
                                       NULL));
        cudaCheckError(cudaMalloc((void**)&buffer2_, buffer_size2_));

        cusparseCheckError(
                cusparseSpGEMM_compute(handle_, opA_, opB_, &alpha, descrA_,
                                       descrB_, &beta, descrC_,
                                       cudaDataType_, alg_, spgemmDesc_,
                                       &buffer_size2_, buffer2_));

        cusparseCheckError(
                cusparseSpMatGetSize(descrC_, &C_num_rows_, &C_num_cols_,
                                     &C_nnz_));

        if (C_mem_allocated_always_) {
          cudaCheckError(cudaFree(C_val_dev_));
          cudaCheckError(cudaFree(C_col_dev_));
        }
        cudaCheckError(cudaMalloc(&C_val_dev_, sizeof(T) * C_nnz_));
        cudaCheckError(cudaMalloc(&C_col_dev_, sizeof(int) * C_nnz_));

        cusparseCheckError(
                cusparseCsrSetPointers(descrC_, C_row_dev_, C_col_dev_,
                                       C_val_dev_));
        cusparseCheckError(
                cusparseSpGEMM_copy(handle_, opA_, opB_, &alpha, descrA_,
                                    descrB_, &beta, descrC_, cudaDataType_,
                                    alg_, spgemmDesc_));

        cudaCheckError(cudaMemcpyAsync(A_val_, A_val_dev_, sizeof(T) *
        A_nnz_, cudaMemcpyDeviceToHost, s1_));
        cudaCheckError(cudaMemcpyAsync(A_col_, A_col_dev_, sizeof(int) *
        A_nnz_, cudaMemcpyDeviceToHost, s1_));
        cudaCheckError(cudaMemcpyAsync(A_row_, A_row_dev_, sizeof(int) *
        (m_ + 1), cudaMemcpyDeviceToHost, s1_));

        cudaCheckError(cudaMemcpyAsync(B_val_, B_val_dev_, sizeof(T) *
        B_nnz_, cudaMemcpyDeviceToHost, s2_));
        cudaCheckError(cudaMemcpyAsync(B_col_, B_col_dev_, sizeof(int) *
        B_nnz_, cudaMemcpyDeviceToHost, s2_));
        cudaCheckError(cudaMemcpyAsync(B_row_, B_row_dev_, sizeof(int) *
        (k_ + 1), cudaMemcpyDeviceToHost, s2_));

        if (C_mem_allocated_always_) {
          free(C_val_);
          free(C_col_);
        }
        C_val_ = (T*)malloc(sizeof(T) * C_nnz_);
        C_col_ = (int*)malloc(sizeof(int) * C_nnz_);
        C_mem_allocated_always_ = true;

        cudaCheckError(cudaMemcpyAsync(C_val_, C_val_dev_, sizeof(T) *
        C_nnz_, cudaMemcpyDeviceToHost, s3_));
        cudaCheckError(cudaMemcpyAsync(C_col_, C_col_dev_, sizeof(int) *
        C_nnz_, cudaMemcpyDeviceToHost, s3_));
        cudaCheckError(cudaMemcpyAsync(C_row_, C_row_dev_, sizeof(int) *
        (n_ + 1), cudaMemcpyDeviceToHost, s3_));
        cudaCheckError(cudaDeviceSynchronize());

        // Freeing memory
        cudaCheckError(cudaFree(buffer1_));
        cudaCheckError(cudaFree(buffer2_));
        buffer_size1_ = 0;
        buffer_size2_ = 0;
        break;
      }
      case gpuOffloadType::once: {
        cusparseCheckError(
                cusparseSpGEMM_workEstimation(handle_, opA_, opB_, &alpha,
                                              descrA_, descrB_, &beta,
                                              descrC_, cudaDataType_, alg_,
                                              spgemmDesc_, &buffer_size1_,
                                              NULL));
        cudaCheckError(cudaMalloc((void**)&buffer1_, buffer_size1_));
        cusparseCheckError(
                cusparseSpGEMM_workEstimation(handle_, opA_, opB_, &alpha,
                                              descrA_, descrB_, &beta,
                                              descrC_, cudaDataType_, alg_,
                                              spgemmDesc_, &buffer_size1_,
                                              buffer1_));
        cusparseCheckError(
                cusparseSpGEMM_compute(handle_, opA_, opB_, &alpha, descrA_,
                                       descrB_, &beta, descrC_, cudaDataType_,
                                       alg_, spgemmDesc_, &buffer_size2_,
                                       NULL));
        cudaCheckError(cudaMalloc((void**)&buffer2_, buffer_size2_));

        cusparseCheckError(
                cusparseSpGEMM_compute(handle_, opA_, opB_, &alpha, descrA_,
                               descrB_, &beta, descrC_, cudaDataType_,
                               alg_, spgemmDesc_, &buffer_size2_, buffer2_));

        cusparseCheckError(
                cusparseSpMatGetSize(descrC_, &C_num_rows_, &C_num_cols_,
                                     &C_nnz_));

        if (C_mem_allocated_once_) {
          cudaCheckError(cudaFree(C_val_dev_));
          cudaCheckError(cudaFree(C_col_dev_));
        }
        cudaCheckError(cudaMalloc(&C_val_dev_, sizeof(T) * C_nnz_));
        cudaCheckError(cudaMalloc(&C_col_dev_, sizeof(int) * C_nnz_));
        C_mem_allocated_once_ = true;

        cusparseCheckError(
                cusparseCsrSetPointers(descrC_, C_row_dev_, C_col_dev_,
                                       C_val_dev_));
        cusparseCheckError(
                cusparseSpGEMM_copy(handle_, opA_, opB_, &alpha,
                                    descrA_, descrB_, &beta, descrC_,
                                    cudaDataType_, alg_, spgemmDesc_));

        // Freeing memory
        cudaCheckError(cudaFree(buffer1_));
        cudaCheckError(cudaFree(buffer2_));
        buffer_size1_ = 0;
        buffer_size2_ = 0;
        break;
      }
      case gpuOffloadType::unified: {
        cusparseCheckError(
                cusparseSpGEMM_workEstimation(handle_, opA_, opB_, &alpha,
                                              descrA_, descrB_, &beta,
                                              descrC_, cudaDataType_,
                                              alg_, spgemmDesc_, &buffer_size1_,
                                              NULL));
        cudaCheckError(cudaMallocManaged((void**)&buffer1_, buffer_size1_));
        cusparseCheckError(
                cusparseSpGEMM_workEstimation(handle_, opA_, opB_, &alpha,
                                              descrA_, descrB_, &beta,
                                              descrC_, cudaDataType_,
                                              alg_, spgemmDesc_, &buffer_size1_,
                                              buffer1_));
        cusparseCheckError(
                cusparseSpGEMM_compute(handle_, opA_, opB_, &alpha, descrA_,
                                       descrB_, &beta, descrC_, cudaDataType_,
                                       alg_, spgemmDesc_, &buffer_size2_,
                                       NULL));
        cudaCheckError(cudaMallocManaged((void**)&buffer2_, buffer_size2_));

        cusparseCheckError(
                cusparseSpGEMM_compute(handle_, opA_, opB_, &alpha, descrA_,
                               descrB_, &beta, descrC_, cudaDataType_,
                               alg_, spgemmDesc_, &buffer_size2_, buffer2_));

        cusparseCheckError(
                cusparseSpMatGetSize(descrC_, &C_num_rows_, &C_num_cols_,
                                     &C_nnz_));

        if (C_mem_allocated_unified_) {
          cudaCheckError(cudaFree(C_val_));
          cudaCheckError(cudaFree(C_col_));
        }

        cudaCheckError(cudaMallocManaged(&C_val_, sizeof(T) * C_nnz_));
        cudaCheckError(cudaMallocManaged(&C_col_, sizeof(int) * C_nnz_));
        C_mem_allocated_unified_ = true;

        cusparseCheckError(
                cusparseCsrSetPointers(descrC_, C_row_, C_col_, C_val_));
        cusparseCheckError(
                cusparseSpGEMM_copy(handle_, opA_, opB_, &alpha, descrA_,
                                    descrB_, &beta, descrC_, cudaDataType_,
                                    alg_, spgemmDesc_));


        // Freeing memory
        cudaCheckError(cudaFree(buffer1_));
        cudaCheckError(cudaFree(buffer2_));
        buffer_size1_ = 0;
        buffer_size2_ = 0;
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
        cudaCheckError(cudaMemcpyAsync(A_val_, A_val_dev_, sizeof(T) *
        A_nnz_, cudaMemcpyDeviceToHost, s1_));
        cudaCheckError(cudaMemcpyAsync(A_col_, A_col_dev_, sizeof(int) *
        A_nnz_, cudaMemcpyDeviceToHost, s1_));
        cudaCheckError(cudaMemcpyAsync(A_row_, A_row_dev_, sizeof(int) *
        (m_ + 1), cudaMemcpyDeviceToHost, s1_));

        cudaCheckError(cudaMemcpyAsync(B_val_, B_val_dev_, sizeof(T) *
        B_nnz_, cudaMemcpyDeviceToHost, s2_));
        cudaCheckError(cudaMemcpyAsync(B_col_, B_col_dev_, sizeof(int) *
        B_nnz_, cudaMemcpyDeviceToHost, s2_));
        cudaCheckError(cudaMemcpyAsync(B_row_, B_row_dev_, sizeof(int) *
        (k_ + 1), cudaMemcpyDeviceToHost, s2_));

        C_val_ = (T*)malloc(sizeof(T) * C_nnz_);
        C_col_ = (int*)malloc(sizeof(int) * C_nnz_);
        cudaCheckError(cudaMemcpyAsync(C_val_, C_val_dev_, sizeof(T) *
        C_nnz_, cudaMemcpyDeviceToHost, s3_));
        cudaCheckError(cudaMemcpyAsync(C_col_, C_col_dev_, sizeof(int) *
        C_nnz_, cudaMemcpyDeviceToHost, s3_));
        cudaCheckError(cudaMemcpyAsync(C_row_, C_row_dev_, sizeof(int) *
        (n_ + 1), cudaMemcpyDeviceToHost, s3_));
        cudaCheckError(cudaDeviceSynchronize());

        cusparseCheckError(cusparseDestroySpMat(descrA_));
        cusparseCheckError(cusparseDestroySpMat(descrB_));
        cusparseCheckError(cusparseDestroySpMat(descrC_));

        break;
      }
      case gpuOffloadType::unified: {
        // Ensure all data resides on host once work has completed
        cudaCheckError(cudaMemPrefetchAsync(A_val_, sizeof(T) * A_nnz_,
                                            cudaCpuDeviceId, s1_));
        cudaCheckError(cudaMemPrefetchAsync(A_col_, sizeof(int) * A_nnz_,
                                            cudaCpuDeviceId, s1_));
        cudaCheckError(cudaMemPrefetchAsync(A_row_, sizeof(int) * (m_ + 1),
                                            cudaCpuDeviceId, s1_));

        cudaCheckError(cudaMemPrefetchAsync(B_val_, sizeof(T) * B_nnz_,
                                            cudaCpuDeviceId, s2_));
        cudaCheckError(cudaMemPrefetchAsync(B_col_, sizeof(int) * B_nnz_,
                                            cudaCpuDeviceId, s2_));
        cudaCheckError(cudaMemPrefetchAsync(B_row_, sizeof(int) * (k_ + 1),
                                            cudaCpuDeviceId, s2_));

        cudaCheckError(cudaMemPrefetchAsync(C_val_, sizeof(T) * C_nnz_,
                                            cudaCpuDeviceId, s3_));
        cudaCheckError(cudaMemPrefetchAsync(C_col_, sizeof(int) * C_nnz_,
                                            cudaCpuDeviceId, s3_));
        cudaCheckError(cudaMemPrefetchAsync(C_row_, sizeof(int) * (n_ + 1),
                                            cudaCpuDeviceId, s3_));
        // Ensure device has finished all work.
        cudaCheckError(cudaDeviceSynchronize());

        cusparseCheckError(cusparseDestroySpMat(descrA_));
        cusparseCheckError(cusparseDestroySpMat(descrB_));
        cusparseCheckError(cusparseDestroySpMat(descrC_));
        break;
      }
    }
    cusparseCheckError(cusparseSpGEMM_destroyDescr(spgemmDesc_));
  }

  /** Do any necessary cleanup (free pointers, close library handles, etc.)
   * after Kernel has been called. */
  void postCallKernelCleanup() override {
    // Destroy the handle
    cusparseCheckError(cusparseDestroy(handle_));

    // Destroy streams after use
    cudaCheckError(cudaStreamDestroy(s1_));
    cudaCheckError(cudaStreamDestroy(s2_));
    cudaCheckError(cudaStreamDestroy(s3_));

    free(A_);
    free(B_);

    if (offload_ == gpuOffloadType::unified) {
      cudaCheckError(cudaFree(A_val_));
      cudaCheckError(cudaFree(A_col_));
      cudaCheckError(cudaFree(A_row_));
      cudaCheckError(cudaFree(B_val_));
      cudaCheckError(cudaFree(B_col_));
      cudaCheckError(cudaFree(B_row_));
      cudaCheckError(cudaFree(C_val_));
      cudaCheckError(cudaFree(C_col_));
      cudaCheckError(cudaFree(C_row_));
    } else {
      free(A_val_);
      free(A_col_);
      free(A_row_);
      free(B_val_);
      free(B_col_);
      free(B_row_);
      free(C_val_);
      free(C_col_);
      free(C_row_);
      cudaCheckError(cudaFree(A_val_dev_));
      cudaCheckError(cudaFree(A_col_dev_));
      cudaCheckError(cudaFree(A_row_dev_));
      cudaCheckError(cudaFree(B_val_dev_));
      cudaCheckError(cudaFree(B_col_dev_));
      cudaCheckError(cudaFree(B_row_dev_));
      cudaCheckError(cudaFree(C_val_dev_));
      cudaCheckError(cudaFree(C_col_dev_));
      cudaCheckError(cudaFree(C_row_dev_));
    }
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

	/** CSR format vectors for matrices A, B and C on the host */
	T* A_val_;
	int* A_col_;
  int* A_row_;
  int64_t A_num_rows_;
  int64_t A_num_cols_;

  T* B_val_;
  int* B_col_;
  int* B_row_;
  int64_t B_num_rows_;
  int64_t B_num_cols_;

  int64_t C_num_rows_;
  int64_t C_num_cols_;

  /** CSR format vectors for matrices A, B and C on the device. */
	T* A_val_dev_;
  T* B_val_dev_;
  T* C_val_dev_;
	int* A_col_dev_;
  int* A_row_dev_;
  int* B_col_dev_;
  int* B_row_dev_;
  int* C_col_dev_;
  int* C_row_dev_;

  bool C_mem_allocated_always_;
  bool C_mem_allocated_once_;
  bool C_mem_allocated_unified_;

  /** The constant value Alpha. */
  const T alpha = ALPHA;

  /** The constant value Beta. */
  const T beta = BETA;


	// Create descriptors for matrices A->C
	cusparseSpMatDescr_t descrA_, descrB_, descrC_;

	// Data type depends on kernel being run
	cudaDataType_t cudaDataType_;

	cusparseSpGEMMDescr_t spgemmDesc_;

	size_t buffer_size1_ = 0;
	size_t buffer_size2_ = 0;
  void* buffer1_ = NULL;
	void* buffer2_ = NULL;

  cusparseOperation_t opA_ = CUSPARSE_OPERATION_NON_TRANSPOSE;
  cusparseOperation_t opB_ = CUSPARSE_OPERATION_NON_TRANSPOSE;
  cusparseSpGEMMAlg_t alg_ = CUSPARSE_SPGEMM_DEFAULT;
  cusparseIndexType_t rType_ = CUSPARSE_INDEX_32I;
  cusparseIndexType_t cType_ = CUSPARSE_INDEX_32I;
  cusparseIndexBase_t indType_ = CUSPARSE_INDEX_BASE_ZERO;
};
}  // namespace gpu
#endif