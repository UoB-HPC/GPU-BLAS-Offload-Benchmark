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
  using spmm<T>::m_;
  using spmm<T>::n_;
  using spmm<T>::k_;
  using spmm<T>::offload_;
  using spmm<T>::sparsity_;
  using spmm<T>::type_;
  using spmm<T>::C_nnz_;
  using spmm<T>::C_rows_;
  using spmm<T>::C_cols_;
  using spmm<T>::C_vals_;

  ~spmm_gpu() {
    if (alreadyInitialised_) {
      alreadyInitialised_ = false;
      cusparseCheckError(cusparseDestroy(handle_));

      cudaCheckError(cudaStreamDestroy(s1_));
      cudaCheckError(cudaStreamDestroy(s2_));
      cudaCheckError(cudaStreamDestroy(s3_));
      cudaCheckError(cudaStreamDestroy(s4_));
      cudaCheckError(cudaStreamDestroy(s5_));
      cudaCheckError(cudaStreamDestroy(s6_));
    }
  }

  /** Initialise the required data structures.
   * `offload` refers to the data offload type:
   *  - Once:    Move data from host to device before all iterations & move from
   *             device to host after all iterations
   *  - Always:  Move data from host to device and device to host each iteration
   *  - Unified: Initialise data as unified memory; no data movement semantics
   *             required */
  void initialise(gpuOffloadType offload, int n, int m, int k, 
                  double sparsity, matrixType type, 
                  bool binary = false) override {
    if (!alreadyInitialised_) {
      alreadyInitialised_ = true;
      cusparseCheckError(cusparseCreate(&handle_));
      
      cudaCheckError(cudaStreamCreate(&s1_));
      cudaCheckError(cudaStreamCreate(&s2_));
      cudaCheckError(cudaStreamCreate(&s3_));
      cudaCheckError(cudaStreamCreate(&s4_));
      cudaCheckError(cudaStreamCreate(&s5_));
      cudaCheckError(cudaStreamCreate(&s6_));

      cusparseCheckError(cusparseSetStream(handle_, s1_));

      // Get device identifier
      cudaCheckError(cudaGetDevice(&gpuDevice_));
    }

    offload_ = offload;
    sparsity_ = sparsity;
    type_ = type;

    m_ = m;
    n_ = n;
    k_ = k;

    /** Determine the number of nnz elements in A and B */
    A_nnz_ = 1 + (uint64_t)((double)m_ * (double)k_ * (1.0 - sparsity_));
    B_nnz_ = 1 + (uint64_t)((double)k_ * (double)n_ * (1.0 - sparsity_));

    opA_ = CUSPARSE_OPERATION_NON_TRANSPOSE;
    opB_ = CUSPARSE_OPERATION_NON_TRANSPOSE;
    alg_ = CUSPARSE_SPGEMM_DEFAULT;
    index_ = CUSPARSE_INDEX_32I;
    base_ = CUSPARSE_INDEX_BASE_ZERO;
    if (std::is_same_v<T, float>) dataType_ = CUDA_R_32F;
    else if (std::is_same_v<T, double>) dataType_ = CUDA_R_64F;
    else {
      std::cerr << "INVALID DATA TYPE PASSED TO cuSPARSE" << std::endl;
      exit(1);
    }

    initInputMatrices();
  }

 protected:
  void toSparseFormat() override {
    // Allocate CSR arrays
    if (offload_ == gpuOffloadType::unified) {
      cudaCheckError(cudaMallocManaged(&A_vals_, sizeof(T) * A_nnz_));
      cudaCheckError(cudaMallocManaged(&A_cols_, sizeof(int32_t) * A_nnz_));
      cudaCheckError(cudaMallocManaged(&A_rows_, sizeof(int32_t) * (m_ + 1)));
      cudaCheckError(cudaMallocManaged(&B_vals_, sizeof(T) * B_nnz_));
      cudaCheckError(cudaMallocManaged(&B_cols_, sizeof(int32_t) * B_nnz_));
      cudaCheckError(cudaMallocManaged(&B_rows_, sizeof(int32_t) * (k_ + 1)));
      cudaCheckError(cudaMallocManaged(&C_rows_32_, sizeof(int32_t) * (m_ + 1)));
      C_vals_ = nullptr;
      C_cols_32_ = nullptr;
    } else {
      A_vals_ = (T*)malloc(sizeof(T) * A_nnz_);
      A_cols_ = (int32_t*)malloc(sizeof(int32_t) * A_nnz_);
      A_rows_ = (int32_t*)malloc(sizeof(int32_t) * (m_ + 1));
      B_vals_ = (T*)malloc(sizeof(T) * B_nnz_);
      B_cols_ = (int32_t*)malloc(sizeof(int32_t) * B_nnz_);
      B_rows_ = (int32_t*)malloc(sizeof(int32_t) * (k_ + 1));
      C_rows_32_ = (int32_t*)malloc(sizeof(int32_t) * (m_ + 1));
      C_vals_ = nullptr;
      C_cols_32_ = nullptr;

      cudaCheckError(cudaMalloc((void**)&A_vals_dev_, sizeof(T) * A_nnz_));
      cudaCheckError(cudaMalloc((void**)&A_cols_dev_, sizeof(int32_t) * A_nnz_));
      cudaCheckError(cudaMalloc((void**)&A_rows_dev_, sizeof(int32_t) * (m_ + 1)));
      cudaCheckError(cudaMalloc((void**)&B_vals_dev_, sizeof(T) * B_nnz_));
      cudaCheckError(cudaMalloc((void**)&B_cols_dev_, sizeof(int32_t) * B_nnz_));
      cudaCheckError(cudaMalloc((void**)&B_rows_dev_, sizeof(int32_t) * (k_ + 1)));
      cudaCheckError(cudaMalloc((void**)&C_rows_dev_, sizeof(int32_t) * (m_ + 1)));
      C_vals_dev_ = nullptr;
      C_cols_dev_ = nullptr;
    }
    cudaCheckError(cudaDeviceSynchronize());
    int seedOffset = 0;
    if (type_ == matrixType::rmat) {
      rMatCSR<T, int32_t>(A_vals_, A_cols_, A_rows_, m_, k_, A_nnz_, SEED + seedOffset++);
      rMatCSR<T, int32_t>(B_vals_, B_cols_, B_rows_, k_, n_, B_nnz_, SEED + seedOffset++);
    } else if (type_ == matrixType::random) {
      randomCSR<T, int32_t>(A_vals_, A_cols_, A_rows_, m_, k_, A_nnz_, SEED + seedOffset++);
      randomCSR<T, int32_t>(B_vals_, B_cols_, B_rows_, k_, n_, B_nnz_, SEED + seedOffset++);
    } else {
      std::cerr << "Matrix type not supported" << std::endl;
      exit(1);
    }

    while (calcCNNZ<int32_t>(m_, A_nnz_, A_rows_, A_cols_, k_, B_nnz_, B_rows_, B_cols_) == 0) {
      if (type_ == matrixType::rmat) {
        rMatCSR<T, int32_t>(A_vals_, A_cols_, A_rows_, m_, k_, A_nnz_, SEED + seedOffset++);
        rMatCSR<T, int32_t>(B_vals_, B_cols_, B_rows_, k_, n_, B_nnz_, SEED + seedOffset++);
      } else if (type_ == matrixType::random) {
        randomCSR<T, int32_t>(A_vals_, A_cols_, A_rows_, m_, k_, A_nnz_, SEED + seedOffset++);
        randomCSR<T, int32_t>(B_vals_, B_cols_, B_rows_, k_, n_, B_nnz_, SEED + seedOffset++);
      } else {
        std::cerr << "Matrix type not supported" << std::endl;
        exit(1);
      }
    }
  }

 private:
  /** Perform any required steps before calling the GEMM kernel that should
   * be timed. */
  void preLoopRequirements() override {
    switch(offload_) {
      case gpuOffloadType::always: {
        break;
      }
      case gpuOffloadType::once: {
        cudaCheckError(cudaMemcpyAsync(A_vals_dev_, A_vals_, sizeof(T) * A_nnz_, cudaMemcpyHostToDevice, s1_));
        cudaCheckError(cudaMemcpyAsync(A_cols_dev_, A_cols_, sizeof(int32_t) * A_nnz_, cudaMemcpyHostToDevice, s2_));
        cudaCheckError(cudaMemcpyAsync(A_rows_dev_, A_rows_, sizeof(int32_t) * (m_ + 1), cudaMemcpyHostToDevice, s3_));
        cudaCheckError(cudaMemcpyAsync(B_vals_dev_, B_vals_, sizeof(T) * B_nnz_, cudaMemcpyHostToDevice, s4_));
        cudaCheckError(cudaMemcpyAsync(B_cols_dev_, B_cols_, sizeof(int32_t) * B_nnz_, cudaMemcpyHostToDevice, s5_));
        cudaCheckError(cudaMemcpyAsync(B_rows_dev_, B_rows_, sizeof(int32_t) * (k_ + 1), cudaMemcpyHostToDevice, s6_));
        cudaCheckError(cudaDeviceSynchronize());
        break;
      }
      case gpuOffloadType::unified: {
        // Prefetch memory to device
        cudaCheckError(cudaMemPrefetchAsync(A_vals_, sizeof(T) * A_nnz_, gpuDevice_, s1_));
        cudaCheckError(cudaMemPrefetchAsync(A_cols_, sizeof(int32_t) * A_nnz_, gpuDevice_, s2_));
        cudaCheckError(cudaMemPrefetchAsync(A_rows_, sizeof(int32_t) * (m_ + 1), gpuDevice_, s3_));
        cudaCheckError(cudaMemPrefetchAsync(B_vals_, sizeof(T) * B_nnz_, gpuDevice_, s4_));
        cudaCheckError(cudaMemPrefetchAsync(B_cols_, sizeof(int32_t) * B_nnz_, gpuDevice_, s5_));
        cudaCheckError(cudaMemPrefetchAsync(B_rows_, sizeof(int32_t) * (k_ + 1), gpuDevice_, s6_));
        cudaCheckError(cudaDeviceSynchronize());
        break;
      }
    }
  }

  /** Make a call to the BLAS Library Kernel. */
  void callSpmm() override {
    switch(offload_) {
      case gpuOffloadType::always: {
        if (C_allocated) {
          free(C_vals_);
          free(C_cols_32_);
          C_allocated = false;
        }

        cudaCheckError(cudaMemcpyAsync(A_vals_dev_, A_vals_, sizeof(T) * A_nnz_, cudaMemcpyHostToDevice, s1_));
        cudaCheckError(cudaMemcpyAsync(A_cols_dev_, A_cols_, sizeof(int32_t) * A_nnz_, cudaMemcpyHostToDevice, s2_));
        cudaCheckError(cudaMemcpyAsync(A_rows_dev_, A_rows_, sizeof(int32_t) * (m_ + 1), cudaMemcpyHostToDevice, s3_));
        cudaCheckError(cudaMemcpyAsync(B_vals_dev_, B_vals_, sizeof(T) * B_nnz_, cudaMemcpyHostToDevice, s4_));
        cudaCheckError(cudaMemcpyAsync(B_cols_dev_, B_cols_, sizeof(int32_t) * B_nnz_, cudaMemcpyHostToDevice, s5_));
        cudaCheckError(cudaMemcpyAsync(B_rows_dev_, B_rows_, sizeof(int32_t) * (k_ + 1), cudaMemcpyHostToDevice, s6_));
        cudaCheckError(cudaDeviceSynchronize());

        // Make matrix descriptors
        cusparseCheckError(cusparseCreateCsr(&A_descr_, 
                                             m_, 
                                             k_, 
                                             A_nnz_, 
                                             A_rows_dev_,
                                             A_cols_dev_, 
                                             A_vals_dev_, 
                                             index_, 
                                             index_,
                                             base_, 
                                             dataType_));
        cusparseCheckError(cusparseCreateCsr(&B_descr_, 
                                             k_, 
                                             n_, 
                                             B_nnz_, 
                                             B_rows_dev_,
                                             B_cols_dev_, 
                                             B_vals_dev_, 
                                             index_, 
                                             index_,
                                             base_, 
                                             dataType_));
        cusparseCheckError(cusparseCreateCsr(&C_descr_, 
                                             m_, 
                                             n_, 
                                             0, 
                                             nullptr,
                                             nullptr, 
                                             nullptr, 
                                             index_, 
                                             index_,
                                             base_, 
                                             dataType_));
        
        cusparseCheckError(cusparseSpGEMM_createDescr(&spgemmDescr_));

        size_t bufferSize1 = 0;
        cusparseCheckError(cusparseSpGEMM_workEstimation(handle_, 
                                                         opA_, 
                                                         opB_, 
                                                         &alpha,
                                                         A_descr_, 
                                                         B_descr_, 
                                                         &beta,
                                                         C_descr_, 
                                                         dataType_, 
                                                         alg_,
                                                         spgemmDescr_, 
                                                         &bufferSize1,
                                                         nullptr));

        void* dBuffer1 = nullptr;
        cudaCheckError(cudaMalloc((void**)&dBuffer1, bufferSize1));

        cusparseCheckError(cusparseSpGEMM_workEstimation(handle_, 
                                                         opA_, 
                                                         opB_, 
                                                         &alpha,
                                                         A_descr_, 
                                                         B_descr_, 
                                                         &beta,
                                                         C_descr_, 
                                                         dataType_, 
                                                         alg_,
                                                         spgemmDescr_, 
                                                         &bufferSize1,
                                                         dBuffer1));

        size_t bufferSize2 = 0;
        cusparseCheckError(cusparseSpGEMM_compute(handle_, 
                                                  opA_, 
                                                  opB_, 
                                                  &alpha, 
                                                  A_descr_,
                                                  B_descr_, 
                                                  &beta, 
                                                  C_descr_, 
                                                  dataType_,
                                                  alg_, 
                                                  spgemmDescr_, 
                                                  &bufferSize2,
                                                  nullptr));

        void* dBuffer2 = nullptr;
        cudaCheckError(cudaMalloc((void**)&dBuffer2, bufferSize2));

        cusparseCheckError(cusparseSpGEMM_compute(handle_, 
                                                  opA_, 
                                                  opB_, 
                                                  &alpha, 
                                                  A_descr_,
                                                  B_descr_, 
                                                  &beta, 
                                                  C_descr_,
                                                  dataType_, 
                                                  alg_, 
                                                  spgemmDescr_,
                                                  &bufferSize2, 
                                                  dBuffer2));

        cusparseCheckError(cusparseSpMatGetSize(C_descr_, 
                                                &C_num_rows_, 
                                                &C_num_cols_,
                                                &C_nnz_));

        cudaCheckError(cudaMalloc(&C_vals_dev_, sizeof(T) * C_nnz_));
        cudaCheckError(cudaMalloc(&C_cols_dev_, sizeof(int32_t) * C_nnz_));

        cusparseCheckError(cusparseCsrSetPointers(C_descr_, 
                                                  C_rows_dev_, 
                                                  C_cols_dev_,
                                                  C_vals_dev_));

        cusparseCheckError(cusparseSpGEMM_copy(handle_, 
                                               opA_, 
                                               opB_, 
                                               &alpha, 
                                               A_descr_,
                                               B_descr_, 
                                               &beta, 
                                               C_descr_, 
                                               dataType_,
                                               alg_, 
                                               spgemmDescr_));

        // Freeing memory
        cudaCheckError(cudaFree(dBuffer1));
        cudaCheckError(cudaFree(dBuffer2));
        cusparseCheckError(cusparseSpGEMM_destroyDescr(spgemmDescr_));
        cusparseCheckError(cusparseDestroySpMat(A_descr_));
        cusparseCheckError(cusparseDestroySpMat(B_descr_));
        cusparseCheckError(cusparseDestroySpMat(C_descr_));

        C_vals_ = (T*)malloc(sizeof(T) * C_nnz_);
        C_cols_32_ = (int32_t*)malloc(sizeof(int32_t) * C_nnz_);
        C_allocated = true;
        
        cudaCheckError(cudaMemcpyAsync(C_rows_32_, C_rows_dev_, sizeof(int32_t) * (m_ + 1), cudaMemcpyDeviceToHost, s1_));
        cudaCheckError(cudaMemcpyAsync(C_cols_32_, C_cols_dev_, sizeof(int32_t) * C_nnz_, cudaMemcpyDeviceToHost, s2_));
        cudaCheckError(cudaMemcpyAsync(C_vals_, C_vals_dev_, sizeof(T) * C_nnz_, cudaMemcpyDeviceToHost, s3_));
        cudaCheckError(cudaDeviceSynchronize());
        
        cudaCheckError(cudaFree(C_vals_dev_));
        cudaCheckError(cudaFree(C_cols_dev_));
        break;
      }
      case gpuOffloadType::once: {
        if (C_allocated) {
          cudaCheckError(cudaFree(C_vals_dev_));
          cudaCheckError(cudaFree(C_cols_dev_));
          C_allocated = false;
        }
        // Make matrix descriptors
        cusparseCheckError(cusparseCreateCsr(&A_descr_, 
                                             m_, 
                                             k_, 
                                             A_nnz_, 
                                             A_rows_dev_,
                                             A_cols_dev_, 
                                             A_vals_dev_, 
                                             index_, 
                                             index_,
                                             base_, 
                                             dataType_));
        cusparseCheckError(cusparseCreateCsr(&B_descr_, 
                                             k_, 
                                             n_, 
                                             B_nnz_, 
                                             B_rows_dev_,
                                             B_cols_dev_, 
                                             B_vals_dev_, 
                                             index_, 
                                             index_,
                                             base_, 
                                             dataType_));
        cusparseCheckError(cusparseCreateCsr(&C_descr_, 
                                             m_, 
                                             n_, 
                                             0, 
                                             nullptr,
                                             nullptr, 
                                             nullptr, 
                                             index_, 
                                             index_,
                                             base_, 
                                             dataType_));

        cusparseCheckError(cusparseSpGEMM_createDescr(&spgemmDescr_));

        size_t bufferSize1 = 0;
        cusparseCheckError(cusparseSpGEMM_workEstimation(handle_, 
                                                         opA_, 
                                                         opB_, 
                                                         &alpha,
                                                         A_descr_, 
                                                         B_descr_, 
                                                         &beta,
                                                         C_descr_, 
                                                         dataType_, 
                                                         alg_,
                                                         spgemmDescr_, 
                                                         &bufferSize1,
                                                         nullptr));

        void* dBuffer1 = nullptr;
        cudaCheckError(cudaMalloc((void**)&dBuffer1, bufferSize1));

        cusparseCheckError(cusparseSpGEMM_workEstimation(handle_, 
                                                         opA_, 
                                                         opB_, 
                                                         &alpha,
                                                         A_descr_, 
                                                         B_descr_, 
                                                         &beta,
                                                         C_descr_, 
                                                         dataType_, 
                                                         alg_,
                                                         spgemmDescr_, 
                                                         &bufferSize1,
                                                         dBuffer1));

        size_t bufferSize2 = 0;
        cusparseCheckError(cusparseSpGEMM_compute(handle_, 
                                                  opA_, 
                                                  opB_, 
                                                  &alpha, 
                                                  A_descr_,
                                                  B_descr_, 
                                                  &beta, 
                                                  C_descr_, 
                                                  dataType_,
                                                  alg_, 
                                                  spgemmDescr_, 
                                                  &bufferSize2,
                                                  nullptr));

        void* dBuffer2 = nullptr;
        cudaCheckError(cudaMalloc((void**)&dBuffer2, bufferSize2));

        cusparseCheckError(cusparseSpGEMM_compute(handle_, 
                                                  opA_, 
                                                  opB_, 
                                                  &alpha, 
                                                  A_descr_,
                                                  B_descr_, 
                                                  &beta, 
                                                  C_descr_,
                                                  dataType_, 
                                                  alg_, 
                                                  spgemmDescr_,
                                                  &bufferSize2, 
                                                  dBuffer2));

        cusparseCheckError(cusparseSpMatGetSize(C_descr_, 
                                                &C_num_rows_, 
                                                &C_num_cols_,
                                                &C_nnz_));

        cudaCheckError(cudaMalloc(&C_vals_dev_, sizeof(T) * C_nnz_));
        cudaCheckError(cudaMalloc(&C_cols_dev_, sizeof(int32_t) * C_nnz_));
        C_allocated = true;

        cusparseCheckError(cusparseCsrSetPointers(C_descr_, 
                                                  C_rows_dev_, 
                                                  C_cols_dev_,
                                                  C_vals_dev_));

        cusparseCheckError(cusparseSpGEMM_copy(handle_, 
                                               opA_, 
                                               opB_, 
                                               &alpha, 
                                               A_descr_,
                                               B_descr_, 
                                               &beta, 
                                               C_descr_, 
                                               dataType_,
                                               alg_, 
                                               spgemmDescr_));

        // Freeing memory
        cudaCheckError(cudaFree(dBuffer1));
        cudaCheckError(cudaFree(dBuffer2));
        cusparseCheckError(cusparseSpGEMM_destroyDescr(spgemmDescr_));
        cusparseCheckError(cusparseDestroySpMat(A_descr_));
        cusparseCheckError(cusparseDestroySpMat(B_descr_));
        cusparseCheckError(cusparseDestroySpMat(C_descr_));
        break;
      }
      case gpuOffloadType::unified: {
        if (C_allocated) {
          cudaCheckError(cudaFree(C_cols_32_));
          cudaCheckError(cudaFree(C_vals_));
          C_allocated = false;
        }

        // Make matrix descriptors
        cusparseCheckError(cusparseCreateCsr(&A_descr_, 
                                             m_, 
                                             k_, 
                                             A_nnz_, 
                                             A_rows_,
                                             A_cols_, 
                                             A_vals_, 
                                             index_, 
                                             index_,
                                             base_, 
                                             dataType_));
        cusparseCheckError(cusparseCreateCsr(&B_descr_, 
                                             k_, 
                                             n_, 
                                             B_nnz_, 
                                             B_rows_,
                                             B_cols_, 
                                             B_vals_, 
                                             index_, 
                                             index_,
                                             base_, 
                                             dataType_));
        cusparseCheckError(cusparseCreateCsr(&C_descr_, 
                                             m_, 
                                             n_, 
                                             0, 
                                             nullptr,
                                             nullptr, 
                                             nullptr, 
                                             index_, 
                                             index_,
                                             base_, 
                                             dataType_));

        cusparseCheckError(cusparseSpGEMM_createDescr(&spgemmDescr_));

        size_t bufferSize1 = 0;
        cusparseCheckError(cusparseSpGEMM_workEstimation(handle_, 
                                                         opA_, 
                                                         opB_, 
                                                         &alpha,
                                                         A_descr_, 
                                                         B_descr_, 
                                                         &beta,
                                                         C_descr_, 
                                                         dataType_, 
                                                         alg_,
                                                         spgemmDescr_, 
                                                         &bufferSize1,
                                                         nullptr));

        void* dBuffer1 = nullptr;
        cudaCheckError(cudaMalloc((void**)&dBuffer1, bufferSize1));

        cusparseCheckError(cusparseSpGEMM_workEstimation(handle_, 
                                                         opA_, 
                                                         opB_, 
                                                         &alpha,
                                                         A_descr_, 
                                                         B_descr_, 
                                                         &beta,
                                                         C_descr_, 
                                                         dataType_, 
                                                         alg_,
                                                         spgemmDescr_, 
                                                         &bufferSize1,
                                                         dBuffer1));

        size_t bufferSize2 = 0;
        cusparseCheckError(cusparseSpGEMM_compute(handle_, 
                                                  opA_, 
                                                  opB_, 
                                                  &alpha, 
                                                  A_descr_,
                                                  B_descr_, 
                                                  &beta, 
                                                  C_descr_, 
                                                  dataType_,
                                                  alg_, 
                                                  spgemmDescr_, 
                                                  &bufferSize2,
                                                  nullptr));

        void* dBuffer2 = nullptr;
        cudaCheckError(cudaMalloc((void**)&dBuffer2, bufferSize2));

        cusparseCheckError(cusparseSpGEMM_compute(handle_, 
                                                  opA_, 
                                                  opB_, 
                                                  &alpha, 
                                                  A_descr_,
                                                  B_descr_, 
                                                  &beta, 
                                                  C_descr_,
                                                  dataType_, 
                                                  alg_, 
                                                  spgemmDescr_,
                                                  &bufferSize2, 
                                                  dBuffer2));

        cusparseCheckError(cusparseSpMatGetSize(C_descr_, 
                                                &C_num_rows_, 
                                                &C_num_cols_,
                                                &C_nnz_));

        cudaCheckError(cudaMallocManaged(&C_vals_, sizeof(T) * C_nnz_));
        cudaCheckError(cudaMallocManaged(&C_cols_32_, sizeof(int32_t) * C_nnz_));
        C_allocated = true;

        cusparseCheckError(cusparseCsrSetPointers(C_descr_, 
                                                  C_rows_32_, 
                                                  C_cols_32_,
                                                  C_vals_));

        cusparseCheckError(cusparseSpGEMM_copy(handle_, 
                                               opA_, 
                                               opB_, 
                                               &alpha, 
                                               A_descr_,
                                               B_descr_, 
                                               &beta, 
                                               C_descr_, 
                                               dataType_,
                                               alg_, 
                                               spgemmDescr_));

        // Freeing memory
        cudaCheckError(cudaFree(dBuffer1));
        cudaCheckError(cudaFree(dBuffer2));
        cusparseCheckError(cusparseSpGEMM_destroyDescr(spgemmDescr_));
        cusparseCheckError(cusparseDestroySpMat(A_descr_));
        cusparseCheckError(cusparseDestroySpMat(B_descr_));
        cusparseCheckError(cusparseDestroySpMat(C_descr_));
        cudaCheckError(cudaDeviceSynchronize());
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
        C_vals_ = (T*)malloc(sizeof(T) * C_nnz_);
        C_cols_32_ = (int32_t*)malloc(sizeof(int32_t) * C_nnz_);
        
        cudaCheckError(cudaMemcpyAsync(C_rows_32_, C_rows_dev_, sizeof(int32_t) * (m_ + 1), cudaMemcpyDeviceToHost, s1_));
        cudaCheckError(cudaMemcpyAsync(C_cols_32_, C_cols_dev_, sizeof(int32_t) * C_nnz_, cudaMemcpyDeviceToHost, s2_));
        cudaCheckError(cudaMemcpyAsync(C_vals_, C_vals_dev_, sizeof(T) * C_nnz_, cudaMemcpyDeviceToHost, s3_));
        cudaCheckError(cudaDeviceSynchronize());

        cudaCheckError(cudaFree(C_vals_dev_));
        cudaCheckError(cudaFree(C_cols_dev_));
        C_allocated = false;
        break;
      }
      case gpuOffloadType::unified: {
        cudaCheckError(cudaDeviceSynchronize());
        cudaCheckError(safeCudaMemPrefetchAsync(C_vals_, sizeof(T) * C_nnz_, cudaCpuDeviceId, 0));
        cudaCheckError(safeCudaMemPrefetchAsync(C_cols_32_, sizeof(int32_t) * C_nnz_, cudaCpuDeviceId, 0));
        cudaCheckError(safeCudaMemPrefetchAsync(C_rows_32_, sizeof(int32_t) * (m_ + 1), cudaCpuDeviceId, 0));
        break;
      }
    }
  }

  /** Do any necessary cleanup (free pointers, close library handles, etc.)
   * after Kernel has been called. */
  void postCallKernelCleanup() override {
    switch (offload_) {
      case gpuOffloadType::always: {
        if (C_allocated) {
          free(C_vals_);
          free(C_cols_32_);
          C_allocated = false;
        }
        free(A_vals_);
        free(A_cols_);
        free(A_rows_);
        free(B_vals_);
        free(B_cols_);
        free(B_rows_);
        free(C_rows_32_);

        cudaCheckError(cudaFree(A_vals_dev_));
        cudaCheckError(cudaFree(A_cols_dev_));
        cudaCheckError(cudaFree(A_rows_dev_));
        cudaCheckError(cudaFree(B_vals_dev_));
        cudaCheckError(cudaFree(B_cols_dev_));
        cudaCheckError(cudaFree(B_rows_dev_));
        cudaCheckError(cudaFree(C_rows_dev_));
        break;
      }
      case gpuOffloadType::once: {
        free(A_vals_);
        free(A_cols_);
        free(A_rows_);
        free(B_vals_);
        free(B_cols_);
        free(B_rows_);
        free(C_vals_);
        free(C_cols_32_);
        free(C_rows_32_);

        cudaCheckError(cudaFree(A_vals_dev_));
        cudaCheckError(cudaFree(A_cols_dev_));
        cudaCheckError(cudaFree(A_rows_dev_));
        cudaCheckError(cudaFree(B_vals_dev_));
        cudaCheckError(cudaFree(B_cols_dev_));
        cudaCheckError(cudaFree(B_rows_dev_));
        cudaCheckError(cudaFree(C_rows_dev_));
        break;
      }
      case gpuOffloadType::unified: {
        if (C_allocated) {
          cudaCheckError(cudaFree(C_vals_));
          cudaCheckError(cudaFree(C_cols_32_));
          C_allocated = false;
        }
        cudaCheckError(cudaFree(A_vals_));
        cudaCheckError(cudaFree(A_cols_));
        cudaCheckError(cudaFree(A_rows_));
        cudaCheckError(cudaFree(B_vals_));
        cudaCheckError(cudaFree(B_cols_));
        cudaCheckError(cudaFree(B_rows_));
        cudaCheckError(cudaFree(C_rows_32_));
        break;
      }
    }
  }

  bool print_ = false;

  bool alreadyInitialised_ = false;

  /** Handle used when calling cuBLAS. */
  cusparseHandle_t handle_;

  /** CUDA Streams - used to asynchronously move data between host and device. */
  cudaStream_t s1_;
  cudaStream_t s2_;
  cudaStream_t s3_;
  cudaStream_t s4_;
  cudaStream_t s5_;
  cudaStream_t s6_;
  
  /** The ID of the target GPU Device. */
  int gpuDevice_;

	/** CSR format vectors for matrices A, B and C on the host */
	T* A_vals_;
	int32_t* A_cols_;
  int32_t* A_rows_;
  int32_t A_num_rows_;
  int32_t A_num_cols_;

  T* B_vals_;
  int32_t* B_cols_;
  int32_t* B_rows_;
  int32_t B_num_rows_;
  int32_t B_num_cols_;

  int64_t C_num_rows_;
  int64_t C_num_cols_;

  /** CSR format vectors for matrices A, B and C on the device. */
	T* A_vals_dev_;
  T* B_vals_dev_;
  T* C_vals_dev_;
	int32_t* A_cols_dev_;
  int32_t* A_rows_dev_;
  int32_t* B_cols_dev_;
  int32_t* B_rows_dev_;
  int32_t* C_cols_dev_;
  int32_t* C_rows_dev_;

  int32_t* C_cols_32_;
  int32_t* C_rows_32_;

  bool C_allocated = false;

  /** The constant value Alpha. */
  const T alpha = ALPHA;

  /** The constant value Beta. */
  const T beta = BETA;


	// Create descriptors for matrices A->C
	cusparseSpMatDescr_t A_descr_, B_descr_, C_descr_;

	cusparseSpGEMMDescr_t spgemmDescr_;

  cusparseOperation_t opA_;
  cusparseOperation_t opB_;
  cusparseSpGEMMAlg_t alg_;
  cusparseIndexType_t index_;
  cusparseIndexBase_t base_;
	cudaDataType_t dataType_;
};
}  // namespace gpu
#endif