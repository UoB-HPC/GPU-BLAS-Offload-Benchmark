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
  using spmm<T>::A_;
  using spmm<T>::B_;
  using spmm<T>::C_;
  using spmm<T>::offload_;
  using spmm<T>::sparsity_;
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
  void initialise(gpuOffloadType offload, int n, int m, int k, 
                  double sparsity, bool binary = false) override {
    print_ = (n >= 2580);
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

    if (print_) std::cout << "\tAllocating dense matrix arrays" << std::endl;
    A_ = (T*)malloc(sizeof(T) * m_ * k_);
    B_ = (T*)malloc(sizeof(T) * k_ * n_);
    C_ = (T*)calloc(m_ * n_, sizeof(T));
    cusparseCheckError(cusparseCreate(&handle_));

    initInputMatrices();

    print_ = false;
    if (print_) {
      std::cout << "===============Initialised=================" << std::endl;
      std::cout << "___________________________________________" << std::endl;
      std::cout << "A =" << std::endl;
      std::cout << "[";
      for (int64_t i = 0; i < (m_ * k_); i++) {
        std::cout << A_[i];
        if ((i % k_) < (k_ - 1)) std::cout << ", ";
        else if (i != ((m_ * k_) - 1)) std::cout << std::endl;
      }
      std::cout << "]" << std::endl;

      std::cout << "B =" << std::endl;
      std::cout << "[";
      for (int64_t i = 0; i < (k_ * n_); i++) {
        std::cout << B_[i];
        if ((i % n_) < (n_ - 1)) std::cout << ", ";
        else if (i != ((k_ * n_) - 1)) std::cout << std::endl;
      }
      std::cout << "]" << std::endl << std::endl;

      std::cout << "___________________________________________" << std::endl;
      std::cout << "===============Sparsified==================" << std::endl;
      std::cout << "___________________________________________" << std::endl;
      std::cout << "A nnz = " << A_nnz_ << std::endl;
      std::cout << "A rows = [";
      for (int64_t i = 0; i < (m_ + 1); i++) {
        std::cout << A_rows_[i];
        if (i < m_) std::cout << ", ";
      }
      std::cout << "]" << std::endl;
      std::cout << "A cols = [";
      for (int64_t i = 0; i < A_nnz_; i++) {
        std::cout << A_cols_[i];
        if (i < (A_nnz_ - 1)) std::cout << ", ";
      }
      std::cout << "]" << std::endl;
      std::cout << "A vals = [";
      for (int64_t i = 0; i < A_nnz_; i++) {
        std::cout << A_vals_[i];
        if (i < (A_nnz_ - 1)) std::cout << ", ";
      }
      std::cout << "]" << std::endl;
      
      std::cout << "B nnz = " << B_nnz_ << std::endl;
      std::cout << "B rows = [";
      for (int64_t i = 0; i < (k_ + 1); i++) {
        std::cout << B_rows_[i];
        if (i < k_) std::cout << ", ";
      }
      std::cout << "]" << std::endl;
      std::cout << "B cols = [";
      for (int64_t i = 0; i < B_nnz_; i++) {
        std::cout << B_cols_[i];
        if (i < (B_nnz_ - 1)) std::cout << ", ";
      }
      std::cout << "]" << std::endl;
      std::cout << "B vals = [";
      for (int64_t i = 0; i < B_nnz_; i++) {
        std::cout << B_vals_[i];
        if (i < (B_nnz_ - 1)) std::cout << ", ";
      }
      std::cout << "]" << std::endl << std::endl;
      std::cout << "___________________________________________" << std::endl;
    }
    print_ = (n >= 2580);
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
      cudaCheckError(cudaMallocManaged(&C_rows_32_, sizeof(int32_t) * (n_ + 1)));
      C_vals_ = nullptr;
      C_cols_32_ = nullptr;
    } else {
      A_vals_ = (T*)malloc(sizeof(T) * A_nnz_);
      A_cols_ = (int32_t*)malloc(sizeof(int32_t) * A_nnz_);
      A_rows_ = (int32_t*)malloc(sizeof(int32_t) * (m_ + 1));
      B_vals_ = (T*)malloc(sizeof(T) * B_nnz_);
      B_cols_ = (int32_t*)malloc(sizeof(int32_t) * B_nnz_);
      B_rows_ = (int32_t*)malloc(sizeof(int32_t) * (k_ + 1));
      C_rows_32_ = (int32_t*)malloc(sizeof(int32_t) * (n_ + 1));
      C_vals_ = nullptr;
      C_cols_32_ = nullptr;

      cudaCheckError(cudaMalloc((void**)&A_vals_dev_, sizeof(T) * A_nnz_));
      cudaCheckError(cudaMalloc((void**)&A_cols_dev_, sizeof(int32_t) * A_nnz_));
      cudaCheckError(cudaMalloc((void**)&A_rows_dev_, sizeof(int32_t) * (m_ + 1)));
      cudaCheckError(cudaMalloc((void**)&B_vals_dev_, sizeof(T) * B_nnz_));
      cudaCheckError(cudaMalloc((void**)&B_cols_dev_, sizeof(int32_t) * B_nnz_));
      cudaCheckError(cudaMalloc((void**)&B_rows_dev_, sizeof(int32_t) * (k_ + 1)));
      cudaCheckError(cudaMalloc((void**)&C_rows_dev_, sizeof(int32_t) * (n_ + 1)));
      C_vals_dev_ = nullptr;
      C_cols_dev_ = nullptr;
    }
    cudaCheckError(cudaDeviceSynchronize());
    // Load A into CSR
    // Load A into CSR
    int32_t nnz_encountered = 0;
    for (int32_t row = 0; row < m_; row++) {
      A_rows_[row] = nnz_encountered;
      for (int32_t col = 0; col < k_; col++) {
        if (A_[(row * k_) + col] != 0.0) { // fixed here
          A_cols_[nnz_encountered] = col;
          A_vals_[nnz_encountered] = A_[(row * k_) + col];
          nnz_encountered++;
        }
      }
    }
    A_rows_[m_] = nnz_encountered;

    // Load B into CSR
    nnz_encountered = 0;
    for (int32_t row = 0; row < k_; row++) {
      B_rows_[row] = nnz_encountered;
      for (int32_t col = 0; col < n_; col++) {
        if (B_[(row * n_) + col] != 0.0) {
          B_cols_[nnz_encountered] = col;
          B_vals_[nnz_encountered] = B_[(row * n_) + col];
          nnz_encountered++;
        }
      }
    }
    B_rows_[k_] = nnz_encountered;
  }

 private:
  /** Perform any required steps before calling the GEMM kernel that should
   * be timed. */
  void preLoopRequirements() override {
    if (print_) std::cout << "Pre-loop stuff" << std::endl;
    switch(offload_) {
      case gpuOffloadType::always: {
        break;
      }
      case gpuOffloadType::once: {
        if (print_) std::cout << "\tCopying data to GPU" << std::endl;
        cudaCheckError(cudaMemcpy(A_vals_dev_, A_vals_, sizeof(T) * A_nnz_, cudaMemcpyHostToDevice));
        cudaCheckError(cudaMemcpy(A_cols_dev_, A_cols_, sizeof(int32_t) * A_nnz_, cudaMemcpyHostToDevice));
        cudaCheckError(cudaMemcpy(A_rows_dev_, A_rows_, sizeof(int32_t) * (m_ + 1), cudaMemcpyHostToDevice));
        cudaCheckError(cudaMemcpy(B_vals_dev_, B_vals_, sizeof(T) * B_nnz_, cudaMemcpyHostToDevice));
        cudaCheckError(cudaMemcpy(B_cols_dev_, B_cols_, sizeof(int32_t) * B_nnz_, cudaMemcpyHostToDevice));
        cudaCheckError(cudaMemcpy(B_rows_dev_, B_rows_, sizeof(int32_t) * (k_ + 1), cudaMemcpyHostToDevice));
        cudaCheckError(cudaMemcpy(C_rows_dev_, C_rows_32_, sizeof(int32_t) * (n_ + 1), cudaMemcpyHostToDevice));
        cudaCheckError(cudaDeviceSynchronize());
        break;
      }
      case gpuOffloadType::unified: {
        if (print_) std::cout << "\tPrefetching data to GPU" << std::endl;
        // Prefetch memory to device
        cudaCheckError(cudaMemPrefetchAsync(A_vals_, sizeof(T) * A_nnz_, gpuDevice_, s1_));
        cudaCheckError(cudaMemPrefetchAsync(A_cols_, sizeof(int32_t) * A_nnz_, gpuDevice_, s1_));
        cudaCheckError(cudaMemPrefetchAsync(A_rows_, sizeof(int32_t) * (m_ + 1), gpuDevice_, s1_));
        cudaCheckError(cudaMemPrefetchAsync(B_vals_, sizeof(T) * B_nnz_, gpuDevice_, s2_));
        cudaCheckError(cudaMemPrefetchAsync(B_cols_, sizeof(int32_t) * B_nnz_, gpuDevice_, s2_));
        cudaCheckError(cudaMemPrefetchAsync(B_rows_, sizeof(int32_t) * (k_ + 1), gpuDevice_, s2_));
        cudaCheckError(cudaMemPrefetchAsync(C_rows_32_, sizeof(int32_t) * (m_ + 1), gpuDevice_, s3_));
        cudaCheckError(cudaDeviceSynchronize());
        break;
      }
    }
  }

  /** Make a call to the BLAS Library Kernel. */
  void callSpmm() override {
    if (print_) std::cout << "callSpmm" << std::endl; 
    switch(offload_) {
      case gpuOffloadType::always: {
        if (C_allocated) {
          if (print_) std::cout << "\tFreeing vals and cols for C" << std::endl;
          free(C_vals_);
          free(C_cols_32_);
          C_allocated = false;
        }

        if (print_) std::cout << "\tCopying data to GPU" << std::endl;
        cudaCheckError(cudaMemcpy(A_vals_dev_, A_vals_, sizeof(T) * A_nnz_, cudaMemcpyHostToDevice));
        cudaCheckError(cudaMemcpy(A_cols_dev_, A_cols_, sizeof(int32_t) * A_nnz_, cudaMemcpyHostToDevice));
        cudaCheckError(cudaMemcpy(A_rows_dev_, A_rows_, sizeof(int32_t) * (m_ + 1), cudaMemcpyHostToDevice));
        cudaCheckError(cudaMemcpy(B_vals_dev_, B_vals_, sizeof(T) * B_nnz_, cudaMemcpyHostToDevice));
        cudaCheckError(cudaMemcpy(B_cols_dev_, B_cols_, sizeof(int32_t) * B_nnz_, cudaMemcpyHostToDevice));
        cudaCheckError(cudaMemcpy(B_rows_dev_, B_rows_, sizeof(int32_t) * (k_ + 1), cudaMemcpyHostToDevice));
        cudaCheckError(cudaMemcpy(C_rows_dev_, C_rows_32_, sizeof(int32_t) * (n_ + 1), cudaMemcpyHostToDevice));
        cudaCheckError(cudaDeviceSynchronize());

        // Make matrix descriptors
        if (print_) std::cout << "\tMaking descriptors for sparce matrices" << std::endl;
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
                                             C_rows_dev_,
                                             nullptr, 
                                             nullptr, 
                                             index_, 
                                             index_,
                                             base_, 
                                             dataType_));

        if (print_) std::cout << "\tMaking GEMM descriptor" << std::endl;
        cusparseCheckError(cusparseSpGEMM_createDescr(&spgemmDescr_));

        size_t bufferSize1 = 0;
        if (print_) std::cout << "\tWork estimation -- getting buffer size" << std::endl;
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

        if (print_) std::cout << "\tAllocating buffer 1" << std::endl;
        void* dBuffer1 = nullptr;
        cudaCheckError(cudaMalloc((void**)&dBuffer1, bufferSize1));

        if (print_) std::cout << "\tWork estimation -- using buffer" << std::endl;
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
        if (print_) std::cout << "\tCompute -- getting buffer size" << std::endl;
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

        if (print_) std::cout << "\tAllocating buffer 2" << std::endl;
        void* dBuffer2 = nullptr;
        cudaCheckError(cudaMalloc((void**)&dBuffer2, bufferSize2));

        if (print_) std::cout << "\tCompute -- actual" << std::endl; 
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

        if (print_) std::cout << "\tGetting C sizes" << std::endl;
        cusparseCheckError(cusparseSpMatGetSize(C_descr_, 
                                                &C_num_rows_, 
                                                &C_num_cols_,
                                                &C_nnz_));

        if (print_) std::cout << "\tAllocating C device arrays" << std::endl;
        cudaCheckError(cudaMalloc(&C_vals_dev_, sizeof(T) * C_nnz_));
        cudaCheckError(cudaMalloc(&C_cols_dev_, sizeof(int32_t) * C_nnz_));

        if (print_) std::cout << "\tSetting C pointers" << std::endl;
        cusparseCheckError(cusparseCsrSetPointers(C_descr_, 
                                                  C_rows_dev_, 
                                                  C_cols_dev_,
                                                  C_vals_dev_));

        if (print_) std::cout << "\tCopying C" << std::endl;
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
        if (print_) std::cout << "\tFreeing buffers" << std::endl;
        cudaCheckError(cudaFree(dBuffer1));
        cudaCheckError(cudaFree(dBuffer2));
        cusparseCheckError(cusparseSpGEMM_destroyDescr(spgemmDescr_));
        cusparseCheckError(cusparseDestroySpMat(A_descr_));
        cusparseCheckError(cusparseDestroySpMat(B_descr_));
        cusparseCheckError(cusparseDestroySpMat(C_descr_));

        if (print_) std::cout << "\tAllocating host C arrays" << std::endl;
        C_vals_ = (T*)malloc(sizeof(T) * C_nnz_);
        C_cols_32_ = (int32_t*)malloc(sizeof(int32_t) * C_nnz_);
        C_allocated = true;

        if (print_) std::cout << "\tCopying results back to the CPU" << std::endl;
        cudaCheckError(cudaMemcpy(C_rows_32_, C_rows_dev_, sizeof(int32_t) * (n_ + 1), cudaMemcpyDeviceToHost));
        cudaCheckError(cudaMemcpy(C_cols_32_, C_cols_dev_, sizeof(int32_t) * C_nnz_, cudaMemcpyDeviceToHost));
        cudaCheckError(cudaMemcpy(C_vals_, C_vals_dev_, sizeof(T) * C_nnz_, cudaMemcpyDeviceToHost));
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
        if (print_) std::cout << "\tMaking descriptors for sparce matrices" << std::endl;
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
                                             C_rows_dev_,
                                             nullptr, 
                                             nullptr, 
                                             index_, 
                                             index_,
                                             base_, 
                                             dataType_));

        if (print_) std::cout << "\tMaking GEMM descriptor" << std::endl;
        cusparseCheckError(cusparseSpGEMM_createDescr(&spgemmDescr_));

        size_t bufferSize1 = 0;
        if (print_) std::cout << "\tWork estimation -- getting buffer size" << std::endl;
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

        if (print_) std::cout << "\tAllocating buffer 1" << std::endl;
        void* dBuffer1 = nullptr;
        cudaCheckError(cudaMalloc((void**)&dBuffer1, bufferSize1));

        if (print_) std::cout << "\tWork estimation -- using buffer" << std::endl;
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
        if (print_) std::cout << "\tCompute -- getting buffer size" << std::endl;
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

        if (print_) std::cout << "\tAllocating buffer 2" << std::endl;
        void* dBuffer2 = nullptr;
        cudaCheckError(cudaMalloc((void**)&dBuffer2, bufferSize2));

        if (print_) std::cout << "\tCompute -- actual" << std::endl; 
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

        if (print_) std::cout << "\tGetting C sizes" << std::endl;
        cusparseCheckError(cusparseSpMatGetSize(C_descr_, 
                                                &C_num_rows_, 
                                                &C_num_cols_,
                                                &C_nnz_));

        if (print_) std::cout << "\tAllocating C device arrays" << std::endl;
        cudaCheckError(cudaMalloc(&C_vals_dev_, sizeof(T) * C_nnz_));
        cudaCheckError(cudaMalloc(&C_cols_dev_, sizeof(int32_t) * C_nnz_));
        C_allocated = true;

        if (print_) std::cout << "\tSetting C pointers" << std::endl;
        cusparseCheckError(cusparseCsrSetPointers(C_descr_, 
                                                  C_rows_dev_, 
                                                  C_cols_dev_,
                                                  C_vals_dev_));

        if (print_) std::cout << "\tCopying C" << std::endl;
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
        if (print_) std::cout << "\tFreeing buffers" << std::endl;
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
        if (print_) std::cout << "\tMaking descriptors for sparce matrices" << std::endl;
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
                                             C_rows_32_,
                                             nullptr, 
                                             nullptr, 
                                             index_, 
                                             index_,
                                             base_, 
                                             dataType_));

        if (print_) std::cout << "\tMaking GEMM descriptor" << std::endl;
        cusparseCheckError(cusparseSpGEMM_createDescr(&spgemmDescr_));

        size_t bufferSize1 = 0;
        if (print_) std::cout << "\tWork estimation -- getting buffer size" << std::endl;
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

        if (print_) std::cout << "\tAllocating buffer 1" << std::endl;
        void* dBuffer1 = nullptr;
        cudaCheckError(cudaMalloc((void**)&dBuffer1, bufferSize1));

        if (print_) std::cout << "\tWork estimation -- using buffer" << std::endl;
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
        if (print_) std::cout << "\tCompute -- getting buffer size" << std::endl;
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

        if (print_) std::cout << "\tAllocating buffer 2" << std::endl;
        void* dBuffer2 = nullptr;
        cudaCheckError(cudaMalloc((void**)&dBuffer2, bufferSize2));

        if (print_) std::cout << "\tCompute -- actual" << std::endl; 
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

        if (print_) std::cout << "\tGetting C sizes" << std::endl;
        cusparseCheckError(cusparseSpMatGetSize(C_descr_, 
                                                &C_num_rows_, 
                                                &C_num_cols_,
                                                &C_nnz_));

        if (print_) std::cout << "\tAllocating C device arrays, nnz = " << C_nnz_ << std::endl;
        cudaCheckError(cudaMallocManaged(&C_vals_, sizeof(T) * C_nnz_));
        cudaCheckError(cudaMallocManaged(&C_cols_32_, sizeof(int32_t) * C_nnz_));
        C_allocated = true;

        if (print_) std::cout << "\tSetting C pointers" << std::endl;
        cusparseCheckError(cusparseCsrSetPointers(C_descr_, 
                                                  C_rows_32_, 
                                                  C_cols_32_,
                                                  C_vals_));

        if (print_) std::cout << "\tCopying C" << std::endl;
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
        if (print_) std::cout << "\tFreeing buffers" << std::endl;
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
    if (print_) std::cout << "Post-loop stuff" << std::endl;
    switch(offload_) {
      case gpuOffloadType::always: {
        break;
      }
      case gpuOffloadType::once: {
        if (print_) std::cout << "\tAllocating host C arrays" << std::endl;
        C_vals_ = (T*)malloc(sizeof(T) * C_nnz_);
        C_cols_32_ = (int32_t*)malloc(sizeof(int32_t) * C_nnz_);
        
        if (print_) std::cout << "\tCopying results back to the CPU" << std::endl;
        cudaCheckError(cudaMemcpy(C_rows_32_, C_rows_dev_, sizeof(int32_t) * (n_ + 1), cudaMemcpyDeviceToHost));
        cudaCheckError(cudaMemcpy(C_cols_32_, C_cols_dev_, sizeof(int32_t) * C_nnz_, cudaMemcpyDeviceToHost));
        cudaCheckError(cudaMemcpy(C_vals_, C_vals_dev_, sizeof(T) * C_nnz_, cudaMemcpyDeviceToHost));
        cudaCheckError(cudaDeviceSynchronize());

        if (print_) std::cout << "\tFreeing device C arrays" << std::endl;
        cudaCheckError(cudaFree(C_vals_dev_));
        cudaCheckError(cudaFree(C_cols_dev_));
        C_allocated = false;
        break;
      }
      case gpuOffloadType::unified: {
        cudaCheckError(cudaDeviceSynchronize());
        if (print_) std::cout << "\tPrefetching results back to CPU" << std::endl;
        cudaCheckError(safeCudaMemPrefetchAsync(C_vals_, sizeof(T) * C_nnz_, cudaCpuDeviceId, 0));
        cudaCheckError(safeCudaMemPrefetchAsync(C_cols_32_, sizeof(int32_t) * C_nnz_, cudaCpuDeviceId, 0));
        cudaCheckError(safeCudaMemPrefetchAsync(C_rows_32_, sizeof(int32_t) * (n_ + 1), cudaCpuDeviceId, 0));
        // Ensure device has finished all work.
        cudaCheckError(cudaDeviceSynchronize());
        if (print_) std::cout << "\tSuccessfully prefetched" << std::endl;
        break;
      }
    }
  }

  /** Do any necessary cleanup (free pointers, close library handles, etc.)
   * after Kernel has been called. */
  void postCallKernelCleanup() override {
    if (print_) std::cout << "Post-kernel cleanup" << std::endl;
    switch (offload_) {
      case gpuOffloadType::always: {
        if (print_) std::cout << "\tFreeing temp C arrays" << std::endl;
        if (C_allocated) {
          free(C_vals_);
          free(C_cols_32_);
          C_allocated = false;
        }
        if (print_) std::cout << "\tFree main arrays" << std::endl;
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
        if (print_) std::cout << "\tFree main arrays" << std::endl;
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
          if (print_) std::cout << "\tFree temp C arrays" << std::endl;
          cudaCheckError(cudaFree(C_vals_));
          cudaCheckError(cudaFree(C_cols_32_));
          C_allocated = false;
        }
        if (print_) std::cout << "\tFree perm C arrays" << std::endl;
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
    if (print_) std::cout << "\tFreeing handle and streams" << std::endl;
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
  void printDenseMatrix(T* M, int32_t rows, int32_t cols) {
    for (int32_t row = 0; row < rows; row++) {
      std::cout << "| ";
      for (int32_t col = 0; col < cols; col++) {
        std::cout << M[(row * cols) + col] << " | ";
      }
      std::cout << std::endl;
    }
  }

  void printCSR(T* values, int32_t* col_indices, int32_t* row_pointers, int32_t nnz,
                int32_t rows, int32_t cols) {
    std::cout << "\tRow pointers__" << std::endl;
    for (int32_t p = 0; p < (rows + 1); p++) {
      std::cout << row_pointers[p] << ", ";
    }
    std::cout << std::endl << "\tColumn Indices__" << std::endl;
    for (int32_t i = 0; i < nnz; i++) {
      std::cout << col_indices[i] << ", ";
    }
    std::cout << std::endl << "\tValues__" << std::endl;
    for (int32_t v = 0; v < nnz; v++) {
      std::cout << values[v] << ", ";
    }
    std::cout << std::endl;
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