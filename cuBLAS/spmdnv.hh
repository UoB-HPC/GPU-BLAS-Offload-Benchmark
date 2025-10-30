#pragma once

#ifdef GPU_CUBLAS
#include <cusparse_v2.h>
#include <cuda_runtime_api.h>
#include <type_traits>
#include <random>
#include <iostream>

#include "../include/kernels/GPU/spmdnv.hh"
#include "../include/utilities.hh"
#include "common.hh"

namespace gpu {
/** A class for SpMDnV GPU BLAS kernels. */
template <typename T>
class spmdnv_gpu : public spmdnv<T> {
 public:
  using spmdnv<T>::spmdnv;
  using spmdnv<T>::initInputMatrixVector;
  using spmdnv<T>::nnz_;
  using spmdnv<T>::m_;
  using spmdnv<T>::n_;
  using spmdnv<T>::x_;
  using spmdnv<T>::y_;
  using spmdnv<T>::offload_;
  using spmdnv<T>::sparsity_;
  using spmdnv<T>::type_;

  ~spmdnv_gpu() {
    if (initialised_) {
      cusparseCheckError(cusparseDestroy(handle_));

      cudaCheckError(cudaStreamDestroy(s1_));
      cudaCheckError(cudaStreamDestroy(s2_));
      cudaCheckError(cudaStreamDestroy(s3_));
      cudaCheckError(cudaStreamDestroy(s4_));
      cudaCheckError(cudaStreamDestroy(s5_));
      
      initialised_ = false;
    }
  }

  void initialise(gpuOffloadType offload, int m, int n, 
                  double sparsity, matrixType type) override {
    if (!initialised_) {
      initialised_ = true;
      cusparseCheckError(cusparseCreate(&handle_));
      
      cudaCheckError(cudaStreamCreate(&s1_));
      cudaCheckError(cudaStreamCreate(&s2_));
      cudaCheckError(cudaStreamCreate(&s3_));
      cudaCheckError(cudaStreamCreate(&s4_));
      cudaCheckError(cudaStreamCreate(&s5_));

      cusparseCheckError(cusparseSetStream(handle_, s1_));

      // Get device identifier
      cudaCheckError(cudaGetDevice(&gpuDevice_));
    }

    offload_ = offload;
    sparsity_ = sparsity;
    type_ = type;


    // Setting cusparse metadata
    if (std::is_same_v<T, float>) {
      dataType_ = CUDA_R_32F;
    } else if (std::is_same_v<T, double>) {
      dataType_ = CUDA_R_64F;
    } else {
      std::cerr << "INVALID DATA TYPE PASSED TO cuSPARSE" << std::endl;
      exit(1);
    }
    opA_ = opB_ = CUSPARSE_OPERATION_NON_TRANSPOSE;
    alg_ = CUSPARSE_SPMV_ALG_DEFAULT;
    index_ = CUSPARSE_INDEX_64I;
    base_ = CUSPARSE_INDEX_BASE_ZERO;

    m_ = m;
    n_ = n;
    nnz_ = 1 + (uint64_t)((double)m_ * (double)n_ * (1.0 - sparsity_));

    // Allocate dense data structures
    if (offload_ == gpuOffloadType::unified) {
      cudaCheckError(cudaMallocManaged(&x_, n_ * sizeof(T)));
      cudaCheckError(cudaMallocManaged(&y_, m_ * sizeof(T)));
      cudaCheckError(cudaDeviceSynchronize());
    } else {
      x_ = (T*)malloc(n_ * sizeof(T));
      y_ = (T*)malloc(m_ * sizeof(T));

      cudaCheckError(cudaMalloc((void**)&x_dev_, n_ * sizeof(T)));
      cudaCheckError(cudaMalloc((void**)&y_dev_, m_ * sizeof(T)));
      cudaCheckError(cudaDeviceSynchronize());
    }

    initInputMatrixVector();
  }

protected:

  void toSparseFormat() override {
    if (offload_ == gpuOffloadType::always) {
      A_vals_store_ = (T*)malloc(sizeof(T) * nnz_);
      A_cols_store_ = (int64_t*)malloc(sizeof(int64_t) * nnz_);
      A_rows_store_ = (int64_t*)malloc(sizeof(int64_t) * (m_ + 1));

      if (type_ == matrixType::random) {
        randomCSR<T, int64_t>(A_vals_store_, A_cols_store_, A_rows_store_, m_, n_, nnz_);
      } else if (type_ == matrixType::rmat) {
        rMatCSR<T, int64_t>(A_vals_store_, A_cols_store_, A_rows_store_, m_, n_, nnz_);
      } else if (type_ == matrixType::finiteElements) {
        finiteElementCSR<T, int64_t>(A_vals_store_, A_cols_store_, A_rows_store_, m_, n_, nnz_);
      } else {
        std::cerr << "Matrix type not supported" << std::endl;
        exit(1);
      }
    }


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

    memcpy(A_vals_, A_vals_store_, sizeof(T) * nnz_);
    memcpy(A_cols_, A_cols_store_, sizeof(int64_t) * nnz_);
    memcpy(A_rows_, A_rows_store_, sizeof(int64_t) * (m_ + 1));
    cudaCheckError(cudaDeviceSynchronize());
  }

 private:
  void preLoopRequirements() override {
    switch(offload_) {
      case gpuOffloadType::always: {
        break;
      }
      case gpuOffloadType::once: {
        cudaCheckError(cudaMemcpyAsync(A_vals_dev_, A_vals_, nnz_ * sizeof(T), cudaMemcpyHostToDevice, s1_));
        cudaCheckError(cudaMemcpyAsync(A_cols_dev_, A_cols_, nnz_ * sizeof(int64_t), cudaMemcpyHostToDevice, s2_));
        cudaCheckError(cudaMemcpyAsync(A_rows_dev_, A_rows_, (m_ + 1) * sizeof(int64_t), cudaMemcpyHostToDevice, s3_));
        cudaCheckError(cudaMemcpyAsync(x_dev_, x_, n_ * sizeof(T), cudaMemcpyHostToDevice, s4_));
        cudaCheckError(cudaMemcpyAsync(y_dev_, y_, m_ * sizeof(T), cudaMemcpyHostToDevice, s5_));
        cudaCheckError(cudaDeviceSynchronize());
        break;
      }
      case gpuOffloadType::unified: {
        // Prefetch memory to device
        cudaCheckError(cudaMemPrefetchAsync(A_vals_, nnz_ * sizeof(T), gpuDevice_, s1_));
        cudaCheckError(cudaMemPrefetchAsync(A_cols_, nnz_ * sizeof(int64_t), gpuDevice_, s2_));
        cudaCheckError(cudaMemPrefetchAsync(A_rows_, (m_ + 1) * sizeof(int64_t), gpuDevice_, s3_));
        cudaCheckError(cudaMemPrefetchAsync(x_, n_ * sizeof(T), gpuDevice_, s4_));
        cudaCheckError(cudaMemPrefetchAsync(y_, m_ * sizeof(T), gpuDevice_, s5_));
        cudaCheckError(cudaDeviceSynchronize());
        break;
      }
    }
  }

  /** Make a call to the BLAS Library Kernel. */
  void callSpMDnV() override {
    switch(offload_) {
      case gpuOffloadType::always: {
        cudaCheckError(cudaMemcpyAsync(A_vals_dev_, A_vals_, nnz_ * sizeof(T), cudaMemcpyHostToDevice, s1_));
        cudaCheckError(cudaMemcpyAsync(A_cols_dev_, A_cols_, nnz_ * sizeof(int64_t), cudaMemcpyHostToDevice, s2_));
        cudaCheckError(cudaMemcpyAsync(A_rows_dev_, A_rows_, (m_ + 1) * sizeof(int64_t), cudaMemcpyHostToDevice, s3_));
        cudaCheckError(cudaMemcpyAsync(x_dev_, x_, n_ * sizeof(T), cudaMemcpyHostToDevice, s4_));
        cudaCheckError(cudaMemcpyAsync(y_dev_, y_, m_ * sizeof(T), cudaMemcpyHostToDevice, s5_));

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

        if (bufferSize > 0) cudaCheckError(cudaMalloc(&dBuffer, bufferSize));
        cudaCheckError(cudaDeviceSynchronize());

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

        cusparseCheckError(cusparseDestroySpMat(A_descr_));
        cusparseCheckError(cusparseDestroyDnVec(x_descr_));
        cusparseCheckError(cusparseDestroyDnVec(y_descr_));

        cudaCheckError(cudaDeviceSynchronize());
        if (dBuffer != nullptr) cudaCheckError(cudaFree(dBuffer));

        cudaCheckError(cudaMemcpyAsync(y_, y_dev_, m_ * sizeof(T), cudaMemcpyDeviceToHost, s2_));
        cudaCheckError(cudaDeviceSynchronize());
        break;
      }
      case gpuOffloadType::once: {
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

        if (bufferSize > 0) cudaCheckError(cudaMalloc(&dBuffer, bufferSize));
        cudaCheckError(cudaDeviceSynchronize());

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

        cusparseCheckError(cusparseDestroySpMat(A_descr_));
        cusparseCheckError(cusparseDestroyDnVec(x_descr_));
        cusparseCheckError(cusparseDestroyDnVec(y_descr_));
        cudaCheckError(cudaDeviceSynchronize());
        if (dBuffer != nullptr) cudaCheckError(cudaFree(dBuffer));
        cudaCheckError(cudaDeviceSynchronize());
        break;
      }
      case gpuOffloadType::unified: {
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

        // TODO -- cusparseSpMV_preprocess()

        if (bufferSize > 0) cudaCheckError(cudaMalloc(&dBuffer, bufferSize));
        cudaCheckError(cudaDeviceSynchronize());

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

  /** Perform any required steps after calling the SpMDnV kernel that should
   * be timed. */
  void postLoopRequirements() override {
    switch(offload_) {
      case gpuOffloadType::always: {
        break;
      }
      case gpuOffloadType::once: {
        cudaCheckError(cudaMemcpyAsync(y_, y_dev_, sizeof(T) * m_, cudaMemcpyDeviceToHost, s3_));
        break;
      }
      case gpuOffloadType::unified: {
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
      free(A_vals_store_);
      free(A_cols_store_);
      free(A_rows_store_);
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
  }

  bool initialised_ = false;

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
  cudaStream_t s4_;
  cudaStream_t s5_;

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
  /** CSR format vectors for storage of matrix between offload type runs */
  T* A_vals_store_;
  int64_t* A_cols_store_;
  int64_t* A_rows_store_;

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