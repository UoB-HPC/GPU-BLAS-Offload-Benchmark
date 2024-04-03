#pragma once

#ifdef GPU_CUBLAS
#include <cublas_v2.h>
#include <cuda_runtime.h>

#include "../include/kernels/GPU/gemv.hh"
#include "../include/utilities.hh"
#include "common.hh"

namespace gpu {
/** A class for GEMV GPU BLAS kernels. */
template <typename T>
class gemv_gpu : public gemv<T> {
 public:
  using gemv<T>::gemv;
  using gemv<T>::initInputMatrixVector;
  using gemv<T>::m_;
  using gemv<T>::n_;
  using gemv<T>::A_;
  using gemv<T>::x_;
  using gemv<T>::y_;
  using gemv<T>::offload_;
  using gemv<T>::vecIncrement_;

  ~gemv_gpu() {
    if (alreadyInitialised_) {
      // Destroy the handle
      cublasDestroy(handle_);

      // Destroy streams after use
      cudaCheckError(cudaStreamDestroy(s1_));
      cudaCheckError(cudaStreamDestroy(s2_));
      cudaCheckError(cudaStreamDestroy(s3_));
    }
  }

  /** Initialise the required data structures.
   * `offload` refers to the data offload type:
   *  - Once:    Move data from host to device before all iterations & move from
   *             device to host after all iterations
   *  - Always:  Move data from host to device and device to host each iteration
   *  - Unified: Initialise data as unified memory; no data movement semantics
   *             required */
  void initialise(gpuOffloadType offload, int m, int n) override {
    if (!alreadyInitialised_) {
      alreadyInitialised_ = true;
      // Perform set-up which doesn't need to happen every problem size change.
      // Create a handle for CUBLAS
      cublasStatus_t status = cublasCreate(&handle_);
      if (status != CUBLAS_STATUS_SUCCESS) {
        std::cout << "Failed to make cublas handle: " << status << std::endl;
        exit(1);
      }

      // Get device identifier
      cudaCheckError(cudaGetDevice(&gpuDevice_));

      // Initialise 3 streams to asynchronously move data between host and
      // device
      cudaCheckError(cudaStreamCreate(&s1_));
      cudaCheckError(cudaStreamCreate(&s2_));
      cudaCheckError(cudaStreamCreate(&s3_));
    }

    offload_ = offload;
    m_ = m;
    n_ = n;

    if (offload_ == gpuOffloadType::unified) {
      cudaCheckError(cudaMallocManaged(&A_, sizeof(T) * m_ * n_));
      cudaCheckError(cudaMallocManaged(&x_, sizeof(T) * n_));
      cudaCheckError(cudaMallocManaged(&y_, sizeof(T) * m_));
    } else {
      // Allocate matrices on host
      cudaCheckError(cudaMallocHost((void**)&A_, sizeof(T) * m_ * n_));
      cudaCheckError(cudaMallocHost((void**)&x_, sizeof(T) * n_));
      cudaCheckError(cudaMallocHost((void**)&y_, sizeof(T) * m_));
      // Allocate matrices on device
      cudaCheckError(cudaMalloc((void**)&A_device_, sizeof(T) * m_ * n_));
      cudaCheckError(cudaMalloc((void**)&x_device_, sizeof(T) * n_));
      cudaCheckError(cudaMalloc((void**)&y_device_, sizeof(T) * m_));
    }

    // Initialise the host input matrix and vector (A_ and x_)
    initInputMatrixVector();
  }

 private:
  /** Perform any required steps before calling the GEMV kernel that should
   * be timed. */
  void preLoopRequirements() override {
    switch (offload_) {
      case gpuOffloadType::always: {
        // Offload data each iteration - no requirements
        break;
      }
      case gpuOffloadType::once: {
        // Offload input data from host to the device.
        cudaCheckError(cudaMemcpyAsync(A_device_, A_, sizeof(T) * m_ * n_,
                                       cudaMemcpyHostToDevice, s1_));
        cudaCheckError(cudaMemcpyAsync(x_device_, x_, sizeof(T) * n_,
                                       cudaMemcpyHostToDevice, s2_));
        cudaCheckError(cudaMemcpyAsync(y_device_, y_, sizeof(T) * m_,
                                       cudaMemcpyHostToDevice, s3_));
        break;
      }
      case gpuOffloadType::unified: {
        // Prefetch input data to device
        cudaCheckError(
            cudaMemPrefetchAsync(A_, sizeof(T) * m_ * n_, gpuDevice_, s1_));
        cudaCheckError(
            cudaMemPrefetchAsync(x_, sizeof(T) * n_, gpuDevice_, s2_));
        cudaCheckError(
            cudaMemPrefetchAsync(y_, sizeof(T) * m_, gpuDevice_, s3_));
        break;
      }
    }
  }

  /** Make a call to the BLAS Library Kernel. */
  void callGemv() override {
    switch (offload_) {
      case gpuOffloadType::always: {
        // Offload input data from host to the device.
        cudaCheckError(cudaMemcpyAsync(A_device_, A_, sizeof(T) * m_ * n_,
                                       cudaMemcpyHostToDevice, s1_));
        cudaCheckError(cudaMemcpyAsync(x_device_, x_, sizeof(T) * n_,
                                       cudaMemcpyHostToDevice, s2_));
        cudaCheckError(cudaMemcpyAsync(y_device_, y_, sizeof(T) * m_,
                                       cudaMemcpyHostToDevice, s3_));
        // Call cuBLAS GEMV kernel
        if constexpr (std::is_same_v<T, float>) {
          cublasStatus_t stat = cublasSgemv(
              handle_, CUBLAS_OP_N, m_, n_, &alpha, A_device_, std::max(1, m_),
              x_device_, vecIncrement_, &beta, y_device_, vecIncrement_);
          if (stat != CUBLAS_STATUS_SUCCESS) {
            std::cout << "cuBLAS error:" << stat << std::endl;
            exit(1);
          }
        } else if constexpr (std::is_same_v<T, double>) {
          cublasStatus_t stat = cublasDgemv(
              handle_, CUBLAS_OP_N, m_, n_, &alpha, A_device_, std::max(1, m_),
              x_device_, vecIncrement_, &beta, y_device_, vecIncrement_);
          if (stat != CUBLAS_STATUS_SUCCESS) {
            std::cout << "cuBLAS error:" << stat << std::endl;
            exit(1);
          }
        }
        // Offload output data from device to host
        cudaCheckError(cudaMemcpyAsync(y_, y_device_, sizeof(T) * m_,
                                       cudaMemcpyDeviceToHost, s3_));
        // Ensure device has finished all work.
        cudaCheckError(cudaDeviceSynchronize());
        break;
      }
      case gpuOffloadType::once: {
        // Call cuBLAS GEMV kernel
        if constexpr (std::is_same_v<T, float>) {
          cublasStatus_t stat = cublasSgemv(
              handle_, CUBLAS_OP_N, m_, n_, &alpha, A_device_, std::max(1, m_),
              x_device_, vecIncrement_, &beta, y_device_, vecIncrement_);
          if (stat != CUBLAS_STATUS_SUCCESS) {
            std::cout << "cuBLAS error:" << stat << std::endl;
            exit(1);
          }
        } else if constexpr (std::is_same_v<T, double>) {
          cublasStatus_t stat = cublasDgemv(
              handle_, CUBLAS_OP_N, m_, n_, &alpha, A_device_, std::max(1, m_),
              x_device_, vecIncrement_, &beta, y_device_, vecIncrement_);
          if (stat != CUBLAS_STATUS_SUCCESS) {
            std::cout << "cuBLAS error:" << stat << std::endl;
            exit(1);
          }
        }
        break;
      }
      case gpuOffloadType::unified: {
        // Call cuBLAS GEMV kernel
        if constexpr (std::is_same_v<T, float>) {
          cublasStatus_t stat = cublasSgemv(
              handle_, CUBLAS_OP_N, m_, n_, &alpha, A_, std::max(1, m_), x_,
              vecIncrement_, &beta, y_, vecIncrement_);
          if (stat != CUBLAS_STATUS_SUCCESS) {
            std::cout << "cuBLAS error:" << stat << std::endl;
            exit(1);
          }
        } else if constexpr (std::is_same_v<T, double>) {
          cublasStatus_t stat = cublasDgemv(
              handle_, CUBLAS_OP_N, m_, n_, &alpha, A_, std::max(1, m_), x_,
              vecIncrement_, &beta, y_, vecIncrement_);
          if (stat != CUBLAS_STATUS_SUCCESS) {
            std::cout << "cuBLAS error:" << stat << std::endl;
            exit(1);
          }
        }
        break;
      }
    }
  }

  /** Perform any required steps after calling the GEMV kernel that should
   * be timed. */
  void postLoopRequirements() override {
    switch (offload_) {
      case gpuOffloadType::always: {
        // Offload data each iteration - no requirements
        break;
      }
      case gpuOffloadType::once: {
        // Offload output data from device to host
        cudaCheckError(cudaMemcpyAsync(y_, y_device_, sizeof(T) * m_,
                                       cudaMemcpyDeviceToHost, s3_));
        // Ensure device has finished all work.
        cudaCheckError(cudaDeviceSynchronize());
        break;
      }
      case gpuOffloadType::unified: {
        // Ensure all output data resides on host once work has completed
        cudaCheckError(
            cudaMemPrefetchAsync(y_, sizeof(T) * m_, cudaCpuDeviceId, s3_));
        // Ensure device has finished all work.
        cudaCheckError(cudaDeviceSynchronize());
        break;
      }
    }
  }

  /** Do any necessary cleanup (free pointers, close library handles, etc.)
   * after Kernel has been called. */
  void postCallKernelCleanup() override {
    if (offload_ == gpuOffloadType::unified) {
      cudaFree(A_);
      cudaFree(x_);
      cudaFree(y_);
    } else {
      // Free the memory held on host and device
      cudaFreeHost((void*)A_);
      cudaFreeHost((void*)x_);
      cudaFreeHost((void*)y_);
      cudaFree(A_device_);
      cudaFree(x_device_);
      cudaFree(y_device_);
    }
  }

  /** Whether the initialise function has been called before. */
  bool alreadyInitialised_ = false;

  /** Handle used when calling cuBLAS. */
  cublasHandle_t handle_;

  /** CUDA Stream 1 - used to asynchronously move data between host and device.
   */
  cudaStream_t s1_;

  /** CUDA Stream 2 - used to asynchronously move data between host and device.
   */
  cudaStream_t s2_;

  /** CUDA Stream 3 - used to asynchronously move data between host and device.
   */
  cudaStream_t s3_;

  /** The ID of the target GPU Device. */
  int gpuDevice_;

  /** Input matrix A, held on the device. */
  T* A_device_;

  /** Input vector x, held on the device. */
  T* x_device_;

  /** Input vector y, held on the device. */
  T* y_device_;

  /** The constant value Alpha. */
  const T alpha = ALPHA;

  /** The constant value Beta. */
  const T beta = BETA;
};
}  // namespace gpu
#endif