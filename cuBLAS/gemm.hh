#pragma once

#ifdef GPU_CUBLAS
#include <cublas_v2.h>
#include <cuda_runtime.h>

#include "../include/kernels/GPU/gemm.hh"
#include "../include/utilities.hh"
#include "common.hh"

namespace gpu {
/** A class for GEMM GPU BLAS kernels. */
template <typename T>
class gemm_gpu : public gemm<T> {
 public:
  using gemm<T>::gemm;
  using gemm<T>::initInputMatrices;
  using gemm<T>::m_;
  using gemm<T>::n_;
  using gemm<T>::k_;
  using gemm<T>::A_;
  using gemm<T>::B_;
  using gemm<T>::C_;
  using gemm<T>::offload_;

  ~gemm_gpu() {
    if (alreadyInitialised_) {
      // Destroy the handle
      cublasCheckError(cublasDestroy(handle_));

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
  void initialise(gpuOffloadType offload, int m, int n, int k) override {
    if (!alreadyInitialised_) {
      alreadyInitialised_ = true;
      // Perform set-up which doesn't need to happen every problem size change.
      // Create a handle for CUBLAS
      cublasCheckError(cublasCreate(&handle_));

      // Enable Tensor Cores
      cublasCheckError(cublasSetMathMode(handle_, CUBLAS_TENSOR_OP_MATH));

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
    k_ = k;

    if (offload_ == gpuOffloadType::unified) {
      cudaCheckError(cudaMallocManaged(&A_, sizeof(T) * m_ * k_));
      cudaCheckError(cudaMallocManaged(&B_, sizeof(T) * k_ * n_));
      cudaCheckError(cudaMallocManaged(&C_, sizeof(T) * m_ * n_));
    } else {
      // Allocate matrices on host
      cudaCheckError(cudaMallocHost((void**)&A_, sizeof(T) * m_ * k_));
      cudaCheckError(cudaMallocHost((void**)&B_, sizeof(T) * k_ * n_));
      cudaCheckError(cudaMallocHost((void**)&C_, sizeof(T) * m_ * n_));
      // Allocate matrices on device
      cudaCheckError(cudaMalloc((void**)&A_device_, sizeof(T) * m_ * k_));
      cudaCheckError(cudaMalloc((void**)&B_device_, sizeof(T) * k_ * n_));
      cudaCheckError(cudaMalloc((void**)&C_device_, sizeof(T) * m_ * n_));
    }

    // Initialise the host input matricies
    initInputMatrices();
  }

 private:
  /** Perform any required steps before calling the GEMM kernel that should
   * be timed. */
  void preLoopRequirements() override {
    switch (offload_) {
      case gpuOffloadType::always: {
        // Offload data each iteration - no requirements
        break;
      }
      case gpuOffloadType::once: {
        // Offload input data from host to the device.
        cudaCheckError(cudaMemcpyAsync(A_device_, A_, sizeof(T) * m_ * k_,
                                       cudaMemcpyHostToDevice, s1_));
        cudaCheckError(cudaMemcpyAsync(B_device_, B_, sizeof(T) * k_ * n_,
                                       cudaMemcpyHostToDevice, s2_));
        cudaCheckError(cudaMemcpyAsync(C_device_, C_, sizeof(T) * m_ * n_,
                                       cudaMemcpyHostToDevice, s3_));
        break;
      }
      case gpuOffloadType::unified: {
        // Prefetch input data to device
        cudaCheckError(
            cudaMemPrefetchAsync(A_, sizeof(T) * m_ * k_, gpuDevice_, s1_));
        cudaCheckError(
            cudaMemPrefetchAsync(B_, sizeof(T) * k_ * n_, gpuDevice_, s2_));
        cudaCheckError(
            cudaMemPrefetchAsync(C_, sizeof(T) * m_ * n_, gpuDevice_, s3_));
        break;
      }
    }
  }

  /** Make a call to the BLAS Library Kernel. */
  void callGemm() override {
    switch (offload_) {
      case gpuOffloadType::always: {
        // Offload input data from host to the device.
        cudaCheckError(cudaMemcpyAsync(A_device_, A_, sizeof(T) * m_ * k_,
                                       cudaMemcpyHostToDevice, s1_));
        cudaCheckError(cudaMemcpyAsync(B_device_, B_, sizeof(T) * k_ * n_,
                                       cudaMemcpyHostToDevice, s2_));
        cudaCheckError(cudaMemcpyAsync(C_device_, C_, sizeof(T) * m_ * n_,
                                       cudaMemcpyHostToDevice, s3_));
        // Call cuBLAS GEMM kernel
        if constexpr (std::is_same_v<T, float>) {
          cublasCheckError(cublasGemmEx(
              handle_, CUBLAS_OP_N, CUBLAS_OP_N, m_, n_, k_, (void*)&alpha,
              (void*)A_device_, CUDA_R_32F, std::max(1, m_), (void*)B_device_,
              CUDA_R_32F, std::max(1, k_), (void*)&beta, (void*)C_device_,
              CUDA_R_32F, std::max(1, m_), CUBLAS_COMPUTE_32F,
              CUBLAS_GEMM_DEFAULT));
        } else if constexpr (std::is_same_v<T, double>) {
          cublasCheckError(cublasGemmEx(
              handle_, CUBLAS_OP_N, CUBLAS_OP_N, m_, n_, k_, (void*)&alpha,
              (void*)A_device_, CUDA_R_64F, std::max(1, m_), (void*)B_device_,
              CUDA_R_64F, std::max(1, k_), (void*)&beta, (void*)C_device_,
              CUDA_R_64F, std::max(1, m_), CUBLAS_COMPUTE_64F,
              CUBLAS_GEMM_DEFAULT));
        }
        // Offload output data from device to host
        cudaCheckError(cudaMemcpyAsync(C_, C_device_, sizeof(T) * m_ * n_,
                                       cudaMemcpyDeviceToHost, s3_));
        // Ensure device has finished all work.
        cudaCheckError(cudaDeviceSynchronize());
        break;
      }
      case gpuOffloadType::once: {
        // Call cuBLAS GEMM kernel
        if constexpr (std::is_same_v<T, float>) {
          cublasCheckError(cublasGemmEx(
              handle_, CUBLAS_OP_N, CUBLAS_OP_N, m_, n_, k_, (void*)&alpha,
              (void*)A_device_, CUDA_R_32F, std::max(1, m_), (void*)B_device_,
              CUDA_R_32F, std::max(1, k_), (void*)&beta, (void*)C_device_,
              CUDA_R_32F, std::max(1, m_), CUBLAS_COMPUTE_32F,
              CUBLAS_GEMM_DEFAULT));
        } else if constexpr (std::is_same_v<T, double>) {
          cublasCheckError(cublasGemmEx(
              handle_, CUBLAS_OP_N, CUBLAS_OP_N, m_, n_, k_, (void*)&alpha,
              (void*)A_device_, CUDA_R_64F, std::max(1, m_), (void*)B_device_,
              CUDA_R_64F, std::max(1, k_), (void*)&beta, (void*)C_device_,
              CUDA_R_64F, std::max(1, m_), CUBLAS_COMPUTE_64F,
              CUBLAS_GEMM_DEFAULT));
        }
        break;
      }
      case gpuOffloadType::unified: {
        // Call cuBLAS GEMM kernel
        if constexpr (std::is_same_v<T, float>) {
          cublasCheckError(cublasGemmEx(
              handle_, CUBLAS_OP_N, CUBLAS_OP_N, m_, n_, k_, (void*)&alpha,
              (void*)A_, CUDA_R_32F, std::max(1, m_), (void*)B_, CUDA_R_32F,
              std::max(1, k_), (void*)&beta, (void*)C_, CUDA_R_32F,
              std::max(1, m_), CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT));
        } else if constexpr (std::is_same_v<T, double>) {
          cublasCheckError(cublasGemmEx(
              handle_, CUBLAS_OP_N, CUBLAS_OP_N, m_, n_, k_, (void*)&alpha,
              (void*)A_, CUDA_R_64F, std::max(1, m_), (void*)B_, CUDA_R_64F,
              std::max(1, k_), (void*)&beta, (void*)C_, CUDA_R_64F,
              std::max(1, m_), CUBLAS_COMPUTE_64F, CUBLAS_GEMM_DEFAULT));
        }
        break;
      }
    }
  }

  /** Perform any required steps after calling the GEMM kernel that should
   * be timed. */
  void postLoopRequirements() override {
    switch (offload_) {
      case gpuOffloadType::always: {
        // Offload data each iteration - no requirements
        break;
      }
      case gpuOffloadType::once: {
        // Offload output data from device to host
        cudaCheckError(cudaMemcpyAsync(C_, C_device_, sizeof(T) * m_ * n_,
                                       cudaMemcpyDeviceToHost, s3_));
        // Ensure device has finished all work.
        cudaCheckError(cudaDeviceSynchronize());
        break;
      }
      case gpuOffloadType::unified: {
        // Ensure all output data resides on host once work has completed
        cudaCheckError(cudaMemPrefetchAsync(C_, sizeof(T) * m_ * n_,
                                            cudaCpuDeviceId, s3_));
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
      cudaFree(B_);
      cudaFree(C_);
    } else {
      // Free the memory held on host and device
      cudaFreeHost((void*)A_);
      cudaFreeHost((void*)B_);
      cudaFreeHost((void*)C_);
      cudaFree(A_device_);
      cudaFree(B_device_);
      cudaFree(C_device_);
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

  /** Input matrix B, held on the device. */
  T* B_device_;

  /** Input matrix C, held on the device. */
  T* C_device_;

  /** The constant value Alpha. */
  const T alpha = ALPHA;

  /** The constant value Beta. */
  const T beta = BETA;
};
}  // namespace gpu
#endif