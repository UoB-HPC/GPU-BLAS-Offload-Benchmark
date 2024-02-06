#pragma once

#ifdef GPU_CUBLAS
#include <cublas_v2.h>
#include <cuda_runtime.h>

#include "../include/GPU/gemm.hh"
#include "../include/utilities.hh"
#include "common.hh"

namespace gpu {
/** A class for GEMM GPU BLAS kernels. */
template <typename T>
class gemm_gpu : public gemm<T> {
 public:
  using gemm<T>::gemm;
  using gemm<T>::m_;
  using gemm<T>::n_;
  using gemm<T>::k_;

  /** Initialise the required data structures.
   * `offloadOnce` refers to whether the data should be offloaded to/from the
   * GPU every iteration, or offloaded once before all iterations and collected
   * after all iterations.
   *  - TRUE = offload before all iterations, collect after all iterations
   *  - FALSE = offload to/from each iteration */
  virtual void initialise(bool offloadOnce, int m, int n, int k) {
    offloadOnce_ = offloadOnce;

    m_ = m;
    n_ = n;
    k_ = k;

    // Create a handle for CUBLAS
    cublasCreate(&handle_);

    A_ = (T*)malloc(sizeof(T) * m_ * k_);
    B_ = (T*)malloc(sizeof(T) * k_ * n_);
    C_ = (T*)malloc(sizeof(T) * m_ * n_);

    // Initialise the host matricies
    for (int y = 0; y < m_; y++) {
      for (int x = 0; x < k_; x++) {
        A_[y * k_ + x] = (((T)(rand() % 10000) / 100.0) - 30.0);
      }
    }
    for (int y = 0; y < k_; y++) {
      for (int x = 0; x < n_; x++) {
        B_[y * n_ + x] = (((T)(rand() % 10000) / 100.0) - 30.0);
      }
    }

    // Allocate matrices on device
    cudaCheckError(cudaMalloc((void**)&A_device_, sizeof(T) * m_ * k_));
    cudaCheckError(cudaMalloc((void**)&B_device_, sizeof(T) * k_ * n_));
    cudaCheckError(cudaMalloc((void**)&C_device_, sizeof(T) * m_ * n_));
  }

 private:
  /** Make a call to the BLAS Library Kernel. */
  virtual void callKernel(const int iterations) {
    const T alpha = ALPHA;
    const T beta = BETA;

    if (offloadOnce_) {
      // Offload data from host to the device.
      cudaCheckError(cudaMemcpy(A_device_, A_, sizeof(T) * m_ * k_,
                                cudaMemcpyHostToDevice));
      cudaCheckError(cudaMemcpy(B_device_, B_, sizeof(T) * k_ * n_,
                                cudaMemcpyHostToDevice));
      cudaCheckError(cudaMemcpy(C_device_, C_, sizeof(T) * m_ * n_,
                                cudaMemcpyHostToDevice));
      // Call GPU BLAS library GEMM kernels
      for (int i = 0; i < iterations; i++) {
        if constexpr (std::is_same_v<T, float>) {
          cublasStatus_t stat = cublasSgemm(
              handle_, CUBLAS_OP_N, CUBLAS_OP_N, m_, n_, k_, &alpha, A_device_,
              MAX(1, m_), B_device_, MAX(1, k_), &beta, C_device_, MAX(1, m_));
          if (stat != CUBLAS_STATUS_SUCCESS) {
            std::cout << "cuBLAS error:" << stat << std::endl;
            exit(1);
          }
        } else if constexpr (std::is_same_v<T, double>) {
          cublasStatus_t stat = cublasDgemm(
              handle_, CUBLAS_OP_N, CUBLAS_OP_N, m_, n_, k_, &alpha, A_device_,
              MAX(1, m_), B_device_, MAX(1, k_), &beta, C_device_, MAX(1, m_));
          if (stat != CUBLAS_STATUS_SUCCESS) {
            std::cout << "cuBLAS error:" << stat << std::endl;
            exit(1);
          }
        }
      }
      // Offload data from device to host
      cudaCheckError(cudaMemcpy(A_, A_device_, sizeof(T) * m_ * k_,
                                cudaMemcpyDeviceToHost));
      cudaCheckError(cudaMemcpy(B_, B_device_, sizeof(T) * k_ * n_,
                                cudaMemcpyDeviceToHost));
      cudaCheckError(cudaMemcpy(C_, C_device_, sizeof(T) * m_ * n_,
                                cudaMemcpyDeviceToHost));
      callConsume();
    } else {
      for (int i = 0; i < iterations; i++) {
        // Offload data from host to the device.
        cudaCheckError(cudaMemcpy(A_device_, A_, sizeof(T) * m_ * k_,
                                  cudaMemcpyHostToDevice));
        cudaCheckError(cudaMemcpy(B_device_, B_, sizeof(T) * k_ * n_,
                                  cudaMemcpyHostToDevice));
        cudaCheckError(cudaMemcpy(C_device_, C_, sizeof(T) * m_ * n_,
                                  cudaMemcpyHostToDevice));
        if constexpr (std::is_same_v<T, float>) {
          cublasStatus_t stat = cublasSgemm(
              handle_, CUBLAS_OP_N, CUBLAS_OP_N, m_, n_, k_, &alpha, A_device_,
              MAX(1, m_), B_device_, MAX(1, k_), &beta, C_device_, MAX(1, m_));
          if (stat != CUBLAS_STATUS_SUCCESS) {
            std::cout << "cuBLAS error:" << stat << std::endl;
            exit(1);
          }
        } else if constexpr (std::is_same_v<T, double>) {
          cublasStatus_t stat = cublasDgemm(
              handle_, CUBLAS_OP_N, CUBLAS_OP_N, m_, n_, k_, &alpha, A_device_,
              MAX(1, m_), B_device_, MAX(1, k_), &beta, C_device_, MAX(1, m_));
          if (stat != CUBLAS_STATUS_SUCCESS) {
            std::cout << "cuBLAS error:" << stat << std::endl;
            exit(1);
          }
        }
        // Offload data from device to host
        cudaCheckError(cudaMemcpy(A_, A_device_, sizeof(T) * m_ * k_,
                                  cudaMemcpyDeviceToHost));
        cudaCheckError(cudaMemcpy(B_, B_device_, sizeof(T) * k_ * n_,
                                  cudaMemcpyDeviceToHost));
        cudaCheckError(cudaMemcpy(C_, C_device_, sizeof(T) * m_ * n_,
                                  cudaMemcpyDeviceToHost));
        callConsume();
      }
    }
  }

  /** Call the extern consume() function. */
  void callConsume() override { consume((void*)A_, (void*)B_, (void*)C_); }

  /** Do any necessary cleanup (free pointers, close library handles, etc.)
   * after Kernel has been called. */
  virtual void postCallKernelCleanup() override {
    // Destroy the handle
    cublasDestroy(handle_);
    // Free the memory held on host and device
    free(A_);
    free(B_);
    free(C_);
    cudaFree(A_device_);
    cudaFree(B_device_);
    cudaFree(C_device_);
  }

  /** Whether or not matrices A, B, and C should be moved from host to device,
   * and from device back to host, every iteration or once before & after all
   * iterations.*/
  bool offloadOnce_;

  /** Handle used when calling cuBLAS. */
  cublasHandle_t handle_;

  /** Input matrix A. */
  T* A_;

  /** Input matrix B. */
  T* B_;

  /** Input matrix C. */
  T* C_;

  /** Input matrix A, held on the device. */
  T* A_device_;

  /** Input matrix B, held on the device. */
  T* B_device_;

  /** Input matrix C, held on the device. */
  T* C_device_;
};
}  // namespace gpu
#endif