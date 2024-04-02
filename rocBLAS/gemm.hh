#pragma once

#ifdef GPU_ROCBLAS
#include <hip/hip_runtime_api.h>
#include <rocblas/rocblas.h>

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
      rocblas_destroy_handle(handle_);

      // Destroy streams after use
      hipCheckError(hipStreamDestroy(s1_));
      hipCheckError(hipStreamDestroy(s2_));
      hipCheckError(hipStreamDestroy(s3_));
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
      // Create a handle for rocBLAS
      rocblas_status status = rocblas_create_handle(&handle_);
      if (status != rocblas_status_success) {
        std::cout << "Failed to make rocBLAS handle: " << status << std::endl;
        exit(1);
      }

      // Get device identifier
      hipCheckError(hipGetDevice(&gpuDevice_));

      // Initialise 3 streams to asynchronously move data between host and
      // device
      hipCheckError(hipStreamCreate(&s1_));
      hipCheckError(hipStreamCreate(&s2_));
      hipCheckError(hipStreamCreate(&s3_));

      // Enable passing alpha parameter from pointer to host memory
      status = rocblas_set_pointer_mode(handle_, rocblas_pointer_mode_host);
      if (status != rocblas_status_success) {
        std::cout << "Failed to set rocBLAS pointer mode: " << status
                  << std::endl;
        exit(1);
      }
    }

    offload_ = offload;
    m_ = m;
    n_ = n;
    k_ = k;

    if (offload_ == gpuOffloadType::unified) {
      hipCheckError(hipMallocManaged(&A_, sizeof(T) * m_ * k_));
      hipCheckError(hipMallocManaged(&B_, sizeof(T) * k_ * n_));
      hipCheckError(hipMallocManaged(&C_, sizeof(T) * m_ * n_));
    } else {
      // Allocate matrices on host
      hipCheckError(hipHostMalloc((void**)&A_, sizeof(T) * m_ * k_));
      hipCheckError(hipHostMalloc((void**)&B_, sizeof(T) * k_ * n_));
      hipCheckError(hipHostMalloc((void**)&C_, sizeof(T) * m_ * n_));
      // Allocate matrices on device
      hipCheckError(hipMalloc((void**)&A_device_, sizeof(T) * m_ * k_));
      hipCheckError(hipMalloc((void**)&B_device_, sizeof(T) * k_ * n_));
      hipCheckError(hipMalloc((void**)&C_device_, sizeof(T) * m_ * n_));
    }

    // Initialise the host input matricies (A_ and B_)
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
        hipCheckError(hipMemcpyAsync(A_device_, A_, sizeof(T) * m_ * k_,
                                     hipMemcpyHostToDevice, s1_));
        hipCheckError(hipMemcpyAsync(B_device_, B_, sizeof(T) * k_ * n_,
                                     hipMemcpyHostToDevice, s2_));
        hipCheckError(hipMemcpyAsync(C_device_, C_, sizeof(T) * m_ * n_,
                                     hipMemcpyHostToDevice, s3_));
        break;
      }
      case gpuOffloadType::unified: {
        // Prefetch input data to device
        hipCheckError(
            hipMemPrefetchAsync(A_, sizeof(T) * m_ * k_, gpuDevice_, s1_));
        hipCheckError(
            hipMemPrefetchAsync(B_, sizeof(T) * k_ * n_, gpuDevice_, s2_));
        hipCheckError(
            hipMemPrefetchAsync(C_, sizeof(T) * m_ * n_, gpuDevice_, s3_));
        break;
      }
    }
  }

  /** Make a call to the BLAS Library Kernel. */
  void callGemm() override {
    switch (offload_) {
      case gpuOffloadType::always: {
        // Offload input data from host to the device.
        hipCheckError(hipMemcpyAsync(A_device_, A_, sizeof(T) * m_ * k_,
                                     hipMemcpyHostToDevice, s1_));
        hipCheckError(hipMemcpyAsync(B_device_, B_, sizeof(T) * k_ * n_,
                                     hipMemcpyHostToDevice, s2_));
        hipCheckError(hipMemcpyAsync(C_device_, C_, sizeof(T) * m_ * n_,
                                     hipMemcpyHostToDevice, s3_));
        // Call rocBLAS GEMM kernel
        if constexpr (std::is_same_v<T, float>) {
          rocblas_status stat =
              rocblas_sgemm(handle_, transA_, transB_, m_, n_, k_, &alpha,
                            A_device_, std::max(1, m_), B_device_,
                            std::max(1, k_), &beta, C_device_, std::max(1, m_));
          if (stat != rocblas_status_success) {
            std::cout << "rocBLAS error:" << rocblas_status_to_string(stat)
                      << std::endl;
            exit(1);
          }
        } else if constexpr (std::is_same_v<T, double>) {
          rocblas_status stat =
              rocblas_dgemm(handle_, transA_, transB_, m_, n_, k_, &alpha,
                            A_device_, std::max(1, m_), B_device_,
                            std::max(1, k_), &beta, C_device_, std::max(1, m_));
          if (stat != rocblas_status_success) {
            std::cout << "rocBLAS error:" << rocblas_status_to_string(stat)
                      << std::endl;
            exit(1);
          }
        }
        // Offload output data from device to host
        hipCheckError(hipMemcpyAsync(C_, C_device_, sizeof(T) * m_ * n_,
                                     hipMemcpyDeviceToHost, s3_));
        // Ensure device has finished all work.
        hipCheckError(hipDeviceSynchronize());
        break;
      }
      case gpuOffloadType::once: {
        // Call rocBLAS GEMM kernel
        if constexpr (std::is_same_v<T, float>) {
          rocblas_status stat =
              rocblas_sgemm(handle_, transA_, transB_, m_, n_, k_, &alpha,
                            A_device_, std::max(1, m_), B_device_,
                            std::max(1, k_), &beta, C_device_, std::max(1, m_));
          if (stat != rocblas_status_success) {
            std::cout << "rocBLAS error:" << rocblas_status_to_string(stat)
                      << std::endl;
            exit(1);
          }
        } else if constexpr (std::is_same_v<T, double>) {
          rocblas_status stat =
              rocblas_dgemm(handle_, transA_, transB_, m_, n_, k_, &alpha,
                            A_device_, std::max(1, m_), B_device_,
                            std::max(1, k_), &beta, C_device_, std::max(1, m_));
          if (stat != rocblas_status_success) {
            std::cout << "rocBLAS error:" << rocblas_status_to_string(stat)
                      << std::endl;
            exit(1);
          }
        }
        break;
      }
      case gpuOffloadType::unified: {
        // Call rocBLAS GEMM kernel
        if constexpr (std::is_same_v<T, float>) {
          rocblas_status stat = rocblas_sgemm(
              handle_, transA_, transB_, m_, n_, k_, &alpha, A_,
              std::max(1, m_), B_, std::max(1, k_), &beta, C_, std::max(1, m_));
          if (stat != rocblas_status_success) {
            std::cout << "rocBLAS error:" << rocblas_status_to_string(stat)
                      << std::endl;
            exit(1);
          }
        } else if constexpr (std::is_same_v<T, double>) {
          rocblas_status stat = rocblas_dgemm(
              handle_, transA_, transB_, m_, n_, k_, &alpha, A_,
              std::max(1, m_), B_, std::max(1, k_), &beta, C_, std::max(1, m_));
          if (stat != rocblas_status_success) {
            std::cout << "rocBLAS error:" << rocblas_status_to_string(stat)
                      << std::endl;
            exit(1);
          }
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
        hipCheckError(hipMemcpyAsync(C_, C_device_, sizeof(T) * m_ * n_,
                                     hipMemcpyDeviceToHost, s3_));
        // Ensure device has finished all work.
        hipCheckError(hipDeviceSynchronize());
        break;
      }
      case gpuOffloadType::unified: {
        // Ensure all output data resides on host once work has completed
        hipCheckError(
            hipMemPrefetchAsync(C_, sizeof(T) * m_ * n_, hipCpuDeviceId, s3_));
        // Ensure device has finished all work.
        hipCheckError(hipDeviceSynchronize());
        break;
      }
    }
  }

  /** Do any necessary cleanup (free pointers, close library handles, etc.)
   * after Kernel has been called. */
  void postCallKernelCleanup() override {
    if (offload_ == gpuOffloadType::unified) {
      hipCheckError(hipFree(A_));
      hipCheckError(hipFree(B_));
      hipCheckError(hipFree(C_));
    } else {
      // Free the memory held on host and device
      hipCheckError(hipHostFree((void*)A_));
      hipCheckError(hipHostFree((void*)B_));
      hipCheckError(hipHostFree((void*)C_));
      hipCheckError(hipFree(A_device_));
      hipCheckError(hipFree(B_device_));
      hipCheckError(hipFree(C_device_));
    }
  }

  /** Whether the initialise function has been called before. */
  bool alreadyInitialised_ = false;

  /** Handle used when calling rocBLAS. */
  rocblas_handle handle_;

  /** HIP Stream 1 - used to asynchronously move data between host and device.
   */
  hipStream_t s1_;

  /** HIP Stream 2 - used to asynchronously move data between host and device.
   */
  hipStream_t s2_;

  /** HIP Stream 3 - used to asynchronously move data between host and device.
   */
  hipStream_t s3_;

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

  /** Weather or not matrix A should be transposed. */
  const rocblas_operation transA_ = rocblas_operation_none;

  /** Weather or not matrix B should be transposed. */
  const rocblas_operation transB_ = rocblas_operation_none;
};
}  // namespace gpu
#endif