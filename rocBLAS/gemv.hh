#pragma once

#ifdef GPU_ROCBLAS
#include <hip/hip_runtime_api.h>
#include <rocblas/rocblas.h>

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

    if (offload_ == gpuOffloadType::unified) {
      hipCheckError(hipMallocManaged(&A_, sizeof(T) * m_ * n_));
      hipCheckError(hipMallocManaged(&x_, sizeof(T) * n_));
      hipCheckError(hipMallocManaged(&y_, sizeof(T) * m_));
    } else {
      // Allocate matrices on host
      hipCheckError(hipHostMalloc((void**)&A_, sizeof(T) * m_ * n_));
      hipCheckError(hipHostMalloc((void**)&x_, sizeof(T) * n_));
      hipCheckError(hipHostMalloc((void**)&y_, sizeof(T) * m_));
      // Allocate matrices on device
      hipCheckError(hipMalloc((void**)&A_device_, sizeof(T) * m_ * n_));
      hipCheckError(hipMalloc((void**)&x_device_, sizeof(T) * n_));
      hipCheckError(hipMalloc((void**)&y_device_, sizeof(T) * m_));
    }

    // Initialise the host data structures
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
        hipCheckError(hipMemcpyAsync(A_device_, A_, sizeof(T) * m_ * n_,
                                     hipMemcpyHostToDevice, s1_));
        hipCheckError(hipMemcpyAsync(x_device_, x_, sizeof(T) * n_,
                                     hipMemcpyHostToDevice, s2_));
        hipCheckError(hipMemcpyAsync(y_device_, y_, sizeof(T) * m_,
                                     hipMemcpyHostToDevice, s3_));
        break;
      }
      case gpuOffloadType::unified: {
        // Prefetch input data to device
        hipCheckError(
            hipMemPrefetchAsync(A_, sizeof(T) * m_ * n_, gpuDevice_, s1_));
        hipCheckError(hipMemPrefetchAsync(x_, sizeof(T) * n_, gpuDevice_, s2_));
        hipCheckError(hipMemPrefetchAsync(y_, sizeof(T) * m_, gpuDevice_, s3_));
        break;
      }
    }
  }

  /** Make a call to the BLAS Library Kernel. */
  void callGemm() override {
    switch (offload_) {
      case gpuOffloadType::always: {
        // Offload input data from host to the device.
        hipCheckError(hipMemcpyAsync(A_device_, A_, sizeof(T) * m_ * n_,
                                     hipMemcpyHostToDevice, s1_));
        hipCheckError(hipMemcpyAsync(x_device_, x_, sizeof(T) * n_,
                                     hipMemcpyHostToDevice, s2_));
        hipCheckError(hipMemcpyAsync(y_device_, y_, sizeof(T) * m_,
                                     hipMemcpyHostToDevice, s3_));
        // Call rocBLAS GEMV kernel
        if constexpr (std::is_same_v<T, float>) {
          rocblas_status stat = rocblas_sgemv(
              handle_, transA_, m_, n_, &alpha, A_device_, std::max(1, m_),
              x_device_, vecIncrement_, &beta, y_device_, vecIncrement_);
          if (stat != rocblas_status_success) {
            std::cout << "rocBLAS error:" << rocblas_status_to_string(stat)
                      << std::endl;
            exit(1);
          }
        } else if constexpr (std::is_same_v<T, double>) {
          rocblas_status stat = rocblas_dgemv(
              handle_, transA_, m_, n_, &alpha, A_device_, std::max(1, m_),
              x_device_, vecIncrement_, &beta, y_device_, vecIncrement_);
          if (stat != rocblas_status_success) {
            std::cout << "rocBLAS error:" << rocblas_status_to_string(stat)
                      << std::endl;
            exit(1);
          }
        }
        // Offload output data from device to host
        hipCheckError(hipMemcpyAsync(y_, y_device_, sizeof(T) * m_,
                                     hipMemcpyDeviceToHost, s3_));
        // Ensure device has finished all work.
        hipCheckError(hipDeviceSynchronize());
        break;
      }
      case gpuOffloadType::once: {
        // Call rocBLAS GEMV kernel
        if constexpr (std::is_same_v<T, float>) {
          rocblas_status stat = rocblas_sgemv(
              handle_, transA_, m_, n_, &alpha, A_device_, std::max(1, m_),
              x_device_, vecIncrement_, &beta, y_device_, vecIncrement_);
          if (stat != rocblas_status_success) {
            std::cout << "rocBLAS error:" << rocblas_status_to_string(stat)
                      << std::endl;
            exit(1);
          }
        } else if constexpr (std::is_same_v<T, double>) {
          rocblas_status stat = rocblas_dgemv(
              handle_, transA_, m_, n_, &alpha, A_device_, std::max(1, m_),
              x_device_, vecIncrement_, &beta, y_device_, vecIncrement_);
          if (stat != rocblas_status_success) {
            std::cout << "rocBLAS error:" << rocblas_status_to_string(stat)
                      << std::endl;
            exit(1);
          }
        }
        break;
      }
      case gpuOffloadType::unified: {
        // Call rocBLAS GEMV kernel
        if constexpr (std::is_same_v<T, float>) {
          rocblas_status stat = rocblas_sgemv(
              handle_, transA_, m_, n_, &alpha, A_, std::max(1, m_), x_,
              vecIncrement_, &beta, y_, vecIncrement_);
          if (stat != rocblas_status_success) {
            std::cout << "rocBLAS error:" << rocblas_status_to_string(stat)
                      << std::endl;
            exit(1);
          }
        } else if constexpr (std::is_same_v<T, double>) {
          rocblas_status stat = rocblas_dgemv(
              handle_, transA_, m_, n_, &alpha, A_, std::max(1, m_), x_,
              vecIncrement_, &beta, y_, vecIncrement_);
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
        hipCheckError(hipMemcpyAsync(y_, y_device_, sizeof(T) * m_,
                                     hipMemcpyDeviceToHost, s3_));
        // Ensure device has finished all work.
        hipCheckError(hipDeviceSynchronize());
        break;
      }
      case gpuOffloadType::unified: {
        // Ensure all output data resides on host once work has completed
        hipCheckError(
            hipMemPrefetchAsync(y_, sizeof(T) * m_, hipCpuDeviceId, s3_));
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
      hipCheckError(hipFree(x_));
      hipCheckError(hipFree(y_));
    } else {
      // Free the memory held on host and device
      hipCheckError(hipHostFree((void*)A_));
      hipCheckError(hipHostFree((void*)x_));
      hipCheckError(hipHostFree((void*)y_));
      hipCheckError(hipFree(A_device_));
      hipCheckError(hipFree(x_device_));
      hipCheckError(hipFree(y_device_));
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

  /** Input vecotr x, held on the device. */
  T* x_device_;

  /** Input vector y, held on the device. */
  T* y_device_;

  /** The constant value Alpha. */
  const T alpha = ALPHA;

  /** The constant value Beta. */
  const T beta = BETA;

  /** Weather or not matrix A should be transposed. */
  const rocblas_operation transA_ = rocblas_operation_none;
};
}  // namespace gpu
#endif