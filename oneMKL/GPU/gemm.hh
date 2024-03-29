#pragma once

#ifdef GPU_ONEMKL

#include "../../include/kernels/GPU/gemm.hh"
#include "../../include/utilities.hh"
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
      try {
        myGpu_ = sycl::device(sycl::gpu_selector_v);
      } catch (const std::exception& e) {
        std::cerr << "ERROR - No GPU device found: " << e.what() << '\n';
        std::terminate();
      }
      gpuQueue_ = sycl::queue(myGpu_, exception_handler);
    }

    offload_ = offload;
    m_ = m;
    n_ = n;
    k_ = k;

    if (offload_ == gpuOffloadType::unified) {
      A_ = (T*)sycl::malloc_shared(sizeof(T) * m_ * k_, gpuQueue_);
      B_ = (T*)sycl::malloc_shared(sizeof(T) * k_ * n_, gpuQueue_);
      C_ = (T*)sycl::malloc_shared(sizeof(T) * m_ * n_, gpuQueue_);
    } else {
      // Allocate matrices on host (64-byte aligned)
      A_ = (T*)sycl::malloc_host(sizeof(T) * m_ * k_, gpuQueue_);
      B_ = (T*)sycl::malloc_host(sizeof(T) * k_ * n_, gpuQueue_);
      C_ = (T*)sycl::malloc_host(sizeof(T) * m_ * n_, gpuQueue_);
      // Allocate matrices on device
      A_device_ = (T*)sycl::malloc_device(sizeof(T) * m_ * k_, gpuQueue_);
      B_device_ = (T*)sycl::malloc_device(sizeof(T) * k_ * n_, gpuQueue_);
      C_device_ = (T*)sycl::malloc_device(sizeof(T) * m_ * n_, gpuQueue_);
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
        gpuQueue_.memcpy(A_device_, A_, sizeof(T) * m_ * k_);
        gpuQueue_.memcpy(B_device_, B_, sizeof(T) * k_ * n_);
        gpuQueue_.memcpy(C_device_, C_, sizeof(T) * m_ * n_);
        gpuQueue_.wait_and_throw();
        break;
      }
      case gpuOffloadType::unified: {
        // Prefetch memory to device --- prefetch broken / not working
        // gpuQueue_.prefetch(A_, sizeof(T) * m_ * k_);
        // gpuQueue_.prefetch(B_, sizeof(T) * k_ * n_);
        // gpuQueue_.prefetch(C_, sizeof(T) * m_ * n_);
        gpuQueue_.wait_and_throw();
        break;
      }
    }
  }

  /** Make a call to the BLAS Library Kernel. */
  void callGemm() override {
    switch (offload_) {
      case gpuOffloadType::always: {
        // Offload input data from host to the device.
        gpuQueue_.memcpy(A_device_, A_, sizeof(T) * m_ * k_);
        gpuQueue_.memcpy(B_device_, B_, sizeof(T) * k_ * n_);
        gpuQueue_.memcpy(C_device_, C_, sizeof(T) * m_ * n_);
        gpuQueue_.wait_and_throw();
        // Call oneMKL GEMM kernel
        try {
          oneapi::mkl::blas::column_major::gemm(
              gpuQueue_, transA_, transB_, (int64_t)m_, (int64_t)n_,
              (int64_t)k_, alpha, A_device_, (int64_t)std::max(1, m_),
              B_device_, (int64_t)std::max(1, k_), beta, C_device_,
              (int64_t)std::max(1, m_))
              .wait_and_throw();
        } catch (sycl::exception const& e) {
          std::cout << "ERROR - Caught synchronous SYCL exception during GEMM "
                       "(Always):\n"
                    << e.what() << std::endl
                    << "OpenCL status: " << e.code().value() << std::endl;
        }
        // Offload output data from device to host
        gpuQueue_.memcpy(C_, C_device_, sizeof(T) * m_ * n_);
        gpuQueue_.wait_and_throw();
        break;
      }
      case gpuOffloadType::once: {
        // Call oneMKL GEMM kernel
        try {
          oneapi::mkl::blas::column_major::gemm(
              gpuQueue_, transA_, transB_, (int64_t)m_, (int64_t)n_,
              (int64_t)k_, alpha, A_device_, (int64_t)std::max(1, m_),
              B_device_, (int64_t)std::max(1, k_), beta, C_device_,
              (int64_t)std::max(1, m_))
              .wait_and_throw();
        } catch (sycl::exception const& e) {
          std::cout << "ERROR - Caught synchronous SYCL exception during GEMM "
                       "(Once):\n"
                    << e.what() << std::endl
                    << "OpenCL status: " << e.code().value() << std::endl;
        }
        break;
      }
      case gpuOffloadType::unified: {
        // Call oneMKL GEMM kernel
        try {
          oneapi::mkl::blas::column_major::gemm(
              gpuQueue_, transA_, transB_, (int64_t)m_, (int64_t)n_,
              (int64_t)k_, alpha, A_, (int64_t)std::max(1, m_), B_,
              (int64_t)std::max(1, k_), beta, C_, (int64_t)std::max(1, m_), {})
              .wait_and_throw();
        } catch (sycl::exception const& e) {
          std::cout << "ERROR - Caught synchronous SYCL exception during GEMM "
                       "(Unified):\n"
                    << e.what() << std::endl
                    << "OpenCL status: " << e.code().value() << std::endl;
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
        gpuQueue_.memcpy(C_, C_device_, sizeof(T) * m_ * n_);
        gpuQueue_.wait_and_throw();
        break;
      }
      case gpuOffloadType::unified: {
        // TODO - Ensure all output data resides on host once work has completed
        gpuQueue_.wait_and_throw();
        break;
      }
    }
  }

  /** Do any necessary cleanup (free pointers, close library handles, etc.)
   * after Kernel has been called. */
  void postCallKernelCleanup() override {
    sycl::free(A_, gpuQueue_);
    sycl::free(B_, gpuQueue_);
    sycl::free(C_, gpuQueue_);
    if (offload_ != gpuOffloadType::unified) {
      sycl::free(A_device_, gpuQueue_);
      sycl::free(B_device_, gpuQueue_);
      sycl::free(C_device_, gpuQueue_);
    }
  }

  /** Whether the initialise function has been called before. */
  bool alreadyInitialised_ = false;

  /** The GPU Device. */
  sycl::device myGpu_;

  /** The SYCL execution queue*/
  sycl::queue gpuQueue_;

  /** Input matrix A, held on the device. */
  T* A_device_;

  /** Input matrix B, held on the device. */
  T* B_device_;

  /** Input matrix C, held on the device. */
  T* C_device_;

  /** Weather or not matrix A should be transposed. */
  oneapi::mkl::transpose transA_ = oneapi::mkl::transpose::nontrans;

  /** Weather or not matrix B should be transposed. */
  oneapi::mkl::transpose transB_ = oneapi::mkl::transpose::nontrans;

  /** The constant value Alpha. */
  const T alpha = ALPHA;

  /** The constant value Beta. */
  const T beta = BETA;
};
}  // namespace gpu

#endif