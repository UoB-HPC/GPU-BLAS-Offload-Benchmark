#pragma once

#ifdef GPU_ONEMKL

#include "../../include/kernels/GPU/gemv.hh"
#include "../../include/utilities.hh"
#include "common.hh"

namespace gpu {
/** A class for GEMV GPU BLAS kernels. */
template <typename T>
class gemv_gpu : public gemv<T> {
 public:
  using gemv<T>::gemv;
  using gemv<T>::initInputMatrixVector;
  using gemm<T>::cacheLineWidth_;
  using gemv<T>::m_;
  using gemv<T>::n_;
  using gemv<T>::A_;
  using gemv<T>::x_;
  using gemv<T>::y_;
  using gemv<T>::offload_;
  using gemv<T>::vecIncrement_;

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

    if (offload_ == gpuOffloadType::unified) {
      A_ = (T*)sycl::malloc_shared(sizeof(T) * m_ * n_, gpuQueue_);
      x_ = (T*)sycl::malloc_shared(sizeof(T) * n_, gpuQueue_);
      y_ = (T*)sycl::malloc_shared(sizeof(T) * m_, gpuQueue_);
    } else {
      // Allocate matrices on host
      A_ = (T*)sycl::malloc_host(sizeof(T) * m_ * n_, gpuQueue_);
      x_ = (T*)sycl::malloc_host(sizeof(T) * n_, gpuQueue_);
      y_ = (T*)sycl::malloc_host(sizeof(T) * m_, gpuQueue_);
      // Allocate matrices on device
      A_device_ = (T*)sycl::malloc_device(sizeof(T) * m_ * n_, gpuQueue_);
      x_device_ = (T*)sycl::malloc_device(sizeof(T) * n_, gpuQueue_);
      y_device_ = (T*)sycl::malloc_device(sizeof(T) * m_, gpuQueue_);
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
        gpuQueue_.memcpy(A_device_, A_, sizeof(T) * m_ * n_);
        gpuQueue_.memcpy(x_device_, x_, sizeof(T) * n_);
        gpuQueue_.memcpy(y_device_, y_, sizeof(T) * m_);
        gpuQueue_.wait_and_throw();
        break;
      }
      case gpuOffloadType::unified: {
        // Prefetch memory to device --- prefetch broken / not working
        gpuQueue_.prefetch(A_, sizeof(T) * m_ * n_);
        gpuQueue_.prefetch(x_, sizeof(T) * n_);
        gpuQueue_.prefetch(y_, sizeof(T) * m_);
        gpuQueue_.wait_and_throw();
        break;
      }
    }
  }

  /** Make a call to the BLAS Library Kernel. */
  void callGemv() override {
    switch (offload_) {
      case gpuOffloadType::always: {
        // Offload input data from host to the device.
        gpuQueue_.memcpy(A_device_, A_, sizeof(T) * m_ * n_);
        gpuQueue_.memcpy(x_device_, x_, sizeof(T) * n_);
        gpuQueue_.memcpy(y_device_, y_, sizeof(T) * m_);
        gpuQueue_.wait_and_throw();
        // Call oneMKL GEMV kernel
        try {
          oneapi::mkl::blas::column_major::gemv(
              gpuQueue_, transA_, (int64_t)m_, (int64_t)n_, alpha, A_device_,
              (int64_t)std::max(1, m_), x_device_, vecIncrement_, beta,
              y_device_, vecIncrement_, {})
              .wait_and_throw();
        } catch (sycl::exception const& e) {
          std::cout << "ERROR - Caught synchronous SYCL exception during GEMV "
                       "(Always):\n"
                    << e.what() << std::endl
                    << "OpenCL status: " << e.code().value() << std::endl;
        }
        // Offload output data from device to host
        gpuQueue_.memcpy(y_, y_device_, sizeof(T) * m_);
        gpuQueue_.wait_and_throw();
        break;
      }
      case gpuOffloadType::once: {
        // Call oneMKL GEMV kernel
        try {
          oneapi::mkl::blas::column_major::gemv(
              gpuQueue_, transA_, (int64_t)m_, (int64_t)n_, alpha, A_device_,
              (int64_t)std::max(1, m_), x_device_, vecIncrement_, beta,
              y_device_, vecIncrement_, {})
              .wait_and_throw();
        } catch (sycl::exception const& e) {
          std::cout << "ERROR - Caught synchronous SYCL exception during GEMV "
                       "(Once):\n"
                    << e.what() << std::endl
                    << "OpenCL status: " << e.code().value() << std::endl;
        }
        break;
      }
      case gpuOffloadType::unified: {
        // Call oneMKL GEMV kernel
        try {
          oneapi::mkl::blas::column_major::gemv(
              gpuQueue_, transA_, (int64_t)m_, (int64_t)n_, alpha, A_,
              (int64_t)std::max(1, m_), x_, vecIncrement_, beta, y_,
              vecIncrement_, {})
              .wait_and_throw();
        } catch (sycl::exception const& e) {
          std::cout << "ERROR - Caught synchronous SYCL exception during GEMV "
                       "(Unified):\n"
                    << e.what() << std::endl
                    << "OpenCL status: " << e.code().value() << std::endl;
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
        gpuQueue_.memcpy(y_, y_device_, sizeof(T) * m_);
        gpuQueue_.wait_and_throw();
        break;
      }
      case gpuOffloadType::unified: {
        // Ensure all data resides on host once work has completed
        for (uint64_t i = 0; i < (sizeof(T) * m_); i += cacheLineWidth_) {
          _mm_prefetch(y_ + i, _MM_HINT_NTA);
        }
        gpuQueue_.wait_and_throw();
        break;
      }
    }
  }

  /** Do any necessary cleanup (free pointers, close library handles, etc.)
   * after Kernel has been called. */
  void postCallKernelCleanup() override {
    sycl::free(A_, gpuQueue_);
    sycl::free(x_, gpuQueue_);
    sycl::free(y_, gpuQueue_);
    if (offload_ != gpuOffloadType::unified) {
      sycl::free(A_device_, gpuQueue_);
      sycl::free(x_device_, gpuQueue_);
      sycl::free(y_device_, gpuQueue_);
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

  /** Input vector x, held on the device. */
  T* x_device_;

  /** Input vector y, held on the device. */
  T* y_device_;

  /** Weather or not matrix A should be transposed. */
  oneapi::mkl::transpose transA_ = oneapi::mkl::transpose::nontrans;

  /** The constant value Alpha. */
  const T alpha = ALPHA;

  /** The constant value Beta. */
  const T beta = BETA;
};
}  // namespace gpu

#endif