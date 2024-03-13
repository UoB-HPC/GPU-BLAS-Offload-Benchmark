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
      A_ = (T*)malloc_shared(sizeof(T) * m_ * k_, myGpu_,
                             gpuQueue_.get_context());
      B_ = (T*)malloc_shared(sizeof(T) * k_ * n_, myGpu_,
                             gpuQueue_.get_context());
      C_ = (T*)malloc_shared(sizeof(T) * m_ * n_, myGpu_,
                             gpuQueue_.get_context());
    } else {
      // Allocate matrices on host
      A_ = (T*)malloc(sizeof(T) * m_ * k_);
      B_ = (T*)malloc(sizeof(T) * k_ * n_);
      C_ = (T*)malloc(sizeof(T) * m_ * n_);
      // Allocate matrices on device
      A_buffer_ = sycl::buffer<T, 1>(A_, m_ * k_);
      B_buffer_ = sycl::buffer<T, 1>(B_, k_ * n_);
      C_buffer_ = sycl::buffer<T, 1>(C_, m_ * n_);
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
        // TODO - Offload data from host to the device.
        break;
      }
      case gpuOffloadType::unified: {
        // TODO - Prefetch memory to device
        break;
      }
    }
  }

  /** Make a call to the BLAS Library Kernel. */
  void callGemm() override {
    switch (offload_) {
      case gpuOffloadType::always: {
        // TODO - Offload data from host to the device.
        // Call cuBLAS GEMM kernel
        try {
          oneapi::mkl::blas::gemm(gpuQueue_, transA_, transB_, (int64_t)m_,
                                  (int64_t)n_, (int64_t)k_, alpha, A_buffer_,
                                  (int64_t)std::max(1, m_), B_buffer_,
                                  (int64_t)std::max(1, k_), beta, C_buffer_,
                                  (int64_t)std::max(1, m_));
        } catch (sycl::exception const& e) {
          std::cout << "ERROR - Caught synchronous SYCL exception during GEMM "
                       "(Always):\n"
                    << e.what() << std::endl
                    << "OpenCL status: " << get_error_code(e) << std::endl;
        }
        // TODO - Offload data from device to host
        break;
      }
      case gpuOffloadType::once: {
        // Call cuBLAS GEMM kernel
        try {
          oneapi::mkl::blas::gemm(gpuQueue_, transA_, transB_, (int64_t)m_,
                                  (int64_t)n_, (int64_t)k_, alpha, A_buffer_,
                                  (int64_t)std::max(1, m_), B_buffer_,
                                  (int64_t)std::max(1, k_), beta, C_buffer_,
                                  (int64_t)std::max(1, m_));
        } catch (sycl::exception const& e) {
          std::cout << "ERROR - Caught synchronous SYCL exception during GEMM "
                       "(Once):\n"
                    << e.what() << std::endl
                    << "OpenCL status: " << get_error_code(e) << std::endl;
        }
        break;
      }
      case gpuOffloadType::unified: {
        // Call cuBLAS GEMM kernel
        try {
          gemmDone_ = oneapi::mkl::blas::gemm(
              gpuQueue_, transA_, transB_, (int64_t)m_, (int64_t)n_,
              (int64_t)k_, alpha, A_, (int64_t)std::max(1, m_), B_,
              (int64_t)std::max(1, k_), beta, C_, (int64_t)std::max(1, m_),
              gemmDependencies_);
        } catch (sycl::exception const& e) {
          std::cout << "ERROR - Caught synchronous SYCL exception during GEMM "
                       "(Unified):\n"
                    << e.what() << std::endl
                    << "OpenCL status: " << get_error_code(e) << std::endl;
        }

        gemm_done.wait();
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
        // TODO - Offload data from device to host
        break;
      }
      case gpuOffloadType::unified: {
        // TODO - Ensure all data resides on host once work has completed
        break;
      }
    }
  }

  /** Do any necessary cleanup (free pointers, close library handles, etc.)
   * after Kernel has been called. */
  void postCallKernelCleanup() override {
    if (offload_ == gpuOffloadType::unified) {
      free(A_, gpuQueue_.get_context());
      free(B_, gpuQueue_.get_context());
      free(C_, gpuQueue_.get_context());
    } else {
      // Free the memory held on host and device
      free(A_);
      free(B_);
      free(C_);
    }
  }

  /** Whether the initialise function has been called before. */
  bool alreadyInitialised_ = false;

  /** The GPU Device. */
  sycl::device myGpu_;

  /** The SYCL execution queue*/
  sycl::queue gpuQueue_;

  /** Device buffer for matrix A. */
  sycl::buffer<T, 1> A_buffer_;

  /** Device buffer for matrix B. */
  sycl::buffer<T, 1> B_buffer_;

  /** Device buffer for matrix C. */
  sycl::buffer<T, 1> C_buffer_;

  /** SYCL event indicating when the GEMM has completed. Used for Unified Memory
   * operation. */
  sycl::event gemmDone_;

  /** A vector of SYCL events passed into the unified memory GEMM oneMKL call.
   */
  std::vector<sycl::event> gemmDependencies_;

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