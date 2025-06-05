#pragma once

#ifdef GPU_ONEMKL

#include "../../include/kernels/GPU/spgemv.hh"
#include "../../include/utilities.hh"
#include "common.hh"

namespace gpu {
template <typename T>
class spgemv_gpu : public spgemv<T> {
public:
    using spgemv<T>::spgemv;
    using spgemv<T>::initInputMatrixVector;
    using spgemv<T>::nnz_;
    using spgemv<T>::m_;
    using spgemv<T>::n_;
    using spgemv<T>::A_;
    using spgemv<T>::x_;
    using spgemv<T>::y_;
    using spgemv<T>::offload_;
    using spgemv<T>::sparsity_;

    void initialise(gpuOffloadType offload, int m, int n, double sparsity)
    override {
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
      sparsity_ = sparsity;
      m_ = m;
      n_ = n;

      index_ = oneapi::mkl::index_base::zero;
      operation_ = oneapi::mkl::transpose::nontrans;

      nnz_ = 1 + (uint64_t)((double)m_ * (double)n_ * (1.0 - sparsity_));

      if (offload_ == gpuOffloadType::unified) {
        A_ = (T*)sycl::malloc_shared(sizeof(T) * m_ * n_, gpuQueue_);
        A_vals_ = (T*)sycl::malloc_shared(sizeof(T) * nnz_, gpuQueue_);
        A_cols_ = (int64_t*)sycl::malloc_shared(sizeof(int64_t) * nnz_,
                                                gpuQueue_);
        A_rows_ = (int64_t*)sycl::malloc_shared(sizeof(int64_t) * (m_ + 1),
                                                gpuQueue_);
        x_ = (T*)sycl::malloc_shared(sizeof(T) * n_, gpuQueue_);
        y_ = (T*)sycl::malloc_shared(sizeof(T) * m_, gpuQueue_);
      } else {
        A_ = (T*)sycl::malloc_host(sizeof(T) * m_ * n_, gpuQueue_);
        A_vals_ = (T*)sycl::malloc_host(sizeof(T) * nnz_, gpuQueue_);
        A_cols_ = (int64_t*)sycl::malloc_host(sizeof(int64_t) * nnz_,
                                              gpuQueue_);
        A_rows_ = (int64_t*)sycl::malloc_host(sizeof(int64_t) * (m_ + 1),
                                              gpuQueue_);
        x_ = (T*)sycl::malloc_host(sizeof(T) * n_, gpuQueue_);
        y_ = (T*)sycl::malloc_host(sizeof(T) * m_, gpuQueue_);
      }

      initInputMatrixVector();
    }


protected:
    void toSparseFormat() override {
      int64_t nnz_encountered = 0;

      A_rows_[0] = 0;

      for (int64_t row = 0; row < m_; row++) {
        A_rows_[row + 1] = nnz_encountered;
        for (int64_t col = 0; col < n_; col++) {
          if (A_[(row * n_) + col] != 0.0) {
            A_cols_[nnz_encountered] = col;
            A_vals_[nnz_encountered] = static_cast<T>(A_[(row * n_) + col]);
            nnz_encountered++;
          }
        }
      }
    }

private:
    void preLoopRequirements() override {
      switch(offload_) {
        case gpuOffloadType::always: {
          break;
        }
        case gpuOffloadType::once: {
          A_vals_device_ = new sycl::buffer<T, 1>(A_vals_,
                                                  sycl::range<1>(nnz_));
          A_cols_device_ = new sycl::buffer<int64_t, 1>(A_cols_,
                                                        sycl::range<1>(nnz_));
          A_rows_device_ = new sycl::buffer<int64_t, 1>(A_rows_,
                                                        sycl::range<1>(m_ + 1));

          oneapi::mkl::sparse::init_matrix_handle(&A_device_);
          oneapi::mkl::sparse::set_csr_data(gpuQueue_,
                                            A_device_,
                                            m_,
                                            n_,
                                            index_,
                                            *A_rows_device_,
                                            *A_cols_device_,
                                            *A_vals_device_);

          x_device_ = new sycl::buffer<T, 1>(x_, sycl::range<1>(n_));
          y_device_ = new sycl::buffer<T, 1>(y_, sycl::range<1>(m_));
          gpuQueue_.wait_and_throw();
          break;
        }
        case gpuOffloadType::unified: {
          oneapi::mkl::sparse::init_matrix_handle(&A_device_);
          oneapi::mkl::sparse::set_csr_data(gpuQueue_,
                                            A_device_,
                                            m_,
                                            n_,
                                            index_,
                                            A_rows_,
                                            A_cols_,
                                            A_vals_);
          gpuQueue_.wait_and_throw();
          break;
        }
      }
    }

    void callSpgemv() override {
      switch (offload_) {
        case gpuOffloadType::always: {
          // Do transfer etc.
          A_vals_device_ = new sycl::buffer<T, 1>(A_vals_,
                                                  sycl::range<1>(nnz_));
          A_cols_device_ = new sycl::buffer<int64_t, 1>(A_cols_,
                                                        sycl::range<1>(nnz_));
          A_rows_device_ = new sycl::buffer<int64_t, 1>(A_rows_,
                                                        sycl::range<1>(m_ + 1));

          oneapi::mkl::sparse::init_matrix_handle(&A_device_);
          oneapi::mkl::sparse::set_csr_data(gpuQueue_,
                                            A_device_,
                                            m_,
                                            n_,
                                            index_,
                                            *A_rows_device_,
                                            *A_cols_device_,
                                            *A_vals_device_);

          x_device_ = new sycl::buffer<T, 1>(x_, sycl::range<1>(n_));
          y_device_ = new sycl::buffer<T, 1>(y_, sycl::range<1>(m_));
          gpuQueue_.wait_and_throw();
          // Do computation
          try {
            oneapi::mkl::sparse::gemv(gpuQueue_,
                                      operation_,
                                      alpha,
                                      A_device_,
                                      *x_device_,
                                      beta,
                                      *y_device_);
          } catch (sycl::exception const& e) {
            std::cout << "ERROR - Caught synchronous SYCL exception during "
                         "SPGEMV (Once):\n"
                      << e.what() << std::endl
                      << "OpenCL status: " << e.code().value() << std::endl;
          }
          // Do cleanup
          oneapi::mkl::sparse::release_matrix_handle(gpuQueue_, &A_device_);
          break;
        }
        case gpuOffloadType::once: {
          try {
            oneapi::mkl::sparse::gemv(gpuQueue_,
                                      operation_,
                                      alpha,
                                      A_device_,
                                      *x_device_,
                                      beta,
                                      *y_device_);
          } catch (sycl::exception const& e) {
            std::cout << "ERROR - Caught synchronous SYCL exception during "
                         "SPGEMV (Once):\n"
                      << e.what() << std::endl
                      << "OpenCL status: " << e.code().value() << std::endl;
          }
          break;
        }
        case gpuOffloadType::unified: {
          try {
            oneapi::mkl::sparse::gemv(gpuQueue_,
                                      operation_,
                                      alpha,
                                      A_device_,
                                      x_,
                                      beta,
                                      y_);
          } catch (sycl::exception const& e) {
            std::cout << "ERROR - Caught synchronous SYCL exception during "
                         "SPGEMV (Unified):\n"
                      << e.what() << std::endl
                      << "OpenCL status: " << e.code().value() << std::endl;
          }
          break;
        }
      }
    }

    void postLoopRequirements() override {
      if (offload_ != gpuOffloadType::always) {
        oneapi::mkl::sparse::release_matrix_handle(gpuQueue_, &A_device_);
      }
    }

    void postCallKernelCleanup() override {
      sycl::free(A_, gpuQueue_);
      sycl::free(A_vals_, gpuQueue_);
      sycl::free(A_cols_, gpuQueue_);
      sycl::free(A_rows_, gpuQueue_);
      sycl::free(x_, gpuQueue_);
      sycl::free(y_, gpuQueue_);
    }


    /** Whether the initialise function has been called before. */
    bool alreadyInitialised_ = false;

    /** The GPU Device. */
    sycl::device myGpu_;

    /** The SYCL execution queue*/
    sycl::queue gpuQueue_;

    oneapi::mkl::index_base index_;
    oneapi::mkl::transpose operation_;

    T* A_vals_;
    int64_t* A_cols_;
    int64_t* A_rows_;

    oneapi::mkl::sparse::matrix_handle_t A_device_;

    sycl::buffer<T, 1>* A_vals_device_;
    sycl::buffer<int64_t, 1>* A_cols_device_;
    sycl::buffer<int64_t, 1>* A_rows_device_;
    sycl::buffer<T, 1>* x_device_;
    sycl::buffer<T, 1>* y_device_;

    const T alpha = ALPHA;
    const T beta = BETA;
};
}

#endif
