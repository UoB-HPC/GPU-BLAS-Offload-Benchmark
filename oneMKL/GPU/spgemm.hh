#pragma once

#ifdef GPU_ONEMKL

#include "../../include/kernels/GPU/spgemm.hh"
#include "../../include/utilities.hh"
#include "common.hh"

#include <iostream>

namespace gpu {
template <typename T>
class spgemm_gpu : public spgemm<T> {
public:
    using spgemm<T>::spgemm;
    using spgemm<T>::initInputMatrices;
    using spgemm<T>::nnz_;
    using spgemm<T>::m_;
    using spgemm<T>::n_;
    using spgemm<T>::k_;
    using spgemm<T>::A_;
    using spgemm<T>::B_;
    using spgemm<T>::C_;
    using spgemm<T>::offload_;
    using spgemm<T>::sparsity_;

    void initialise(gpuOffloadType offload, int m, int n, int k,
                    double sparsity, bool binary = false) override {
      std::cout << "checking already init, ";
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

      std::cout << "setting up metadata,";

      offload_ = offload;
      sparsity_ = sparsity;
      m_ = m;
      n_ = n;
      k_ = k;

      layout_ = oneapi::mkl::layout::row_major;
      operationA_ = oneapi::mkl::transpose::nontrans;
      operationB_ = oneapi::mkl::transpose::nontrans;
      index_ = oneapi::mkl::index_base::zero;


      nnz_ = 1 + (uint64_t)((double)m_ * (double)k_ * (1.0 - sparsity_));
      std::cout << " allocating space,";
      if (offload_ == gpuOffloadType::unified) {
        A_ = (T*)sycl::malloc_shared(sizeof(T) * m_ * k_, gpuQueue_);
        A_vals_ = (T*)sycl::malloc_shared(sizeof(T) * nnz_, gpuQueue_);
        A_cols_ = (int64_t*)sycl::malloc_shared(sizeof(int64_t) * nnz_,
                                                gpuQueue_);
        A_rows_ = (int64_t*)sycl::malloc_shared(sizeof(int64_t) * (m_ + 1),
                                                gpuQueue_);
        B_ = (T*)sycl::malloc_shared(sizeof(T) * k_ * n_, gpuQueue_);
        C_ = (T*)sycl::malloc_shared(sizeof(T) * m_ * n_, gpuQueue_);
      } else {
        A_ = (T*)sycl::malloc_host(sizeof(T) * m_ * k_, gpuQueue_);
        A_vals_ = (T*)sycl::malloc_host(sizeof(T) * nnz_, gpuQueue_);
        A_cols_ = (int64_t*)sycl::malloc_host(sizeof(int64_t) * nnz_,
                                              gpuQueue_);
        A_rows_ = (int64_t*)sycl::malloc_host(sizeof(int64_t) * (m_ + 1),
                                              gpuQueue_);
        B_ = (T*)sycl::malloc_host(sizeof(T) * k_ * n_, gpuQueue_);
        C_ = (T*)sycl::malloc_host(sizeof(T) * m_ * n_, gpuQueue_);
      }
      initInputMatrices();
    }


protected:
    void toSparseFormat() override {
      int64_t nnz_encountered = 0;

      A_rows_[0] = 0;

      for (int64_t row = 0; row < m_; row++) {
        A_rows_[row + 1] = nnz_encountered;
        for (int64_t col = 0; col < k_; col++) {
          if (A_[(row * k_) + col] != 0.0) {
            A_cols_[nnz_encountered] = col;
            A_vals_[nnz_encountered] = static_cast<T>(A_[(row * k_) + col]);
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
                                            k_,
                                            index_,
                                            *A_rows_device_,
                                            *A_cols_device_,
                                            *A_vals_device_);

          B_device_ = new sycl::buffer<T, 1>(B_, sycl::range<1>(n_ * k_));
          C_device_ = new sycl::buffer<T, 1>(C_, sycl::range<1>(n_ * m_));

          gpuQueue_.wait_and_throw();
          break;
        }
        case gpuOffloadType::unified: {
          oneapi::mkl::sparse::init_matrix_handle(&A_device_);
          oneapi::mkl::sparse::set_csr_data(gpuQueue_,
                                            A_device_,
                                            m_,
                                            k_,
                                            index_,
                                            A_rows_,
                                            A_cols_,
                                            A_vals_);
          gpuQueue_.wait_and_throw();
          break;
        }
      }
    }

    void callSpgemm() override {
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
                                            k_,
                                            index_,
                                            *A_rows_device_,
                                            *A_cols_device_,
                                            *A_vals_device_);

          B_device_ = new sycl::buffer<T, 1>(B_, sycl::range<1>(n_ * k_));
          C_device_ = new sycl::buffer<T, 1>(C_, sycl::range<1>(n_ * m_));

          gpuQueue_.wait_and_throw();
          // Do computation
          try {
            oneapi::mkl::sparse::gemm(gpuQueue_,
                                      layout_,
                                      operationA_,
                                      operationB_,
                                      alpha,
                                      A_device_,
                                      *B_device_,
                                      n_,
                                      n_,
                                      beta,
                                      *C_device_,
                                      n_);
          } catch (sycl::exception const& e) {
            std::cout << "ERROR - Caught synchronous SYCL exception during "
                         "SPGEMM (Once):\n"
                      << e.what() << std::endl
                      << "OpenCL status: " << e.code().value() << std::endl;
          }
          // Do cleanup
          oneapi::mkl::sparse::release_matrix_handle(gpuQueue_, &A_device_);

          delete A_vals_device_;
          delete A_cols_device_;
          delete A_rows_device_;
          delete B_device_;
          delete C_device_;

          break;
        }
        case gpuOffloadType::once: {
          try {
            oneapi::mkl::sparse::gemm(gpuQueue_,
                                      layout_,
                                      operationA_,
                                      operationB_,
                                      alpha,
                                      A_device_,
                                      *B_device_,
                                      n_,
                                      n_,
                                      beta,
                                      *C_device_,
                                      n_);
          } catch (sycl::exception const& e) {
            std::cout << "ERROR - Caught synchronous SYCL exception during "
                         "SPGEMM (Once):\n"
                      << e.what() << std::endl
                      << "OpenCL status: " << e.code().value() << std::endl;
          }
          break;
        }
        case gpuOffloadType::unified: {
          try {
            oneapi::mkl::sparse::gemm(gpuQueue_,
                                      layout_,
                                      operationA_,
                                      operationB_,
                                      alpha,
                                      A_device_,
                                      B_,
                                      n_,
                                      n_,
                                      beta,
                                      C_,
                                      n_);
          } catch (sycl::exception const& e) {
            std::cout << "ERROR - Caught synchronous SYCL exception during "
                         "SPGEMM (Unified):\n"
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
      if (offload_ == gpuOffloadType::once) {
        delete A_vals_device_;
        delete A_cols_device_;
        delete A_rows_device_;
        delete B_device_;
        delete C_device_;
      }
    }

    void postCallKernelCleanup() override {

      /**
       *
        A_ = (T*)sycl::malloc_shared(sizeof(T) * m_ * k_, gpuQueue_);
        A_vals_ = (T*)sycl::malloc_shared(sizeof(T) * nnz_, gpuQueue_);
        A_cols_ = (int64_t*)sycl::malloc_shared(sizeof(int64_t) * nnz_,
                                                gpuQueue_);
        A_rows_ = (int64_t*)sycl::malloc_shared(sizeof(int64_t) * (m_ + 1),
                                                gpuQueue_);
        B_ = (T*)sycl::malloc_shared(sizeof(T) * k_ * n_, gpuQueue_);
        C_ = (T*)sycl::malloc_shared(sizeof(T) * m_ * n_, gpuQueue_);
       */
      sycl::free(A_, gpuQueue_);
      sycl::free(A_vals_, gpuQueue_);
      sycl::free(A_cols_, gpuQueue_);
      sycl::free(A_rows_, gpuQueue_);
      sycl::free(B_, gpuQueue_);
      sycl::free(C_, gpuQueue_);
    }

    /** Whether the initialise function has been called before. */
    bool alreadyInitialised_ = false;

    /** The GPU Device. */
    sycl::device myGpu_;

    /** The SYCL execution queue*/
    sycl::queue gpuQueue_;

    oneapi::mkl::layout layout_;
    oneapi::mkl::transpose operationA_;
    oneapi::mkl::transpose operationB_;
    oneapi::mkl::index_base index_;

    T* A_vals_;
    int64_t* A_cols_;
    int64_t* A_rows_;

    oneapi::mkl::sparse::matrix_handle_t A_device_;

    sycl::buffer<T, 1>* A_vals_device_;
    sycl::buffer<int64_t, 1>* A_cols_device_;
    sycl::buffer<int64_t, 1>* A_rows_device_;
    sycl::buffer<T, 1>* B_device_;
    sycl::buffer<T, 1>* C_device_;

    const T alpha = ALPHA;
    const T beta = BETA;
};
}

#endif
