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
      switch (offload) {
        case gpuOffloadType::always: {
          if (print_) std::cout << "===========  ALWAYS  ===========" << std::endl;
          break;
        }
        case gpuOffloadType::once: {
          if (print_) std::cout << "===========   ONCE   ===========" << std::endl;
          break;
        }
        case gpuOffloadType::unified: {
          if (print_) std::cout << "===========  UNIFIED ===========" << std::endl;
          break;
        }
      }
      if (print_) std::cout << "Initialising " << m << "x" << n << std::endl;
      if (!alreadyInitialised_) {
        alreadyInitialised_ = true;
        // Perform set-up which doesn't need to happen every problem size change.
        try {
          myGpu_ = sycl::device(sycl::gpu_selector_v);
        } catch (const std::exception& e) {
          std::cerr << "ERROR - No GPU device found: " << e.what() << '\n';
          std::terminate();
        }
        auto exception_handler = [](sycl::exception_list exceptions) {
          for (std::exception_ptr const &e : exceptions) {
            try {
              std::rethrow_exception(e);
            } catch (sycl::exception const &e) {
              std::cout << "Caught asynchronous SYCL exception during sparse::gemv:\n" << e.what() << std::endl;
            }
          }
        };  

        gpuQueue_ = sycl::queue(myGpu_, exception_handler);
      }

      offload_ = offload;
      sparsity_ = sparsity;
      m_ = m;
      n_ = n;

      index_ = oneapi::mkl::index_base::zero;
      operation_ = oneapi::mkl::transpose::nontrans;

      nnz_ = 1 + (uint64_t)((double)m_ * (double)n_ * (1.0 - sparsity_));

      if (print_) std::cout << "Allocating arrays" << std::endl;
      switch (offload_) {
        case gpuOffloadType::unified: {
          A_cols_ = (int64_t*)sycl::malloc_shared(sizeof(int64_t) * nnz_, gpuQueue_);
          A_rows_ = (int64_t*)sycl::malloc_shared(sizeof(int64_t) * (m_ + 1), gpuQueue_);
          A_ = (T*)sycl::malloc_shared(sizeof(T) * m_ * n_, gpuQueue_);
          A_vals_ = (T*)sycl::malloc_shared(sizeof(T) * nnz_, gpuQueue_);
          x_ = (T*)sycl::malloc_shared(sizeof(T) * n_, gpuQueue_);
          y_ = (T*)sycl::malloc_shared(sizeof(T) * m_, gpuQueue_);

          if (!A_ || !A_vals_ || !A_cols_ || !A_rows_ || !x_ || !y_) {
            std::cerr << "ERROR - Failed to allocate memory for GPU SPGEMV" << std::endl;
            exit(1);
          }
          break;
        }
        case gpuOffloadType::always:
        case gpuOffloadType::once: {
          A_ = (T*)sycl::malloc_host(sizeof(T) * m_ * n_, gpuQueue_);
          A_vals_ = (T*)sycl::malloc_host(sizeof(T) * nnz_, gpuQueue_);
          A_cols_ = (int64_t*)sycl::malloc_host(sizeof(int64_t) * nnz_,
                                                gpuQueue_);
          A_rows_ = (int64_t*)sycl::malloc_host(sizeof(int64_t) * (m_ + 1),
                                                gpuQueue_);
          x_ = (T*)sycl::malloc_host(sizeof(T) * n_, gpuQueue_);
          y_ = (T*)sycl::malloc_host(sizeof(T) * m_, gpuQueue_);
          if (!A_ || !A_vals_ || !A_cols_ || !A_rows_ || !x_ || !y_) {
            std::cerr << "ERROR - Failed to allocate host memory" << std::endl;
            exit(1);
          }

          A_vals_device_ = (T*)sycl::malloc_device(sizeof(T) * nnz_, gpuQueue_);
          A_cols_device_ = (int64_t*)sycl::malloc_device(sizeof(int64_t) * nnz_, gpuQueue_);
          A_rows_device_ = (int64_t*)sycl::malloc_device(sizeof(int64_t) * (m_ + 1), gpuQueue_);
          x_device_ = (T*)sycl::malloc_device(sizeof(T) * n_, gpuQueue_);
          y_device_ = (T*)sycl::malloc_device(sizeof(T) * m_, gpuQueue_);
          if (!A_vals_device_ || !A_cols_device_ || !A_rows_device_ || !x_device_ || !y_device_) {
            std::cerr << "ERROR - Failed to allocate device memory" << std::endl;
            exit(1);
          }
          break;
        }
      }
      gpuQueue_.wait_and_throw();

      if (print_) std::cout << "Initialising matrices" << std::endl;
      initInputMatrixVector();
      gpuQueue_.wait_and_throw();
    }


protected:
    void toSparseFormat() override {
      gpuQueue_.wait_and_throw();
      if (print_) std::cout << "Making sparse now" << std::endl;

      int64_t nnz_encountered = 0;
      A_rows_[0] = 0;

      constexpr double eps = 1e-12;

      for (int64_t row = 0; row < m_; row++) {
        for (int64_t col = 0; col < n_; col++) {
          double val = A_[(row * n_) + col];
          if (std::abs(val) > eps) {
            A_cols_[nnz_encountered] = col;
            A_vals_[nnz_encountered] = static_cast<T>(val);
            nnz_encountered++;
          }
        }
        A_rows_[row + 1] = nnz_encountered;
      }
    }

private:
    void preLoopRequirements() override {
      if (offload_ == gpuOffloadType::once) {
        if (print_) std::cout << "\tMoving data to GPU" << std::endl;
        gpuQueue_.memcpy(A_vals_device_, A_vals_, sizeof(T) * nnz_);
        gpuQueue_.memcpy(A_cols_device_, A_cols_, sizeof(int64_t) * nnz_);
        gpuQueue_.memcpy(A_rows_device_, A_rows_, sizeof(int64_t) * (m_ + 1));
        gpuQueue_.memcpy(x_device_, x_, sizeof(T) * n_);
        gpuQueue_.wait_and_throw();
      }
    }

    void callSpgemv() override {
      switch (offload_) {
        case gpuOffloadType::always: {
          if (print_) std::cout << "Moving data to GPU" << std::endl;
          gpuQueue_.memcpy(A_vals_device_, A_vals_, sizeof(T) * nnz_);
          gpuQueue_.memcpy(A_cols_device_, A_cols_, sizeof(int64_t) * nnz_);
          gpuQueue_.memcpy(A_rows_device_, A_rows_, sizeof(int64_t) * (m_ + 1));
          gpuQueue_.memcpy(x_device_, x_, sizeof(T) * n_);
          gpuQueue_.wait_and_throw();
          // Do computation
          try {
            if (print_) std::cout << "Initialising matrix handle" << std::endl;
            oneapi::mkl::sparse::init_matrix_handle(&handle_);
            if (!handle_) {
              std::cerr << "ERROR - Failed to initialise matrix handle" << std::endl;
              exit(1);
            }
            gpuQueue_.wait_and_throw();
            auto set = oneapi::mkl::sparse::set_csr_data(gpuQueue_,
                                                              handle_,
                                                              m_,
                                                              n_,
                                                              index_,
                                                              A_rows_,
                                                              A_cols_,
                                                              A_vals_,
                                                              {});
            
            if (print_) std::cout << "Optimising handle" << std::endl;
            auto optimise = oneapi::mkl::sparse::optimize_gemv(gpuQueue_,
                                                               operation_,
                                                               handle_,
                                                               {set});

            if (print_) std::cout << "Calling SPGEMV kernel" << std::endl;
            auto gemv = oneapi::mkl::sparse::gemv(gpuQueue_,
                                                  operation_,
                                                  alpha,
                                                  handle_,
                                                  x_device_,
                                                  beta,
                                                  y_device_,
                                                  {optimise});
                                                  
            if (print_) std::cout << "Releasing matrix handle" << std::endl;
            auto release = oneapi::mkl::sparse::release_matrix_handle(gpuQueue_, &handle_, {gemv});
            release.wait_and_throw();
            handle_ = nullptr; // Reset handle to avoid double free
          } catch (sycl::exception const& e) {std::cout << "ERROR - Caught synchronous SYCL exception during SPGEMV (Once):\n" << e.what() << std::endl << "OpenCL status: " << e.code().value() << std::endl;}
          gpuQueue_.memcpy(y_, y_device_, sizeof(T) * m_);
          break;
        }
        case gpuOffloadType::once: {
          try {
            if (print_) std::cout << "Initialising matrix handle" << std::endl;
            oneapi::mkl::sparse::init_matrix_handle(&handle_);
            if (!handle_) {
              std::cerr << "ERROR - Failed to initialise matrix handle" << std::endl;
              exit(1);
            }
            gpuQueue_.wait_and_throw();
            auto set = oneapi::mkl::sparse::set_csr_data(gpuQueue_,
                                                         handle_,
                                                         m_,
                                                         n_,
                                                         index_,
                                                         A_rows_,
                                                         A_cols_,
                                                         A_vals_,
                                                         {});
            
            if (print_) std::cout << "Optimising handle" << std::endl;
            auto optimise = oneapi::mkl::sparse::optimize_gemv(gpuQueue_,
                                                               operation_,
                                                               handle_,
                                                               {set});

            if (print_) std::cout << "Calling SPGEMV kernel" << std::endl;
            auto gemv = oneapi::mkl::sparse::gemv(gpuQueue_,
                                                  operation_,
                                                  alpha,
                                                  handle_,
                                                  x_device_,
                                                  beta,
                                                  y_device_,
                                                  {optimise});
            auto release = oneapi::mkl::sparse::release_matrix_handle(gpuQueue_, &handle_, {gemv});
            release.wait_and_throw();
            
            handle_ = nullptr; // Reset handle to avoid double free
          } catch (sycl::exception const& e) {std::cout << "ERROR - Caught synchronous SYCL exception during SPGEMV (Once):\n" << e.what() << std::endl << "OpenCL status: " << e.code().value() << std::endl;}
          break;
        }
        case gpuOffloadType::unified: {
          try {
            std::vector<int64_t*> int_ptr_vec;
            int_ptr_vec.push_back(A_cols_);
            int_ptr_vec.push_back(A_rows_);
            std::vector<T*> float_ptr_vec;
            float_ptr_vec.push_back(A_vals_);
            float_ptr_vec.push_back(x_);
            float_ptr_vec.push_back(y_);


            if (print_) std::cout << "Initialising matrix handle" << std::endl;
            handle_ = nullptr;
            oneapi::mkl::sparse::init_matrix_handle(&handle_);
            if (!handle_) {
              std::cerr << "ERROR - Failed to initialise matrix handle" << std::endl;
              exit(1);
            }
            gpuQueue_.wait_and_throw();

            auto set = oneapi::mkl::sparse::set_csr_data(gpuQueue_,
                                                         handle_,
                                                         m_,
                                                         n_,
                                                         index_,
                                                         A_rows_,
                                                         A_cols_,
                                                         A_vals_,
                                                         {});
            
            if (print_) std::cout << "Optimising handle" << std::endl;
            auto optimise = oneapi::mkl::sparse::optimize_gemv(gpuQueue_,
                                                               operation_,
                                                               handle_,
                                                               {set});

            if (print_) std::cout << "Calling SPGEMV kernel" << std::endl;
            auto gemv = oneapi::mkl::sparse::gemv(gpuQueue_,
                                                  operation_,
                                                  alpha,
                                                  handle_,
                                                  x_,
                                                  beta,
                                                  y_,
                                                  {optimise});

            auto release = oneapi::mkl::sparse::release_matrix_handle(gpuQueue_, &handle_, {gemv});
            release.wait_and_throw();

            handle_ = nullptr; // Reset handle to avoid double free
          } catch (sycl::exception const& e) {std::cout << "ERROR - Caught synchronous SYCL exception during SPGEMV (Unified):\n" << e.what() << std::endl << "OpenCL status: " << e.code().value() << std::endl;}
          break;
        }
      }
    }

    void postLoopRequirements() override {
      if (offload_ == gpuOffloadType::once) {
        if (print_) std::cout << "\tMoving data back to host" << std::endl;
        gpuQueue_.memcpy(y_, y_device_, sizeof(T) * m_);
        gpuQueue_.wait_and_throw();
      }
    }

    void postCallKernelCleanup() override {
      if (print_) std::cout << "Freeing arrays" << std::endl;
      switch (offload_) {
        case gpuOffloadType::always:
        case gpuOffloadType::once: {
          sycl::free(A_vals_device_, gpuQueue_);
          sycl::free(A_cols_device_, gpuQueue_);
          sycl::free(A_rows_device_, gpuQueue_);
          sycl::free(x_device_, gpuQueue_);
          sycl::free(y_device_, gpuQueue_);
        }
        case gpuOffloadType::unified: {
          sycl::free(A_, gpuQueue_);
          sycl::free(A_vals_, gpuQueue_);
          sycl::free(A_cols_, gpuQueue_);
          sycl::free(A_rows_, gpuQueue_);
          sycl::free(x_, gpuQueue_);
          sycl::free(y_, gpuQueue_);
          break;
        }
      }
      gpuQueue_.wait_and_throw();
    }

    bool print_ = true;

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

    oneapi::mkl::sparse::matrix_handle_t handle_;

    T* A_vals_device_;
    int64_t* A_cols_device_;
    int64_t* A_rows_device_;
    T* x_device_;
    T* y_device_;

    const T alpha = ALPHA;
    const T beta = BETA;
};
}

#endif
