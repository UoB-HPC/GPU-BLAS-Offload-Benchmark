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
    using spgemv<T>::x_;
    using spgemv<T>::y_;
    using spgemv<T>::offload_;
    using spgemv<T>::sparsity_;
    using spgemv<T>::type_;


    void initialise(gpuOffloadType offload, int m, int n, double sparsity, 
                    matrixType type)
    override {
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
            std::cerr << "Caught asynchronous SYCL exception during sparse::gemv:\n" << e.what() << std::endl;
          }
        }
      };  

      gpuQueue_ = sycl::queue(myGpu_, exception_handler);
      context_ = gpuQueue_.get_context();
      
      x_ = nullptr;
      y_ = nullptr;

      offload_ = offload;
      sparsity_ = sparsity;
      type_ = type;
      m_ = m;
      n_ = n;

      index_ = oneapi::mkl::index_base::zero;
      operation_ = oneapi::mkl::transpose::nontrans;

      nnz_ = 1 + (uint64_t)((double)m_ * (double)n_ * (1.0 - sparsity_));

      if (offload_ == gpuOffloadType::unified) {
        x_ = sycl::malloc_shared<T>(n_, gpuQueue_);
        y_ = sycl::malloc_shared<T>(m_, gpuQueue_);
        if (!x_ || !y_) {
          std::cerr << "ERROR - Failed to allocate memory for GPU SPGEMV" << std::endl;
          exit(1);
        }
      } else {
        x_ = sycl::malloc_host<T>(n_, gpuQueue_);
        y_ = sycl::malloc_host<T>(m_, gpuQueue_);
        x_device_ = sycl::malloc_device<T>(n_, gpuQueue_);
        y_device_ = sycl::malloc_device<T>(m_, gpuQueue_);
        if (!x_ || !y_) {
          std::cerr << "ERROR - Failed to allocate host memory" << std::endl;
          exit(1);
        }
      }
      
      gpuQueue_.wait_and_throw();

      initInputMatrixVector();
      gpuQueue_.wait_and_throw();
    }


protected:
    void toSparseFormat() override {
      gpuQueue_.wait_and_throw();
      if (offload_ == gpuOffloadType::unified) {
        A_vals_ = sycl::malloc_shared<T>(nnz_, gpuQueue_);
        A_cols_ = sycl::malloc_shared<int64_t>(nnz_, gpuQueue_);
        A_rows_ = sycl::malloc_shared<int64_t>(m_ + 1, gpuQueue_);
      } else {
        A_vals_ = sycl::malloc_host<T>(nnz_, gpuQueue_);
        A_cols_ = sycl::malloc_host<int64_t>(nnz_, gpuQueue_);
        A_rows_ = sycl::malloc_host<int64_t>(m_ + 1, gpuQueue_);
        A_vals_device_ = (T*)sycl::malloc_device(nnz_ * sizeof(T), gpuQueue_);
        A_cols_device_ = (int64_t*)sycl::malloc_device(nnz_ * sizeof(int64_t), gpuQueue_);
        A_rows_device_ = (int64_t*)sycl::malloc_device((m_ + 1) * sizeof(int64_t), gpuQueue_);
      }

      if (type_ == matrixType::rmat) {
        rMatCSR<T, int64_t>(A_vals_, A_cols_, A_rows_, m_, n_, nnz_);
      } else if (type_ == matrixType::random) {
        randomCSR<T, int64_t>(A_vals_, A_cols_, A_rows_, m_, n_, nnz_);
      } else {
        std::cerr << "Matrix type not supported" << std::endl;
        exit(1);
      }
    }

private:
    void preLoopRequirements() override {
      if (offload_ == gpuOffloadType::once) {
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
          gpuQueue_.memcpy(A_vals_device_, A_vals_, sizeof(T) * nnz_);
          gpuQueue_.memcpy(A_cols_device_, A_cols_, sizeof(int64_t) * nnz_);
          gpuQueue_.memcpy(A_rows_device_, A_rows_, sizeof(int64_t) * (m_ + 1));
          gpuQueue_.memcpy(x_device_, x_, sizeof(T) * n_);
          gpuQueue_.wait_and_throw();
          // Do computation
          try {
            oneapi::mkl::sparse::init_matrix_handle(&handle_);
            auto set = oneapi::mkl::sparse::set_csr_data(gpuQueue_,
                                                         handle_,
                                                         m_,
                                                         n_,
                                                         index_,
                                                         A_rows_device_,
                                                         A_cols_device_,
                                                         A_vals_device_);
            
            auto optimise = oneapi::mkl::sparse::optimize_gemv(gpuQueue_,
                                                               operation_,
                                                               handle_,
                                                               {set});

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
          } catch (sycl::exception const& e) {
            gpuQueue_.wait();
            oneapi::mkl::sparse::release_matrix_handle(gpuQueue_, &handle_);
            std::cerr << "ERROR - Caught synchronous SYCL exception during SPGEMV (Once):\n" << e.what() << std::endl << "OpenCL status: " << e.code().value() << std::endl;
          } catch (std::exception const &e) {
            std::cerr << "\t\tCaught std exception:\n" << e.what() << std::endl;
            gpuQueue_.wait();
            oneapi::mkl::sparse::release_matrix_handle(gpuQueue_, &handle_);
            exit(1);
          }
          gpuQueue_.memcpy(y_, y_device_, sizeof(T) * m_);
          break;
        }
        case gpuOffloadType::once: {
          try {
            oneapi::mkl::sparse::init_matrix_handle(&handle_);
            if (handle_ == nullptr) {
              std::cerr << "ERROR - Failed to initialise matrix handle" << std::endl;
              exit(1);
            }
            gpuQueue_.wait_and_throw();
            auto set = oneapi::mkl::sparse::set_csr_data(gpuQueue_,
                                                         handle_,
                                                         m_,
                                                         n_,
                                                         index_,
                                                         A_rows_device_,
                                                         A_cols_device_,
                                                         A_vals_device_,
                                                         {});
            
            auto optimise = oneapi::mkl::sparse::optimize_gemv(gpuQueue_,
                                                               operation_,
                                                               handle_,
                                                               {set});

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
          } catch (sycl::exception const& e) {
            gpuQueue_.wait();
            oneapi::mkl::sparse::release_matrix_handle(gpuQueue_, &handle_);
            std::cerr << "ERROR - Caught synchronous SYCL exception during SPGEMV (Once):\n" << e.what() << std::endl << "OpenCL status: " << e.code().value() << std::endl;
          } catch (std::exception const &e) {
            std::cerr << "\t\tCaught std exception:\n" << e.what() << std::endl;
            gpuQueue_.wait();
            oneapi::mkl::sparse::release_matrix_handle(gpuQueue_, &handle_);
            exit(1);
          }
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


            handle_ = nullptr;
            oneapi::mkl::sparse::init_matrix_handle(&handle_);
            if (handle_ == nullptr) {
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
            
            auto optimise = oneapi::mkl::sparse::optimize_gemv(gpuQueue_,
                                                               operation_,
                                                               handle_,
                                                               {set});

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
          } catch (sycl::exception const& e) {
            gpuQueue_.wait();
            oneapi::mkl::sparse::release_matrix_handle(gpuQueue_, &handle_);
            std::cerr << "ERROR - Caught synchronous SYCL exception during SPGEMV (Once):\n" << e.what() << std::endl << "OpenCL status: " << e.code().value() << std::endl;
          } catch (std::exception const &e) {
            std::cerr << "\t\tCaught std exception:\n" << e.what() << std::endl;
            gpuQueue_.wait();
            oneapi::mkl::sparse::release_matrix_handle(gpuQueue_, &handle_).wait();
            exit(1);
          }
          break;
        }
      }
    }

    void postLoopRequirements() override {
      if (offload_ == gpuOffloadType::once) {
        gpuQueue_.memcpy(y_, y_device_, sizeof(T) * m_);
        gpuQueue_.wait_and_throw();
      }
    }

    void postCallKernelCleanup() override {
      switch (offload_) {
        case gpuOffloadType::always:
        case gpuOffloadType::once: {
          if (A_vals_ != nullptr) {
            sycl::free(A_vals_, context_);
            A_vals_ = nullptr;
          }
          if (A_cols_ != nullptr) {
            sycl::free(A_cols_, context_);
            A_cols_ = nullptr;
          }
          if (A_rows_ != nullptr) {
            sycl::free(A_rows_, context_);
            A_rows_ = nullptr;
          }
          if (A_vals_device_ != nullptr) {
            sycl::free(A_vals_device_, context_);
            A_vals_device_ = nullptr;
          }
          if (A_cols_device_ != nullptr) {
            sycl::free(A_cols_device_, context_);
            A_cols_device_ = nullptr;
          }
          if (A_rows_device_ != nullptr) {
            sycl::free(A_rows_device_, context_);
            A_rows_device_ = nullptr;
          }
          if (x_ != nullptr) {
            sycl::free(x_, context_);
            x_ = nullptr;
          }
          if (y_ != nullptr) {
            sycl::free(y_, context_);
            y_ = nullptr;
          }
          if (x_device_ != nullptr) {
            sycl::free(x_device_, context_);
            x_device_ = nullptr;
          }
          if (y_device_ != nullptr) {
            sycl::free(y_device_, context_);
            y_device_ = nullptr;
          }
        }
        case gpuOffloadType::unified: {
          if (A_vals_ != nullptr) {
            sycl::free(A_vals_, context_);
            A_vals_ = nullptr;
          }
          if (A_cols_ != nullptr) {
            sycl::free(A_cols_, context_);
            A_cols_ = nullptr;
          }
          if (A_rows_ != nullptr) {
            sycl::free(A_rows_, context_);
            A_rows_ = nullptr;
          }
          if (x_ != nullptr) {
            sycl::free(x_, context_);
            x_ = nullptr;
          }
          if (y_ != nullptr) {
            sycl::free(y_, context_);
            y_ = nullptr;
          }
          gpuQueue_.wait_and_throw();
          break;
        }
      }
      gpuQueue_.wait_and_throw();
    }

    /** Whether the initialise function has been called before. */
    bool alreadyInitialised_ = false;

    /** The GPU Device. */
    sycl::device myGpu_;

    /** The SYCL execution queue*/
    sycl::queue gpuQueue_;

    sycl::context context_;

    oneapi::mkl::index_base index_;
    oneapi::mkl::transpose operation_;

    T* A_vals_ = nullptr;
    int64_t* A_cols_ = nullptr;
    int64_t* A_rows_ = nullptr;

    oneapi::mkl::sparse::matrix_handle_t handle_ = nullptr;

    T* A_vals_device_ = nullptr;
    int64_t* A_cols_device_ = nullptr;
    int64_t* A_rows_device_ = nullptr;
    T* x_device_ = nullptr;
    T* y_device_ = nullptr;

    const T alpha = ALPHA;
    const T beta = BETA;
};
}

#endif
