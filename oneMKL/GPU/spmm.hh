#pragma once

#ifdef GPU_ONEMKL

#include <memory>
#include "../../include/kernels/GPU/spmm.hh"
#include "../../include/utilities.hh"
#include "common.hh"

namespace gpu {
template <typename T>
class spmm_gpu : public spmm<T> {
public:
    using spmm<T>::spmm;
    using spmm<T>::initInputMatrices;
    using spmm<T>::nnzA_;
    using spmm<T>::nnzB_;
    using spmm<T>::nnzC_;
    using spmm<T>::m_;
    using spmm<T>::n_;
    using spmm<T>::k_;
    using spmm<T>::A_;
    using spmm<T>::B_;
    using spmm<T>::C_;
    using spmm<T>::offload_;
    using spmm<T>::sparsity_;

    ~spmm_gpu() {
      // Release descriptor if initialized
      oneapi::mkl::sparse::release_matmat_descr(&description_);
      // Wait for all operations to complete before destroying the queue
      gpuQueue_.wait_and_throw();
    }

    void initialise(gpuOffloadType offload, int m, int n, int k,
                    double sparsity, bool binary = false) override {
      if (offload == gpuOffloadType::always) return;
      // Perform set-up which doesn't need to happen every problem size change.
      try {
        myGpu_ = sycl::device(sycl::gpu_selector_v);
      } catch (const std::exception& e) {
        std::cerr << "ERROR - No GPU device found: " << e.what() << std::endl;
        std::terminate();
      }

      try {
        gpuQueue_ = sycl::queue(myGpu_, exception_handler);
      } catch (const sycl::exception& e) {
        std::cerr << "ERROR - Failed to create queue: " << e.what() << std::endl;
        throw;
      }

//      std::cout << ".. setting metadata";
      offload_ = offload;
      sparsity_ = sparsity;
      m_ = m;
      n_ = n;
      k_ = k;

      layout_ = oneapi::mkl::layout::row_major;
      operationA_ = oneapi::mkl::transpose::nontrans;
      operationB_ = oneapi::mkl::transpose::nontrans;
      index_ = oneapi::mkl::index_base::zero;

      try {
        nnzA_ = 1 + static_cast<uint64_t>(static_cast<double>(m_) * static_cast<double>(k_) * (1.0 - sparsity_));
        nnzB_ = 1 + static_cast<uint64_t>(static_cast<double>(k_) * static_cast<double>(n_) * (1.0 - sparsity_));

        // Verify no overflow occurred
        if (nnzA_ > std::numeric_limits<int64_t>::max() ||
            nnzB_ > std::numeric_limits<int64_t>::max()) {
          throw std::overflow_error("Matrix dimensions result in overflow");
        }
      } catch (const std::exception& e) {
        std::cerr << "ERROR - Invalid matrix dimensions: " << e.what() << std::endl;
        throw;
      }

      if (offload_ == gpuOffloadType::unified) {
        try {
          // Todo -- do we need A_, B_, and C_ in unified memory?  Surely
          //  just CPU memory would work.  Consider.
          A_ = safe_malloc_shared<T>(m_ * k_, gpuQueue_, "A matrix");
          A_vals_ = safe_malloc_shared<T>(nnzA_, gpuQueue_, "A values");
          A_cols_ = safe_malloc_shared<int64_t>(nnzA_, gpuQueue_, "A columns");
          A_rows_ = safe_malloc_shared<int64_t>(m_ + 1, gpuQueue_, "A rows");

          B_ = safe_malloc_shared<T>(k_ * n_, gpuQueue_, "B matrix");
          B_vals_ = safe_malloc_shared<T>(nnzB_, gpuQueue_, "B values");
          B_cols_ = safe_malloc_shared<int64_t>(nnzB_, gpuQueue_, "B columns");
          B_rows_ = safe_malloc_shared<int64_t>(k_ + 1, gpuQueue_, "B rows");

          C_ = safe_malloc_shared<T>(m_ * n_, gpuQueue_, "C matrix");
          C_rows_ = safe_malloc_shared<int64_t>(m_ + 1, gpuQueue_, "C rows");

          safe_wait(gpuQueue_, "unified memory allocation");
        } catch (const std::exception& e) {
          throw;
        }
      } else {
        // For 'once' mode, allocate host memory only
        try {
//          std::cout << ".. host malloc";
          A_ = static_cast<T*>(sycl::malloc_host(sizeof(T) * m_ * k_, gpuQueue_));
          if (!A_) throw std::bad_alloc();
          A_vals_ = static_cast<T*>(sycl::malloc_host(sizeof(T) * nnzA_, gpuQueue_));
          if (!A_vals_) throw std::bad_alloc();
          A_cols_ = static_cast<int64_t*>(sycl::malloc_host(sizeof(int64_t) * nnzA_, gpuQueue_));
          if (!A_cols_) throw std::bad_alloc();
          A_rows_ = static_cast<int64_t*>(sycl::malloc_host(sizeof(int64_t) * (m_ + 1), gpuQueue_));
          if (!A_rows_) throw std::bad_alloc();

          B_ = static_cast<T*>(sycl::malloc_host(sizeof(T) * k_ * n_, gpuQueue_));
          if (!B_) throw std::bad_alloc();
          B_vals_ = static_cast<T*>(sycl::malloc_host(sizeof(T) * nnzB_, gpuQueue_));
          if (!B_vals_) throw std::bad_alloc();
          B_cols_ = static_cast<int64_t*>(sycl::malloc_host(sizeof(int64_t) * nnzB_, gpuQueue_));
          if (!B_cols_) throw std::bad_alloc();
          B_rows_ = static_cast<int64_t*>(sycl::malloc_host(sizeof(int64_t) * (k_ + 1), gpuQueue_));
          if (!B_rows_) throw std::bad_alloc();

          C_ = static_cast<T*>(sycl::malloc_host(sizeof(T) * m_ * n_, gpuQueue_));
          if (!C_) throw std::bad_alloc();
          C_rows_ = static_cast<int64_t*>(sycl::malloc_host(sizeof(int64_t) * (m_ + 1), gpuQueue_));
          if (!C_rows_) throw std::bad_alloc();

          // Initialize C array pointers to nullptr
          C_cols_ = nullptr;
          C_vals_ = nullptr;

          safe_wait(gpuQueue_, "host memory allocation");
        } catch (const std::exception& e) {
          throw;
        }
      }

//      std::cout << ".. initialising input matrices";
      try {
        initInputMatrices();
      } catch (const std::exception& e) {
        std::cerr << "ERROR - Matrix initialization failed: " << e.what() << std::endl;
        throw;
      }
//      std::cout << ".. DONE" << std::endl;
    }

protected:
    void toSparseFormat() override {
      int64_t nnz_encountered = 0;

//      std::cout << ".. to sparse A";
      // Convert A to CSR format
      A_rows_[0] = 0;
      for (int64_t row = 0; row < m_; row++) {
        for (int64_t col = 0; col < k_; col++) {
          if (A_[(row * k_) + col] != 0.0) {
            A_cols_[nnz_encountered] = col;
            A_vals_[nnz_encountered] = static_cast<T>(A_[(row * k_) + col]);
            nnz_encountered++;
          }
        }
        A_rows_[row + 1] = nnz_encountered;
      }

      // Verify A conversion
      if (nnz_encountered != nnzA_) {
        std::cerr << "Warning: A matrix has " << nnz_encountered
                  << " non-zeros, expected " << nnzA_ << std::endl;
        nnzA_ = nnz_encountered;  // Update to actual count
      }

//      std::cout << " B";
      // Convert B to CSR format
      nnz_encountered = 0;
      B_rows_[0] = 0;
      for (int64_t row = 0; row < k_; row++) {
        for (int64_t col = 0; col < n_; col++) {
          if (B_[(row * n_) + col] != 0.0) {
            B_cols_[nnz_encountered] = col;
            B_vals_[nnz_encountered] = static_cast<T>(B_[(row * n_) + col]);
            nnz_encountered++;
          }
        }
        B_rows_[row + 1] = nnz_encountered;
      }

      // Verify B conversion
      if (nnz_encountered != nnzB_) {
        std::cerr << "Warning: B matrix has " << nnz_encountered
                  << " non-zeros, expected " << nnzB_ << std::endl;
        nnzB_ = nnz_encountered;  // Update to actual count
      }

//      std::cout << " and C";
      // Initialize C_rows_ for CSR format
      for (int64_t i = 0; i <= m_; i++) {
        C_rows_[i] = 0;
      }

      // Ensure synchronization for unified memory
      if (offload_ == gpuOffloadType::unified) {
        gpuQueue_.wait();
      }
    }

private:
    void preLoopRequirements() override {
      if (offload_ == gpuOffloadType::always) return;
      // Initialize the descriptor if not already done
      oneapi::mkl::sparse::init_matmat_descr(&description_);

      switch(offload_) {
        case gpuOffloadType::always: {
          // TODO -- currently empty
          break;
        }
        case gpuOffloadType::once: {
          // Allocate device memory using USM (explicit device allocation)
          A_rows_device_ = (int64_t*)sycl::malloc_device(sizeof(int64_t) * (m_ + 1), gpuQueue_);
          A_cols_device_ = (int64_t*)sycl::malloc_device(sizeof(int64_t) * nnzA_, gpuQueue_);
          A_vals_device_ = (T*)sycl::malloc_device(sizeof(T) * nnzA_, gpuQueue_);

          B_rows_device_ = (int64_t*)sycl::malloc_device(sizeof(int64_t) * (k_ + 1), gpuQueue_);
          B_cols_device_ = (int64_t*)sycl::malloc_device(sizeof(int64_t) * nnzB_, gpuQueue_);
          B_vals_device_ = (T*)sycl::malloc_device(sizeof(T) * nnzB_, gpuQueue_);

          // Copy data from host to device
          gpuQueue_.memcpy(A_rows_device_, A_rows_, sizeof(int64_t) * (m_ + 1)).wait();
          gpuQueue_.memcpy(A_cols_device_, A_cols_, sizeof(int64_t) * nnzA_).wait();
          gpuQueue_.memcpy(A_vals_device_, A_vals_, sizeof(T) * nnzA_).wait();

          gpuQueue_.memcpy(B_rows_device_, B_rows_, sizeof(int64_t) * (k_ + 1)).wait();
          gpuQueue_.memcpy(B_cols_device_, B_cols_, sizeof(int64_t) * nnzB_).wait();
          gpuQueue_.memcpy(B_vals_device_, B_vals_, sizeof(T) * nnzB_).wait();

          // Initialize matrix handles for A and B
          oneapi::mkl::sparse::init_matrix_handle(&A_handle_);
          oneapi::mkl::sparse::init_matrix_handle(&B_handle_);

          // Set CSR data for A and B with device pointers
          oneapi::mkl::sparse::set_csr_data(gpuQueue_, A_handle_, m_, k_, index_,
                                            A_rows_device_, A_cols_device_, A_vals_device_);
          oneapi::mkl::sparse::set_csr_data(gpuQueue_, B_handle_, k_, n_, index_,
                                            B_rows_device_, B_cols_device_, B_vals_device_);

          // Sort matrices to ensure they're in proper format
          oneapi::mkl::sparse::sort_matrix(gpuQueue_, A_handle_);
          oneapi::mkl::sparse::sort_matrix(gpuQueue_, B_handle_);

          // Wait to ensure data is set
          gpuQueue_.wait_and_throw();
          break;
        }
        case gpuOffloadType::unified: {
          // Initialize matrix handles for A and B only
          oneapi::mkl::sparse::init_matrix_handle(&A_handle_);
          oneapi::mkl::sparse::init_matrix_handle(&B_handle_);

          // Set CSR data for A and B
          oneapi::mkl::sparse::set_csr_data(gpuQueue_, A_handle_, m_, k_, index_,
                                            A_rows_, A_cols_, A_vals_);
          oneapi::mkl::sparse::set_csr_data(gpuQueue_, B_handle_, k_, n_, index_,
                                            B_rows_, B_cols_, B_vals_);

          // Sort matrices to ensure they're in proper format
          oneapi::mkl::sparse::sort_matrix(gpuQueue_, A_handle_);
          oneapi::mkl::sparse::sort_matrix(gpuQueue_, B_handle_);

          // Wait to ensure data is set
          gpuQueue_.wait_and_throw();
          break;
        }
      }
      safe_wait(gpuQueue_, "Pre loop requirements");
    }

    void callSpmm() override {
      switch (offload_) {
        case gpuOffloadType::always: {
          // TODO -- currently empty
          break;
        }
        case gpuOffloadType::once: {
          // Pre-allocate C arrays with conservative estimate
          int64_t max_nnzC = std::min((int64_t)(m_ * n_),
                                      std::min((int64_t)(nnzA_ * nnzB_),
                                               (int64_t)(2.0 * (nnzA_ + nnzB_))));

          // Allocate device memory for C (explicit device memory)
          int64_t* C_cols_device = (int64_t*)sycl::malloc_device(sizeof(int64_t) * max_nnzC, gpuQueue_);
          T* C_vals_device = (T*)sycl::malloc_device(sizeof(T) * max_nnzC, gpuQueue_);
          int64_t* C_rows_device = (int64_t*)sycl::malloc_device(sizeof(int64_t) * (m_ + 1), gpuQueue_);

          // Initialize C_rows on device to zeros
          gpuQueue_.memset(C_rows_device, 0, sizeof(int64_t) * (m_ + 1)).wait();

          oneapi::mkl::sparse::init_matrix_handle(&C_handle_);
          oneapi::mkl::sparse::set_csr_data(gpuQueue_, C_handle_, m_, n_,
                                            index_, C_rows_device, C_cols_device, C_vals_device);

          // Step 1: Get work estimation buffer size
          int64_t temp_buffer_size = 0;
          int64_t* temp_buffer_size_device = (int64_t*)sycl::malloc_device(sizeof(int64_t), gpuQueue_);
          void* temp_buffer = nullptr;
          std::vector<sycl::event> dependencies;

          request_ = oneapi::mkl::sparse::matmat_request::get_work_estimation_buf_size;

          try {
            auto event = oneapi::mkl::sparse::matmat(gpuQueue_, A_handle_, B_handle_,
                                                    C_handle_, request_, description_,
                                                    temp_buffer_size_device, nullptr,
                                                    dependencies);
            event.wait();

            // Copy size back to host
            gpuQueue_.memcpy(&temp_buffer_size, temp_buffer_size_device, sizeof(int64_t)).wait();
          } catch (sycl::exception const& e) {
            std::cerr << "ERROR - Work estimation buffer size: " << e.what()
                      << std::endl;
            if (temp_buffer_size_device != nullptr) {
              sycl::free(temp_buffer_size_device, gpuQueue_);
              temp_buffer_size_device = nullptr;
            }
            if (C_cols_device != nullptr) {
              sycl::free(C_cols_device, gpuQueue_);
              C_cols_device = nullptr;
            }
            if (C_vals_device != nullptr) {
              sycl::free(C_vals_device, gpuQueue_);
              C_vals_device = nullptr;
            }
            if (C_cols_device != nullptr) {
              sycl::free(C_rows_device, gpuQueue_);
              C_rows_device = nullptr;
            }
            oneapi::mkl::sparse::release_matrix_handle(gpuQueue_, &C_handle_);
            throw;
          }

          // Allocate temp buffer on device if needed
          if (temp_buffer_size > 0) {
            temp_buffer = sycl::malloc_device(temp_buffer_size, gpuQueue_);
          }

          // Step 2: Perform work estimation
          request_ = oneapi::mkl::sparse::matmat_request::work_estimation;
          try {
            auto event = oneapi::mkl::sparse::matmat(gpuQueue_, A_handle_,
                                                    B_handle_, C_handle_,
                                                    request_, description_,
                                                    temp_buffer_size_device,
                                                    temp_buffer,
                                                    dependencies);
            event.wait();
          } catch (sycl::exception const& e) {
            std::cerr << "ERROR - Work estimation: " << e.what() << std::endl;
            if (temp_buffer) {
              sycl::free(temp_buffer, gpuQueue_);
              temp_buffer = nullptr;
            }
            if (temp_buffer_size_device != nullptr) {
              sycl::free(temp_buffer_size_device, gpuQueue_);
              temp_buffer_size_device = nullptr;
            }
            if (C_cols_device != nullptr) {
              sycl::free(C_cols_device, gpuQueue_);
              C_cols_device = nullptr;
            }
            if (C_vals_device != nullptr) {
              sycl::free(C_vals_device, gpuQueue_);
              C_vals_device = nullptr;
            }
            if (C_cols_device != nullptr) {
              sycl::free(C_rows_device, gpuQueue_);
              C_rows_device = nullptr;
            }
            oneapi::mkl::sparse::release_matrix_handle(gpuQueue_, &C_handle_);
            throw;
          }

          // Step 3: Get compute buffer size
          request_ = oneapi::mkl::sparse::matmat_request::get_compute_buf_size;
          int64_t compute_buffer_size = 0;
          int64_t* compute_buffer_size_device = (int64_t*)sycl::malloc_device(sizeof(int64_t), gpuQueue_);

          try {
            auto event = oneapi::mkl::sparse::matmat(gpuQueue_, A_handle_,
                                                    B_handle_, C_handle_,
                                                    request_, description_,
                                                    compute_buffer_size_device,
                                                    nullptr,
                                                    dependencies);
            event.wait();

            // Copy size back to host
            gpuQueue_.memcpy(&compute_buffer_size, compute_buffer_size_device, sizeof(int64_t)).wait();
          } catch (sycl::exception const& e) {
            std::cerr << "ERROR - Get compute buffer size: " << e.what()
                      << std::endl;
            if (compute_buffer_size_device != nullptr) {
              sycl::free(compute_buffer_size_device, gpuQueue_);
              compute_buffer_size_device = nullptr;
            }
            if (temp_buffer) {
              sycl::free(temp_buffer, gpuQueue_);
              temp_buffer = nullptr;
            }
            if (temp_buffer_size_device != nullptr) {
              sycl::free(temp_buffer_size_device, gpuQueue_);
              temp_buffer_size_device = nullptr;
            }
            if (C_cols_device != nullptr) {
              sycl::free(C_cols_device, gpuQueue_);
              C_cols_device = nullptr;
            }
            if (C_vals_device != nullptr) {
              sycl::free(C_vals_device, gpuQueue_);
              C_vals_device = nullptr;
            }
            if (C_cols_device != nullptr) {
              sycl::free(C_rows_device, gpuQueue_);
              C_rows_device = nullptr;
            }
            oneapi::mkl::sparse::release_matrix_handle(gpuQueue_, &C_handle_);
            throw;
          }

          // Allocate compute buffer on device if needed
          void* compute_buffer = nullptr;
          if (compute_buffer_size > 0) {
            compute_buffer = sycl::malloc_device(compute_buffer_size, gpuQueue_);
          }

          // Step 4: Compute
          request_ = oneapi::mkl::sparse::matmat_request::compute;
          try {
            auto event = oneapi::mkl::sparse::matmat(gpuQueue_, A_handle_, B_handle_,
                                                    C_handle_, request_, description_,
                                                    compute_buffer_size_device,
                                                    compute_buffer,
                                                    dependencies);
            event.wait();
          } catch (sycl::exception const& e) {
            std::cerr << "ERROR - Compute: " << e.what() << std::endl;
            if (compute_buffer_size_device != nullptr) {
              sycl::free(compute_buffer_size_device, gpuQueue_);
              compute_buffer_size_device = nullptr;
            }
            if (compute_buffer != nullptr) {
              sycl::free(compute_buffer, gpuQueue_);
              compute_buffer = nullptr;
            }
            if (temp_buffer) {
              sycl::free(temp_buffer, gpuQueue_);
              temp_buffer = nullptr;
            }
            if (temp_buffer_size_device != nullptr) {
              sycl::free(temp_buffer_size_device, gpuQueue_);
              temp_buffer_size_device = nullptr;
            }
            if (C_cols_device != nullptr) {
              sycl::free(C_cols_device, gpuQueue_);
              C_cols_device = nullptr;
            }
            if (C_vals_device != nullptr) {
              sycl::free(C_vals_device, gpuQueue_);
              C_vals_device = nullptr;
            }
            if (C_cols_device != nullptr) {
              sycl::free(C_rows_device, gpuQueue_);
              C_rows_device = nullptr;
            }
            oneapi::mkl::sparse::release_matrix_handle(gpuQueue_, &C_handle_);
            throw;
          }

          // Step 5: Finalize
          request_ = oneapi::mkl::sparse::matmat_request::finalize;
          try {
            auto event = oneapi::mkl::sparse::matmat(gpuQueue_, A_handle_,
                                                     B_handle_, C_handle_,
                                                     request_, description_,
                                                     nullptr, nullptr,
                                                     dependencies);
            event.wait();
          } catch (sycl::exception const& e) {
            std::cerr << "ERROR - Finalize: " << e.what() << std::endl;
            throw;
          }

          // Copy C_rows back to host to get nnzC
          gpuQueue_.memcpy(C_rows_, C_rows_device,
                           sizeof(int64_t) * (m_ + 1)).wait();
          nnzC_ = C_rows_[m_];

          // Optional: Copy C data back to host if needed for verification
          if (nnzC_ > 0 && nnzC_ <= max_nnzC) {
            C_cols_ = (int64_t*)sycl::malloc_host(sizeof(int64_t) * nnzC_, gpuQueue_);
            C_vals_ = (T*)sycl::malloc_host(sizeof(T) * nnzC_, gpuQueue_);

            gpuQueue_.memcpy(C_rows_, C_rows_device, sizeof(int64_t) * (m_ + 1)).wait();
            gpuQueue_.memcpy(C_cols_, C_cols_device, sizeof(int64_t) * nnzC_).wait();
            gpuQueue_.memcpy(C_vals_, C_vals_device, sizeof(T) * nnzC_).wait();
          }

          // Clean up device memory
          if (temp_buffer != nullptr) {
//            std::cout << ".. freeing temp_buffer";
            sycl::free(temp_buffer, gpuQueue_);
            temp_buffer = nullptr;
          }
          if (compute_buffer != nullptr) {
//            std::cout << ".. freeing compute_buffer";
            sycl::free(compute_buffer, gpuQueue_);
            compute_buffer = nullptr;
          }
          if (temp_buffer_size_device != nullptr) {
//            std::cout << ".. freeing temp_buffer_size_device";
            sycl::free(temp_buffer_size_device, gpuQueue_);
            temp_buffer_size_device = nullptr;
          }
          if (compute_buffer_size_device != nullptr) {
//            std::cout << ".. freeing compute_buffer_size_device";
            sycl::free(compute_buffer_size_device, gpuQueue_);
            compute_buffer_size_device = nullptr;
          }
          if (C_cols_device != nullptr) {
//            std::cout << ".. freeing C_cols_device";
            sycl::free(C_cols_device, gpuQueue_);
            C_cols_device = nullptr;
          }
          if (C_vals_device != nullptr) {
//            std::cout << ".. freeing C_vals_device";
            sycl::free(C_vals_device, gpuQueue_);
            C_vals_device = nullptr;
          }
          if (C_rows_device != nullptr) {
//            std::cout << ".. freeing C_rows_device";
            sycl::free(C_rows_device, gpuQueue_);
            C_rows_device = nullptr;
          }
          // Release C handle
          oneapi::mkl::sparse::release_matrix_handle(gpuQueue_, &C_handle_);
          if (C_cols_ != nullptr) {
            sycl::free(C_cols_, gpuQueue_);
            C_cols_ = nullptr;
          }
          if (C_vals_ != nullptr) {
            sycl::free(C_vals_, gpuQueue_);
            C_vals_ = nullptr;
          }

          break;
        }
        case gpuOffloadType::unified: {
          // Unified memory implementation with proper memory management
          int64_t temp_buffer_size = 0;
          void* temp_buffer = nullptr;
          std::vector<sycl::event> dependencies;

          // Initialize C matrix handle for this iteration
          oneapi::mkl::sparse::init_matrix_handle(&C_handle_);

          // Pre-allocate C arrays with conservative estimate
          int64_t max_nnzC = std::min((int64_t)(m_ * n_),
                                      std::min((int64_t)(nnzA_ * nnzB_),
                                               (int64_t)(2.0 * (nnzA_ + nnzB_))));

          // Allocate C arrays
          C_vals_ = (T*)sycl::malloc_shared(sizeof(T) * max_nnzC, gpuQueue_);
          C_cols_ = (int64_t*)sycl::malloc_shared(sizeof(int64_t) * max_nnzC,
                                                  gpuQueue_);

          // Initialize C_rows to zero
          gpuQueue_.memset(C_rows_, 0, sizeof(int64_t) * (m_ + 1)).wait();

          // Set CSR data for C with pre-allocated arrays
          oneapi::mkl::sparse::set_csr_data(gpuQueue_, C_handle_, m_, n_,
                                            index_, C_rows_, C_cols_, C_vals_);

          // Step 1: Work estimation to determine C structure
          request_ = oneapi::mkl::sparse::matmat_request::get_work_estimation_buf_size;
          try {
            auto event = oneapi::mkl::sparse::matmat(gpuQueue_,
                                                     A_handle_,
                                                     B_handle_,
                                                     C_handle_,
                                                     request_,
                                                     description_,
                                                     &temp_buffer_size,
                                                     temp_buffer,
                                                     dependencies);
            event.wait();
          } catch (sycl::exception const& e) {
            std::cerr << "ERROR - Work estimation buffer size: " << e.what()
            << std::endl;
            oneapi::mkl::sparse::release_matrix_handle(gpuQueue_, &C_handle_);
            throw;
          }

          // Allocate temporary buffer if needed
          if (temp_buffer_size > 0) {
            temp_buffer = sycl::malloc_shared(temp_buffer_size, gpuQueue_);
          } else {
            temp_buffer = nullptr;
          }

          // Step 2: Perform work estimation
          request_ = oneapi::mkl::sparse::matmat_request::work_estimation;
          try {
            auto event = oneapi::mkl::sparse::matmat(gpuQueue_, A_handle_,
                                                     B_handle_, C_handle_,
                                                     request_, description_,
                                                     &temp_buffer_size,
                                                     temp_buffer, dependencies);
            event.wait();
          } catch (sycl::exception const& e) {
            std::cerr << "ERROR - Work estimation: " << e.what() << std::endl;
            if (temp_buffer) {
              sycl::free(temp_buffer, gpuQueue_);
              temp_buffer = nullptr;
            }
            oneapi::mkl::sparse::release_matrix_handle(gpuQueue_, &C_handle_);
            throw;
          }

          // Step 3: Get compute buffer size
          request_ = oneapi::mkl::sparse::matmat_request::get_compute_buf_size;
          int64_t compute_buffer_size = 0;
          try {
            auto event = oneapi::mkl::sparse::matmat(gpuQueue_, A_handle_,
                                                     B_handle_, C_handle_,
                                                     request_, description_,
                                                     &compute_buffer_size,
                                                     nullptr, dependencies);
            event.wait();
          } catch (sycl::exception const& e) {
            std::cerr << "ERROR - Get compute buffer size: " << e.what()
            << std::endl;
            if (temp_buffer) {
              sycl::free(temp_buffer, gpuQueue_);
              temp_buffer = nullptr;
            }
            oneapi::mkl::sparse::release_matrix_handle(gpuQueue_, &C_handle_);
            throw;
          }

          // Allocate compute buffer if needed (separate from work estimation buffer)
          void* compute_buffer = nullptr;
          if (compute_buffer_size > 0) {
            compute_buffer = sycl::malloc_shared(compute_buffer_size, gpuQueue_);
          } else {
            compute_buffer = nullptr;
          }

          // Step 4: Compute
          request_ = oneapi::mkl::sparse::matmat_request::compute;
          try {
            auto event = oneapi::mkl::sparse::matmat(gpuQueue_, A_handle_,
                                                     B_handle_, C_handle_,
                                                     request_, description_,
                                                     &compute_buffer_size,
                                                     compute_buffer, dependencies);
            event.wait();
          } catch (sycl::exception const& e) {
            std::cerr << "ERROR - Compute: " << e.what() << std::endl;
            if (temp_buffer) {
              sycl::free(temp_buffer, gpuQueue_);
              temp_buffer = nullptr;
            }
            if (compute_buffer) {
              sycl::free(compute_buffer, gpuQueue_);
              compute_buffer = nullptr;
            }
            oneapi::mkl::sparse::release_matrix_handle(gpuQueue_, &C_handle_);
            throw;
          }

          // Step 5: Finalize
          request_ = oneapi::mkl::sparse::matmat_request::finalize;
          try {
            auto event = oneapi::mkl::sparse::matmat(gpuQueue_, A_handle_,
                                                     B_handle_, C_handle_,
                                                     request_, description_,
                                                     nullptr, nullptr,
                                                     dependencies);
            event.wait();
          } catch (sycl::exception const& e) {
            std::cerr << "ERROR - Finalize: " << e.what() << std::endl;
          }

          // Get actual nnzC after computation
          gpuQueue_.wait();
          nnzC_ = C_rows_[m_];

          try {
            // Clean up temporary buffers
            if (temp_buffer) {
//              std::cout << ".. freeing temp_buffer";
              sycl::free(temp_buffer, gpuQueue_);
              temp_buffer = nullptr;
            }
            if (compute_buffer) {
//              std::cout << ".. freeing compute_buffer";
              sycl::free(compute_buffer, gpuQueue_);
              compute_buffer = nullptr;
            }
            // Free previous allocations if they exist
            if (C_vals_) {
//              std::cout << ".. freeing C_vals_";
              sycl::free(C_vals_, gpuQueue_);
              C_vals_ = nullptr;
            }
            if (C_cols_) {
//              std::cout << ".. freeing C_cols_";
              sycl::free(C_cols_, gpuQueue_);
              C_cols_ = nullptr;
            }
            // Release C handle
            oneapi::mkl::sparse::release_matrix_handle(gpuQueue_, &C_handle_);
          } catch (sycl::exception const& e) {
            std::cerr << "ERROR - Cleaning up callSpmm: " << e.what() <<
            std::endl;
          }
          break;
        }
      }
      safe_wait(gpuQueue_, "Spmm call");
    }

    void postLoopRequirements() override {
      switch(offload_) {
        case gpuOffloadType::always: {
          // TODO -- currently empty
          break;
        }
        case gpuOffloadType::once: {
          // Release matrix handles
          try {
//            std::cout << ".. releasing A_handle_";
            oneapi::mkl::sparse::release_matrix_handle(gpuQueue_, &A_handle_);
//            std::cout << ".. releasing B_handle_";
            oneapi::mkl::sparse::release_matrix_handle(gpuQueue_, &B_handle_);
//            std::cout << ".. releasing matmat decription";
            oneapi::mkl::sparse::release_matmat_descr(&description_);
            // Free device memory
            if (A_rows_device_) {
//              std::cout << ".. freeing A_rows_device_";
              sycl::free(A_rows_device_, gpuQueue_);
              A_rows_device_ = nullptr;
            }
            if (A_cols_device_) {
//              std::cout << ".. freeing A_cols_device_";
              sycl::free(A_cols_device_, gpuQueue_);
              A_cols_device_ = nullptr;
            }
            if (A_vals_device_) {
//              std::cout << ".. freeing A_vals_device_";
              sycl::free(A_vals_device_, gpuQueue_);
              A_vals_device_ = nullptr;
            }
            if (B_rows_device_) {
//              std::cout << ".. freeing B_rows_device_";
              sycl::free(B_rows_device_, gpuQueue_);
              B_rows_device_ = nullptr;
            }
            if (B_cols_device_) {
//              std::cout << ".. freeing B_cols_device_";
              sycl::free(B_cols_device_, gpuQueue_);
              B_cols_device_ = nullptr;
            }
            if (B_vals_device_) {
//              std::cout << ".. freeing B_vals_device_";
              sycl::free(B_vals_device_, gpuQueue_);
              B_vals_device_ = nullptr;
            }
            gpuQueue_.wait();
          } catch  (sycl::exception const& e) {
            std::cerr << "ERROR - Cleaning up device: " << e.what() <<
            std::endl;
          }

          break;
        }
        case gpuOffloadType::unified: {
          // Release matrix handles
          try {
            oneapi::mkl::sparse::release_matmat_descr(&description_);
            if (A_handle_) {
//              std::cout << ".. releasing A_handle_";
              oneapi::mkl::sparse::release_matrix_handle(gpuQueue_, &A_handle_);
              A_handle_ = nullptr;
            }
            if (B_handle_) {
//              std::cout << ".. releasing B_handle_";
              oneapi::mkl::sparse::release_matrix_handle(gpuQueue_, &B_handle_);
              B_handle_ = nullptr;
            }
            gpuQueue_.wait();
          } catch  (sycl::exception const& e) {
            std::cerr << "ERROR - Cleaning up handles: " << e.what() <<
            std::endl;
          }
          break;
        }
      }
      safe_wait(gpuQueue_, "Post loop requirements");
    }

    void postCallKernelCleanup() override {
      if (offload_ == gpuOffloadType::unified) {
        if (A_) {
          sycl::free(A_, gpuQueue_);
          A_ = nullptr;
        }
        if (A_vals_) {
          sycl::free(A_vals_, gpuQueue_);
          A_vals_ = nullptr;
        }
        if (A_cols_) {
          sycl::free(A_cols_, gpuQueue_);
          A_cols_ = nullptr;
        }
        if (A_rows_) {
          sycl::free(A_rows_, gpuQueue_);
          A_rows_ = nullptr;
        }
        if (B_) {
          sycl::free(B_, gpuQueue_);
          B_ = nullptr;
        }
        if (B_vals_) {
          sycl::free(B_vals_, gpuQueue_);
          B_vals_ = nullptr;
        }
        if (B_cols_) {
          sycl::free(B_cols_, gpuQueue_);
          B_cols_ = nullptr;
        }
        if (B_rows_) {
          sycl::free(B_rows_, gpuQueue_);
          B_rows_ = nullptr;
        }
        if (C_) {
          sycl::free(C_, gpuQueue_);
          C_ = nullptr;
        }
        if (C_rows_) {
          sycl::free(C_rows_, gpuQueue_);
          C_rows_ = nullptr;
        }
      } else if (offload_ == gpuOffloadType::once) {
        if (A_) {
          sycl::free(A_, gpuQueue_);
          A_ = nullptr;
        }
        if (A_vals_) {
          sycl::free(A_vals_, gpuQueue_);
          A_vals_ = nullptr;
        }
        if (A_cols_) {
          sycl::free(A_cols_, gpuQueue_);
          A_cols_ = nullptr;
        }
        if (A_rows_) {
          sycl::free(A_rows_, gpuQueue_);
          A_rows_ = nullptr;
        }
        if (B_) {
          sycl::free(B_, gpuQueue_);
          B_ = nullptr;
        }
        if (B_vals_) {
          sycl::free(B_vals_, gpuQueue_);
          B_vals_ = nullptr;
        }
        if (B_cols_) {
          sycl::free(B_cols_, gpuQueue_);
          B_cols_ = nullptr;
        }
        if (B_rows_) {
          sycl::free(B_rows_, gpuQueue_);
          B_rows_ = nullptr;
        }
        if (C_) {
          sycl::free(C_, gpuQueue_);
          C_ = nullptr;
        }
        if (C_rows_) {
          sycl::free(C_rows_, gpuQueue_);
          C_rows_ = nullptr;
        }
      }
    }

    // Template function to allocate shared memory
    template <typename U>
    U* safe_malloc_shared(size_t size, sycl::queue& q, const std::string& var_name) {
      try {
        U* ptr = sycl::malloc_shared<U>(size, q);
        if (!ptr) {
          throw std::runtime_error("Failed to allocate shared memory for " + var_name);
        }
        return ptr;
      } catch (const sycl::exception& e) {
        std::cerr << "SYCL allocation error for " << var_name << ": " << e.what() << std::endl;
        throw;
      }
    }

    void safe_wait(sycl::queue& q, const std::string& operation) {
      try {
        q.wait_and_throw();
      } catch (const sycl::exception& e) {
        std::cerr << "SYCL synchronization error during " << operation << ": " << e.what() << std::endl;
        throw;
      }
    }



    // Member variables
    bool alreadyInitialised_ = false;
    bool descriptor_initialized_ = false;


    sycl::device myGpu_;
    sycl::queue gpuQueue_;

    oneapi::mkl::index_base index_;
    oneapi::mkl::transpose operationA_;
    oneapi::mkl::transpose operationB_;
    oneapi::mkl::sparse::matmat_request request_;
    oneapi::mkl::sparse::matmat_descr_t description_;
    oneapi::mkl::layout layout_;

    // Matrix data pointers (host memory)
    T* A_vals_ = nullptr;
    int64_t* A_cols_ = nullptr;
    int64_t* A_rows_ = nullptr;

    T* B_vals_ = nullptr;
    int64_t* B_cols_ = nullptr;
    int64_t* B_rows_ = nullptr;

    T* C_vals_ = nullptr;
    int64_t* C_cols_ = nullptr;
    int64_t* C_rows_ = nullptr;

    // Device memory pointers (for 'once' mode)
    T* A_vals_device_ = nullptr;
    int64_t* A_cols_device_ = nullptr;
    int64_t* A_rows_device_ = nullptr;

    T* B_vals_device_ = nullptr;
    int64_t* B_cols_device_ = nullptr;
    int64_t* B_rows_device_ = nullptr;

    // Matrix handles
    oneapi::mkl::sparse::matrix_handle_t A_handle_ = nullptr;
    oneapi::mkl::sparse::matrix_handle_t B_handle_ = nullptr;
    oneapi::mkl::sparse::matrix_handle_t C_handle_ = nullptr;

    const T alpha = ALPHA;
    const T beta = BETA;
};
}

#endif
