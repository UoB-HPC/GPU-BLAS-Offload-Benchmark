#pragma once

#ifdef GPU_ONEMKL

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
      if (descriptor_initialized_) {
        oneapi::mkl::sparse::release_matmat_descr(&description_);
        descriptor_initialized_ = false;
      }
    }

    void initialise(gpuOffloadType offload, int m, int n, int k,
                    double sparsity, bool binary = false) override {
      std::cout << ".. checking already init";
      if (!alreadyInitialised_) {
        alreadyInitialised_ = true;
        // Perform set-up which doesn't need to happen every problem size change.
        try {
          myGpu_ = sycl::device(sycl::gpu_selector_v);
        } catch (const std::exception& e) {
          std::cerr << "ERROR - No GPU device found: " << e.what() << std::endl;
          std::terminate();
        }
        gpuQueue_ = sycl::queue(myGpu_, exception_handler);
      }

      std::cout << ".. setting metadata";
      offload_ = offload;
      sparsity_ = sparsity;
      m_ = m;
      n_ = n;
      k_ = k;

      layout_ = oneapi::mkl::layout::row_major;
      operationA_ = oneapi::mkl::transpose::nontrans;
      operationB_ = oneapi::mkl::transpose::nontrans;
      index_ = oneapi::mkl::index_base::zero;

      nnzA_ = 1 + (uint64_t)((double)m_ * (double)k_ * (1.0 - sparsity_));
      nnzB_ = 1 + (uint64_t)((double)k_ * (double)n_ * (1.0 - sparsity_));

      // For unified memory, don't pre-allocate C arrays
      if (offload_ == gpuOffloadType::unified) {
        std::cout <<".. unified malloc";
        A_ = (T*)sycl::malloc_shared(sizeof(T) * m_ * k_, gpuQueue_);
        A_vals_ = (T*)sycl::malloc_shared(sizeof(T) * nnzA_, gpuQueue_);
        A_cols_ = (int64_t*)sycl::malloc_shared(sizeof(int64_t) * nnzA_,
                                                gpuQueue_);
        A_rows_ = (int64_t*)sycl::malloc_shared(sizeof(int64_t) * (m_ + 1),
                                                gpuQueue_);
        gpuQueue_.wait_and_throw();

        B_ = (T*)sycl::malloc_shared(sizeof(T) * k_ * n_, gpuQueue_);
        B_vals_ = (T*)sycl::malloc_shared(sizeof(T) * nnzB_, gpuQueue_);
        B_cols_ = (int64_t*)sycl::malloc_shared(sizeof(int64_t) * nnzB_,
                                                gpuQueue_);
        B_rows_ = (int64_t*)sycl::malloc_shared(sizeof(int64_t) * (k_ + 1),
                                                gpuQueue_);
        gpuQueue_.wait_and_throw();

        C_ = (T*)sycl::malloc_shared(sizeof(T) * m_ * n_, gpuQueue_);
        C_rows_ = (int64_t*)sycl::malloc_shared(sizeof(int64_t) * (m_ + 1),
                                                gpuQueue_);
        // Don't pre-allocate C_cols_ and C_vals_ for unified memory
        C_cols_ = nullptr;
        C_vals_ = nullptr;
        gpuQueue_.wait_and_throw();

      } else {
        std::cout << ".. host malloc";
        A_ = (T*)sycl::malloc_host(sizeof(T) * m_ * k_, gpuQueue_);
        A_vals_ = (T*)sycl::malloc_host(sizeof(T) * nnzA_, gpuQueue_);
        A_cols_ = (int64_t*)sycl::malloc_host(sizeof(int64_t) * nnzA_,
                                              gpuQueue_);
        A_rows_ = (int64_t*)sycl::malloc_host(sizeof(int64_t) * (m_ + 1),
                                              gpuQueue_);
        gpuQueue_.wait_and_throw();

        B_ = (T*)sycl::malloc_host(sizeof(T) * k_ * n_, gpuQueue_);
        B_vals_ = (T*)sycl::malloc_host(sizeof(T) * nnzB_, gpuQueue_);
        B_cols_ = (int64_t*)sycl::malloc_host(sizeof(int64_t) * nnzB_,
                                              gpuQueue_);
        B_rows_ = (int64_t*)sycl::malloc_host(sizeof(int64_t) * (k_ + 1),
                                              gpuQueue_);
        gpuQueue_.wait_and_throw();

        C_ = (T*)sycl::malloc_host(sizeof(T) * m_ * n_, gpuQueue_);
        C_rows_ = (int64_t*)sycl::malloc_host(sizeof(int64_t) * (m_ + 1),
                                              gpuQueue_);
        gpuQueue_.wait_and_throw();
      }

      std::cout << ".. initialising input matrices";
      initInputMatrices();
      gpuQueue_.wait_and_throw();
      std::cout << ".. DONE";
    }

protected:
    void toSparseFormat() override {
      int64_t nnz_encountered = 0;

      std::cout << ".. to sparse A";
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

      std::cout << " B";
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

      std::cout << " and C";
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
      // Initialize the descriptor if not already done
      if (!descriptor_initialized_) {
        oneapi::mkl::sparse::init_matmat_descr(&description_);
        descriptor_initialized_ = true;
      }

      switch(offload_) {
        case gpuOffloadType::always: {
          // Nothing to do here - handles created in callSpmm
          break;
        }
        case gpuOffloadType::once: {
          // Create buffers and initialize matrix handles
          A_vals_device_ = new sycl::buffer<T, 1>(A_vals_, sycl::range<1>(nnzA_));
          A_cols_device_ = new sycl::buffer<int64_t, 1>(A_cols_, sycl::range<1>(nnzA_));
          A_rows_device_ = new sycl::buffer<int64_t, 1>(A_rows_, sycl::range<1>(m_ + 1));

          oneapi::mkl::sparse::init_matrix_handle(&A_device_);
          oneapi::mkl::sparse::set_csr_data(gpuQueue_, A_device_, m_, k_, index_,
                                            *A_rows_device_, *A_cols_device_, *A_vals_device_);

          B_vals_device_ = new sycl::buffer<T, 1>(B_vals_, sycl::range<1>(nnzB_));
          B_cols_device_ = new sycl::buffer<int64_t, 1>(B_cols_, sycl::range<1>(nnzB_));
          B_rows_device_ = new sycl::buffer<int64_t, 1>(B_rows_, sycl::range<1>(k_ + 1));

          oneapi::mkl::sparse::init_matrix_handle(&B_device_);
          oneapi::mkl::sparse::set_csr_data(gpuQueue_, B_device_, k_, n_, index_,
                                            *B_rows_device_, *B_cols_device_, *B_vals_device_);

          C_rows_device_ = new sycl::buffer<int64_t, 1>(C_rows_, sycl::range<1>(m_ + 1));

          gpuQueue_.wait_and_throw();
          break;
        }
        case gpuOffloadType::unified: {
          // Initialize matrix handles for A and B only
          oneapi::mkl::sparse::init_matrix_handle(&A_device_);
          oneapi::mkl::sparse::init_matrix_handle(&B_device_);

          // Set CSR data for A and B
          oneapi::mkl::sparse::set_csr_data(gpuQueue_, A_device_, m_, k_, index_,
                                            A_rows_, A_cols_, A_vals_);
          oneapi::mkl::sparse::set_csr_data(gpuQueue_, B_device_, k_, n_, index_,
                                            B_rows_, B_cols_, B_vals_);

          // Sort matrices to ensure they're in proper format
          oneapi::mkl::sparse::sort_matrix(gpuQueue_, A_device_);
          oneapi::mkl::sparse::sort_matrix(gpuQueue_, B_device_);

          // Wait to ensure data is set
          gpuQueue_.wait_and_throw();
          break;
        }
      }
    }

    void callSpmm() override {
      switch (offload_) {
        case gpuOffloadType::always: {
          // Transfer data to the GPU, and set up data structures
          A_vals_device_ = new sycl::buffer<T, 1>(A_vals_,
                                                  sycl::range<1>(nnzA_));
          A_cols_device_ = new sycl::buffer<int64_t, 1>(A_cols_,
                                                        sycl::range<1>(nnzA_));
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
          oneapi::mkl::sparse::sort_matrix(gpuQueue_,
                                           A_device_);

          B_vals_device_ = new sycl::buffer<T, 1>(B_vals_,
                                                  sycl::range<1>(nnzB_));
          B_cols_device_ = new sycl::buffer<int64_t, 1>(B_cols_,
                                                        sycl::range<1>(nnzB_));
          B_rows_device_ = new sycl::buffer<int64_t, 1>(B_rows_,
                                                        sycl::range<1>(k_ + 1));

          oneapi::mkl::sparse::init_matrix_handle(&B_device_);
          oneapi::mkl::sparse::set_csr_data(gpuQueue_,
                                            B_device_,
                                            k_,
                                            n_,
                                            index_,
                                            *B_rows_device_,
                                            *B_cols_device_,
                                            *B_vals_device_);
          oneapi::mkl::sparse::sort_matrix(gpuQueue_,
                                           B_device_);

          C_rows_device_ = new sycl::buffer<int64_t, 1>(C_rows_,
                                                        sycl::range<1>(m_ + 1));

          oneapi::mkl::sparse::init_matrix_handle(&C_device_);
          oneapi::mkl::sparse::set_csr_data(gpuQueue_,
                                            C_device_,
                                            m_,
                                            n_,
                                            index_,
                                            *C_rows_device_,
                                            *C_cols_device_,
                                            *C_vals_device_);
          gpuQueue_.wait_and_throw();

          // Do computation
          request_ = oneapi::mkl::sparse::matmat_request
                  ::get_work_estimation_buf_size;
          try {
            oneapi::mkl::sparse::matmat(gpuQueue_,
                                        A_device_,
                                        B_device_,
                                        C_device_,
                                        request_,
                                        description_,
                                        device_temp_buffer_1_size_,
                                        device_temp_buffer_1_);
          } catch (sycl::exception const& e) {
            std::cout << "ERROR - Caught synchronous SYCL exception during "
                         "SPMM (Always):\n"
                      << e.what() << std::endl
                      << "OpenCL status: " << e.code().value() << std::endl;
          }

          request_ = oneapi::mkl::sparse::matmat_request
                  ::get_work_estimation_buf_size;
          try {
            oneapi::mkl::sparse::matmat(gpuQueue_,
                                        A_device_,
                                        B_device_,
                                        C_device_,
                                        request_,
                                        description_,
                                        device_temp_buffer_2_size_,
                                        device_temp_buffer_2_);
          } catch (sycl::exception const& e) {
            std::cout << "ERROR - Caught synchronous SYCL exception during "
                         "SPMM (Always):\n"
                      << e.what() << std::endl
                      << "OpenCL status: " << e.code().value() << std::endl;
          }

          request_ = oneapi::mkl::sparse::matmat_request
                  ::get_work_estimation_buf_size;
          try {
            oneapi::mkl::sparse::matmat(gpuQueue_,
                                        A_device_,
                                        B_device_,
                                        C_device_,
                                        request_,
                                        description_,
                                        NULL,
                                        NULL);
          } catch (sycl::exception const& e) {
            std::cout << "ERROR - Caught synchronous SYCL exception during "
                         "SPMM (Always):\n"
                      << e.what() << std::endl
                      << "OpenCL status: " << e.code().value() << std::endl;
          }
          // Do cleanup
          oneapi::mkl::sparse::release_matrix_handle(gpuQueue_, &A_device_);
          oneapi::mkl::sparse::release_matrix_handle(gpuQueue_, &B_device_);
          oneapi::mkl::sparse::release_matrix_handle(gpuQueue_, &C_device_);

          delete A_vals_device_;
          delete A_cols_device_;
          delete A_rows_device_;
          delete B_vals_device_;
          delete B_cols_device_;
          delete B_rows_device_;
          delete C_vals_device_;
          delete C_cols_device_;
          delete C_rows_device_;

          break;
        }
        case gpuOffloadType::once: {
          /**
           * STEP 1 -- Allocate C amtrix row pointer and C matrix handle
           */
          oneapi::mkl::sparse::init_matrix_handle(&C_device_);
          oneapi::mkl::sparse::set_csr_data(gpuQueue_,
                                            C_device_,
                                            m_,
                                            n_,
                                            index_,
                                            *C_rows_device_,
                                            *C_cols_device_,
                                            *C_vals_device_);

          /**
           * STEP 2 -- Work estimation
           */
          request_ = oneapi::mkl::sparse::matmat_request
                  ::get_work_estimation_buf_size;
          try {
            oneapi::mkl::sparse::matmat(gpuQueue_,
                                        A_device_,
                                        B_device_,
                                        C_device_,
                                        request_,
                                        description_,
                                        device_temp_buffer_1_size_,
                                        device_temp_buffer_1_);
          } catch (sycl::exception const& e) {
            std::cout << "ERROR - Caught synchronous SYCL exception during "
                         "SPMM (Once):\n"
                      << e.what() << std::endl
                      << "OpenCL status: " << e.code().value() << std::endl;
          }

          /**
           * STEP 3 -- Compute
           */
          request_ = oneapi::mkl::sparse::matmat_request
                  ::get_work_estimation_buf_size;
          try {
            oneapi::mkl::sparse::matmat(gpuQueue_,
                                        A_device_,
                                        B_device_,
                                        C_device_,
                                        request_,
                                        description_,
                                        device_temp_buffer_2_size_,
                                        device_temp_buffer_2_);
          } catch (sycl::exception const& e) {
            std::cout << "ERROR - Caught synchronous SYCL exception during "
                         "SPMM (Once):\n"
                      << e.what() << std::endl
                      << "OpenCL status: " << e.code().value() << std::endl;
          }

          /**
           * STEP 4 -- Finalisation
           */
          request_ = oneapi::mkl::sparse::matmat_request
                  ::get_work_estimation_buf_size;
          try {
            oneapi::mkl::sparse::matmat(gpuQueue_,
                                        A_device_,
                                        B_device_,
                                        C_device_,
                                        request_,
                                        description_,
                                        NULL,
                                        NULL);
          } catch (sycl::exception const& e) {
            std::cout << "ERROR - Caught synchronous SYCL exception during "
                         "SPMM (Once):\n"
                      << e.what() << std::endl
                      << "OpenCL status: " << e.code().value() << std::endl;
          }

          /**
           * STEP 5 -- Releasing C
           */
          oneapi::mkl::sparse::release_matrix_handle(gpuQueue_,
                                                     &C_device_);

          break;
        }
        case gpuOffloadType::unified: {
          // Unified memory implementation with proper memory management
          int64_t temp_buffer_size = 0;
          void* temp_buffer = nullptr;
          std::vector<sycl::event> dependencies;

          // Initialize C matrix handle for this iteration
          oneapi::mkl::sparse::init_matrix_handle(&C_device_);

          // Step 1: Work estimation to determine C structure
          // First, get the work estimation buffer size
          request_ = oneapi::mkl::sparse::matmat_request::get_work_estimation_buf_size;
          try {
            auto event = oneapi::mkl::sparse::matmat(gpuQueue_,
                                                     A_device_,
                                                     B_device_,
                                                     C_device_,
                                                     request_,
                                                     description_,
                                                     &temp_buffer_size,
                                                     temp_buffer,
                                                     dependencies);
            event.wait();
          } catch (sycl::exception const& e) {
            std::cerr << "ERROR - Work estimation buffer size: " << e.what()
                      << std::endl;
            oneapi::mkl::sparse::release_matrix_handle(gpuQueue_, &C_device_);
            throw;
          }

          // Allocate temporary buffer if needed
          if (temp_buffer_size > 0) {
            temp_buffer = sycl::malloc_shared(temp_buffer_size, gpuQueue_);
          }

          // Step 2: Perform work estimation
          request_ = oneapi::mkl::sparse::matmat_request::work_estimation;
          try {
            auto event = oneapi::mkl::sparse::matmat(gpuQueue_, A_device_,
                                                     B_device_, C_device_,
                                                     request_, description_,
                                                     &temp_buffer_size,
                                                     temp_buffer, dependencies);
            event.wait();
          } catch (sycl::exception const& e) {
            std::cerr << "ERROR - Work estimation: " << e.what() << std::endl;
            if (temp_buffer) sycl::free(temp_buffer, gpuQueue_);
            oneapi::mkl::sparse::release_matrix_handle(gpuQueue_, &C_device_);
            throw;
          }

          // Step 3: Query the number of non-zeros in C
          // This is implementation-specific - you may need to query the handle
          // or use a different API call to get the expected nnzC
          // For now, we'll use a conservative estimate
          int64_t expected_nnzC = std::min((int64_t)(m_ * n_),
                                          (int64_t)(1.5 * (nnzA_ + nnzB_)));

          // Allocate C arrays based on expected size
          if (C_vals_ != nullptr) {
            sycl::free(C_vals_, gpuQueue_);
          }
          if (C_cols_ != nullptr) {
            sycl::free(C_cols_, gpuQueue_);
          }
          C_vals_ = (T*)sycl::malloc_shared(sizeof(T) * expected_nnzC, gpuQueue_);
          C_cols_ = (int64_t*)sycl::malloc_shared(sizeof(int64_t) * expected_nnzC,
                                                  gpuQueue_);
          gpuQueue_.wait();

          // Step 4: Set CSR data for C with newly allocated arrays
          oneapi::mkl::sparse::set_csr_data(gpuQueue_, C_device_, m_, n_,
                                            index_, C_rows_, C_cols_, C_vals_);

          // Step 5: Get compute buffer size
          request_ = oneapi::mkl::sparse::matmat_request::get_compute_buf_size;
          try {
            auto event = oneapi::mkl::sparse::matmat(gpuQueue_, A_device_,
                                                     B_device_, C_device_,
                                                     request_, description_,
                                                     &temp_buffer_size,
                                                     temp_buffer, dependencies);
            event.wait();
          } catch (sycl::exception const& e) {
            std::cerr << "ERROR - Get compute buffer size: " << e.what()
                      << std::endl;
            if (temp_buffer) sycl::free(temp_buffer, gpuQueue_);
            oneapi::mkl::sparse::release_matrix_handle(gpuQueue_, &C_device_);
            throw;
          }

          // Reallocate temp buffer if size changed
          if (temp_buffer) {
            sycl::free(temp_buffer, gpuQueue_);
            temp_buffer = nullptr;
          }
          if (temp_buffer_size > 0) {
            temp_buffer = sycl::malloc_shared(temp_buffer_size, gpuQueue_);
          }

          // Step 6: Compute
          request_ = oneapi::mkl::sparse::matmat_request::compute;
          try {
            auto event = oneapi::mkl::sparse::matmat(gpuQueue_, A_device_,
                                                     B_device_, C_device_,
                                                     request_, description_,
                                                     &temp_buffer_size,
                                                     temp_buffer, dependencies);
            event.wait();
          } catch (sycl::exception const& e) {
            std::cerr << "ERROR - Compute: " << e.what() << std::endl;
            if (temp_buffer) sycl::free(temp_buffer, gpuQueue_);
            oneapi::mkl::sparse::release_matrix_handle(gpuQueue_, &C_device_);
            throw;
          }

          // Step 7: Finalize
          request_ = oneapi::mkl::sparse::matmat_request::finalize;
          try {
            auto event = oneapi::mkl::sparse::matmat(gpuQueue_, A_device_, B_device_,
                                                     C_device_, request_, description_,
                                                     nullptr, nullptr, dependencies);
            event.wait();
          } catch (sycl::exception const& e) {
            std::cerr << "ERROR - Finalize: " << e.what() << std::endl;
          }

          // Get actual nnzC
          gpuQueue_.wait();
          nnzC_ = C_rows_[m_];

          // Clean up temporary buffer
          if (temp_buffer) {
            sycl::free(temp_buffer, gpuQueue_);
          }

          // Release C handle
          oneapi::mkl::sparse::release_matrix_handle(gpuQueue_, &C_device_);

          // Free C arrays after each iteration to prevent memory accumulation
          if (C_vals_) {
            sycl::free(C_vals_, gpuQueue_);
            C_vals_ = nullptr;
          }
          if (C_cols_) {
            sycl::free(C_cols_, gpuQueue_);
            C_cols_ = nullptr;
          }

          break;
        }
      }
    }

    void postLoopRequirements() override {
      switch(offload_) {
        case gpuOffloadType::always: {
          // Nothing to do - handles are created/destroyed in callSpmm
          break;
        }
        case gpuOffloadType::once: {
          // Release matrix handles and delete buffers
          oneapi::mkl::sparse::release_matrix_handle(gpuQueue_, &A_device_);
          oneapi::mkl::sparse::release_matrix_handle(gpuQueue_, &B_device_);

          delete A_vals_device_;
          delete A_cols_device_;
          delete A_rows_device_;
          delete B_vals_device_;
          delete B_cols_device_;
          delete B_rows_device_;
          delete C_rows_device_;

          // Note: C_vals_device_ and C_cols_device_ might not be allocated
          if (C_vals_device_) delete C_vals_device_;
          if (C_cols_device_) delete C_cols_device_;
          break;
        }
        case gpuOffloadType::unified: {
          // Release A and B handles
          oneapi::mkl::sparse::release_matrix_handle(gpuQueue_, &A_device_);
          oneapi::mkl::sparse::release_matrix_handle(gpuQueue_, &B_device_);

          // Ensure C arrays are freed if they haven't been already
          if (C_vals_) {
            sycl::free(C_vals_, gpuQueue_);
            C_vals_ = nullptr;
          }
          if (C_cols_) {
            sycl::free(C_cols_, gpuQueue_);
            C_cols_ = nullptr;
          }
          break;
        }
      }
    }

    void postCallKernelCleanup() override {
      // Free all allocated memory
      sycl::free(A_, gpuQueue_);
      sycl::free(A_vals_, gpuQueue_);
      sycl::free(A_cols_, gpuQueue_);
      sycl::free(A_rows_, gpuQueue_);
      sycl::free(B_, gpuQueue_);
      sycl::free(B_vals_, gpuQueue_);
      sycl::free(B_cols_, gpuQueue_);
      sycl::free(B_rows_, gpuQueue_);
      sycl::free(C_, gpuQueue_);
      sycl::free(C_rows_, gpuQueue_);

      // These should already be null, but double-check
      if (C_vals_) {
        sycl::free(C_vals_, gpuQueue_);
        C_vals_ = nullptr;
      }
      if (C_cols_) {
        sycl::free(C_cols_, gpuQueue_);
        C_cols_ = nullptr;
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

    // Matrix data pointers
    T* A_vals_ = nullptr;
    int64_t* A_cols_ = nullptr;
    int64_t* A_rows_ = nullptr;

    T* B_vals_ = nullptr;
    int64_t* B_cols_ = nullptr;
    int64_t* B_rows_ = nullptr;

    T* C_vals_ = nullptr;
    int64_t* C_cols_ = nullptr;
    int64_t* C_rows_ = nullptr;

    // Matrix handles
    oneapi::mkl::sparse::matrix_handle_t A_device_ = nullptr;
    oneapi::mkl::sparse::matrix_handle_t B_device_ = nullptr;
    oneapi::mkl::sparse::matrix_handle_t C_device_ = nullptr;

    // Buffer pointers for "once" offload mode
    sycl::buffer<T, 1>* A_vals_device_ = nullptr;
    sycl::buffer<int64_t, 1>* A_cols_device_ = nullptr;
    sycl::buffer<int64_t, 1>* A_rows_device_ = nullptr;

    sycl::buffer<T, 1>* B_vals_device_ = nullptr;
    sycl::buffer<int64_t, 1>* B_cols_device_ = nullptr;
    sycl::buffer<int64_t, 1>* B_rows_device_ = nullptr;

    sycl::buffer<T, 1>* C_vals_device_ = nullptr;
    sycl::buffer<int64_t, 1>* C_cols_device_ = nullptr;
    sycl::buffer<int64_t, 1>* C_rows_device_ = nullptr;

    const T alpha = ALPHA;
    const T beta = BETA;
};
}

#endif