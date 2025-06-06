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
      if (alreadyInitialised_) {
        oneapi::mkl::sparse::release_matmat_descr(&description_);
      }
    }

    void initialise(gpuOffloadType offload, int m, int n, int k,
                    double sparsity, bool binary = false) override {
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

        // Initialize the descriptor once
        if (!descriptor_initialised_) {
          oneapi::mkl::sparse::init_matmat_descr(&description_);
          descriptor_initialised_ = true;
        }
      }

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

      if (offload_ == gpuOffloadType::unified) {
        A_ = (T*)sycl::malloc_shared(sizeof(T) * m_ * k_, gpuQueue_);
        A_vals_ = (T*)sycl::malloc_shared(sizeof(T) * nnzA_, gpuQueue_);
        A_cols_ = (int64_t*)sycl::malloc_shared(sizeof(int64_t) * nnzA_,
                                                gpuQueue_);
        A_rows_ = (int64_t*)sycl::malloc_shared(sizeof(int64_t) * (m_ + 1),
                                                gpuQueue_);

        B_ = (T*)sycl::malloc_shared(sizeof(T) * k_ * n_, gpuQueue_);
        B_vals_ = (T*)sycl::malloc_shared(sizeof(T) * nnzB_, gpuQueue_);
        B_cols_ = (int64_t*)sycl::malloc_shared(sizeof(int64_t) * nnzB_,
                                                gpuQueue_);
        B_rows_ = (int64_t*)sycl::malloc_shared(sizeof(int64_t) * (k_ + 1),
                                                gpuQueue_);

        C_ = (T*)sycl::malloc_shared(sizeof(T) * m_ * n_, gpuQueue_);
        C_rows_ = (int64_t*)sycl::malloc_shared(sizeof(int64_t) * (m_ + 1),
                                                gpuQueue_);

        dependencies_ = (std::vector<sycl::event>*)sycl::malloc_shared(
                sizeof(std::vector<sycl::event>*),
                gpuQueue_);

      } else {
        A_ = (T*)sycl::malloc_host(sizeof(T) * m_ * k_, gpuQueue_);
        A_vals_ = (T*)sycl::malloc_host(sizeof(T) * nnzA_, gpuQueue_);
        A_cols_ = (int64_t*)sycl::malloc_host(sizeof(int64_t) * nnzA_,
                                              gpuQueue_);
        A_rows_ = (int64_t*)sycl::malloc_host(sizeof(int64_t) * (m_ + 1),
                                              gpuQueue_);

        B_ = (T*)sycl::malloc_host(sizeof(T) * k_ * n_, gpuQueue_);
        B_vals_ = (T*)sycl::malloc_host(sizeof(T) * nnzB_, gpuQueue_);
        B_cols_ = (int64_t*)sycl::malloc_host(sizeof(int64_t) * nnzB_,
                                              gpuQueue_);
        B_rows_ = (int64_t*)sycl::malloc_host(sizeof(int64_t) * (k_ + 1),
                                              gpuQueue_);

        C_ = (T*)sycl::malloc_host(sizeof(T) * m_ * n_, gpuQueue_);
        C_rows_ = (int64_t*)sycl::malloc_host(sizeof(int64_t) * (m_ + 1),
                                              gpuQueue_);
      }

      initInputMatrices();

      if (offload_ == gpuOffloadType::unified) {
        // Debug: Verify CSR structure
        std::cout << "\nDebug - Matrix A: " << m_ << "x" << k_
                  << ", nnz=" << nnzA_ << std::endl;
        std::cout << "A_rows_[0]=" << A_rows_[0]
                  << ", A_rows_[" << m_ << "]=" << A_rows_[m_] << std::endl;

        std::cout << "Debug - Matrix B: " << k_ << "x" << n_
                  << ", nnz=" << nnzB_ << std::endl;
        std::cout << "B_rows_[0]=" << B_rows_[0]
                  << ", B_rows_[" << k_ << "]=" << B_rows_[k_] << std::endl;

        // Verify the CSR format is correct
        if (A_rows_[m_] != nnzA_) {
          std::cerr << "ERROR: A_rows_[m_] != nnzA_" << std::endl;
        }
        if (B_rows_[k_] != nnzB_) {
          std::cerr << "ERROR: B_rows_[k_] != nnzB_" << std::endl;
        }
      }
    }


protected:
    void toSparseFormat() override {
      int64_t nnz_encountered = 0;

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
    }

private:
    void preLoopRequirements() override {
      switch(offload_) {
        case gpuOffloadType::always: {
          break;
        }
        case gpuOffloadType::once: {
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

          gpuQueue_.wait_and_throw();
          break;
        }
        case gpuOffloadType::unified: {
          // IMPORTANT: Initialize the descriptor first
          oneapi::mkl::sparse::init_matmat_descr(&description_);

          // Initialize matrix handles
          oneapi::mkl::sparse::init_matrix_handle(&A_device_);
          oneapi::mkl::sparse::init_matrix_handle(&B_device_);

          // Then set the CSR data
          oneapi::mkl::sparse::set_csr_data(gpuQueue_,
                                            A_device_,
                                            m_,
                                            k_,
                                            index_,
                                            A_rows_,
                                            A_cols_,
                                            A_vals_);

          oneapi::mkl::sparse::set_csr_data(gpuQueue_,
                                            B_device_,
                                            k_,
                                            n_,
                                            index_,
                                            B_rows_,
                                            B_cols_,
                                            B_vals_);

          // Wait to ensure data is set
          gpuQueue_.wait_and_throw();

          // Important: Sort the matrices after setting CSR data
          oneapi::mkl::sparse::sort_matrix(gpuQueue_, A_device_);
          oneapi::mkl::sparse::sort_matrix(gpuQueue_, B_device_);

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
          /**
           * STEP 1 -- Initialize C matrix handle
           */
          oneapi::mkl::sparse::init_matrix_handle(&C_device_);
          gpuQueue_.wait_and_throw();

          /**
           * STEP 2 -- Work estimation to get buffer sizes
           */
          request_ = oneapi::mkl::sparse::matmat_request::get_work_estimation_buf_size;
          int64_t temp_buffer_size = 0;
          void* temp_buffer = nullptr;
          std::vector<sycl::event> dependencies;

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
            std::cout << "ERROR - Work estimation buffer size: " << e.what() << std::endl;
            // Add more debug info
            std::cout << "A_device_ initialized: " << (A_device_ != nullptr) << std::endl;
            std::cout << "B_device_ initialized: " << (B_device_ != nullptr) << std::endl;
            std::cout << "C_device_ initialized: " << (C_device_ != nullptr) << std::endl;
            throw; // Re-throw to see full error
          }

          // Allocate temp buffer
          if (temp_buffer_size > 0) {
            temp_buffer = sycl::malloc_shared(temp_buffer_size, gpuQueue_);
          }

          /**
           * STEP 3 -- Work estimation with allocated buffer
           */
          request_ = oneapi::mkl::sparse::matmat_request::work_estimation;
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
            std::cout << "ERROR - Work estimation phase: " << e.what() << std::endl;
          }

          /**
           * STEP 4 -- Get compute buffer size and NNZ
           */
          request_ = oneapi::mkl::sparse::matmat_request::get_compute_buf_size;
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
            std::cout << "ERROR - Get compute buffer size: " << e.what() << std::endl;
          }
          // Since we can't query nnzC before computation, we need to allocate a conservative estimate
          // For SpMM, worst case is nnzC = m * n (fully dense result)
          // A more realistic estimate based on sparsity patterns:
          int64_t estimated_nnzC = std::min((int64_t)(m_ * n_),
                                            (int64_t)(nnzA_ * nnzB_ / k_));

          // Allocate C arrays with conservative estimate
          C_cols_ = (int64_t*)sycl::malloc_shared(sizeof(int64_t) * estimated_nnzC, gpuQueue_);
          C_vals_ = (T*)sycl::malloc_shared(sizeof(T) * estimated_nnzC, gpuQueue_);

          // Set CSR data for C with estimated arrays
          oneapi::mkl::sparse::set_csr_data(gpuQueue_,
                                            C_device_,
                                            m_,
                                            n_,
                                            index_,
                                            C_rows_,
                                            C_cols_,
                                            C_vals_);

          // Reallocate temp buffer if needed
          if (temp_buffer_size > 0) {
            sycl::free(temp_buffer, gpuQueue_);
            temp_buffer = sycl::malloc_shared(temp_buffer_size, gpuQueue_);
          }

          /**
           * STEP 5 -- Compute (this will populate C structure)
           */
          request_ = oneapi::mkl::sparse::matmat_request::compute;
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
            std::cout << "ERROR - Compute phase: " << e.what() << std::endl;
          }

          // After computation, determine actual nnzC from C_rows array
          // The last element of C_rows contains the total number of non-zeros
          gpuQueue_.wait(); // Ensure computation is complete
          nnzC_ = C_rows_[m_]; // CSR format: rows[m] contains total nnz

          // If you need to resize the arrays to actual size (optional):
          if (nnzC_ < estimated_nnzC) {
            // Create new correctly sized arrays
            int64_t* new_C_cols = (int64_t*)sycl::malloc_shared(sizeof(int64_t) * nnzC_, gpuQueue_);
            T* new_C_vals = (T*)sycl::malloc_shared(sizeof(T) * nnzC_, gpuQueue_);

            // Copy data
            gpuQueue_.memcpy(new_C_cols, C_cols_, sizeof(int64_t) * nnzC_).wait();
            gpuQueue_.memcpy(new_C_vals, C_vals_, sizeof(T) * nnzC_).wait();

            // Free old arrays and update pointers
            sycl::free(C_cols_, gpuQueue_);
            sycl::free(C_vals_, gpuQueue_);
            C_cols_ = new_C_cols;
            C_vals_ = new_C_vals;
          }

          /**
           * STEP 6 -- Finalize
           */
          request_ = oneapi::mkl::sparse::matmat_request::finalize;
          try {
            auto event = oneapi::mkl::sparse::matmat(gpuQueue_,
                                                     A_device_,
                                                     B_device_,
                                                     C_device_,
                                                     request_,
                                                     description_,
                                                     nullptr,
                                                     nullptr,
                                                     dependencies);
            event.wait();
          } catch (sycl::exception const& e) {
            std::cout << "ERROR - Finalize: " << e.what() << std::endl;
          }

          // Clean up
          if (temp_buffer) {
            sycl::free(temp_buffer, gpuQueue_);
          }

          oneapi::mkl::sparse::release_matrix_handle(gpuQueue_, &C_device_);
          gpuQueue_.wait_and_throw();
          break;
        }
      }
    }

    void postLoopRequirements() override {
      switch(offload_) {
        case gpuOffloadType::always: {
          // Nothing to do here as handles are created/destroyed in callSpmm
          break;
        }
        case gpuOffloadType::once: {
          oneapi::mkl::sparse::release_matrix_handle(gpuQueue_, &A_device_);
          oneapi::mkl::sparse::release_matrix_handle(gpuQueue_, &B_device_);
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
        case gpuOffloadType::unified: {
          oneapi::mkl::sparse::release_matrix_handle(gpuQueue_, &A_device_);
          oneapi::mkl::sparse::release_matrix_handle(gpuQueue_, &B_device_);
          // Note: Don't release C_device_ here as it's released in callSpmm
          break;
        }
      }
    }

    void postCallKernelCleanup() override {
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

      if (offload_ == gpuOffloadType::unified) {
        if (C_vals_) sycl::free(C_vals_, gpuQueue_);
        if (C_cols_) sycl::free(C_cols_, gpuQueue_);
      }
    }

    /** Whether the initialise function has been called before. */
    bool alreadyInitialised_ = false;
    bool descriptor_initialised_ = false;

    /** The GPU Device. */
    sycl::device myGpu_;

    /** The SYCL execution queue*/
    sycl::queue gpuQueue_;
    oneapi::mkl::index_base index_;
    oneapi::mkl::transpose operationA_;
    oneapi::mkl::transpose operationB_;
    oneapi::mkl::sparse::matmat_request request_;
    oneapi::mkl::sparse::matmat_descr_t description_;
    oneapi::mkl::layout layout_;

    sycl::buffer<int64_t, 1>* device_temp_buffer_1_size_;
    sycl::buffer<uint8_t, 1>* device_temp_buffer_1_;
    int64_t* usm_temp_buffer_1_size_;
    void* usm_temp_buffer_1_;
    sycl::buffer<int64_t, 1>* device_temp_buffer_2_size_;
    sycl::buffer<uint8_t, 1>* device_temp_buffer_2_;
    int64_t* usm_temp_buffer_2_size_;
    void* usm_temp_buffer_2_;
    sycl::buffer<int64_t, 1>* device_nnz_buffer_size_;
    int64_t* usm_nnz_buffer_size_;

    std::vector<sycl::event>* dependencies_;

    T* A_vals_ = nullptr;
    int64_t* A_cols_ = nullptr;
    int64_t* A_rows_ = nullptr;

    T* B_vals_ = nullptr;
    int64_t* B_cols_ = nullptr;
    int64_t* B_rows_ = nullptr;

    T* C_vals_ = nullptr;
    int64_t* C_cols_ = nullptr;
    int64_t* C_rows_ = nullptr;

    oneapi::mkl::sparse::matrix_handle_t A_device_;
    sycl::buffer<T, 1>* A_vals_device_;
    sycl::buffer<int64_t, 1>* A_cols_device_;
    sycl::buffer<int64_t, 1>* A_rows_device_;

    oneapi::mkl::sparse::matrix_handle_t B_device_;
    sycl::buffer<T, 1>* B_vals_device_;
    sycl::buffer<int64_t, 1>* B_cols_device_;
    sycl::buffer<int64_t, 1>* B_rows_device_;

    oneapi::mkl::sparse::matrix_handle_t C_device_;
    sycl::buffer<T, 1>* C_vals_device_;
    sycl::buffer<int64_t, 1>* C_cols_device_;
    sycl::buffer<int64_t, 1>* C_rows_device_;

    const T alpha = ALPHA;
    const T beta = BETA;
};
}

#endif
