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
      // Clean up temporary buffers first
      deallocateTempBuffers();

      // Release descriptor if initialized
      if (descriptor_initialized_) {
        oneapi::mkl::sparse::release_matmat_descr(&description_);
        descriptor_initialized_ = false;
      }

      // Clean up allocated memory based on offload type
      if (alreadyInitialised_) {
        if (offload_ == gpuOffloadType::unified) {
            // Free all unified memory allocations
          if (A_) { sycl::free(A_, gpuQueue_); A_ = nullptr; }
          if (A_vals_) { sycl::free(A_vals_, gpuQueue_); A_vals_ = nullptr; }
          if (A_cols_) { sycl::free(A_cols_, gpuQueue_); A_cols_ = nullptr; }
          if (A_rows_) { sycl::free(A_rows_, gpuQueue_); A_rows_ = nullptr; }

          if (B_) { sycl::free(B_, gpuQueue_); B_ = nullptr; }
          if (B_vals_) { sycl::free(B_vals_, gpuQueue_); B_vals_ = nullptr; }
          if (B_cols_) { sycl::free(B_cols_, gpuQueue_); B_cols_ = nullptr; }
          if (B_rows_) { sycl::free(B_rows_, gpuQueue_); B_rows_ = nullptr; }

          if (C_) { sycl::free(C_, gpuQueue_); C_ = nullptr; }
          if (C_rows_) { sycl::free(C_rows_, gpuQueue_); C_rows_ = nullptr; }
          // C_vals_ and C_cols_ should already be nullptr from callSpmm cleanup
          if (C_vals_) { sycl::free(C_vals_, gpuQueue_); C_vals_ = nullptr; }
          if (C_cols_) { sycl::free(C_cols_, gpuQueue_); C_cols_ = nullptr; }
        } else {
          // For host memory allocations (always and once modes)
          if (A_) { sycl::free(A_, gpuQueue_); A_ = nullptr; }
          if (A_vals_) { sycl::free(A_vals_, gpuQueue_); A_vals_ = nullptr; }
          if (A_cols_) { sycl::free(A_cols_, gpuQueue_); A_cols_ = nullptr; }
          if (A_rows_) { sycl::free(A_rows_, gpuQueue_); A_rows_ = nullptr; }

          if (B_) { sycl::free(B_, gpuQueue_); B_ = nullptr; }
          if (B_vals_) { sycl::free(B_vals_, gpuQueue_); B_vals_ = nullptr; }
          if (B_cols_) { sycl::free(B_cols_, gpuQueue_); B_cols_ = nullptr; }
          if (B_rows_) { sycl::free(B_rows_, gpuQueue_); B_rows_ = nullptr; }

          if (C_) { sycl::free(C_, gpuQueue_); C_ = nullptr; }
          if (C_rows_) { sycl::free(C_rows_, gpuQueue_); C_rows_ = nullptr; }
        }

        // Wait for all operations to complete before destroying the queue
        gpuQueue_.wait_and_throw();
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

        // Initialize all pointers to nullptr
        A_ = nullptr;
        A_vals_ = nullptr;
        A_cols_ = nullptr;
        A_rows_ = nullptr;

        B_ = nullptr;
        B_vals_ = nullptr;
        B_cols_ = nullptr;
        B_rows_ = nullptr;

        C_ = nullptr;
        C_vals_ = nullptr;
        C_cols_ = nullptr;
        C_rows_ = nullptr;

        A_vals_device_ = nullptr;
        A_cols_device_ = nullptr;
        A_rows_device_ = nullptr;

        B_vals_device_ = nullptr;
        B_cols_device_ = nullptr;
        B_rows_device_ = nullptr;

        C_vals_device_ = nullptr;
        C_cols_device_ = nullptr;
        C_rows_device_ = nullptr;

        A_device_ = nullptr;
        B_device_ = nullptr;
        C_device_ = nullptr;

        device_temp_buffer_1_ = nullptr;
        device_temp_buffer_2_ = nullptr;

        C_vals_temp_ = nullptr;
        C_cols_temp_ = nullptr;
      }

      // Clean up previous allocations
      if (offload_ == gpuOffloadType::unified) {
        if (A_) { sycl::free(A_, gpuQueue_); A_ = nullptr; }
        if (A_vals_) { sycl::free(A_vals_, gpuQueue_); A_vals_ = nullptr; }
        if (A_cols_) { sycl::free(A_cols_, gpuQueue_); A_cols_ = nullptr; }
        if (A_rows_) { sycl::free(A_rows_, gpuQueue_); A_rows_ = nullptr; }

        if (B_) { sycl::free(B_, gpuQueue_); B_ = nullptr; }
        if (B_vals_) { sycl::free(B_vals_, gpuQueue_); B_vals_ = nullptr; }
        if (B_cols_) { sycl::free(B_cols_, gpuQueue_); B_cols_ = nullptr; }
        if (B_rows_) { sycl::free(B_rows_, gpuQueue_); B_rows_ = nullptr; }

        if (C_) { sycl::free(C_, gpuQueue_); C_ = nullptr; }
        if (C_rows_) { sycl::free(C_rows_, gpuQueue_); C_rows_ = nullptr; }
        if (C_vals_) { sycl::free(C_vals_, gpuQueue_); C_vals_ = nullptr; }
        if (C_cols_) { sycl::free(C_cols_, gpuQueue_); C_cols_ = nullptr; }
      } else {
        if (A_) { sycl::free(A_, gpuQueue_); A_ = nullptr; }
        if (A_vals_) { sycl::free(A_vals_, gpuQueue_); A_vals_ = nullptr; }
        if (A_cols_) { sycl::free(A_cols_, gpuQueue_); A_cols_ = nullptr; }
        if (A_rows_) { sycl::free(A_rows_, gpuQueue_); A_rows_ = nullptr; }

        if (B_) { sycl::free(B_, gpuQueue_); B_ = nullptr; }
        if (B_vals_) { sycl::free(B_vals_, gpuQueue_); B_vals_ = nullptr; }
        if (B_cols_) { sycl::free(B_cols_, gpuQueue_); B_cols_ = nullptr; }
        if (B_rows_) { sycl::free(B_rows_, gpuQueue_); B_rows_ = nullptr; }

        if (C_) { sycl::free(C_, gpuQueue_); C_ = nullptr; }
        if (C_rows_) { sycl::free(C_rows_, gpuQueue_); C_rows_ = nullptr; }
      }
      gpuQueue_.wait_and_throw();

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
        // Initialize C array pointers to nullptr
        C_cols_ = nullptr;
        C_vals_ = nullptr;
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
          // Transfer data to the GPU, and set up data structures
          A_vals_device_ = new sycl::buffer<T, 1>(A_vals_, sycl::range<1>(nnzA_));
          A_cols_device_ = new sycl::buffer<int64_t, 1>(A_cols_, sycl::range<1>(nnzA_));
          A_rows_device_ = new sycl::buffer<int64_t, 1>(A_rows_, sycl::range<1>(m_ + 1));

          oneapi::mkl::sparse::init_matrix_handle(&A_device_);
          oneapi::mkl::sparse::set_csr_data(gpuQueue_, A_device_, m_, k_, index_,
                                            *A_rows_device_, *A_cols_device_, *A_vals_device_);
          oneapi::mkl::sparse::sort_matrix(gpuQueue_, A_device_);

          B_vals_device_ = new sycl::buffer<T, 1>(B_vals_, sycl::range<1>(nnzB_));
          B_cols_device_ = new sycl::buffer<int64_t, 1>(B_cols_, sycl::range<1>(nnzB_));
          B_rows_device_ = new sycl::buffer<int64_t, 1>(B_rows_, sycl::range<1>(k_ + 1));

          oneapi::mkl::sparse::init_matrix_handle(&B_device_);
          oneapi::mkl::sparse::set_csr_data(gpuQueue_, B_device_, k_, n_, index_,
                                            *B_rows_device_, *B_cols_device_, *B_vals_device_);
          oneapi::mkl::sparse::sort_matrix(gpuQueue_, B_device_);

          C_rows_device_ = new sycl::buffer<int64_t, 1>(C_rows_, sycl::range<1>(m_ + 1));

          // Pre-allocate C arrays with conservative estimate
          int64_t max_nnzC = std::min((int64_t)(m_ * n_),
                                      std::min((int64_t)(nnzA_ * nnzB_),
                                               (int64_t)(2.0 * (nnzA_ + nnzB_))));

          // Allocate temporary host arrays for C
          C_vals_device_ = new sycl::buffer<T, 1>(sycl::range<1>(max_nnzC));
          C_cols_device_ = new sycl::buffer<int64_t, 1>(sycl::range<1>(max_nnzC));

          // Initialize the buffer contents
          auto vals_acc = C_vals_device_->get_host_access();
          auto cols_acc = C_cols_device_->get_host_access();
          for (int64_t i = 0; i < max_nnzC; i++) {
            vals_acc[i] = T(0);
            cols_acc[i] = -1;
          }

          oneapi::mkl::sparse::init_matrix_handle(&C_device_);
          oneapi::mkl::sparse::set_csr_data(gpuQueue_, C_device_, m_, n_, index_,
                                            *C_rows_device_, *C_cols_device_, *C_vals_device_);

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
          std::cout << "DEBUG: Starting always mode callSpmm" << std::endl;

          // Reset matrix handles to ensure clean state
          A_device_ = nullptr;
          B_device_ = nullptr;
          C_device_ = nullptr;

          // Pre-allocate C arrays with conservative estimate
          int64_t max_nnzC = std::min((int64_t)(m_ * n_),
                                      std::min((int64_t)(nnzA_ * nnzB_),
                                               (int64_t)(2.0 * (nnzA_ + nnzB_))));

          // Allocate temporary host arrays for C FIRST (store as member variables)
          // Clean up any existing allocations
          if (C_vals_temp_ != nullptr) {
            delete[] C_vals_temp_;
            C_vals_temp_ = nullptr;
          }
          if (C_cols_temp_ != nullptr) {
            delete[] C_cols_temp_;
            C_cols_temp_ = nullptr;
          }

          C_vals_temp_ = new T[max_nnzC];
          C_cols_temp_ = new int64_t[max_nnzC];

          // Initialize to zero/invalid values
          std::fill(C_vals_temp_, C_vals_temp_ + max_nnzC, T(0));
          std::fill(C_cols_temp_, C_cols_temp_ + max_nnzC, -1);

          // Create ALL buffers AFTER host arrays are ready
          A_vals_device_ = new sycl::buffer<T, 1>(A_vals_, sycl::range<1>(nnzA_));
          A_cols_device_ = new sycl::buffer<int64_t, 1>(A_cols_, sycl::range<1>(nnzA_));
          A_rows_device_ = new sycl::buffer<int64_t, 1>(A_rows_, sycl::range<1>(m_ + 1));

          B_vals_device_ = new sycl::buffer<T, 1>(B_vals_, sycl::range<1>(nnzB_));
          B_cols_device_ = new sycl::buffer<int64_t, 1>(B_cols_, sycl::range<1>(nnzB_));
          B_rows_device_ = new sycl::buffer<int64_t, 1>(B_rows_, sycl::range<1>(k_ + 1));

          C_rows_device_ = new sycl::buffer<int64_t, 1>(C_rows_, sycl::range<1>(m_ + 1));
          C_vals_device_ = new sycl::buffer<T, 1>(C_vals_temp_, sycl::range<1>(max_nnzC));
          C_cols_device_ = new sycl::buffer<int64_t, 1>(C_cols_temp_, sycl::range<1>(max_nnzC));

          std::cout << "DEBUG: Buffers created, initializing matrix handles" << std::endl;

          // Initialize matrix handles AFTER all buffers are created
          oneapi::mkl::sparse::init_matrix_handle(&A_device_);
          oneapi::mkl::sparse::set_csr_data(gpuQueue_, A_device_, m_, k_, index_,
                                            *A_rows_device_, *A_cols_device_, *A_vals_device_);

          oneapi::mkl::sparse::init_matrix_handle(&B_device_);
          oneapi::mkl::sparse::set_csr_data(gpuQueue_, B_device_, k_, n_, index_,
                                            *B_rows_device_, *B_cols_device_, *B_vals_device_);

          oneapi::mkl::sparse::init_matrix_handle(&C_device_);
          oneapi::mkl::sparse::set_csr_data(gpuQueue_, C_device_, m_, n_, index_,
                                            *C_rows_device_, *C_cols_device_, *C_vals_device_);

          // Sort matrices AFTER all are initialized
          oneapi::mkl::sparse::sort_matrix(gpuQueue_, A_device_);
          oneapi::mkl::sparse::sort_matrix(gpuQueue_, B_device_);

          // Critical synchronization point
          gpuQueue_.wait_and_throw();

          std::cout << "DEBUG: Matrix handles initialized, starting computation" << std::endl;

          // Work estimation phase
          request_ = oneapi::mkl::sparse::matmat_request::get_work_estimation_buf_size;
          int64_t work_size_local = 0;
          sycl::buffer<int64_t, 1> work_size_buffer(&work_size_local, sycl::range<1>(1));

          try {
            oneapi::mkl::sparse::matmat(gpuQueue_, A_device_, B_device_, C_device_,
                                        request_, description_, &work_size_buffer, nullptr);
            gpuQueue_.wait_and_throw();
            device_temp_buffer_1_size_ = work_size_local;
          } catch (sycl::exception const& e) {
            std::cout << "ERROR - Work estimation buffer size: " << e.what() << std::endl;
          }

          std::cout << "DEBUG: Work estimation buffer size: " << device_temp_buffer_1_size_ << std::endl;

          // Work estimation
          request_ = oneapi::mkl::sparse::matmat_request::work_estimation;
          if (device_temp_buffer_1_size_ > 0) {
            std::vector<std::uint8_t> work_buffer_host(device_temp_buffer_1_size_);
            sycl::buffer<int64_t, 1> work_size_buffer_est(&device_temp_buffer_1_size_, sycl::range<1>(1));
            sycl::buffer<std::uint8_t, 1> work_buffer(work_buffer_host.data(),
                                                      sycl::range<1>(device_temp_buffer_1_size_));

            try {
              oneapi::mkl::sparse::matmat(gpuQueue_, A_device_, B_device_, C_device_,
                                          request_, description_, &work_size_buffer_est, &work_buffer);
              gpuQueue_.wait_and_throw();
            } catch (sycl::exception const& e) {
              std::cout << "ERROR - Work estimation: " << e.what() << std::endl;
            }
          } else {
            try {
              oneapi::mkl::sparse::matmat(gpuQueue_, A_device_, B_device_, C_device_,
                                          request_, description_, nullptr, nullptr);
              gpuQueue_.wait_and_throw();
            } catch (sycl::exception const& e) {
              std::cout << "ERROR - Work estimation (no buffer): " << e.what() << std::endl;
            }
          }

          std::cout << "DEBUG: Work estimation complete" << std::endl;

          // Get compute buffer size
          request_ = oneapi::mkl::sparse::matmat_request::get_compute_buf_size;
          int64_t compute_size_local = 0;
          sycl::buffer<int64_t, 1> compute_size_buffer(&compute_size_local, sycl::range<1>(1));

          try {
            oneapi::mkl::sparse::matmat(gpuQueue_, A_device_, B_device_, C_device_,
                                        request_, description_, &compute_size_buffer, nullptr);
            gpuQueue_.wait_and_throw();
            device_temp_buffer_2_size_ = compute_size_local;
          } catch (sycl::exception const& e) {
            std::cout << "ERROR - Compute buffer size: " << e.what() << std::endl;
          }

          std::cout << "DEBUG: Compute buffer size: " << device_temp_buffer_2_size_ << std::endl;

          // Perform actual computation
          request_ = oneapi::mkl::sparse::matmat_request::compute;
          if (device_temp_buffer_2_size_ > 0) {
            std::vector<std::uint8_t> compute_buffer_host(device_temp_buffer_2_size_);
            sycl::buffer<int64_t, 1> compute_size_buffer_comp(&device_temp_buffer_2_size_, sycl::range<1>(1));
            sycl::buffer<std::uint8_t, 1> compute_buffer(compute_buffer_host.data(),
                                                          sycl::range<1>(device_temp_buffer_2_size_));

            try {
              oneapi::mkl::sparse::matmat(gpuQueue_, A_device_, B_device_, C_device_,
                                          request_, description_, &compute_size_buffer_comp, &compute_buffer);
              gpuQueue_.wait_and_throw();
            } catch (sycl::exception const& e) {
              std::cout << "ERROR - Compute: " << e.what() << std::endl;
            }
          } else {
            try {
              oneapi::mkl::sparse::matmat(gpuQueue_, A_device_, B_device_, C_device_,
                                          request_, description_, nullptr, nullptr);
              gpuQueue_.wait_and_throw();
            } catch (sycl::exception const& e) {
              std::cout << "ERROR - Compute (no buffer): " << e.what() << std::endl;
            }
          }

          std::cout << "DEBUG: Compute complete, finalizing" << std::endl;

          // Finalize the computation
          request_ = oneapi::mkl::sparse::matmat_request::finalize;
          try {
            oneapi::mkl::sparse::matmat(gpuQueue_, A_device_, B_device_, C_device_,
                                        request_, description_, nullptr, nullptr);
            gpuQueue_.wait_and_throw();
          } catch (sycl::exception const& e) {
            std::cout << "ERROR - Finalize: " << e.what() << std::endl;
          }

          // Get actual nnzC after computation
          {
            auto C_rows_acc = C_rows_device_->get_host_access();
            nnzC_ = C_rows_acc[m_];
          }

          std::cout << "DEBUG: Finalize complete, cleaning up. nnzC = " << nnzC_ << std::endl;

          // 1. Wait for all GPU operations to complete FIRST
          gpuQueue_.wait_and_throw();

          std::cout << "1, ";

          // 2. Release matrix handles (in reverse order of creation)
          if (C_device_ != nullptr) {
            oneapi::mkl::sparse::release_matrix_handle(gpuQueue_, &C_device_);
            C_device_ = nullptr;
          }
          if (B_device_ != nullptr) {
            oneapi::mkl::sparse::release_matrix_handle(gpuQueue_, &B_device_);
            B_device_ = nullptr;
          }
          if (A_device_ != nullptr) {
            oneapi::mkl::sparse::release_matrix_handle(gpuQueue_, &A_device_);
            A_device_ = nullptr;
          }
          std::cout << "2, ";

          // 3. Wait again to ensure handles are released
          gpuQueue_.wait_and_throw();
          std::cout << "3, ";

          // 4. Delete SYCL buffers (in reverse dependency order)
          // C buffers first (they depend on A and B)
          if (C_cols_device_) { delete C_cols_device_; C_cols_device_ = nullptr; }
          if (C_vals_device_) { delete C_vals_device_; C_vals_device_ = nullptr; }
          if (C_rows_device_) { delete C_rows_device_; C_rows_device_ = nullptr; }

          // B buffers
          if (B_rows_device_) { delete B_rows_device_; B_rows_device_ = nullptr; }
          if (B_cols_device_) { delete B_cols_device_; B_cols_device_ = nullptr; }
          if (B_vals_device_) { delete B_vals_device_; B_vals_device_ = nullptr; }

          // A buffers
          if (A_rows_device_) { delete A_rows_device_; A_rows_device_ = nullptr; }
          if (A_cols_device_) { delete A_cols_device_; A_cols_device_ = nullptr; }
          if (A_vals_device_) { delete A_vals_device_; A_vals_device_ = nullptr; }
          std::cout << "4, ";

          // 5. Final synchronization before deleting host arrays
          gpuQueue_.wait_and_throw();
          std::cout << "5, ";

          // 6. Clean up host temporary arrays LAST
          if (C_vals_temp_ != nullptr) {
            delete[] C_vals_temp_;
            C_vals_temp_ = nullptr;
          }
          if (C_cols_temp_ != nullptr) {
            delete[] C_cols_temp_;
            C_cols_temp_ = nullptr;
          }

          std::cout << "DEBUG: Always mode callSpmm complete" << std::endl;
          break;
        }
        case gpuOffloadType::once: {
          // Do computation
          request_ = oneapi::mkl::sparse::matmat_request::get_work_estimation_buf_size;
          int64_t work_size_local = 0;
          sycl::buffer<int64_t, 1> work_size_buffer(&work_size_local, sycl::range<1>(1));
          try {
            oneapi::mkl::sparse::matmat(gpuQueue_, A_device_, B_device_, C_device_,
                                        request_, description_, &work_size_buffer, nullptr);
          } catch (sycl::exception const& e) {
            std::cout << "ERROR - Caught synchronous SYCL exception during "
                         "SPMM (Once) get_work_estimation_buf_size:\n"
                      << e.what() << std::endl
                      << "OpenCL status: " << e.code().value() << std::endl;
          }

          gpuQueue_.wait_and_throw();
          device_temp_buffer_1_size_ = work_size_local;

          request_ = oneapi::mkl::sparse::matmat_request::work_estimation;
          if (device_temp_buffer_1_size_ > 0) {
            std::vector<std::uint8_t> work_buffer_host(device_temp_buffer_1_size_);

            sycl::buffer<int64_t, 1> work_size_buffer_est(&device_temp_buffer_1_size_, sycl::range<1>(1));
            sycl::buffer<std::uint8_t, 1> work_buffer(work_buffer_host.data(),
                                                      sycl::range<1>(device_temp_buffer_1_size_));

            try {
              oneapi::mkl::sparse::matmat(gpuQueue_, A_device_, B_device_, C_device_,
                                          request_, description_,
                                          &work_size_buffer_est,
                                          &work_buffer);
            } catch (sycl::exception const& e) {
              std::cout << "ERROR - Caught synchronous SYCL exception during "
                           "SPMM (once) work_estimation:\n"
                        << e.what() << std::endl
                        << "OpenCL status: " << e.code().value() << std::endl;
            }
          } else {
            try {
              oneapi::mkl::sparse::matmat(gpuQueue_, A_device_, B_device_, C_device_,
                                          request_, description_, nullptr, nullptr);
            } catch (sycl::exception const& e) {
              std::cout << "ERROR - Caught synchronous SYCL exception during "
                           "SPMM (once) work_estimation:\n"
                        << e.what() << std::endl
                        << "OpenCL status: " << e.code().value() << std::endl;
            }
          }

          // Get compute buffer size
          request_ = oneapi::mkl::sparse::matmat_request::get_compute_buf_size;
          int64_t compute_size_local = 0;
          sycl::buffer<int64_t, 1> compute_size_buffer(&compute_size_local, sycl::range<1>(1));
          try {
            oneapi::mkl::sparse::matmat(gpuQueue_, A_device_, B_device_, C_device_,
                                        request_, description_, &compute_size_buffer, nullptr);
          } catch (sycl::exception const& e) {
            std::cout << "ERROR - Caught synchronous SYCL exception during "
                         "SPMM (Once) get_compute_buf_size:\n"
                      << e.what() << std::endl
                      << "OpenCL status: " << e.code().value() << std::endl;
          }

          gpuQueue_.wait_and_throw();
          device_temp_buffer_2_size_ = compute_size_local;

          // Perform actual computation
          request_ = oneapi::mkl::sparse::matmat_request::compute;
          if (device_temp_buffer_2_size_ > 0) {
            std::vector<std::uint8_t> compute_buffer_host(device_temp_buffer_2_size_);

            sycl::buffer<int64_t, 1> compute_size_buffer_comp(&device_temp_buffer_2_size_, sycl::range<1>(1));
            sycl::buffer<std::uint8_t, 1> compute_buffer(compute_buffer_host.data(),
                                                          sycl::range<1>(device_temp_buffer_2_size_));

            try {
              oneapi::mkl::sparse::matmat(gpuQueue_, A_device_, B_device_, C_device_,
                                          request_, description_,
                                          &compute_size_buffer_comp,
                                          &compute_buffer);
            } catch (sycl::exception const& e) {
              std::cout << "ERROR - Caught synchronous SYCL exception during "
                           "SPMM (Once) compute:\n"
                        << e.what() << std::endl
                        << "OpenCL status: " << e.code().value() << std::endl;
            }
          } else {
            // No compute buffer needed
            try {
              oneapi::mkl::sparse::matmat(gpuQueue_, A_device_, B_device_, C_device_,
                                          request_, description_, nullptr, nullptr);
            } catch (sycl::exception const& e) {
              std::cout << "ERROR - Caught synchronous SYCL exception during "
                           "SPMM (Once) compute:\n"
                        << e.what() << std::endl
                        << "OpenCL status: " << e.code().value() << std::endl;
            }
          }

          // Finalize the computation
          request_ = oneapi::mkl::sparse::matmat_request::finalize;
          try {
            oneapi::mkl::sparse::matmat(gpuQueue_, A_device_, B_device_, C_device_,
                                        request_, description_, nullptr, nullptr);
          } catch (sycl::exception const& e) {
            std::cout << "ERROR - Caught synchronous SYCL exception during "
                         "SPMM (Once) finalize:\n"
                      << e.what() << std::endl
                      << "OpenCL status: " << e.code().value() << std::endl;
          }

          // Get actual nnzC after computation
          gpuQueue_.wait_and_throw();

          // Read back the last element of C_rows to get nnzC
          {
            auto C_rows_acc = C_rows_device_->get_host_access();
            nnzC_ = C_rows_acc[m_];
          }
          break;
        }
        case gpuOffloadType::unified: {
          // Unified memory implementation with proper memory management
          int64_t temp_buffer_size = 0;
          void* temp_buffer = nullptr;
          std::vector<sycl::event> dependencies;

          // Initialize C matrix handle for this iteration
          oneapi::mkl::sparse::init_matrix_handle(&C_device_);

          // Pre-allocate C arrays with conservative estimate
          int64_t max_nnzC = std::min((int64_t)(m_ * n_),
                                      std::min((int64_t)(nnzA_ * nnzB_),
                                               (int64_t)(2.0 * (nnzA_ + nnzB_))));

          // Free previous allocations if they exist
          if (C_vals_ != nullptr) {
            sycl::free(C_vals_, gpuQueue_);
          }
          if (C_cols_ != nullptr) {
            sycl::free(C_cols_, gpuQueue_);
          }

          // Allocate C arrays
          C_vals_ = (T*)sycl::malloc_shared(sizeof(T) * max_nnzC, gpuQueue_);
          C_cols_ = (int64_t*)sycl::malloc_shared(sizeof(int64_t) * max_nnzC, gpuQueue_);
          gpuQueue_.wait();

          // Set CSR data for C with pre-allocated arrays
          oneapi::mkl::sparse::set_csr_data(gpuQueue_, C_device_, m_, n_,
                                            index_, C_rows_, C_cols_, C_vals_);

          // Step 1: Work estimation to determine C structure
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

          // Step 3: Get compute buffer size
          request_ = oneapi::mkl::sparse::matmat_request::get_compute_buf_size;
          int64_t compute_buffer_size = 0;
          try {
            auto event = oneapi::mkl::sparse::matmat(gpuQueue_, A_device_,
                                                     B_device_, C_device_,
                                                     request_, description_,
                                                     &compute_buffer_size,
                                                     nullptr, dependencies);
            event.wait();
          } catch (sycl::exception const& e) {
            std::cerr << "ERROR - Get compute buffer size: " << e.what()
                      << std::endl;
            if (temp_buffer) sycl::free(temp_buffer, gpuQueue_);
            oneapi::mkl::sparse::release_matrix_handle(gpuQueue_, &C_device_);
            throw;
          }

          // Allocate compute buffer if needed (separate from work estimation buffer)
          void* compute_buffer = nullptr;
          if (compute_buffer_size > 0) {
            compute_buffer = sycl::malloc_shared(compute_buffer_size, gpuQueue_);
          }

          // Step 4: Compute
          request_ = oneapi::mkl::sparse::matmat_request::compute;
          try {
            auto event = oneapi::mkl::sparse::matmat(gpuQueue_, A_device_,
                                                     B_device_, C_device_,
                                                     request_, description_,
                                                     &compute_buffer_size,
                                                     compute_buffer, dependencies);
            event.wait();
          } catch (sycl::exception const& e) {
            std::cerr << "ERROR - Compute: " << e.what() << std::endl;
            if (temp_buffer) sycl::free(temp_buffer, gpuQueue_);
            if (compute_buffer) sycl::free(compute_buffer, gpuQueue_);
            oneapi::mkl::sparse::release_matrix_handle(gpuQueue_, &C_device_);
            throw;
          }

          // Step 5: Finalize
          request_ = oneapi::mkl::sparse::matmat_request::finalize;
          try {
            auto event = oneapi::mkl::sparse::matmat(gpuQueue_, A_device_, B_device_,
                                                 C_device_, request_, description_,
                                                 nullptr, nullptr, dependencies);
            event.wait();
          } catch (sycl::exception const& e) {
            std::cerr << "ERROR - Finalize: " << e.what() << std::endl;
          }

          // Get actual nnzC after computation
          gpuQueue_.wait();
          nnzC_ = C_rows_[m_];

          // Clean up temporary buffers
          if (temp_buffer) {
            sycl::free(temp_buffer, gpuQueue_);
          }
          if (compute_buffer) {
            sycl::free(compute_buffer, gpuQueue_);
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

          // Note: C_vals_temp and C_cols_temp were local variables in preLoopRequirements()
          // They should be cleaned up there, not here
          break;
        }
        case gpuOffloadType::unified: {
          // Release A and B handles
          oneapi::mkl::sparse::release_matrix_handle(gpuQueue_, &A_device_);
          oneapi::mkl::sparse::release_matrix_handle(gpuQueue_, &B_device_);

          // Don't free C arrays here - they're already freed in callSpmm
          // Just ensure pointers are null
          C_vals_ = nullptr;
          C_cols_ = nullptr;
          break;
        }
      }
    }

    void postCallKernelCleanup() override {}

    void allocateTempBuffers() {
      if (device_temp_buffer_1_size_ > 0 && device_temp_buffer_1_ == nullptr) {
        device_temp_buffer_1_ = sycl::malloc_device(device_temp_buffer_1_size_, gpuQueue_);
      }
      if (device_temp_buffer_2_size_ > 0 && device_temp_buffer_2_ == nullptr) {
        device_temp_buffer_2_ = sycl::malloc_device(device_temp_buffer_2_size_, gpuQueue_);
      }
    }

    void deallocateTempBuffers() {
      if (device_temp_buffer_1_ != nullptr) {
        sycl::free(device_temp_buffer_1_, gpuQueue_);
        device_temp_buffer_1_ = nullptr;
        device_temp_buffer_1_size_ = 0;
      }
      if (device_temp_buffer_2_ != nullptr) {
        sycl::free(device_temp_buffer_2_, gpuQueue_);
        device_temp_buffer_2_ = nullptr;
        device_temp_buffer_2_size_ = 0;
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

    // Temporary arrays for C in always mode
    T* C_vals_temp_ = nullptr;
    int64_t* C_cols_temp_ = nullptr;

    // Matrix handles
    oneapi::mkl::sparse::matrix_handle_t A_device_ = nullptr;
    oneapi::mkl::sparse::matrix_handle_t B_device_ = nullptr;
    oneapi::mkl::sparse::matrix_handle_t C_device_ = nullptr;

    // Buffer pointers for "once" and "always" offload modes
    sycl::buffer<T, 1>* A_vals_device_ = nullptr;
    sycl::buffer<int64_t, 1>* A_cols_device_ = nullptr;
    sycl::buffer<int64_t, 1>* A_rows_device_ = nullptr;

    sycl::buffer<T, 1>* B_vals_device_ = nullptr;
    sycl::buffer<int64_t, 1>* B_cols_device_ = nullptr;
    sycl::buffer<int64_t, 1>* B_rows_device_ = nullptr;

    sycl::buffer<T, 1>* C_vals_device_ = nullptr;
    sycl::buffer<int64_t, 1>* C_cols_device_ = nullptr;
    sycl::buffer<int64_t, 1>* C_rows_device_ = nullptr;

    int64_t device_temp_buffer_1_size_ = 0;
    void* device_temp_buffer_1_ = nullptr;
    int64_t device_temp_buffer_2_size_ = 0;
    void* device_temp_buffer_2_ = nullptr;

    const T alpha = ALPHA;
    const T beta = BETA;
};
}


#endif
