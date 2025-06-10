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
      if (descriptor_initialized_) {
        oneapi::mkl::sparse::release_matmat_descr(&description_);
        descriptor_initialized_ = false;
      }

      // Clean up allocated memory based on offload type
      if (alreadyInitialised_) {
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

        // Delete buffer objects
        delete A_vals_device_;
        delete A_cols_device_;
        delete A_rows_device_;
        delete B_vals_device_;
        delete B_cols_device_;
        delete B_rows_device_;
        delete C_vals_device_;
        delete C_cols_device_;
        delete C_rows_device_;

        // Wait for all operations to complete before destroying the queue
        gpuQueue_.wait_and_throw();
      }
    }

    void initialise(gpuOffloadType offload, int m, int n, int k,
                double sparsity, bool binary = false) override {
      if (offload == gpuOffloadType::always) return;
      std::cout << ".. checking already init" << std::endl;
      if (!alreadyInitialised_) {
        alreadyInitialised_ = true;
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
      try {
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
        safe_wait(gpuQueue_, "memory cleanup");
      } catch (const sycl::exception& e) {
        std::cerr << "WARNING - Memory cleanup failed: " << e.what() <<
        std::endl;
      }

      std::cout << ".. setting metadata" << std::endl;
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
        if (nnzA_ > std::numeric_limits<int64_t>::max() || nnzB_ > std::numeric_limits<int64_t>::max()) {
          throw std::overflow_error("Matrix dimensions result in overflow");
        }
      } catch (const std::exception& e) {
        std::cerr << "ERROR - Invalid matrix dimensions: " << e.what() << std::endl;
        throw;
      }

      // For unified memory, don't pre-allocate C arrays
      if (offload_ == gpuOffloadType::unified) {
        try {
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
          // Cleanup any successfully allocated memory
          cleanup_allocations();
          throw;
        }
      } else {
        try {
          std::cout << ".. host malloc" << std::endl;
          A_ = static_cast<T*>(sycl::malloc_host(sizeof(T) * m_ * k_,
                                                 gpuQueue_));
          if (!A_) throw std::bad_alloc();
          A_vals_ = static_cast<T*>((T*)sycl::malloc_host(sizeof(T) * nnzA_,
                                                  gpuQueue_));
          if (!A_vals_) throw std::bad_alloc();
          A_cols_ = static_cast<int64_t*>sycl::malloc_host(sizeof(int64_t) *
                  nnzA_, gpuQueue_);
          if (!A_cols_) throw std::badalloc();
          A_rows_ = static_cast<int64_t*>sycl::malloc_host(sizeof(int64_t) *
                  (m_ + 1), gpuQueue_);
          if (!A_rows_) throw std::badalloc();


          B_ = static_cast<T*>(sycl::malloc_host(sizeof(T) * k_ * n_,
                                                 gpuQueue_));
          if (!B_) throw std::bad_alloc();
          B_vals_ = static_cast<T*>((T*)sycl::malloc_host(sizeof(T) * nnzB_,
                                                  gpuQueue_));
          if (!B_vals_) throw std::bad_alloc();
          B_cols_ = static_cast<int64_t*>sycl::malloc_host(sizeof(int64_t) *
                  nnzB_, gpuQueue_);
          if (!B_cols_) throw std::badalloc();
          B_rows_ = static_cast<int64_t*>sycl::malloc_host(sizeof(int64_t) *
                  (k_ + 1), gpuQueue_);
          if (!B_rows_) throw std::badalloc();

          C_ = static_cast<T*>(sycl::malloc_host(sizeof(T) * m_ * n_,
                                                 gpuQueue_));
          if (!C_) throw std::bad_alloc();
          C_rows_ = static_cast<int64_t*>sycl::malloc_host(sizeof(int64_t) *
                  (m_ + 1), gpuQueue_);
          if (!C_rows_) throw std::badalloc();
          // Initialize C array pointers to nullptr
          C_cols_ = nullptr;
          C_vals_ = nullptr;
          safe_wait(gpuQueue_, "unified memory allocation");
        } catch (const st::exception& e) {
          cleanup_allocations();
          throw;
        }
      }

      std::cout << ".. initialising input matrices" << std::endl;
      try {
        initInputMatrices();
      } catch (const std::exception& e) {
        std::cerr << "ERROR - Matrix initialization failed: " << e.what() << std::endl;
        cleanup_allocations();
        throw;
      }
      std::cout << ".. DONE" << std::endl;
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

      std::cout << " and C" << std::endl;
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
          // TODO -- currently empty
          break;
        }
        case gpuOffloadType::once: {
          // Allocate GPU memory for A and B
          A_rows_device_ = new sycl::buffer<int64_t>(A_rows_, sycl::range<1>(m_ + 1));
          A_cols_device_ = new sycl::buffer<int64_t>(A_cols_, sycl::range<1>(nnzA_));
          A_vals_device_ = new sycl::buffer<T>(A_vals_, sycl::range<1>(nnzA_));

          B_rows_device_ = new sycl::buffer<int64_t>(B_rows_, sycl::range<1>(k_ + 1));
          B_cols_device_ = new sycl::buffer<int64_t>(B_cols_, sycl::range<1>(nnzB_));
          B_vals_device_ = new sycl::buffer<T>(B_vals_, sycl::range<1>(nnzB_));

          // Create A, B and C handles, set A and B data
          oneapi::mkl::sparse::init_matrix_handle(&A_device_);
          oneapi::mkl::sparse::set_csr_data(gpuQueue_, A_device_, m_, k_,
                                            index_, *A_rows_device_,
                                            *A_cols_device_, *A_vals_device_);

          oneapi::mkl::sparse::init_matrix_handle(&B_device_);
          oneapi::mkl::sparse::set_csr_data(gpuQueue_, B_device_, k_, n_,
                                            index_, *B_rows_device_,
                                            *B_cols_device_, *B_vals_device_);
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
          // TODO -- currently empty
          break;
        }
        case gpuOffloadType::once: {
          // Pre-allocate C arrays with conservative estimate
          int64_t max_nnzC = std::min((int64_t)(m_ * n_),
                                      std::min((int64_t)(nnzA_ * nnzB_),
                                               (int64_t)(2.0 * (nnzA_ + nnzB_))));

          // Allocate host memory for C
          if (!C_cols_) {
            C_cols_ = (int64_t*)sycl::malloc_host(sizeof(int64_t) * max_nnzC, gpuQueue_);
          }
          if (!C_vals_) {
            C_vals_ = (T*)sycl::malloc_host(sizeof(T) * max_nnzC, gpuQueue_);
          }

          C_rows_device_ = new sycl::buffer<int64_t>(C_rows_, sycl::range<1>(m_ + 1));
          C_cols_device_ = new sycl::buffer<int64_t>(C_cols_, sycl::range<1>(max_nnzC));
          C_vals_device_ = new sycl::buffer<T>(C_vals_, sycl::range<1>(max_nnzC));

          oneapi::mkl::sparse::init_matrix_handle(&C_device_);
          oneapi::mkl::sparse::set_csr_data(gpuQueue_, C_device_, m_, n_,
                                            index_, *C_rows_device_,
                                            *C_cols_device_, *C_vals_device_);

          // Step 1: Get work estimation buffer size using buffer-based API
          sycl::buffer<std::int64_t, 1> size_temp_buffer(sycl::range<1>(1));
          request_ = oneapi::mkl::sparse::matmat_request::get_work_estimation_buf_size;

          try {
            oneapi::mkl::sparse::matmat(gpuQueue_, A_device_, B_device_,
                                        C_device_, request_, description_,
                                        &size_temp_buffer, nullptr);
            gpuQueue_.wait_and_throw();
          } catch (sycl::exception const& e) {
            std::cerr << "ERROR - Work estimation buffer size: " << e.what()
                      << std::endl;
            oneapi::mkl::sparse::release_matrix_handle(gpuQueue_, &C_device_);
            throw;
          }

          // Get the size from the buffer and allocate temp buffer
          auto temp_size_acc = size_temp_buffer.get_host_access(sycl::read_only);
          std::int64_t temp_buffer_size = temp_size_acc[0];

          sycl::buffer<std::uint8_t, 1>* temp_buffer_ptr = nullptr;
          if (temp_buffer_size > 0) {
            temp_buffer_ptr = new sycl::buffer<std::uint8_t, 1>(sycl::range<1>(temp_buffer_size));
          }

          // Step 2: Perform work estimation
          request_ = oneapi::mkl::sparse::matmat_request::work_estimation;
          try {
            oneapi::mkl::sparse::matmat(gpuQueue_, A_device_,
                                       B_device_, C_device_,
                                       request_, description_,
                                       &size_temp_buffer,
                                       temp_buffer_ptr);
            gpuQueue_.wait_and_throw();
          } catch (sycl::exception const& e) {
            std::cerr << "ERROR - Work estimation: " << e.what() << std::endl;
            if (temp_buffer_ptr) delete temp_buffer_ptr;
            oneapi::mkl::sparse::release_matrix_handle(gpuQueue_, &C_device_);
            throw;
          }

          // Step 3: Get compute buffer size
          request_ = oneapi::mkl::sparse::matmat_request::get_compute_buf_size;
          sycl::buffer<std::int64_t, 1> size_compute_buffer(sycl::range<1>(1));

          try {
            oneapi::mkl::sparse::matmat(gpuQueue_, A_device_,
                                       B_device_, C_device_,
                                       request_, description_,
                                       &size_compute_buffer,
                                       nullptr);
            gpuQueue_.wait_and_throw();
          } catch (sycl::exception const& e) {
            std::cerr << "ERROR - Get compute buffer size: " << e.what()
                      << std::endl;
            if (temp_buffer_ptr) delete temp_buffer_ptr;
            oneapi::mkl::sparse::release_matrix_handle(gpuQueue_, &C_device_);
            throw;
          }

          // Get the actual size from the buffer and allocate compute buffer
          auto compute_size_acc = size_compute_buffer.get_host_access(sycl::read_only);
          std::int64_t compute_buffer_size = compute_size_acc[0];
          sycl::buffer<std::uint8_t, 1>* compute_buffer_ptr = nullptr;
          if (compute_buffer_size > 0) {
            compute_buffer_ptr = new sycl::buffer<std::uint8_t, 1>(sycl::range<1>(compute_buffer_size));
          }

          // Step 4: Compute
          request_ = oneapi::mkl::sparse::matmat_request::compute;
          try {
            oneapi::mkl::sparse::matmat(gpuQueue_, A_device_, B_device_,
                                        C_device_, request_, description_,
                                        &size_compute_buffer, compute_buffer_ptr);
            gpuQueue_.wait_and_throw();
          } catch (sycl::exception const& e) {
            std::cerr << "ERROR - Compute: " << e.what() << std::endl;
            if (temp_buffer_ptr) delete temp_buffer_ptr;
            if (compute_buffer_ptr) delete compute_buffer_ptr;
            oneapi::mkl::sparse::release_matrix_handle(gpuQueue_, &C_device_);
            throw;
          }

          // Step 5: Finalize
          request_ = oneapi::mkl::sparse::matmat_request::finalize;
          try {
            oneapi::mkl::sparse::matmat(gpuQueue_, A_device_, B_device_,
                                       C_device_, request_, description_,
                                       &size_compute_buffer, compute_buffer_ptr);
            gpuQueue_.wait_and_throw();
          } catch (sycl::exception const& e) {
            std::cerr << "ERROR - Finalize: " << e.what() << std::endl;
          }

          // Ensure all operations complete before reading results
          gpuQueue_.wait_and_throw();

          // Get actual nnzC after computation - need to copy from device buffer
          {
            auto c_rows_acc = C_rows_device_->get_host_access(sycl::read_only);
            nnzC_ = c_rows_acc[m_];
          }

          // Clean up temporary buffers
          if (temp_buffer_ptr) {
            delete temp_buffer_ptr;
          }
          if (compute_buffer_ptr) {
            delete compute_buffer_ptr;
          }

          // Release C handle and delete buffers
          oneapi::mkl::sparse::release_matrix_handle(gpuQueue_, &C_device_);

          delete C_rows_device_;
          delete C_cols_device_;
          delete C_vals_device_;
          C_rows_device_ = nullptr;
          C_cols_device_ = nullptr;
          C_vals_device_ = nullptr;

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
          // TODO -- currently empty
          break;
        }
        case gpuOffloadType::once: {
          // Release matrix handles
          oneapi::mkl::sparse::release_matrix_handle(gpuQueue_, &A_device_);
          oneapi::mkl::sparse::release_matrix_handle(gpuQueue_, &B_device_);

          // Delete buffer objects
          delete A_rows_device_;
          delete A_cols_device_;
          delete A_vals_device_;
          delete B_rows_device_;
          delete B_cols_device_;
          delete B_vals_device_;

          A_rows_device_ = nullptr;
          A_cols_device_ = nullptr;
          A_vals_device_ = nullptr;
          B_rows_device_ = nullptr;
          B_cols_device_ = nullptr;
          B_vals_device_ = nullptr;

          // Free host memory for C if allocated
          if (C_cols_) {
            sycl::free(C_cols_, gpuQueue_);
            C_cols_ = nullptr;
          }
          if (C_vals_) {
            sycl::free(C_vals_, gpuQueue_);
            C_vals_ = nullptr;
          }
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

    T* safe_malloc_shared(size_t size, sycl::queue& q, const std::string& var_name) {
      try {
        T* ptr = sycl::malloc_shared<T>(size, q);
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