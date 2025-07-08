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
      // Perform set-up which doesn't need to happen every problem size change.
      if (firstRun_) {
        firstRun_ = false;
        try {
          myGpu_ = sycl::device(sycl::gpu_selector_v);
        } catch (const std::exception& e) {
          std::cerr << "ERROR - No GPU device found: " << e.what() << '\n';
          exit(1);
        }
        gpuQueue_ = sycl::queue(myGpu_, exception_handler);
      }  
      
      try {
        // Initialize ALL pointers to nullptr FIRST
        A_ = nullptr;
        B_ = nullptr;
        C_ = nullptr;
        A_vals_ = nullptr;
        A_cols_ = nullptr;
        A_rows_ = nullptr;
        A_vals_device_ = nullptr;
        A_cols_device_ = nullptr;
        A_rows_device_ = nullptr;
        B_device_ = nullptr;
        C_device_ = nullptr;

        if (print_) std::cout << "\t\tsetting up metadata" << std::endl;

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

        if (print_) std::cout << "\t\tallocating space" << std::endl;
        if (offload_ == gpuOffloadType::unified) {
          A_ = (T*)sycl::malloc_shared(sizeof(T) * m_ * k_, gpuQueue_);
          checkPointer(A_, "A_");
          A_vals_ = (T*)sycl::malloc_shared(sizeof(T) * nnz_, gpuQueue_);
          checkPointer(A_vals_, "A_vals_");
          A_cols_ = (int64_t*)sycl::malloc_shared(sizeof(int64_t) * nnz_, gpuQueue_);
          checkPointer(A_cols_, "A_cols_");
          A_rows_ = (int64_t*)sycl::malloc_shared(sizeof(int64_t) * (m_ + 1), gpuQueue_);
          checkPointer(A_rows_, "A_rows_");
          B_ = (T*)sycl::malloc_shared(sizeof(T) * k_ * n_, gpuQueue_);
          checkPointer(B_, "B_");
          C_ = (T*)sycl::malloc_shared(sizeof(T) * m_ * n_, gpuQueue_);
          checkPointer(C_, "C_");
        } else {
          // Host memory allocation
          A_ = (T*)sycl::malloc_host(sizeof(T) * m_ * k_, gpuQueue_);
          checkPointer(A_, "A_");
          A_vals_ = (T*)sycl::malloc_host(sizeof(T) * nnz_, gpuQueue_);
          checkPointer(A_vals_, "A_vals_");
          A_cols_ = (int64_t*)sycl::malloc_host(sizeof(int64_t) * nnz_, gpuQueue_);
          checkPointer(A_cols_, "A_cols_");
          A_rows_ = (int64_t*)sycl::malloc_host(sizeof(int64_t) * (m_ + 1), gpuQueue_);
          checkPointer(A_rows_, "A_rows_");
          B_ = (T*)sycl::malloc_host(sizeof(T) * k_ * n_, gpuQueue_);
          checkPointer(B_, "B_");
          C_ = (T*)sycl::malloc_host(sizeof(T) * m_ * n_, gpuQueue_);
          checkPointer(C_, "C_");

          // Device memory allocation
          A_vals_device_ = (T*)sycl::malloc_device(sizeof(T) * nnz_, gpuQueue_);
          checkPointer(A_vals_device_, "A_vals_device_");
          A_cols_device_ = (int64_t*)sycl::malloc_device(sizeof(int64_t) * nnz_, gpuQueue_);
          checkPointer(A_cols_device_, "A_cols_device_");
          A_rows_device_ = (int64_t*)sycl::malloc_device(sizeof(int64_t) * (m_ + 1), gpuQueue_);
          checkPointer(A_rows_device_, "A_rows_device_");
          B_device_ = (T*)sycl::malloc_device(sizeof(T) * k_ * n_, gpuQueue_);
          checkPointer(B_device_, "B_device_");
          C_device_ = (T*)sycl::malloc_device(sizeof(T) * m_ * n_, gpuQueue_);
          checkPointer(C_device_, "C_device_");
        }
        initInputMatrices();

      } catch (const std::exception& e) {
        std::cerr << "ERROR in initialise(): " << e.what() << std::endl;
              exit(1);
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
    }

private:
    void preLoopRequirements() override {
      switch(offload_) {
        case gpuOffloadType::always: break;
        case gpuOffloadType::once: {
          // Moving memory over to device from host
          if (print_) std::cout << "\t\tCopying data to device for 'once' mode" << std::endl;
          gpuQueue_.memcpy(A_vals_device_, A_vals_, sizeof(T) * nnz_);
          gpuQueue_.memcpy(A_cols_device_, A_cols_, sizeof(int64_t) * nnz_);
          gpuQueue_.memcpy(A_rows_device_, A_rows_, sizeof(int64_t) * (m_ + 1));
          gpuQueue_.memcpy(B_device_, B_, sizeof(T) * k_ * n_);
          gpuQueue_.wait();
          if (print_) std::cout << "\t\tSetting up matrix handle for unified memory" << std::endl;
          oneapi::mkl::sparse::init_matrix_handle(&A_device_);
          if (print_) std::cout << "\t\tLoading data into the matrix handle" << std::endl;
          oneapi::mkl::sparse::set_csr_data(gpuQueue_,
                                            A_device_,
                                            m_,
                                            k_,
                                            index_,
                                            A_rows_device_,
                                            A_cols_device_,
                                            A_vals_device_);
          gpuQueue_.wait_and_throw();
          break;
        }
        case gpuOffloadType::unified: {
          // For unified memory, set up matrix handle once
          if (print_) std::cout << "\t\tSetting up matrix handle for unified memory" << std::endl;
          oneapi::mkl::sparse::init_matrix_handle(&A_device_);
          if (print_) std::cout << "\t\tLoading data into the matrix handle" << std::endl;
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
          // Copy data to device for this iteration
          gpuQueue_.memcpy(A_vals_device_, A_vals_, sizeof(T) * nnz_);
          gpuQueue_.memcpy(A_cols_device_, A_cols_, sizeof(int64_t) * nnz_);
          gpuQueue_.memcpy(A_rows_device_, A_rows_, sizeof(int64_t) * (m_ + 1));
          gpuQueue_.memcpy(B_device_, B_, sizeof(T) * k_ * n_);
          gpuQueue_.wait();

          if (print_) std::cout << "\t\tMaking matrix handle" << std::endl;
          oneapi::mkl::sparse::init_matrix_handle(&A_device_);
          if (print_) std::cout << "\t\tSetting CSR data" << std::endl;
          oneapi::mkl::sparse::set_csr_data(gpuQueue_,
                                            A_device_,
                                            m_,
                                            k_,
                                            index_,
                                            A_rows_device_,
                                            A_cols_device_,
                                            A_vals_device_);
          gpuQueue_.wait();

          // Do computation
          try {
            oneapi::mkl::sparse::gemm(gpuQueue_,
                                      layout_,
                                      operationA_,
                                      operationB_,
                                      alpha,
                                      A_device_,
                                      B_device_,
                                      n_,
                                      n_,
                                      beta,
                                      C_device_,
                                      n_);
            gpuQueue_.wait();
          } catch (sycl::exception const& e) {
            std::cout << "ERROR - Caught synchronous SYCL exception during "
                          "SPGEMM (Always):\n" << e.what() << std::endl <<
                          "OpenCL status: " << e.code().value() << std::endl;
            exit(1);
          }

          // Copy result back to host
          gpuQueue_.memcpy(C_, C_device_, sizeof(T) * m_ * n_);
          gpuQueue_.wait();

          
          // Clean up matrix handle
          oneapi::mkl::sparse::release_matrix_handle(gpuQueue_, &A_device_);
          gpuQueue_.wait();
          break;
        }
        case gpuOffloadType::once: {
          // Buffers already exist, just do computation
          try {
            if (print_) std::cout << "\t\tAbout to call oneapi::mkl::sparse::gemm()" << std::endl;
            oneapi::mkl::sparse::gemm(gpuQueue_,
                                      layout_,
                                      operationA_,
                                      operationB_,
                                      alpha,
                                      A_device_,
                                      B_device_,
                                      n_,
                                      n_,
                                      beta,
                                      C_device_,
                                      n_);
            gpuQueue_.wait();
          } catch (sycl::exception const& e) {
            std::cout << "ERROR - Caught synchronous SYCL exception during "
                          "SPGEMM (Once):\n" << e.what() << std::endl <<
                          "OpenCL status: " << e.code().value() << std::endl;
            exit(1);
          }
          break;
        }
        case gpuOffloadType::unified: {
          // Direct computation with unified memory
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
            gpuQueue_.wait_and_throw();
          } catch (sycl::exception const& e) {
              std::cerr << "ERROR - Caught synchronous SYCL exception during SPGEMM (Unified): " << e.what() << std::endl << "OpenCL status: " << e.code().value() << std::endl;
              exit(1);
          }
          break;
        }
      }
    }

    void postLoopRequirements() override {
      // Clean up buffers that were created for the entire loop duration
      if (offload_ == gpuOffloadType::once) {
        if (print_) std::cout << "\t\tCleaning up 'once' mode resources" << std::endl;
        gpuQueue_.memcpy(C_, C_device_, sizeof(T) * m_ * n_);
        gpuQueue_.wait();
      }
      if (print_) std::cout << "\t\tFinal cleanup" << std::endl;
      
      if (offload_ != gpuOffloadType::always) {
        oneapi::mkl::sparse::release_matrix_handle(gpuQueue_, &A_device_);
      }
      
      if (A_) { sycl::free(A_, gpuQueue_); A_ = nullptr; }
      if (B_) { sycl::free(B_, gpuQueue_); B_ = nullptr; }
      if (C_) { sycl::free(C_, gpuQueue_); C_ = nullptr; }
      if (A_vals_) { sycl::free(A_vals_, gpuQueue_); A_vals_ = nullptr; }
      if (A_cols_) { sycl::free(A_cols_, gpuQueue_); A_cols_ = nullptr; }
      if (A_rows_) { sycl::free(A_rows_, gpuQueue_); A_rows_ = nullptr; }
      
      // Free device memory if allocated
      if (A_vals_device_) { sycl::free(A_vals_device_, gpuQueue_); A_vals_device_ = nullptr; }
      if (A_cols_device_) { sycl::free(A_cols_device_, gpuQueue_); A_cols_device_ = nullptr; }
      if (A_rows_device_) { sycl::free(A_rows_device_, gpuQueue_); A_rows_device_ = nullptr; }
      if (B_device_) { sycl::free(B_device_, gpuQueue_); B_device_ = nullptr; }
      if (C_device_) { sycl::free(C_device_, gpuQueue_); C_device_ = nullptr; } 
      if (print_) std::cout << "\t\tdone" << std::endl;
    }

    void postCallKernelCleanup() override {
    }

    void checkPointer(void* ptr, std::string name) {
      if (ptr == nullptr) {
        std::cout << "Pointer " << name << " is a null pointer" << std::endl;
        exit(1);
      }
    }

    bool firstRun_ = true;

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

    T* A_vals_device_;
    int64_t* A_cols_device_;
    int64_t* A_rows_device_;
    T* B_device_;
    T* C_device_;

    const T alpha = ALPHA;
    const T beta = BETA;

    bool print_ = false;
};
}

#endif
