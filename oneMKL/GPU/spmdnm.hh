#pragma once

#ifdef GPU_ONEMKL

#include "../../include/kernels/GPU/spmdnm.hh"
#include "../../include/utilities.hh"
#include "common.hh"

#include <iostream>

namespace gpu {
template <typename T>
class spmdnm_gpu : public spmdnm<T> {
public:
    using spmdnm<T>::spmdnm;
    using spmdnm<T>::initInputMatrices;
    using spmdnm<T>::nnz_;
    using spmdnm<T>::m_;
    using spmdnm<T>::n_;
    using spmdnm<T>::k_;
    using spmdnm<T>::B_;
    using spmdnm<T>::C_;
    using spmdnm<T>::offload_;
    using spmdnm<T>::sparsity_;
    using spmdnm<T>::type_;

    void initialise(gpuOffloadType offload, int m, int n, int k,
                double sparsity, matrixType type, 
                bool binary = false) override {
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

        offload_ = offload;
        sparsity_ = sparsity;
        type_ = type;
        m_ = m;
        n_ = n;
        k_ = k;

        layout_ = oneapi::mkl::layout::row_major;
        operationA_ = oneapi::mkl::transpose::nontrans;
        operationB_ = oneapi::mkl::transpose::nontrans;
        index_ = oneapi::mkl::index_base::zero;

        nnz_ = 1 + (uint64_t)((double)m_ * (double)k_ * (1.0 - sparsity_));

        if (offload_ == gpuOffloadType::unified) {
          B_ = (T*)sycl::malloc_shared(sizeof(T) * k_ * n_, gpuQueue_);
          C_ = (T*)sycl::malloc_shared(sizeof(T) * m_ * n_, gpuQueue_);
        } else {
          // Host memory allocation
          B_ = (T*)sycl::malloc_host(sizeof(T) * k_ * n_, gpuQueue_);
          C_ = (T*)sycl::malloc_host(sizeof(T) * m_ * n_, gpuQueue_);

          // Device memory allocation
          B_device_ = (T*)sycl::malloc_device(sizeof(T) * k_ * n_, gpuQueue_);
          C_device_ = (T*)sycl::malloc_device(sizeof(T) * m_ * n_, gpuQueue_);
        }
        initInputMatrices();
      } catch (const std::exception& e) {
        std::cerr << "ERROR in initialise(): " << e.what() << std::endl;
        exit(1);
      }
    }


protected:
    void toSparseFormat() override {
      if (offload_ == gpuOffloadType::always) {
        A_vals_store_ = (T*)malloc(nnz_ * sizeof(T));
        A_cols_store_ = (int64_t*)malloc(nnz_ * sizeof(int64_t));
        A_rows_store_ = (int64_t*)malloc((m_ + 1) * sizeof(int64_t));
        if (type_ == matrixType::rmat) {
          rMatCSR<T, int64_t>(A_vals_store_, A_cols_store_, A_rows_store_, m_, k_, nnz_);
        } else if (type_ == matrixType::random) {
          randomCSR<T, int64_t>(A_vals_store_, A_cols_store_, A_rows_store_, m_, k_, nnz_);
        } else if (type_ == matrixType::finiteElements) {
          finiteElementCSR<T, int64_t>(A_vals_store_, A_cols_store_, A_rows_store_, m_, k_, nnz_);
        } else {
          std::cerr << "ERROR - Unknown matrix type" << std::endl;
          exit(1);
        }
      }


      if (offload_ == gpuOffloadType::unified) {
          A_vals_ = (T*)sycl::malloc_shared(sizeof(T) * nnz_, gpuQueue_);
          A_cols_ = (int64_t*)sycl::malloc_shared(sizeof(int64_t) * nnz_, gpuQueue_);
          A_rows_ = (int64_t*)sycl::malloc_shared(sizeof(int64_t) * (m_ + 1), gpuQueue_);
      } else {
          A_vals_ = (T*)sycl::malloc_host(sizeof(T) * nnz_, gpuQueue_);
          A_cols_ = (int64_t*)sycl::malloc_host(sizeof(int64_t) * nnz_, gpuQueue_);
          A_rows_ = (int64_t*)sycl::malloc_host(sizeof(int64_t) * (m_ + 1), gpuQueue_);

          A_vals_device_ = (T*)sycl::malloc_device(sizeof(T) * nnz_, gpuQueue_);
          A_cols_device_ = (int64_t*)sycl::malloc_device(sizeof(int64_t) * nnz_, gpuQueue_);
          A_rows_device_ = (int64_t*)sycl::malloc_device(sizeof(int64_t) * (m_ + 1), gpuQueue_);
      }
      
      memcpy(A_rows_, A_rows_store_, sizeof(int64_t) * (m_ + 1));
      memcpy(A_cols_, A_cols_store_, sizeof(int64_t) * nnz_);
      memcpy(A_vals_, A_vals_store_, sizeof(T) * nnz_);
    }

private:
    void preLoopRequirements() override {
      switch(offload_) {
        case gpuOffloadType::always: break;
        case gpuOffloadType::once: {
          // Moving memory over to device from host
          gpuQueue_.memcpy(A_vals_device_, A_vals_, sizeof(T) * nnz_);
          gpuQueue_.memcpy(A_cols_device_, A_cols_, sizeof(int64_t) * nnz_);
          gpuQueue_.memcpy(A_rows_device_, A_rows_, sizeof(int64_t) * (m_ + 1));
          gpuQueue_.memcpy(B_device_, B_, sizeof(T) * k_ * n_);
          gpuQueue_.wait(); // Is this needed?
          oneapi::mkl::sparse::init_matrix_handle(&A_device_);
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

    void callSpmdnm() override {
      switch (offload_) {
        case gpuOffloadType::always: {
          // Copy data to device for this iteration
          gpuQueue_.memcpy(A_vals_device_, A_vals_, sizeof(T) * nnz_);
          gpuQueue_.memcpy(A_cols_device_, A_cols_, sizeof(int64_t) * nnz_);
          gpuQueue_.memcpy(A_rows_device_, A_rows_, sizeof(int64_t) * (m_ + 1));
          gpuQueue_.memcpy(B_device_, B_, sizeof(T) * k_ * n_);
          gpuQueue_.wait();

          oneapi::mkl::sparse::init_matrix_handle(&A_device_);
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
            std::cerr << "ERROR - Caught synchronous SYCL exception during "
                          "spmdnm (Always):\n" << e.what() << std::endl <<
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
            std::cerr << "ERROR - Caught synchronous SYCL exception during "
                          "spmdnm (Once):\n" << e.what() << std::endl <<
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
              std::cerr << "ERROR - Caught synchronous SYCL exception during spmdnm (Unified): " << e.what() << std::endl << "OpenCL status: " << e.code().value() << std::endl;
              exit(1);
          }
          break;
        }
      }
    }

    void postLoopRequirements() override {
      // Clean up buffers that were created for the entire loop duration
      if (offload_ == gpuOffloadType::once) {
        gpuQueue_.memcpy(C_, C_device_, sizeof(T) * m_ * n_);
        gpuQueue_.wait();
      }
      
      if (offload_ != gpuOffloadType::always) {
        oneapi::mkl::sparse::release_matrix_handle(gpuQueue_, &A_device_);
      }
    }

    void postCallKernelCleanup() override {
      if (offload_ == gpuOffloadType::unified) {
        if (B_) { sycl::free(B_, gpuQueue_); B_ = nullptr; }
        if (C_) { sycl::free(C_, gpuQueue_); C_ = nullptr; }
        if (A_vals_) { sycl::free(A_vals_, gpuQueue_); A_vals_ = nullptr; }
        if (A_cols_) { sycl::free(A_cols_, gpuQueue_); A_cols_ = nullptr; }
        if (A_rows_) { sycl::free(A_rows_, gpuQueue_); A_rows_ = nullptr; }

        free(A_vals_store_);
        free(A_cols_store_);
        free(A_rows_store_);
      } else {
        if (B_) { sycl::free(B_, gpuQueue_); B_ = nullptr; }
        if (C_) { sycl::free(C_, gpuQueue_); C_ = nullptr; }
        if (A_vals_) { sycl::free(A_vals_, gpuQueue_); A_vals_ = nullptr; }
        if (A_cols_) { sycl::free(A_cols_, gpuQueue_); A_cols_ = nullptr; }
        if (A_rows_) { sycl::free(A_rows_, gpuQueue_); A_rows_ = nullptr; }
        
        if (A_vals_device_) { sycl::free(A_vals_device_, gpuQueue_); A_vals_device_ = nullptr; }
        if (A_cols_device_) { sycl::free(A_cols_device_, gpuQueue_); A_cols_device_ = nullptr; }
        if (A_rows_device_) { sycl::free(A_rows_device_, gpuQueue_); A_rows_device_ = nullptr; }
        if (B_device_) { sycl::free(B_device_, gpuQueue_); B_device_ = nullptr; }
        if (C_device_) { sycl::free(C_device_, gpuQueue_); C_device_ = nullptr; } 
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

    T* A_vals_store_;
    int64_t* A_cols_store_;
    int64_t* A_rows_store_;

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
};
}

#endif
