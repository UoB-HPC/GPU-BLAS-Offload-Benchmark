#pragma once

#ifdef GPU_ROCBLAS
#include <hip/hip_runtime_api.h>
#include <rocsparse/rocsparse.h>

#include <memory>
#include "../include/kernels/GPU/spmm.hh"
#include "../include/utilities.hh"
#include "common.hh"

namespace gpu {
template <typename T>
class spmm_gpu : public spmm<T> {
public:
    using spmm<T>::spmm;
    using spmm<T>::initInputMatrices;
    using spmm<T>::nnzA_;
    using spmm<T>::nnzB_;
    using spmm<T>::m_;
    using spmm<T>::n_;
    using spmm<T>::k_;
    using spmm<T>::A_;
    using spmm<T>::B_;
    using spmm<T>::C_;
    using spmm<T>::offload_;
    using spmm<T>::sparsity_;

    void initialise(gpuOffloadType offload, int m, int n, int k,
                    double sparsity, bool binary = false) override {
      
      if (print_) {
        switch (offload) {
          case gpuOffloadType::always: {
            std::cout << "===========  ALWAYS  ===========" << std::endl;
            break;
          }
          case gpuOffloadType::once: {
            std::cout << "===========   ONCE   ===========" << std::endl;
            break;
          }
          case gpuOffloadType::unified: {
            std::cout << "===========  UNIFIED ===========" << std::endl;
            break;
          }
        }
      }


      if (print_) std::cout << "Initialising " << m << "x" << k << " . " << k << "x" << n << std::endl;
      firstRun_ = true;

      m_ = m;
      n_ = n;
      k_ = k;
      sparsity_ = sparsity;
      offload_ = offload;
      nnzA_ = 1 + (uint64_t)((double)m_ * (double)k_ * (1.0 - sparsity_));
      nnzB_ = 1 + (uint64_t)((double)k_ * (double)n_ * (1.0 - sparsity_));
            
      // Setting up rocSPARSE type parameters
      m_roc_ = m_;
      n_roc_ = n_;
      k_roc_ = k_;
      nnzA_roc_ = nnzA_;
      nnzB_roc_ = nnzB_;

      // Set up rocSPARSE metadata
      index_ = rocsparse_index_base_zero;
      type_ = rocsparse_matrix_type_general;
      operation_ = rocsparse_operation_none;

      if (print_) std::cout << "\tAbout to set up handle and hip streams" << std::endl;
      if (!initialised_) {
        status_ = rocsparse_create_handle(&handle_);
        checkStatus("Failed rocsparse_create_handle");

        // Get the GPU
        hipCheckError(hipGetDevice(&gpuDevice_));
        // Make streams for asynchronous GPU comunication
        hipCheckError(hipStreamCreate(&s1_));
        hipCheckError(hipStreamCreate(&s2_));
        hipCheckError(hipStreamCreate(&s3_));
      }

      if (print_) std::cout << "\tAbout to malloc arrays" << std::endl;
      if (offload_ == gpuOffloadType::unified) {
        hipCheckError(hipMallocManaged(&A_, sizeof(T) * m_ * k_));
        hipCheckError(hipMallocManaged(&A_rows_, sizeof(int64_t) * (m_ + 1)));
        hipCheckError(hipMallocManaged(&A_cols_, sizeof(int64_t) * nnzA_));
        hipCheckError(hipMallocManaged(&A_vals_, sizeof(T) * nnzA_));
        hipCheckError(hipMallocManaged(&B_, sizeof(T) * k_ * n_));
        hipCheckError(hipMallocManaged(&B_rows_, sizeof(int64_t) * (k_ + 1)));
        hipCheckError(hipMallocManaged(&B_cols_, sizeof(int64_t) * nnzB_));
        hipCheckError(hipMallocManaged(&B_vals_, sizeof(T) * nnzB_));        
      } else {
        // Host data structures
        hipCheckError(hipHostMalloc((void**)&A_, sizeof(T) * m_ * k_));
        hipCheckError(hipHostMalloc((void**)&A_rows_, sizeof(int64_t) * (m_ + 1)));
        hipCheckError(hipHostMalloc((void**)&A_cols_, sizeof(int64_t) * nnzA_));
        hipCheckError(hipHostMalloc((void**)&A_vals_, sizeof(T) * nnzA_));
        hipCheckError(hipHostMalloc((void**)&B_, sizeof(T) * k_ * n_));
        hipCheckError(hipHostMalloc((void**)&B_rows_, sizeof(int64_t) * (k_ + 1)));
        hipCheckError(hipHostMalloc((void**)&B_cols_, sizeof(int64_t) * nnzB_));
        hipCheckError(hipHostMalloc((void**)&B_vals_, sizeof(T) * nnzB_));
        
        // GPU data structures
        hipCheckError(hipMalloc((void**)&A_rows_device_, sizeof(int64_t) * (m_ + 1)));
        hipCheckError(hipMalloc((void**)&A_cols_device_, sizeof(int64_t) * nnzA_));
        hipCheckError(hipMalloc((void**)&A_vals_device_, sizeof(T) * nnzA_));
        hipCheckError(hipMalloc((void**)&B_rows_device_, sizeof(int64_t) * (k_ + 1)));
        hipCheckError(hipMalloc((void**)&B_cols_device_, sizeof(int64_t) * nnzB_));
        hipCheckError(hipMalloc((void**)&B_vals_device_, sizeof(T) * nnzB_));
      }

      if (print_) std::cout << "\tInitialising matrices" << std::endl;
      initInputMatrices();
    }

protected:
    void toSparseFormat() override {
      if (print_) std::cout << "Making sparse now" << std::endl;
      int64_t nnz_encountered = 0;

      if (print_) std::cout << "\tA into CSR" << std::endl;
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
        std::cerr << "Warning: A matrix has " << nnz_encountered << " non-zeros, expected " << nnzA_ << std::endl;
        nnzA_ = nnz_encountered;  // Update to actual count
      }

      if (print_) std::cout << "\tB into CSR" << std::endl;
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
        std::cerr << "Warning: B matrix has " << nnz_encountered << " non-zeros, expected " << nnzB_ << std::endl;
        nnzB_ = nnz_encountered;  // Update to actual count
      }

      // Ensure synchronization for unified memory
      hipCheckError(hipDeviceSynchronize());
    }

private:
    void preLoopRequirements() override {
    if (print_) std::cout << "pre-loop stuff" << std::endl;
      switch (offload_) {
        case gpuOffloadType::always: {
          break;
        }
        case gpuOffloadType::once: {
          if (print_) std::cout << "\tMoving data to GPU" << std::endl;
          hipCheckError(hipMemcpyAsync(A_rows_device_,
                                       A_rows_,
                                       sizeof(rocsparse_int) * (m_ + 1),
                                       hipMemcpyHostToDevice,
                                       s1_));
          hipCheckError(hipMemcpyAsync(A_cols_device_,
                                       A_cols_,
                                       sizeof(rocsparse_int) * nnzA_,
                                       hipMemcpyHostToDevice,
                                       s1_));
          hipCheckError(hipMemcpyAsync(A_vals_device_,
                                       A_vals_,
                                       sizeof(T) * nnzA_,
                                       hipMemcpyHostToDevice,
                                       s1_));
          hipCheckError(hipMemcpyAsync(B_rows_device_,
                                       B_rows_,
                                       sizeof(rocsparse_int) * (k_ + 1),
                                       hipMemcpyHostToDevice,
                                       s2_));
          hipCheckError(hipMemcpyAsync(B_cols_device_,
                                       B_cols_,
                                       sizeof(rocsparse_int) * nnzB_,
                                       hipMemcpyHostToDevice,
                                       s2_));
          hipCheckError(hipMemcpyAsync(B_vals_device_,
                                       B_vals_,
                                       sizeof(T) * nnzB_,
                                       hipMemcpyHostToDevice,
                                       s2_));
          hipCheckError(hipDeviceSynchronize());
          break;
        }
        case gpuOffloadType::unified: {
          if (print_) std::cout << "\tMoving data to GPU" << std::endl;
          hipCheckError(hipMemPrefetchAsync(A_rows_, 
                                            sizeof(rocsparse_int) * (m_ + 1), 
                                            gpuDevice_, 
                                            s1_));
          hipCheckError(hipMemPrefetchAsync(A_cols_, 
                                            sizeof(rocsparse_int) * nnzA_, 
                                            gpuDevice_, 
                                            s1_));
          hipCheckError(hipMemPrefetchAsync(A_vals_, 
                                            sizeof(T) * nnzA_, 
                                            gpuDevice_, 
                                            s1_));
          hipCheckError(hipMemPrefetchAsync(B_rows_, 
                                            sizeof(rocsparse_int) * (k_ + 1), 
                                            gpuDevice_, 
                                            s2_));
          hipCheckError(hipMemPrefetchAsync(B_cols_, 
                                            sizeof(rocsparse_int) * nnzB_, 
                                            gpuDevice_, 
                                            s2_));
          hipCheckError(hipMemPrefetchAsync(B_vals_, 
                                            sizeof(T) * nnzB_, 
                                            gpuDevice_, 
                                            s2_));
          hipCheckError(hipDeviceSynchronize());
          break;
        }
      }

      // Set the pointer mode for rocsparse
      status_ = rocsparse_set_pointer_mode(handle_,
                                           rocsparse_pointer_mode_host);
    }

    void callSpmm() override {
      switch (offload_) {
        case gpuOffloadType::always: {
          if (print_) std::cout << "\tMoving data to GPU" << std::endl;
          hipCheckError(hipMemcpyAsync(A_rows_device_,
                                       A_rows_,
                                       sizeof(rocsparse_int) * (m_ + 1),
                                       hipMemcpyHostToDevice,
                                       s1_));
          hipCheckError(hipMemcpyAsync(A_cols_device_,
                                       A_cols_,
                                       sizeof(rocsparse_int) * nnzA_,
                                       hipMemcpyHostToDevice,
                                       s1_));
          hipCheckError(hipMemcpyAsync(A_vals_device_,
                                       A_vals_,
                                       sizeof(T) * nnzA_,
                                       hipMemcpyHostToDevice,
                                       s1_));
          hipCheckError(hipMemcpyAsync(B_rows_device_,
                                       B_rows_,
                                       sizeof(rocsparse_int) * (k_ + 1),
                                       hipMemcpyHostToDevice,
                                       s2_));
          hipCheckError(hipMemcpyAsync(B_cols_device_,
                                       B_cols_,
                                       sizeof(rocsparse_int) * nnzB_,
                                       hipMemcpyHostToDevice,
                                       s2_));
          hipCheckError(hipMemcpyAsync(B_vals_device_,
                                       B_vals_,
                                       sizeof(T) * nnzB_,
                                       hipMemcpyHostToDevice,
                                       s2_));
          hipCheckError(hipDeviceSynchronize());
          size_t buffer_size;

          // Set up the rocSPARSE structures for the MM
          status_ = rocsparse_create_mat_descr(&description_A_); // The defaults are for base=0, and type=general.  This is okay for us.
          checkStatus("Failed rocsparse_create_mat_descr");
          status_ = rocsparse_create_mat_descr(&description_B_); // The defaults are for base=0, and type=general.  This is okay for us.
          checkStatus("Failed rocsparse_create_mat_descr");
          status_ = rocsparse_create_mat_descr(&description_C_); // The defaults are for base=0, and type=general.  This is okay for us.
          checkStatus("Failed rocsparse_create_mat_descr");

          status_ = rocsparse_create_mat_info(&info_);
          checkStatus("Failed rocsparse_create_mat_info");

          if constexpr (std::is_same_v<T, float>) {
            status_ = rocsparse_scsrgemm_buffer_size(handle_,
                                                     operation_,
                                                     operation_,
                                                     m_roc_,
                                                     n_roc_,
                                                     k_roc_,
                                                     &alpha,
                                                     description_A_,
                                                     nnzA_roc_,
                                                     A_rows_device_,
                                                     A_cols_device_,
                                                     description_B_,
                                                     nnzB_roc_,
                                                     B_rows_device_,
                                                     B_cols_device_,
                                                     &beta,
                                                     nullptr,
                                                     0,
                                                     nullptr,
                                                     nullptr,
                                                     info_,
                                                     &buffer_size);
            checkStatus("Failed rocsparse_scsrgemm_buffer_size");
          } else if constexpr (std::is_same_v<T, double>) {
            status_ = rocsparse_dcsrgemm_buffer_size(handle_,
                                                     operation_,
                                                     operation_,
                                                     m_roc_,
                                                     n_roc_,
                                                     k_roc_,
                                                     &alpha,
                                                     description_A_,
                                                     nnzA_roc_,
                                                     A_rows_device_,
                                                     A_cols_device_,
                                                     description_B_,
                                                     nnzB_roc_,
                                                     B_rows_device_,
                                                     B_cols_device_,
                                                     &beta,
                                                     nullptr,
                                                     0,
                                                     nullptr,
                                                     nullptr,
                                                     info_,
                                                     &buffer_size);
            checkStatus("Failed rocsparse_dcsrgemm_buffer_size");
          }

          void* buffer;
          hipCheckError(hipMalloc(&buffer, buffer_size));

          rocsparse_int nnzC_roc_;
          hipCheckError(hipMalloc((void**)&C_rows_device_, sizeof(rocsparse_int) * (m_ + 1)));

          status_ = rocsparse_csrgemm_nnz(handle_,
                                          operation_,
                                          operation_,
                                          m_roc_,
                                          n_roc_,
                                          k_roc_,
                                          description_A_,
                                          nnzA_roc_,
                                          A_rows_device_,
                                          A_cols_device_,
                                          description_B_,
                                          nnzB_roc_,
                                          B_rows_device_,
                                          B_cols_device_,
                                          nullptr,
                                          0,
                                          nullptr,
                                          nullptr,
                                          description_C_,
                                          C_rows_device_,
                                          &nnzC_roc_,
                                          info_,
                                          buffer);
          checkStatus("Failed rocsparse_csrgemm_nnz");

          hipCheckError(hipMalloc((void**)&C_cols_device_, sizeof(rocsparse_int) * nnzC_roc_));
          hipCheckError(hipMalloc((void**)&C_vals_device_, sizeof(T) * nnzC_roc_));

          if constexpr (std::is_same_v<T, float>) {
            status_ = rocsparse_scsrgemm(handle_,
                                         operation_,
                                         operation_,
                                         m_roc_,
                                         n_roc_,
                                         k_roc_,
                                         &alpha,
                                         description_A_,
                                         nnzA_roc_,
                                         A_vals_device_,
                                         A_rows_device_,
                                         A_cols_device_,
                                         description_B_,
                                         nnzB_roc_,
                                         B_vals_device_,
                                         B_rows_device_,
                                         B_cols_device_,
                                         &beta,
                                         nullptr,
                                         0,
                                         nullptr,
                                         nullptr,
                                         nullptr,
                                         description_C_,
                                         C_vals_device_,
                                         C_rows_device_,
                                         C_cols_device_,
                                         info_,
                                         buffer);
            checkStatus("Failed rocsparse_scsrgemm");
          } else if constexpr (std::is_same_v<T, double>) {
            status_ = rocsparse_dcsrgemm(handle_,
                                         operation_,
                                         operation_,
                                         m_roc_,
                                         n_roc_,
                                         k_roc_,
                                         &alpha,
                                         description_A_,
                                         nnzA_roc_,
                                         A_vals_device_,
                                         A_rows_device_,
                                         A_cols_device_,
                                         description_B_,
                                         nnzB_roc_,
                                         B_vals_device_,
                                         B_rows_device_,
                                         B_cols_device_,
                                         &beta,
                                         nullptr,
                                         0,
                                         nullptr,
                                         nullptr,
                                         nullptr,
                                         description_C_,
                                         C_vals_device_,
                                         C_rows_device_,
                                         C_cols_device_,
                                         info_,
                                         buffer);
            checkStatus("Failed rocsparse_dcsrgemm");
          }
          // Freeing up buffer
          hipCheckError(hipFree(buffer));



          // Allocate host arrays for C
          hipCheckError(hipHostMalloc((void**)&C_rows_, sizeof(rocsparse_int) * (m_ + 1)));
          hipCheckError(hipHostMalloc((void**)&C_cols_, sizeof(rocsparse_int) * nnzC_roc_));
          hipCheckError(hipHostMalloc((void**)&C_vals_, sizeof(T) * nnzC_roc_));
          hipCheckError(hipDeviceSynchronize());

          // Moving data to CPU
          hipCheckError(hipMemcpyAsync(C_rows_,
                                       C_rows_device_,
                                       sizeof(rocsparse_int) * (m_ + 1),
                                       hipMemcpyDeviceToHost,
                                       s1_));
          hipCheckError(hipMemcpyAsync(C_cols_,
                                       C_cols_device_,
                                       sizeof(rocsparse_int) * nnzC_roc_,
                                       hipMemcpyDeviceToHost,
                                       s2_));
          hipCheckError(hipMemcpyAsync(C_vals_,
                                       C_vals_device_,
                                       sizeof(T) * nnzC_roc_,
                                       hipMemcpyDeviceToHost,
                                       s3_));
          hipCheckError(hipDeviceSynchronize());

          // Freeing stuff up
          hipCheckError(hipFree(C_rows_device_));
          hipCheckError(hipFree(C_cols_device_));
          hipCheckError(hipFree(C_vals_device_));
          hipCheckError(hipFree(C_rows_));
          hipCheckError(hipFree(C_cols_));
          hipCheckError(hipFree(C_vals_));
          break;
        }
        case gpuOffloadType::once: {
          size_t buffer_size;
          // Check if there are old arrays to get rid of
          if (!firstRun_) {
            hipCheckError(hipFree(C_rows_device_));
            hipCheckError(hipFree(C_cols_device_));
            hipCheckError(hipFree(C_vals_device_));
          }

          if (print_) std::cout << "\tCreating rocSPARSE structures" << std::endl;
          // Set up the rocSPARSE structures for the MM
          status_ = rocsparse_create_mat_descr(&description_A_); // The defaults are for base=0, and type=general.  This is okay for us.
          checkStatus("Failed rocsparse_create_mat_descr");
          status_ = rocsparse_create_mat_descr(&description_B_); // The defaults are for base=0, and type=general.  This is okay for us.
          checkStatus("Failed rocsparse_create_mat_descr");
          status_ = rocsparse_create_mat_descr(&description_C_); // The defaults are for base=0, and type=general.  This is okay for us.
          checkStatus("Failed rocsparse_create_mat_descr");

          status_ = rocsparse_create_mat_info(&info_);
          checkStatus("Failed rocsparse_create_mat_info");

          if constexpr (std::is_same_v<T, float>) {
            status_ = rocsparse_scsrgemm_buffer_size(handle_,
                                                     operation_,
                                                     operation_,
                                                     m_roc_,
                                                     n_roc_,
                                                     k_roc_,
                                                     &alpha,
                                                     description_A_,
                                                     nnzA_roc_,
                                                     A_rows_device_,
                                                     A_cols_device_,
                                                     description_B_,
                                                     nnzB_roc_,
                                                     B_rows_device_,
                                                     B_cols_device_,
                                                     &beta,
                                                     nullptr,
                                                     0,
                                                     nullptr,
                                                     nullptr,
                                                     info_,
                                                     &buffer_size);
            checkStatus("Failed rocsparse_scsrgemm_buffer_size");
          } else if constexpr (std::is_same_v<T, double>) {
            status_ = rocsparse_dcsrgemm_buffer_size(handle_,
                                                     operation_,
                                                     operation_,
                                                     m_roc_,
                                                     n_roc_,
                                                     k_roc_,
                                                     &alpha,
                                                     description_A_,
                                                     nnzA_roc_,
                                                     A_rows_device_,
                                                     A_cols_device_,
                                                     description_B_,
                                                     nnzB_roc_,
                                                     B_rows_device_,
                                                     B_cols_device_,
                                                     &beta,
                                                     nullptr,
                                                     0,
                                                     nullptr,
                                                     nullptr,
                                                     info_,
                                                     &buffer_size);
            checkStatus("Failed rocsparse_dcsrgemm_buffer_size");
          }

          void* buffer;
          hipCheckError(hipMalloc(&buffer, buffer_size));

          rocsparse_int nnzC_roc_;
          hipCheckError(hipMalloc((void**)&C_rows_device_, sizeof(rocsparse_int) * (m_ + 1)));

          status_ = rocsparse_csrgemm_nnz(handle_,
                                          operation_,
                                          operation_,
                                          m_roc_,
                                          n_roc_,
                                          k_roc_,
                                          description_A_,
                                          nnzA_roc_,
                                          A_rows_device_,
                                          A_cols_device_,
                                          description_B_,
                                          nnzB_roc_,
                                          B_rows_device_,
                                          B_cols_device_,
                                          nullptr,
                                          0,
                                          nullptr,
                                          nullptr,
                                          description_C_,
                                          C_rows_device_,
                                          &nnzC_roc_,
                                          info_,
                                          buffer);
          checkStatus("Failed rocsparse_csrgemm_nnz");

          hipCheckError(hipMalloc((void**)&C_cols_device_, sizeof(rocsparse_int) * nnzC_roc_));
          hipCheckError(hipMalloc((void**)&C_vals_device_, sizeof(T) * nnzC_roc_));

          if constexpr (std::is_same_v<T, float>) {
            status_ = rocsparse_scsrgemm(handle_,
                                         operation_,
                                         operation_,
                                         m_roc_,
                                         n_roc_,
                                         k_roc_,
                                         &alpha,
                                         description_A_,
                                         nnzA_roc_,
                                         A_vals_device_,
                                         A_rows_device_,
                                         A_cols_device_,
                                         description_B_,
                                         nnzB_roc_,
                                         B_vals_device_,
                                         B_rows_device_,
                                         B_cols_device_,
                                         &beta,
                                         nullptr,
                                         0,
                                         nullptr,
                                         nullptr,
                                         nullptr,
                                         description_C_,
                                         C_vals_device_,
                                         C_rows_device_,
                                         C_cols_device_,
                                         info_,
                                         buffer);
            checkStatus("Failed rocsparse_scsrgemm");
          } else if constexpr (std::is_same_v<T, double>) {
            status_ = rocsparse_dcsrgemm(handle_,
                                         operation_,
                                         operation_,
                                         m_roc_,
                                         n_roc_,
                                         k_roc_,
                                         &alpha,
                                         description_A_,
                                         nnzA_roc_,
                                         A_vals_device_,
                                         A_rows_device_,
                                         A_cols_device_,
                                         description_B_,
                                         nnzB_roc_,
                                         B_vals_device_,
                                         B_rows_device_,
                                         B_cols_device_,
                                         &beta,
                                         nullptr,
                                         0,
                                         nullptr,
                                         nullptr,
                                         nullptr,
                                         description_C_,
                                         C_vals_device_,
                                         C_rows_device_,
                                         C_cols_device_,
                                         info_,
                                         buffer);
            checkStatus("Failed rocsparse_dcsrgemm");
          }
          // Freeing up buffer
          hipCheckError(hipFree(buffer));

          if (print_) std::cout << "\tdestroying rocSPARSE structures" << std::endl;
          // Now clean up
          status_ = rocsparse_destroy_mat_descr(description_A_);
          checkStatus("Failed rocsparse_destroy_mat_descr");
          status_ = rocsparse_destroy_mat_descr(description_B_);
          checkStatus("Failed rocsparse_destroy_mat_descr");
          status_ = rocsparse_destroy_mat_descr(description_C_);
          checkStatus("Failed rocsparse_destroy_mat_descr");
          status_ = rocsparse_destroy_mat_info(info_);
          checkStatus("Failed rocsparse_destroy_mat_info");
          firstRun_ = false;
          break;
        }
        case gpuOffloadType::unified: {
          size_t buffer_size;

          // Check if there are old arrays to get rid of
          if (!firstRun_) {
            hipCheckError(hipFree(C_rows_device_));
            hipCheckError(hipFree(C_cols_device_));
            hipCheckError(hipFree(C_vals_device_));
          }

          // Set up the rocSPARSE structures for the MM
          status_ = rocsparse_create_mat_descr(&description_A_); // The defaults are for base=0, and type=general.  This is okay for us.
          checkStatus("Failed rocsparse_create_mat_descr");
          status_ = rocsparse_create_mat_descr(&description_B_); // The defaults are for base=0, and type=general.  This is okay for us.
          checkStatus("Failed rocsparse_create_mat_descr");
          status_ = rocsparse_create_mat_descr(&description_C_); // The defaults are for base=0, and type=general.  This is okay for us.
          checkStatus("Failed rocsparse_create_mat_descr");

          status_ = rocsparse_create_mat_info(&info_);
          checkStatus("Failed rocsparse_create_mat_info");

          if constexpr (std::is_same_v<T, float>) {
            status_ = rocsparse_scsrgemm_buffer_size(handle_,
                                                     operation_,
                                                     operation_,
                                                     m_roc_,
                                                     n_roc_,
                                                     k_roc_,
                                                     &alpha,
                                                     description_A_,
                                                     nnzA_roc_,
                                                     A_rows_,
                                                     A_cols_,
                                                     description_B_,
                                                     nnzB_roc_,
                                                     B_rows_,
                                                     B_cols_,
                                                     &beta,
                                                     nullptr,
                                                     0,
                                                     nullptr,
                                                     nullptr,
                                                     info_,
                                                     &buffer_size);
            checkStatus("Failed rocsparse_scsrgemm_buffer_size");
          } else if constexpr (std::is_same_v<T, double>) {
            status_ = rocsparse_dcsrgemm_buffer_size(handle_,
                                                     operation_,
                                                     operation_,
                                                     m_roc_,
                                                     n_roc_,
                                                     k_roc_,
                                                     &alpha,
                                                     description_A_,
                                                     nnzA_roc_,
                                                     A_rows_,
                                                     A_cols_,
                                                     description_B_,
                                                     nnzB_roc_,
                                                     B_rows_,
                                                     B_cols_,
                                                     &beta,
                                                     nullptr,
                                                     0,
                                                     nullptr,
                                                     nullptr,
                                                     info_,
                                                     &buffer_size);
            checkStatus("Failed rocsparse_dcsrgemm_buffer_size");
          }

          void* buffer;
          hipCheckError(hipMallocManaged(&buffer, buffer_size));

          rocsparse_int nnzC_roc_;
          hipCheckError(hipMallocManaged(&C_rows_device_, sizeof(rocsparse_int), (m_ + 1)));

          status_ = rocsparse_csrgemm_nnz(handle_,
                                          operation_,
                                          operation_,
                                          m_roc_,
                                          n_roc_,
                                          k_roc_,
                                          description_A_,
                                          nnzA_roc_,
                                          A_rows_,
                                          A_cols_,
                                          description_B_,
                                          nnzB_roc_,
                                          B_rows_,
                                          B_cols_,
                                          nullptr,
                                          0,
                                          nullptr,
                                          nullptr,
                                          description_C_,
                                          C_rows_,
                                          &nnzC_roc_,
                                          info_,
                                          buffer);
          checkStatus("Failed rocsparse_csrgemm_nnz");

          hipCheckError(hipMallocManaged(&C_cols_device_, sizeof(rocsparse_int) * nnzC_roc_));
          hipCheckError(hipMallocManaged(&C_vals_device_, sizeof(T) * nnzC_roc_));

          if constexpr (std::is_same_v<T, float>) {
            status_ = rocsparse_scsrgemm(handle_,
                                         operation_,
                                         operation_,
                                         m_roc_,
                                         n_roc_,
                                         k_roc_,
                                         &alpha,
                                         description_A_,
                                         nnzA_roc_,
                                         A_vals_,
                                         A_rows_,
                                         A_cols_,
                                         description_B_,
                                         nnzB_roc_,
                                         B_vals_,
                                         B_rows_,
                                         B_cols_,
                                         &beta,
                                         nullptr,
                                         0,
                                         nullptr,
                                         nullptr,
                                         nullptr,
                                         description_C_,
                                         C_vals_,
                                         C_rows_,
                                         C_cols_,
                                         info_,
                                         buffer);
            checkStatus("Failed rocsparse_scsrgemm");
          } else if constexpr (std::is_same_v<T, double>) {
            status_ = rocsparse_dcsrgemm(handle_,
                                         operation_,
                                         operation_,
                                         m_roc_,
                                         n_roc_,
                                         k_roc_,
                                         &alpha,
                                         description_A_,
                                         nnzA_roc_,
                                         A_vals_,
                                         A_rows_,
                                         A_cols_,
                                         description_B_,
                                         nnzB_roc_,
                                         B_vals_,
                                         B_rows_,
                                         B_cols_,
                                         &beta,
                                         nullptr,
                                         0,
                                         nullptr,
                                         nullptr,
                                         nullptr,
                                         description_C_,
                                         C_vals_,
                                         C_rows_,
                                         C_cols_,
                                         info_,
                                         buffer);
            checkStatus("Failed rocsparse_dcsrgemm");
          }

          // Freeing up buffer
          hipCheckError(hipFree(buffer));

          if (print_) std::cout << "\tdestroying rocSPARSE structures" << std::endl;
          // Now clean up
          status_ = rocsparse_destroy_mat_descr(description_A_);
          checkStatus("Failed rocsparse_destroy_mat_descr");
          status_ = rocsparse_destroy_mat_descr(description_B_);
          checkStatus("Failed rocsparse_destroy_mat_descr");
          status_ = rocsparse_destroy_mat_descr(description_C_);
          checkStatus("Failed rocsparse_destroy_mat_descr");
          status_ = rocsparse_destroy_mat_info(info_);
          checkStatus("Failed rocsparse_destroy_mat_info");
          firstRun_ = false;
          break;
        }
      }
    }

    void postLoopRequirements() override {
      if (print_) std::cout << "Post-Loop stuff" << std::endl;
      switch(offload_) {
        case gpuOffloadType::always: {
          break;
        }
        case gpuOffloadType::once: {
          if (print_) std::cout << "\tAllocating host arrays for C" << std::endl;
          // Allocate host arrays for C
          hipCheckError(hipHostMalloc((void**)&C_rows_, sizeof(rocsparse_int) * (m_ + 1)));
          hipCheckError(hipHostMalloc((void**)&C_cols_, sizeof(rocsparse_int) * nnzC_roc_));
          hipCheckError(hipHostMalloc((void**)&C_vals_, sizeof(T) * nnzC_roc_));
          hipCheckError(hipDeviceSynchronize());


          if (print_) std::cout << "\tMoving C data to host" << std::endl;
          // Moving data to CPU
          hipCheckError(hipMemcpyAsync(C_rows_,
                                       C_rows_device_,
                                       sizeof(rocsparse_int) * (m_ + 1),
                                       hipMemcpyDeviceToHost,
                                       s1_));
          hipCheckError(hipMemcpyAsync(C_cols_,
                                       C_cols_device_,
                                       sizeof(rocsparse_int) * nnzC_roc_,
                                       hipMemcpyDeviceToHost,
                                       s2_));
          hipCheckError(hipMemcpyAsync(C_vals_,
                                       C_vals_device_,
                                       sizeof(T) * nnzC_roc_,
                                       hipMemcpyDeviceToHost,
                                       s3_));
          hipCheckError(hipDeviceSynchronize());

          if (print_) std::cout << "\tFreeing C arrays" << std::endl;
          // Freeing stuff up
          hipCheckError(hipFree(C_rows_device_));
          hipCheckError(hipFree(C_cols_device_));
          hipCheckError(hipFree(C_vals_device_));
          hipCheckError(hipFree(C_rows_));
          hipCheckError(hipFree(C_cols_));
          hipCheckError(hipFree(C_vals_));
          break;
        }
        case gpuOffloadType::unified: {
          if (print_) std::cout << "\tMoving data to CPU" << std::endl;
          hipCheckError(hipMemPrefetchAsync(C_rows_,
                                            sizeof(rocsparse_int) * (m_ + 1),
                                            hipCpuDeviceId,
                                            s1_));
          hipCheckError(hipMemPrefetchAsync(C_cols_,
                                            sizeof(rocsparse_int) * nnzC_roc_,
                                            hipCpuDeviceId,
                                            s2_));
          hipCheckError(hipMemPrefetchAsync(C_vals_,
                                            sizeof(T) * nnzC_roc_,
                                            hipCpuDeviceId,
                                            s3_));
          hipCheckError(hipDeviceSynchronize());
          if (print_) std::cout << "\tFreeing C arrays" << std::endl;
          hipCheckError(hipFree(C_rows_));
          hipCheckError(hipFree(C_cols_));
          hipCheckError(hipFree(C_vals_));
          break;
        }
      }
    }

    void postCallKernelCleanup() override {
      if (print_) std::cout << "Post-kernel clean up" << std::endl;
      if (offload_ == gpuOffloadType::unified) {
        if (print_) std::cout << "Freeing unified memory arrays for A and B" << std::endl;
        hipCheckError(hipFree(A_));
        hipCheckError(hipFree(A_rows_));
        hipCheckError(hipFree(A_cols_));
        hipCheckError(hipFree(A_vals_));
        hipCheckError(hipFree(B_));
        hipCheckError(hipFree(B_rows_));
        hipCheckError(hipFree(B_cols_));
        hipCheckError(hipFree(B_vals_));
      } else {
        if (print_) std::cout << "Freeing host arrays for A and B" << std::endl;
        hipCheckError(hipHostFree((void*)A_));
        hipCheckError(hipHostFree((void*)A_rows_));
        hipCheckError(hipHostFree((void*)A_cols_));
        hipCheckError(hipHostFree((void*)A_vals_));
        hipCheckError(hipHostFree((void*)B_));
        hipCheckError(hipHostFree((void*)B_rows_));
        hipCheckError(hipHostFree((void*)B_cols_));
        hipCheckError(hipHostFree((void*)B_vals_));
        if (print_) std::cout << "Freeing GPU arrays for A and B" << std::endl;
        hipCheckError(hipFree(A_rows_device_));
        hipCheckError(hipFree(A_cols_device_));
        hipCheckError(hipFree(A_vals_device_));
        hipCheckError(hipFree(B_rows_device_));
        hipCheckError(hipFree(B_cols_device_));
        hipCheckError(hipFree(B_vals_device_));
      }
    }

    void checkStatus(std::string message) {
      if (status_ != rocsparse_status_success) {
        std::cerr << message << std::endl;
        exit(1);
      }
    }

    bool print_ = false;
    bool initialised_ = false;
    bool firstRun_ = false;

    rocsparse_handle handle_;
    rocsparse_operation operation_;
    rocsparse_mat_info info_;
    rocsparse_status status_;
    rocsparse_index_base index_;
    rocsparse_matrix_type type_;


    rocsparse_int m_roc_, n_roc_, k_roc_, nnzA_roc_, nnzB_roc_, nnzC_roc_;
    
    rocsparse_mat_descr description_A_, description_B_, description_C_;

    rocsparse_int* A_rows_;
    rocsparse_int* A_cols_;
    T* A_vals_;
    rocsparse_int* B_rows_;
    rocsparse_int* B_cols_;
    T* B_vals_;
    rocsparse_int* C_rows_;
    rocsparse_int* C_cols_;
    T* C_vals_;


    rocsparse_int* A_rows_device_;
    rocsparse_int* A_cols_device_;
    T* A_vals_device_;
    rocsparse_int* B_rows_device_;
    rocsparse_int* B_cols_device_;
    T* B_vals_device_;
    rocsparse_int* C_rows_device_;
    rocsparse_int* C_cols_device_;
    T* C_vals_device_;

    int gpuDevice_;
    hipStream_t s1_, s2_, s3_;
    

    const T alpha = ALPHA;
    const T beta = BETA;
};
}

#endif
