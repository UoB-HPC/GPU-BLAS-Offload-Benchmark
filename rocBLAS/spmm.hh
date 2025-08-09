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
    using spmm<T>::A_nnz_;
    using spmm<T>::B_nnz_;
    using spmm<T>::m_;
    using spmm<T>::n_;
    using spmm<T>::k_;
    using spmm<T>::A_;
    using spmm<T>::B_;
    using spmm<T>::C_;
    using spmm<T>::offload_;
    using spmm<T>::sparsity_;

    ~spmm_gpu() {
      if (initialised_) {
        status_ = rocsparse_destroy_handle(handle_);
        checkStatus("Failed rocsparse_destroy_handle");
        hipCheckError(hipStreamDestroy(stream_));
        initialised_ = false;
      }
    }

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
      A_nnz_ = 1 + (int64_t)((double)m_ * (double)k_ * (1.0 - sparsity_));
      B_nnz_ = 1 + (int64_t)((double)k_ * (double)n_ * (1.0 - sparsity_));

      // Set up rocSPARSE metadata
      index_ = rocsparse_indextype_i64;
      base_ = rocsparse_index_base_zero;
      type_ = rocsparse_matrix_type_general;
      operation_ = rocsparse_operation_none;
      algorithm_ = rocsparse_spgemm_alg_default;

      if constexpr (std::is_same_v<T, float>) {
        dataType_ = rocsparse_datatype_f32_r;
      } else if constexpr (std::is_same_v<T, double>) {
        dataType_ = rocsparse_datatype_f64_r;
      } else {
        static_assert("Unsupported data type for rocSPARSE");
      }

      if (print_) std::cout << "\tAbout to set up handle and hip streams" << std::endl;
      if (!initialised_) {
        status_ = rocsparse_create_handle(&handle_);
        checkStatus("Failed rocsparse_create_handle");

        // Get the GPU
        hipCheckError(hipGetDevice(&gpuDevice_));
        // Make streams for asynchronous GPU comunication
        hipCheckError(hipStreamCreate(&stream_));

        status_ = rocsparse_set_stream(handle_, stream_);
        checkStatus("Failed rocsparse_get_stream");
      }

      if (print_) std::cout << "\tAbout to malloc arrays" << std::endl;
      if (offload_ == gpuOffloadType::unified) {
        hipCheckError(hipMallocManaged(&A_, sizeof(T) * m_ * k_));
        hipCheckError(hipMallocManaged(&A_rows_, sizeof(int64_t) * (m_ + 1)));
        hipCheckError(hipMallocManaged(&A_cols_, sizeof(int64_t) * A_nnz_));
        hipCheckError(hipMallocManaged(&A_vals_, sizeof(T) * A_nnz_));
        hipCheckError(hipMallocManaged(&B_, sizeof(T) * k_ * n_));
        hipCheckError(hipMallocManaged(&B_rows_, sizeof(int64_t) * (k_ + 1)));
        hipCheckError(hipMallocManaged(&B_cols_, sizeof(int64_t) * B_nnz_));
        hipCheckError(hipMallocManaged(&B_vals_, sizeof(T) * B_nnz_));
        hipCheckError(hipMallocManaged(&D_rows_, sizeof(int64_t) * (m_ + 1)));
        hipCheckError(hipMallocManaged(&D_cols_, sizeof(int64_t) * D_nnz_));
        hipCheckError(hipMallocManaged(&D_vals_, sizeof(T) * D_nnz_));

        hipCheckError(hipDeviceSynchronize());
      } else {
        // Host data structures
        hipCheckError(hipHostMalloc(&A_, sizeof(T) * m_ * k_));
        hipCheckError(hipHostMalloc(&A_rows_, sizeof(int64_t) * (m_ + 1)));
        hipCheckError(hipHostMalloc(&A_cols_, sizeof(int64_t) * A_nnz_));
        hipCheckError(hipHostMalloc(&A_vals_, sizeof(T) * A_nnz_));
        hipCheckError(hipHostMalloc(&B_, sizeof(T) * k_ * n_));
        hipCheckError(hipHostMalloc(&B_rows_, sizeof(int64_t) * (k_ + 1)));
        hipCheckError(hipHostMalloc(&B_cols_, sizeof(int64_t) * B_nnz_));
        hipCheckError(hipHostMalloc(&B_vals_, sizeof(T) * B_nnz_));
        hipCheckError(hipHostMalloc(&D_rows_, sizeof(int64_t) * (m_ + 1)));
        hipCheckError(hipHostMalloc(&D_cols_, sizeof(int64_t) * D_nnz_));
        hipCheckError(hipHostMalloc(&D_vals_, sizeof(T) * D_nnz_));
        hipCheckError(hipDeviceSynchronize());

        // GPU data structures
        hipCheckError(hipMalloc(&A_rows_device_, sizeof(int64_t) * (m_ + 1)));
        hipCheckError(hipMalloc(&A_cols_device_, sizeof(int64_t) * A_nnz_));
        hipCheckError(hipMalloc(&A_vals_device_, sizeof(T) * A_nnz_));
        hipCheckError(hipMalloc(&B_rows_device_, sizeof(int64_t) * (k_ + 1)));
        hipCheckError(hipMalloc(&B_cols_device_, sizeof(int64_t) * B_nnz_));
        hipCheckError(hipMalloc(&B_vals_device_, sizeof(T) * B_nnz_));
        hipCheckError(hipMalloc(&D_rows_device_, sizeof(int64_t) * (m_ + 1)));
        hipCheckError(hipMalloc(&D_cols_device_, sizeof(int64_t) * D_nnz_));
        hipCheckError(hipMalloc(&D_vals_device_, sizeof(T) * D_nnz_));
        hipCheckError(hipDeviceSynchronize());
      }


      if (print_) std::cout << "\tInitialising matrices" << std::endl;
      uint64_t outputNNZ = 0;
      while (outputNNZ == 0) {
        initInputMatrices();
        outputNNZ = calcNNZC();
      }
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
      if (nnz_encountered != A_nnz_) {
        std::cerr << "Warning: A matrix has " << nnz_encountered << " non-zeros, expected " << A_nnz_ << std::endl;
        A_nnz_ = nnz_encountered;  // Update to actual count
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
      if (nnz_encountered != B_nnz_) {
        std::cerr << "Warning: B matrix has " << nnz_encountered << " non-zeros, expected " << B_nnz_ << std::endl;
        B_nnz_ = nnz_encountered;  // Update to actual count
      }

      // Make D a possible matrix
      D_cols_[0] = 0;
      D_vals_[0] = 1.0;
      D_rows_[0] = 0;
      for (size_t i = 1; i < (m_ + 1); i++) {
        D_rows_[i] = 1; 
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
                                       sizeof(int64_t) * (m_ + 1),
                                       hipMemcpyHostToDevice,
                                       stream_));
          hipCheckError(hipMemcpyAsync(A_cols_device_,
                                       A_cols_,
                                       sizeof(int64_t) * A_nnz_,
                                       hipMemcpyHostToDevice,
                                       stream_));
          hipCheckError(hipMemcpyAsync(A_vals_device_,
                                       A_vals_,
                                       sizeof(T) * A_nnz_,
                                       hipMemcpyHostToDevice,
                                       stream_));
          hipCheckError(hipMemcpyAsync(B_rows_device_,
                                       B_rows_,
                                       sizeof(int64_t) * (k_ + 1),
                                       hipMemcpyHostToDevice,
                                       stream_));
          hipCheckError(hipMemcpyAsync(B_cols_device_,
                                       B_cols_,
                                       sizeof(int64_t) * B_nnz_,
                                       hipMemcpyHostToDevice,
                                       stream_));
          hipCheckError(hipMemcpyAsync(B_vals_device_,
                                       B_vals_,
                                       sizeof(T) * B_nnz_,
                                       hipMemcpyHostToDevice,
                                       stream_));
          hipCheckError(hipMemcpyAsync(D_rows_device_,
                                       D_rows_,
                                       sizeof(int64_t) * (m_ + 1),
                                       hipMemcpyHostToDevice,
                                       stream_));
          hipCheckError(hipMemcpyAsync(D_cols_device_,
                                       D_cols_,
                                       sizeof(int64_t) * D_nnz_,
                                       hipMemcpyHostToDevice,
                                       stream_));
          hipCheckError(hipMemcpyAsync(D_vals_device_,
                                       D_vals_,
                                       sizeof(T) * D_nnz_,
                                       hipMemcpyHostToDevice,
                                       stream_));
          hipCheckError(hipDeviceSynchronize());
          break;
        case gpuOffloadType::unified: {
          if (print_) std::cout << "\tMoving data to GPU" << std::endl;
          hipCheckError(hipMemPrefetchAsync(A_rows_, 
                                            sizeof(int64_t) * (m_ + 1), 
                                            gpuDevice_, 
                                            stream_));
          hipCheckError(hipMemPrefetchAsync(A_cols_, 
                                            sizeof(int64_t) * A_nnz_, 
                                            gpuDevice_, 
                                            stream_));
          hipCheckError(hipMemPrefetchAsync(A_vals_, 
                                            sizeof(T) * A_nnz_, 
                                            gpuDevice_, 
                                            stream_));
          hipCheckError(hipMemPrefetchAsync(B_rows_, 
                                            sizeof(int64_t) * (k_ + 1), 
                                            gpuDevice_, 
                                            stream_));
          hipCheckError(hipMemPrefetchAsync(B_cols_, 
                                            sizeof(int64_t) * B_nnz_, 
                                            gpuDevice_, 
                                            stream_));
          hipCheckError(hipMemPrefetchAsync(B_vals_, 
                                            sizeof(T) * B_nnz_, 
                                            gpuDevice_, 
                                            stream_));
          hipCheckError(hipMemPrefetchAsync(D_rows_, 
                                            sizeof(int64_t) * (m_ + 1), 
                                            gpuDevice_, 
                                            stream_));
          hipCheckError(hipMemPrefetchAsync(D_cols_, 
                                            sizeof(int64_t) * D_nnz_, 
                                            gpuDevice_, 
                                            stream_));
          hipCheckError(hipMemPrefetchAsync(D_vals_, 
                                            sizeof(T) * D_nnz_, 
                                            gpuDevice_, 
                                            stream_));
          hipCheckError(hipDeviceSynchronize());
          break;
        }
      }
    }

    void callSpmm() override {
      if (print_) std::cout << "Calling spmm kernel" << std::endl;
      switch (offload_) {
        case gpuOffloadType::unified: {
          size_t buffer_size = 0;
          // Check if there are old arrays to get rid of
          if (!firstRun_) {
            hipCheckError(hipFree(C_rows_));
            hipCheckError(hipFree(C_cols_));
            hipCheckError(hipFree(C_vals_));
            hipCheckError(hipDeviceSynchronize());
          }
          if (print_) std::cout << "\tAllocating C rows" << std::endl;
          hipCheckError(hipMallocManaged(&C_rows_, sizeof(int64_t) * (m_ + 1)));
          hipCheckError(hipDeviceSynchronize());

          // Set up the rocSPARSE structures for the MM
          if (print_) std::cout << "\tCreating csr descriptions" << std::endl;
          status_ = rocsparse_create_csr_descr(&description_A_, m_, k_, A_nnz_, A_rows_, A_cols_,
                                               A_vals_, index_, index_, base_, dataType_);
          checkStatus("Failed rocsparse_create_csr_descr");
          status_ = rocsparse_create_csr_descr(&description_B_, k_, n_, B_nnz_, B_rows_, B_cols_,
                                               B_vals_, index_, index_, base_, dataType_);
          checkStatus("Failed rocsparse_create_csr_descr");
          status_ = rocsparse_create_csr_descr(&description_C_, m_, n_, 0, C_rows_, nullptr,
                                               nullptr, index_, index_, base_, dataType_);
          checkStatus("Failed rocsparse_create_csr_descr");
          status_ = rocsparse_create_csr_descr(&description_D_, m_, n_, D_nnz_, D_rows_, D_cols_,
                                               D_vals_, index_, index_, base_, dataType_);
          checkStatus("Failed rocsparse_create_csr_descr");
          hipCheckError(hipDeviceSynchronize());

          if (print_) std::cout << "\tDetermining buffer size" << std::endl;
          stage_ = rocsparse_spgemm_stage_buffer_size;
          status_ = rocsparse_spgemm(handle_,
                                     operation_,
                                     operation_,
                                     &alpha,
                                     description_A_,
                                     description_B_,
                                     &beta,
                                     description_D_,
                                     description_C_,
                                     dataType_,
                                     algorithm_,
                                     stage_,
                                     &buffer_size,
                                     nullptr);
          checkStatus("Failed rocsparse_spgemm with stage=rocsparse_spgemm_stage_buffer_size");
          hipCheckError(hipDeviceSynchronize());

          if (print_) std::cout << "\tAllocating buffer and C_rows" << std::endl;
          void* buffer;
          hipCheckError(hipMallocManaged(&buffer, buffer_size));
          hipCheckError(hipDeviceSynchronize());

          if (print_) std::cout << "\tDetermining nnz" << std::endl;
          stage_ = rocsparse_spgemm_stage_nnz;
          status_ = rocsparse_spgemm(handle_,
                                     operation_,
                                     operation_,
                                     &alpha,
                                     description_A_,
                                     description_B_,
                                     &beta,
                                     description_D_,
                                     description_C_,
                                     dataType_,
                                     algorithm_,
                                     stage_,
                                     &buffer_size,
                                     buffer);
          checkStatus("Failed rocsparse_spgemm with stage=rocsparse_spgemm_stage_nnz");
          hipCheckError(hipDeviceSynchronize());

          if (print_) std::cout << "\tAllocating rows and vals" << std::endl;
          int64_t rowC, colC;
          status_ = rocsparse_spmat_get_size(description_C_, &rowC, &colC, &C_nnz_);
          checkStatus("Failed rocsparse_spmat_get_size");

          hipCheckError(hipMallocManaged(&C_cols_, sizeof(int64_t) * C_nnz_));
          hipCheckError(hipMallocManaged(&C_vals_, sizeof(T) * C_nnz_));
          hipCheckError(hipDeviceSynchronize());

          status_ = rocsparse_csr_set_pointers(description_C_, C_rows_, C_cols_, C_vals_);
          checkStatus("Failed rocsparse_csr_set_pointers");
          hipCheckError(hipDeviceSynchronize());

          if (print_) std::cout << "\tDoing calculation" << std::endl;
          stage_ = rocsparse_spgemm_stage_compute;
          status_ = rocsparse_spgemm(handle_,
                                     operation_,
                                     operation_,
                                     &alpha,
                                     description_A_,
                                     description_B_,
                                     &beta,
                                     description_D_,
                                     description_C_,
                                     dataType_,
                                     algorithm_,
                                     stage_,
                                     &buffer_size,
                                     buffer);
          checkStatus("Failed rocsparse_spgemm with stage=rocsparse_spgemm_stage_compute");
          hipCheckError(hipDeviceSynchronize());


          if (print_) std::cout << "\tFreeing buffer and descriptions etc." << std::endl;
          // Freeing up buffer
          hipCheckError(hipFree(buffer));
          status_ = rocsparse_destroy_spmat_descr(description_A_);
          checkStatus("Failing rocsparse_destroy_mat_descr for A");
          status_ = rocsparse_destroy_spmat_descr(description_B_);
          checkStatus("Failing rocsparse_destroy_mat_descr for B");
          status_ = rocsparse_destroy_spmat_descr(description_C_);
          checkStatus("Failing rocsparse_destroy_mat_descr for C");
          status_ = rocsparse_destroy_spmat_descr(description_D_);
          checkStatus("Failing rocsparse_destroy_mat_descr for D");
          hipCheckError(hipDeviceSynchronize());
          firstRun_ = false;
          break;
        }
        case gpuOffloadType::always: {
          if (print_) std::cout << "\tMoving data to GPU" << std::endl;
          hipCheckError(hipMemcpyAsync(A_rows_device_,
                                       A_rows_,
                                       sizeof(int64_t) * (m_ + 1),
                                       hipMemcpyHostToDevice,
                                       stream_));
          hipCheckError(hipMemcpyAsync(A_cols_device_,
                                       A_cols_,
                                       sizeof(int64_t) * A_nnz_,
                                       hipMemcpyHostToDevice,
                                       stream_));
          hipCheckError(hipMemcpyAsync(A_vals_device_,
                                       A_vals_,
                                       sizeof(T) * A_nnz_,
                                       hipMemcpyHostToDevice,
                                       stream_));
          hipCheckError(hipMemcpyAsync(B_rows_device_,
                                       B_rows_,
                                       sizeof(int64_t) * (k_ + 1),
                                       hipMemcpyHostToDevice,
                                       stream_));
          hipCheckError(hipMemcpyAsync(B_cols_device_,
                                       B_cols_,
                                       sizeof(int64_t) * B_nnz_,
                                       hipMemcpyHostToDevice,
                                       stream_));
          hipCheckError(hipMemcpyAsync(B_vals_device_,
                                       B_vals_,
                                       sizeof(T) * B_nnz_,
                                       hipMemcpyHostToDevice,
                                       stream_));
          hipCheckError(hipMemcpyAsync(D_rows_device_,
                                       D_rows_,
                                       sizeof(int64_t) * (m_ + 1),
                                       hipMemcpyHostToDevice,
                                       stream_));
          hipCheckError(hipMemcpyAsync(D_cols_device_,
                                       D_cols_,
                                       sizeof(int64_t) * D_nnz_,
                                       hipMemcpyHostToDevice,
                                       stream_));
          hipCheckError(hipMemcpyAsync(D_vals_device_,
                                       D_vals_,
                                       sizeof(T) * D_nnz_,
                                       hipMemcpyHostToDevice,
                                       stream_));
          hipCheckError(hipDeviceSynchronize());
          size_t buffer_size = 0;

          if (print_) std::cout << "\tAllocating C rows" << std::endl;
          hipCheckError(hipMalloc((void**)&C_rows_device_, sizeof(int64_t) * (m_ + 1)));
          hipCheckError(hipDeviceSynchronize());

          // Set up the rocSPARSE structures for the MM
          if (print_) std::cout << "\tCreating csr descriptions" << std::endl;
          status_ = rocsparse_create_csr_descr(&description_A_, m_, k_, A_nnz_, A_rows_device_, A_cols_device_,
                                               A_vals_device_, index_, index_, base_, dataType_);
          checkStatus("Failed rocsparse_create_csr_descr");
          status_ = rocsparse_create_csr_descr(&description_B_, k_, n_, B_nnz_, B_rows_device_, B_cols_device_,
                                               B_vals_device_, index_, index_, base_, dataType_);
          checkStatus("Failed rocsparse_create_csr_descr");
          status_ = rocsparse_create_csr_descr(&description_C_, m_, n_, 0, C_rows_device_, nullptr,
                                               nullptr, index_, index_, base_, dataType_);
          checkStatus("Failed rocsparse_create_csr_descr");
          status_ = rocsparse_create_csr_descr(&description_D_, m_, n_, D_nnz_, D_rows_device_, D_cols_device_,
                                               D_vals_device_, index_, index_, base_, dataType_);
          checkStatus("Failed rocsparse_create_csr_descr");


          if (print_) std::cout << "\tDetermining buffer size" << std::endl;
          stage_ = rocsparse_spgemm_stage_buffer_size;
          status_ = rocsparse_spgemm(handle_,
                                     operation_,
                                     operation_,
                                     &alpha,
                                     description_A_,
                                     description_B_,
                                     &beta,
                                     description_D_,
                                     description_C_,
                                     dataType_,
                                     algorithm_,
                                     stage_,
                                     &buffer_size,
                                     nullptr);
          checkStatus("Failed rocsparse_spgemm with stage=rocsparse_spgemm_stage_buffer_size");

          if (print_) std::cout << "\tAllocating buffer and C_rows" << std::endl;
          void* buffer;
          hipCheckError(hipMalloc(&buffer, buffer_size));

          if (print_) std::cout << "\tDetermining nnz" << std::endl;
          stage_ = rocsparse_spgemm_stage_nnz;
          status_ = rocsparse_spgemm(handle_,
                                     operation_,
                                     operation_,
                                     &alpha,
                                     description_A_,
                                     description_B_,
                                     &beta,
                                     description_D_,
                                     description_C_,
                                     dataType_,
                                     algorithm_,
                                     stage_,
                                     &buffer_size,
                                     buffer);
          checkStatus("Failed rocsparse_spgemm with stage=rocsparse_spgemm_stage_nnz");

          if (print_) std::cout << "\tAllocating rows and vals" << std::endl;
          int64_t rowC, colC;
          status_ = rocsparse_spmat_get_size(description_C_, &rowC, &colC, &C_nnz_);
          checkStatus("Failed rocsparse_spmat_get_size");

          hipCheckError(hipMalloc((void**)&C_cols_device_, sizeof(int64_t) * C_nnz_));
          hipCheckError(hipMalloc((void**)&C_vals_device_, sizeof(T) * C_nnz_));

          status_ = rocsparse_csr_set_pointers(description_C_,
                                               C_rows_device_,
                                               C_cols_device_,
                                               C_vals_device_);
          checkStatus("Failed rocsparse_csr_set_pointers");

          if (print_) std::cout << "\tDoing calculation" << std::endl;
          stage_ = rocsparse_spgemm_stage_compute;
          status_ = rocsparse_spgemm(handle_,
                                     operation_,
                                     operation_,
                                     &alpha,
                                     description_A_,
                                     description_B_,
                                     &beta,
                                     description_D_,
                                     description_C_,
                                     dataType_,
                                     algorithm_,
                                     stage_,
                                     &buffer_size,
                                     buffer);
          checkStatus("Failed rocsparse_spgemm with stage=rocsparse_spgemm_stage_compute");


          if (print_) std::cout << "\tFreeing buffer and descriptions etc." << std::endl;
          // Freeing up buffer
          hipCheckError(hipFree(buffer));
          status_ = rocsparse_destroy_spmat_descr(description_A_);
          checkStatus("Failing rocsparse_destroy_mat_descr for A");
          status_ = rocsparse_destroy_spmat_descr(description_B_);
          checkStatus("Failing rocsparse_destroy_mat_descr for B");
          status_ = rocsparse_destroy_spmat_descr(description_C_);
          checkStatus("Failing rocsparse_destroy_mat_descr for C");
          status_ = rocsparse_destroy_spmat_descr(description_D_);
          checkStatus("Failing rocsparse_destroy_mat_descr for D");

          if (print_) std::cout << "\tAllocating host C arrays" << std::endl;
          // Allocate host arrays for C
          hipCheckError(hipHostMalloc((void**)&C_rows_, sizeof(int64_t) * (m_ + 1)));
          hipCheckError(hipHostMalloc((void**)&C_cols_, sizeof(int64_t) * C_nnz_));
          hipCheckError(hipHostMalloc((void**)&C_vals_, sizeof(T) * C_nnz_));
          hipCheckError(hipDeviceSynchronize());

          // Moving data to CPU
          if (print_) std::cout << "\tTransfering data back to CPU" << std::endl;
          hipCheckError(hipMemcpyAsync(C_rows_,
                                       C_rows_device_,
                                       sizeof(int64_t) * (m_ + 1),
                                       hipMemcpyDeviceToHost,
                                       stream_));
          hipCheckError(hipMemcpyAsync(C_cols_,
                                       C_cols_device_,
                                       sizeof(int64_t) * C_nnz_,
                                       hipMemcpyDeviceToHost,
                                       stream_));
          hipCheckError(hipMemcpyAsync(C_vals_,
                                       C_vals_device_,
                                       sizeof(T) * C_nnz_,
                                       hipMemcpyDeviceToHost,
                                       stream_));
          hipCheckError(hipDeviceSynchronize());

          // Freeing stuff up
          if (print_) std::cout << "\tFreeing C arrays (host and device)" << std::endl;
          hipCheckError(hipFree(C_rows_device_));
          hipCheckError(hipFree(C_cols_device_));
          hipCheckError(hipFree(C_vals_device_));
          hipCheckError(hipFree(C_rows_));
          hipCheckError(hipFree(C_cols_));
          hipCheckError(hipFree(C_vals_));
          hipCheckError(hipDeviceSynchronize());
          break;
        }
        case gpuOffloadType::once: {
          size_t buffer_size;
          // Check if there are old arrays to get rid of
          if (!firstRun_) {
            hipCheckError(hipFree(C_rows_device_));
            hipCheckError(hipFree(C_cols_device_));
            hipCheckError(hipFree(C_vals_device_));
            hipCheckError(hipDeviceSynchronize());
          }

          if (print_) std::cout << "\tAllocating C rows" << std::endl;
          hipCheckError(hipMalloc((void**)&C_rows_device_, sizeof(int64_t) * (m_ + 1)));
          hipCheckError(hipDeviceSynchronize());

          // Set up the rocSPARSE structures for the MM
          if (print_) std::cout << "\tCreating csr descriptions" << std::endl;
          status_ = rocsparse_create_csr_descr(&description_A_,
                                               m_,
                                               k_,
                                               A_nnz_,
                                               A_rows_device_,
                                               A_cols_device_,
                                               A_vals_device_,
                                               index_,
                                               index_,
                                               base_,
                                               dataType_);
          checkStatus("Failed rocsparse_create_csr_descr");
          status_ = rocsparse_create_csr_descr(&description_B_,
                                               k_,
                                               n_,
                                               B_nnz_,
                                               B_rows_device_,
                                               B_cols_device_,
                                               B_vals_device_,
                                               index_,
                                               index_,
                                               base_,
                                               dataType_);
          checkStatus("Failed rocsparse_create_csr_descr");
          status_ = rocsparse_create_csr_descr(&description_C_,
                                               m_,
                                               n_,
                                               0,
                                               C_rows_device_,
                                               nullptr,
                                               nullptr,
                                               index_,
                                               index_,
                                               base_,
                                               dataType_);
          checkStatus("Failed rocsparse_create_csr_descr");
          status_ = rocsparse_create_csr_descr(&description_D_,
                                               m_,
                                               n_,
                                               D_nnz_,
                                               D_rows_device_,
                                               D_cols_device_,
                                               D_vals_device_,
                                               index_,
                                               index_,
                                               base_,
                                               dataType_);
          checkStatus("Failed rocsparse_create_csr_descr");

          if (print_) std::cout << "\tDetermining buffer size" << std::endl;
          stage_ = rocsparse_spgemm_stage_buffer_size;
          status_ = rocsparse_spgemm(handle_,
                                     operation_,
                                     operation_,
                                     &alpha,
                                     description_A_,
                                     description_B_,
                                     &beta,
                                     description_D_,
                                     description_C_,
                                     dataType_,
                                     algorithm_,
                                     stage_,
                                     &buffer_size,
                                     nullptr);
          checkStatus("Failed rocsparse_spgemm with stage=rocsparse_spgemm_stage_buffer_size");

          if (print_) std::cout << "\tAllocating buffer and C_rows" << std::endl;
          void* buffer;
          hipCheckError(hipMalloc(&buffer, buffer_size));

          if (print_) std::cout << "\tDetermining nnz" << std::endl;
          stage_ = rocsparse_spgemm_stage_nnz;
          status_ = rocsparse_spgemm(handle_,
                                     operation_,
                                     operation_,
                                     &alpha,
                                     description_A_,
                                     description_B_,
                                     &beta,
                                     description_D_,
                                     description_C_,
                                     dataType_,
                                     algorithm_,
                                     stage_,
                                     &buffer_size,
                                     buffer);
          checkStatus("Failed rocsparse_spgemm with stage=rocsparse_spgemm_stage_nnz");

          if (print_) std::cout << "\tAllocating rows and vals" << std::endl;
          int64_t rowC, colC;
          status_ = rocsparse_spmat_get_size(description_C_, &rowC, &colC, &C_nnz_);
          checkStatus("Failed rocsparse_spmat_get_size");

          hipCheckError(hipMalloc((void**)&C_cols_device_, sizeof(int64_t) * C_nnz_));
          hipCheckError(hipMalloc((void**)&C_vals_device_, sizeof(T) * C_nnz_));

          status_ = rocsparse_csr_set_pointers(description_C_,
                                               C_rows_device_,
                                               C_cols_device_,
                                               C_vals_device_);
          checkStatus("Failed rocsparse_csr_set_pointers");

          if (print_) std::cout << "\tDoing calculation" << std::endl;
          stage_ = rocsparse_spgemm_stage_compute;
          status_ = rocsparse_spgemm(handle_,
                                     operation_,
                                     operation_,
                                     &alpha,
                                     description_A_,
                                     description_B_,
                                     &beta,
                                     description_D_,
                                     description_C_,
                                     dataType_,
                                     algorithm_,
                                     stage_,
                                     &buffer_size,
                                     buffer);
          checkStatus("Failed rocsparse_spgemm with stage=rocsparse_spgemm_stage_compute");


          if (print_) std::cout << "\tFreeing buffer and descriptions etc." << std::endl;
          // Freeing up buffer
          hipCheckError(hipFree(buffer));
          status_ = rocsparse_destroy_spmat_descr(description_A_);
          checkStatus("Failing rocsparse_destroy_mat_descr for A");
          status_ = rocsparse_destroy_spmat_descr(description_B_);
          checkStatus("Failing rocsparse_destroy_mat_descr for B");
          status_ = rocsparse_destroy_spmat_descr(description_C_);
          checkStatus("Failing rocsparse_destroy_mat_descr for C");
          status_ = rocsparse_destroy_spmat_descr(description_D_);
          checkStatus("Failing rocsparse_destroy_mat_descr for D");
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
          hipCheckError(hipHostMalloc((void**)&C_rows_, sizeof(int64_t) * (m_ + 1)));
          hipCheckError(hipHostMalloc((void**)&C_cols_, sizeof(int64_t) * C_nnz_));
          hipCheckError(hipHostMalloc((void**)&C_vals_, sizeof(T) * C_nnz_));
          hipCheckError(hipDeviceSynchronize());


          if (print_) std::cout << "\tMoving C data to host" << std::endl;
          // Moving data to CPU
          hipCheckError(hipMemcpyAsync(C_rows_,
                                       C_rows_device_,
                                       sizeof(int64_t) * (m_ + 1),
                                       hipMemcpyDeviceToHost,
                                       stream_));
          hipCheckError(hipMemcpyAsync(C_cols_,
                                       C_cols_device_,
                                       sizeof(int64_t) * C_nnz_,
                                       hipMemcpyDeviceToHost,
                                       stream_));
          hipCheckError(hipMemcpyAsync(C_vals_,
                                       C_vals_device_,
                                       sizeof(T) * C_nnz_,
                                       hipMemcpyDeviceToHost,
                                       stream_));
          hipCheckError(hipDeviceSynchronize());

          // Freeing stuff up
          if (print_) std::cout << "\tFreeing C arrays (host and device)" << std::endl;
          hipCheckError(hipFree(C_rows_device_));
          hipCheckError(hipFree(C_cols_device_));
          hipCheckError(hipFree(C_vals_device_));
          hipCheckError(hipFree(C_rows_));
          hipCheckError(hipFree(C_cols_));
          hipCheckError(hipFree(C_vals_));
          hipCheckError(hipDeviceSynchronize());
          break;
        }
        case gpuOffloadType::unified: {
          if (print_) std::cout << "\tMoving data to CPU" << std::endl;
          hipCheckError(hipMemPrefetchAsync(C_rows_,
                                            sizeof(int64_t) * (m_ + 1),
                                            hipCpuDeviceId,
                                            stream_));
          hipCheckError(hipMemPrefetchAsync(C_cols_,
                                            sizeof(int64_t) * C_nnz_,
                                            hipCpuDeviceId,
                                            stream_));
          hipCheckError(hipMemPrefetchAsync(C_vals_,
                                            sizeof(T) * C_nnz_,
                                            hipCpuDeviceId,
                                            stream_));
          hipCheckError(hipDeviceSynchronize());
          if (print_) std::cout << "\tFreeing C arrays" << std::endl;
          hipCheckError(hipFree(C_rows_));
          hipCheckError(hipFree(C_cols_));
          hipCheckError(hipFree(C_vals_));
          hipCheckError(hipDeviceSynchronize());
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
        hipCheckError(hipFree(D_rows_));
        hipCheckError(hipFree(D_cols_));
        hipCheckError(hipFree(D_vals_));
        hipCheckError(hipDeviceSynchronize());
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
        hipCheckError(hipHostFree((void*)D_rows_));
        hipCheckError(hipHostFree((void*)D_cols_));
        hipCheckError(hipHostFree((void*)D_vals_));
        hipCheckError(hipDeviceSynchronize());
        if (print_) std::cout << "Freeing GPU arrays for A and B" << std::endl;
        hipCheckError(hipFree(A_rows_device_));
        hipCheckError(hipFree(A_cols_device_));
        hipCheckError(hipFree(A_vals_device_));
        hipCheckError(hipFree(B_rows_device_));
        hipCheckError(hipFree(B_cols_device_));
        hipCheckError(hipFree(B_vals_device_));
        hipCheckError(hipFree(D_rows_device_));
        hipCheckError(hipFree(D_cols_device_));
        hipCheckError(hipFree(D_vals_device_));
        hipCheckError(hipDeviceSynchronize());
      }
    }

    void checkStatus(std::string message) {
      if (status_ != rocsparse_status_success) {
        std::cerr << message << " error = ";
        switch (status_) {
          case rocsparse_status_success: {
            std::cerr << "Success" << std::endl;
            break;
          }
          case rocsparse_status_invalid_handle: {
            std::cerr << "invalid handle (handle not initialized, invalid or null.)" << std::endl;
            break;
          }
          case rocsparse_status_not_implemented: {
            std::cerr << "not imlpemented (function is not implemented.)" << std::endl;
            break;
          }
          case rocsparse_status_invalid_pointer: {
            std::cerr << "invalid pointer (invalid pointer parameter.)" << std::endl;
            break;
          }
          case rocsparse_status_invalid_size: {
            std::cerr << "invalid size (invalid size parameter.)" << std::endl;
            break;
          }
          case rocsparse_status_memory_error: {
            std::cerr << "memory error (failed memory allocation, copy, dealloc.)" << std::endl;
            break;
          }
          case rocsparse_status_internal_error: {
            std::cerr << "internal error (other internal library failure.)" << std::endl;
            break;
          }
          case rocsparse_status_invalid_value: {
            std::cerr << "invalid value (invalid value parameter.)" << std::endl;
            break;
          }
          case rocsparse_status_arch_mismatch: {
            std::cerr << "arch mismatch (device arch is not supported.)" << std::endl;
            break;
          }
          case rocsparse_status_zero_pivot: {
            std::cerr << "zero pivot (encountered zero pivot.)" << std::endl;
            break;
          }
          case rocsparse_status_not_initialized: {
            std::cerr << "not initialized (decriptor has not been initialized.)" << std::endl;
            break;
          }
          case rocsparse_status_type_mismatch: {
            std::cerr << "type mismatch (index types do not match.)" << std::endl;
            break;
          }
          case rocsparse_status_requires_sorted_storage: {
            std::cerr << "requires sorted storage (sorted storage required.)" << std::endl;
            break;
          }
          case rocsparse_status_thrown_exception: {
            std::cerr << "thrown exception (exception being thrown.)" << std::endl;
            break;
          }
          default: {
            std::cerr << "Not a known status enum" << std::endl;
            break;
          }
        }
      }
    }

    uint64_t calcNNZC() {
      uint64_t nnzSoFar = 0;
      for (size_t row = 0; row < m_ + 1; row++) {
        for (size_t col = 0; col < n_; col++) {
          for (size_t entry = 0; entry < k_; entry++) {
            if (A_[row * k_ + entry] != 0 && B_[entry * n_ + col] != 0) {
              nnzSoFar++;
              break;
            }
          }
        }
      }
      if (print_) std::cout << "Calculated nnzC = " << nnzSoFar << std::endl;
      return nnzSoFar;
    }

    void printMatrices() {
      std::cout << "================ Printing matrices ================" << std::endl;
      std::cout << "A matrix dense:" << std::endl;
      for (size_t i = 0; i < m_; i++) {
        for (size_t j = 0; j < k_; j++) {
          std::cout << A_[i * k_ + j] << " ";
        }
        std::cout << std::endl;
      }
      std::cout << "A matrix CSR:" << std::endl;
      std::cout << "\tRows: ";
      for (size_t i = 0; i < m_ + 1; i++) {
        std::cout << A_rows_[i] << " ";
      }
      std::cout << std::endl;
      std::cout << "\tCols: ";
      for (size_t i = 0; i < A_nnz_; i++) {
        std::cout << A_cols_[i] << " ";
      }
      std::cout << std::endl;
      std::cout << "\tVals: ";
      for (size_t i = 0; i < A_nnz_; i++) {
        std::cout << A_vals_[i] << " ";
      }
      std::cout << std::endl;

      std::cout << "---------------------------------------------------" << std::endl;

      std::cout << "B matrix dense:" << std::endl;
      for (size_t i = 0; i < k_; i++) {
        for (size_t j = 0; j < n_; j++) {
          std::cout << B_[i * n_ + j] << " ";
        }
        std::cout << std::endl;
      }
      std::cout << "B matrix CSR:" << std::endl;
      std::cout << "\tRows: ";
      for (size_t i = 0; i < k_ + 1; i++) {
        std::cout << B_rows_[i] << " ";
      }
      std::cout << std::endl;
      std::cout << "\tCols: ";
      for (size_t i = 0; i < B_nnz_; i++) {
        std::cout << B_cols_[i] << " ";
      }
      std::cout << std::endl;
      std::cout << "\tVals: ";
      for (size_t i = 0; i < B_nnz_; i++) {
        std::cout << B_vals_[i] << " ";
      }
      std::cout << std::endl;

      std::cout << "---------------------------------------------------" << std::endl;

      std::cout << "D matrix CSR:" << std::endl;
      std::cout << "\tRows: ";
      for (size_t i = 0; i < m_ + 1; i++) {
        std::cout << D_rows_[i] << " ";
      }
      std::cout << std::endl;
      std::cout << "\tCols: ";
      for (size_t i = 0; i < nnzD_; i++) {
        std::cout << D_cols_[i] << " ";
      }
      std::cout << std::endl;
      std::cout << "\tVals: ";
      for (size_t i = 0; i < nnzD_; i++) {
        std::cout << D_vals_[i] << " ";
      }
      std::cout << std::endl;

      std::cout << "================ Matrices printed! ================" << std::endl;
    }  

    bool print_ = true;
    bool initialised_ = false;
    bool firstRun_ = false;

    rocsparse_handle handle_;
    rocsparse_operation operation_;
    rocsparse_status status_;
    rocsparse_indextype_ index_;
    rocsparse_index_base base_;
    rocsparse_matrix_type type_;
    rocsparse_datatype dataType_;
    rocsparse_spgemm_stage stage_;
    rocsparse_spgemm_alg algorithm_;
    
    rocsparse_spmat_descr description_A_, description_B_, description_C_, description_D_;

    int64_t* A_rows_;
    int64_t* A_cols_;
    T* A_vals_;
    int64_t* B_rows_;
    int64_t* B_cols_;
    T* B_vals_;

    int64_t* A_rows_device_;
    int64_t* A_cols_device_;
    T* A_vals_device_;
    int64_t* B_rows_device_;
    int64_t* B_cols_device_;
    T* B_vals_device_;
    int64_t* C_rows_device_;
    int64_t* C_cols_device_;
    T* C_vals_device_;

    int64_t* D_rows_;
    int64_t* D_cols_;
    T* D_vals_;
    int64_t* D_rows_device_;
    int64_t* D_cols_device_;
    T* D_vals_device_;
    int64_t nnzD_ = 1;

    int gpuDevice_;
    hipStream_t stream_;
    
    const T alpha = ALPHA;
    const T beta = BETA;
};
}

#endif
