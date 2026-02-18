#pragma once

#ifdef GPU_ROCBLAS
#include <hip/hip_runtime_api.h>
#include <rocsparse/rocsparse.h>

#include <memory>
#include "../include/kernels/GPU/spgemm.hh"
#include "../include/utilities.hh"
#include "common.hh"

namespace gpu {
template <typename T>
class spgemm_gpu : public spgemm<T> {
public:
  using spgemm<T>::spgemm;
  using spgemm<T>::initInputMatrices;
  using spgemm<T>::A_nnz_;
  using spgemm<T>::B_nnz_;
  using spgemm<T>::m_;
  using spgemm<T>::n_;
  using spgemm<T>::k_;
  using spgemm<T>::offload_;
  using spgemm<T>::sparsity_;
  using spgemm<T>::type_;
  using spgemm<T>::C_nnz_;
  using spgemm<T>::C_vals_;
  using spgemm<T>::C_rows_;
  using spgemm<T>::C_cols_;

    ~spgemm_gpu() {
      if (initialised_) {
      rocsparse_destroy_handle(handle_);
      hipCheckError(hipStreamDestroy(s1_));
      hipCheckError(hipStreamDestroy(s2_));
      hipCheckError(hipStreamDestroy(s3_));
      hipCheckError(hipStreamDestroy(s4_));
      hipCheckError(hipStreamDestroy(s5_));
      hipCheckError(hipStreamDestroy(s6_));

      initialised_ = false;
      }
    }

    void initialise(gpuOffloadType offload, int m, int n, int k,
                    double sparsity, matrixType type, 
                    bool binary = false) override {
      if (!initialised_) {
        initialised_ = true;
        rocCheckError(rocsparse_create_handle(&handle_));
        
        hipCheckError(hipStreamCreate(&s1_));
        hipCheckError(hipStreamCreate(&s2_));
        hipCheckError(hipStreamCreate(&s3_));
        hipCheckError(hipStreamCreate(&s4_));
        hipCheckError(hipStreamCreate(&s5_));
        hipCheckError(hipStreamCreate(&s6_));

        rocCheckError(rocsparse_set_stream(handle_, s1_));

        hipCheckError(hipGetDevice(&gpuDevice_));
      }
      type_ = type;
      sparsity_ = sparsity;
      offload_ = offload;

      m_ = m;
      n_ = n;
      k_ = k;
      
      /** Determine the number of nnz elements in A and B */
      A_nnz_ = 1 + (uint64_t)((double)m_ * (double)k_ * (1.0 - sparsity_));
      B_nnz_ = 1 + (uint64_t)((double)k_ * (double)n_ * (1.0 - sparsity_));


      if constexpr (std::is_same_v<T, float>) {
        dataType_ = rocsparse_datatype_f32_r;
      } else if constexpr (std::is_same_v<T, double>) {
        dataType_ = rocsparse_datatype_f64_r;
      } else {
        std::cerr << "INVALID DATA TYPE PASSED TO rocSPARSE" << std::endl;
        exit(1);
      }

      initInputMatrices();
    }

protected:
    void toSparseFormat() override {
    if (offload_ == gpuOffloadType::always) {
      A_vals_store_ = (T*)malloc(sizeof(T) * A_nnz_);
      A_cols_store_ = (int32_t*)malloc(sizeof(int32_t) * A_nnz_);
      A_rows_store_ = (int32_t*)malloc(sizeof(int32_t) * (m_ + 1));
      B_vals_store_ = (T*)malloc(sizeof(T) * B_nnz_);
      B_cols_store_ = (int32_t*)malloc(sizeof(int32_t) * B_nnz_);
      B_rows_store_ = (int32_t*)malloc(sizeof(int32_t) * (k_ + 1));

      int seedOffset = 0;
      do {
        if (type_ == matrixType::rmat) {
          rMatCSR<T, int32_t>(A_vals_store_, A_cols_store_, A_rows_store_, m_, k_, A_nnz_, SEED + seedOffset++);
          rMatCSR<T, int32_t>(B_vals_store_, B_cols_store_, B_rows_store_, k_, n_, B_nnz_, SEED + seedOffset++);
        } else if (type_ == matrixType::random) {
          randomCSR<T, int32_t>(A_vals_store_, A_cols_store_, A_rows_store_, m_, k_, A_nnz_, SEED + seedOffset++);
          randomCSR<T, int32_t>(B_vals_store_, B_cols_store_, B_rows_store_, k_, n_, B_nnz_, SEED + seedOffset++);
        } else if (type_ == matrixType::finiteElements) {
          finiteElementCSR<T, int32_t>(A_vals_store_, A_cols_store_, A_rows_store_, m_, k_, A_nnz_, SEED + seedOffset++);
          finiteElementCSR<T, int32_t>(B_vals_store_, B_cols_store_, B_rows_store_, k_, n_, B_nnz_, SEED + seedOffset++);
        } else {
          std::cerr << "Matrix type not supported" << std::endl;
          exit(1);
        }
      } while (calcCNNZ<int32_t>(m_, A_nnz_, A_rows_store_, A_cols_store_, k_, B_nnz_, B_rows_store_, B_cols_store_) == 0);
    }

    // Allocate CSR arrays
    if (offload_ == gpuOffloadType::unified) {
      hipCheckError(hipMallocManaged(&A_vals_, sizeof(T) * A_nnz_));
      hipCheckError(hipMallocManaged(&A_cols_, sizeof(int32_t) * A_nnz_));
      hipCheckError(hipMallocManaged(&A_rows_, sizeof(int32_t) * (m_ + 1)));
      hipCheckError(hipMallocManaged(&B_vals_, sizeof(T) * B_nnz_));
      hipCheckError(hipMallocManaged(&B_cols_, sizeof(int32_t) * B_nnz_));
      hipCheckError(hipMallocManaged(&B_rows_, sizeof(int32_t) * (k_ + 1)));
      hipCheckError(hipMallocManaged(&C_rows_32_, sizeof(int32_t) * (m_ + 1)));
      C_vals_ = nullptr;
      C_cols_32_ = nullptr;
    } else {
      A_vals_ = (T*)malloc(sizeof(T) * A_nnz_);
      A_cols_ = (int32_t*)malloc(sizeof(int32_t) * A_nnz_);
      A_rows_ = (int32_t*)malloc(sizeof(int32_t) * (m_ + 1));
      B_vals_ = (T*)malloc(sizeof(T) * B_nnz_);
      B_cols_ = (int32_t*)malloc(sizeof(int32_t) * B_nnz_);
      B_rows_ = (int32_t*)malloc(sizeof(int32_t) * (k_ + 1));
      C_rows_32_ = (int32_t*)malloc(sizeof(int32_t) * (m_ + 1));
      C_vals_ = nullptr;
      C_cols_32_ = nullptr;

      hipCheckError(hipMalloc((void**)&A_vals_dev_, sizeof(T) * A_nnz_));
      hipCheckError(hipMalloc((void**)&A_cols_dev_, sizeof(int32_t) * A_nnz_));
      hipCheckError(hipMalloc((void**)&A_rows_dev_, sizeof(int32_t) * (m_ + 1)));
      hipCheckError(hipMalloc((void**)&B_vals_dev_, sizeof(T) * B_nnz_));
      hipCheckError(hipMalloc((void**)&B_cols_dev_, sizeof(int32_t) * B_nnz_));
      hipCheckError(hipMalloc((void**)&B_rows_dev_, sizeof(int32_t) * (k_ + 1)));
      hipCheckError(hipMalloc((void**)&C_rows_dev_, sizeof(int32_t) * (m_ + 1)));
      C_vals_dev_ = nullptr;
      C_cols_dev_ = nullptr;
    }

    // Move data into the correct arrays
    memcpy(A_vals_, A_vals_store_, sizeof(T) * A_nnz_);
    memcpy(A_cols_, A_cols_store_, sizeof(int32_t) * A_nnz_);
    memcpy(A_rows_, A_rows_store_, sizeof(int32_t) * (m_ + 1));
    memcpy(B_vals_, B_vals_store_, sizeof(T) * B_nnz_);
    memcpy(B_cols_, B_cols_store_, sizeof(int32_t) * B_nnz_);
    memcpy(B_rows_, B_rows_store_, sizeof(int32_t) * (k_ + 1));
  }

private:
    void preLoopRequirements() override {
      switch (offload_) {
        case gpuOffloadType::always: {
          break;
        }
      case gpuOffloadType::once: {
        hipCheckError(hipMemcpyAsync(A_vals_dev_, A_vals_, sizeof(T) * A_nnz_, hipMemcpyHostToDevice, s1_));
        hipCheckError(hipMemcpyAsync(A_cols_dev_, A_cols_, sizeof(int32_t) * A_nnz_, hipMemcpyHostToDevice, s2_));
        hipCheckError(hipMemcpyAsync(A_rows_dev_, A_rows_, sizeof(int32_t) * (m_ + 1), hipMemcpyHostToDevice, s3_));
        hipCheckError(hipMemcpyAsync(B_vals_dev_, B_vals_, sizeof(T) * B_nnz_, hipMemcpyHostToDevice, s4_));
        hipCheckError(hipMemcpyAsync(B_cols_dev_, B_cols_, sizeof(int32_t) * B_nnz_, hipMemcpyHostToDevice, s5_));
        hipCheckError(hipMemcpyAsync(B_rows_dev_, B_rows_, sizeof(int32_t) * (k_ + 1), hipMemcpyHostToDevice, s6_));
        break;
      }
      case gpuOffloadType::unified: {
        // Prefetch memory to device
        hipCheckError(hipMemPrefetchAsync(A_vals_, sizeof(T) * A_nnz_, gpuDevice_, s1_));
        hipCheckError(hipMemPrefetchAsync(A_cols_, sizeof(int32_t) * A_nnz_, gpuDevice_, s2_));
        hipCheckError(hipMemPrefetchAsync(A_rows_, sizeof(int32_t) * (m_ + 1), gpuDevice_, s3_));
        hipCheckError(hipMemPrefetchAsync(B_vals_, sizeof(T) * B_nnz_, gpuDevice_, s4_));
        hipCheckError(hipMemPrefetchAsync(B_cols_, sizeof(int32_t) * B_nnz_, gpuDevice_, s5_));
        hipCheckError(hipMemPrefetchAsync(B_rows_, sizeof(int32_t) * (k_ + 1), gpuDevice_, s6_));
        break;
      }
      }
    }

    void callSpgemm() override {
      switch (offload_) {
        case gpuOffloadType::always: {
          if (C_allocated_) {
            free(C_vals_);
            free(C_cols_32_);
            C_allocated_ = false;
          }

          hipCheckError(hipMemcpyAsync(A_vals_dev_, A_vals_, sizeof(T) * A_nnz_, hipMemcpyHostToDevice, s1_));
          hipCheckError(hipMemcpyAsync(A_cols_dev_, A_cols_, sizeof(int32_t) * A_nnz_, hipMemcpyHostToDevice, s2_));
          hipCheckError(hipMemcpyAsync(A_rows_dev_, A_rows_, sizeof(int32_t) * (m_ + 1), hipMemcpyHostToDevice, s3_));
          hipCheckError(hipMemcpyAsync(B_vals_dev_, B_vals_, sizeof(T) * B_nnz_, hipMemcpyHostToDevice, s4_));
          hipCheckError(hipMemcpyAsync(B_cols_dev_, B_cols_, sizeof(int32_t) * B_nnz_, hipMemcpyHostToDevice, s5_));
          hipCheckError(hipMemcpyAsync(B_rows_dev_, B_rows_, sizeof(int32_t) * (k_ + 1), hipMemcpyHostToDevice, s6_));
          hipCheckError(hipDeviceSynchronize());
          
          rocCheckError(rocsparse_create_csr_descr(&A_descr_,
                                                   m_,
                                                   k_,
                                                   A_nnz_,
                                                   A_rows_dev_,
                                                   A_cols_dev_,
                                                   A_vals_dev_,
                                                   index_,
                                                   index_,
                                                   base_,
                                                   dataType_));

          rocCheckError(rocsparse_create_csr_descr(&B_descr_,
                                                   k_,
                                                   n_,
                                                   B_nnz_,
                                                   B_rows_dev_,
                                                   B_cols_dev_,
                                                   B_vals_dev_,
                                                   index_,
                                                   index_,
                                                   base_,
                                                   dataType_));

          rocCheckError(rocsparse_create_csr_descr(&C_descr_,
                                                   m_,
                                                   n_,
                                                   0,
                                                   C_rows_dev_,
                                                   nullptr,
                                                   nullptr,
                                                   index_,
                                                   index_,
                                                   base_,
                                                   dataType_));
          size_t buffer_size = 0;
          void* temp_buffer;

          rocCheckError(rocsparse_spgemm(handle_,
                                         operation_,
                                         operation_,
                                         &alpha,
                                         A_descr_,
                                         B_descr_,
                                         &beta,
                                         C_descr_,
                                         C_descr_,
                                         dataType_,
                                         algorithm_,
                                         rocsparse_spgemm_stage_buffer_size,
                                         &buffer_size,
                                         nullptr));

          hipCheckError(hipMalloc(&temp_buffer, buffer_size));

          rocCheckError(rocsparse_spgemm(handle_,
                                         operation_,
                                         operation_,
                                         &alpha,
                                         A_descr_,
                                         B_descr_,
                                         &beta,
                                         C_descr_,
                                         C_descr_,
                                         dataType_,
                                         algorithm_,
                                         rocsparse_spgemm_stage_nnz,
                                         &buffer_size,
                                         temp_buffer));

          int64_t temp_m;
          int64_t temp_n;
          rocCheckError(rocsparse_spmat_get_size(C_descr_, &temp_m, &temp_n, &C_nnz_));

          hipCheckError(hipMalloc((void**)&C_cols_dev_, sizeof(int32_t) * C_nnz_));
          hipCheckError(hipMalloc((void**)&C_vals_dev_, sizeof(T) * C_nnz_));

          rocCheckError(rocsparse_csr_set_pointers(C_descr_, C_rows_dev_, C_cols_dev_, C_vals_dev_));

          rocCheckError(rocsparse_spgemm(handle_,
                                         operation_,
                                         operation_,
                                         &alpha,
                                         A_descr_,
                                         B_descr_,
                                         &beta,
                                         C_descr_,
                                         C_descr_,
                                         dataType_,
                                         algorithm_,
                                         rocsparse_spgemm_stage_compute,
                                         &buffer_size,
                                         temp_buffer));

          hipCheckError(hipFree(temp_buffer));
          rocCheckError(rocsparse_destroy_spmat_descr(A_descr_));
          rocCheckError(rocsparse_destroy_spmat_descr(B_descr_));
          rocCheckError(rocsparse_destroy_spmat_descr(C_descr_));

          C_vals_ = (T*)malloc(sizeof(T) * C_nnz_);
          C_cols_32_ = (int32_t*)malloc(sizeof(int32_t) * C_nnz_);
          C_allocated_ = true;

          hipCheckError(hipMemcpyAsync(C_rows_32_, C_rows_dev_, sizeof(int32_t) * (m_ + 1), hipMemcpyDeviceToHost, s1_));
          hipCheckError(hipMemcpyAsync(C_cols_32_, C_cols_dev_, sizeof(int32_t) * C_nnz_, hipMemcpyDeviceToHost, s2_));
          hipCheckError(hipMemcpyAsync(C_vals_, C_vals_dev_, sizeof(T) * C_nnz_, hipMemcpyDeviceToHost, s3_));

          hipCheckError(hipDeviceSynchronize());
          hipCheckError(hipFree(C_cols_dev_));
          hipCheckError(hipFree(C_vals_dev_));
          break;
        }

        case gpuOffloadType::once: {
          if (C_allocated_) {
            hipCheckError(hipFree(C_vals_dev_));
            hipCheckError(hipFree(C_cols_dev_));
            C_allocated_ = false;
          }
          
          rocCheckError(rocsparse_create_csr_descr(&A_descr_,
                                                   m_,
                                                   k_,
                                                   A_nnz_,
                                                   A_rows_dev_,
                                                   A_cols_dev_,
                                                   A_vals_dev_,
                                                   index_,
                                                   index_,
                                                   base_,
                                                   dataType_));

          rocCheckError(rocsparse_create_csr_descr(&B_descr_,
                                                   k_,
                                                   n_,
                                                   B_nnz_,
                                                   B_rows_dev_,
                                                   B_cols_dev_,
                                                   B_vals_dev_,
                                                   index_,
                                                   index_,
                                                   base_,
                                                   dataType_));

          rocCheckError(rocsparse_create_csr_descr(&C_descr_,
                                                    m_,
                                                    n_,
                                                    0,
                                                    C_rows_dev_,
                                                    nullptr,
                                                    nullptr,
                                                    index_,
                                                    index_,
                                                    base_,
                                                    dataType_));
          size_t buffer_size = 0;
          void* temp_buffer;

          rocCheckError(rocsparse_spgemm(handle_,
                                         operation_,
                                         operation_,
                                         &alpha,
                                         A_descr_,
                                         B_descr_,
                                         &beta,
                                         C_descr_,
                                         C_descr_,
                                         dataType_,
                                         algorithm_,
                                         rocsparse_spgemm_stage_buffer_size,
                                         &buffer_size,
                                         nullptr));

          hipCheckError(hipMalloc(&temp_buffer, buffer_size));

          rocCheckError(rocsparse_spgemm(handle_,
                                         operation_,
                                         operation_,
                                         &alpha,
                                         A_descr_,
                                         B_descr_,
                                         &beta,
                                         C_descr_,
                                         C_descr_,
                                         dataType_,
                                         algorithm_,
                                         rocsparse_spgemm_stage_nnz,
                                         &buffer_size,
                                         temp_buffer));

          int64_t temp_m, temp_n;
          rocCheckError(rocsparse_spmat_get_size(C_descr_, &temp_m, &temp_n, &C_nnz_));

          hipCheckError(hipMalloc((void**)&C_cols_dev_, sizeof(int32_t) * C_nnz_));
          hipCheckError(hipMalloc((void**)&C_vals_dev_, sizeof(T) * C_nnz_));

          rocCheckError(rocsparse_csr_set_pointers(C_descr_, C_rows_dev_, C_cols_dev_, C_vals_dev_));

          rocCheckError(rocsparse_spgemm(handle_,
                                         operation_,
                                         operation_,
                                         &alpha,
                                         A_descr_,
                                         B_descr_,
                                         &beta,
                                         C_descr_,
                                         C_descr_,
                                         dataType_,
                                         algorithm_,
                                         rocsparse_spgemm_stage_compute,
                                         &buffer_size,
                                         temp_buffer));

          hipCheckError(hipFree(temp_buffer));
          rocCheckError(rocsparse_destroy_spmat_descr(A_descr_));
          rocCheckError(rocsparse_destroy_spmat_descr(B_descr_));
          rocCheckError(rocsparse_destroy_spmat_descr(C_descr_));
          break;                      
        }
        case gpuOffloadType::unified: {
          if (C_allocated_) {
            hipCheckError(hipFree(C_vals_));
            hipCheckError(hipFree(C_cols_32_));
            C_allocated_ = false;
          }
          
          rocCheckError(rocsparse_create_csr_descr(&A_descr_,
                                                   m_,
                                                   k_,
                                                   A_nnz_,
                                                   A_rows_,
                                                   A_cols_,
                                                   A_vals_,
                                                   index_,
                                                   index_,
                                                   base_,
                                                   dataType_));

          rocCheckError(rocsparse_create_csr_descr(&B_descr_,
                                                   k_,
                                                   n_,
                                                   B_nnz_,
                                                   B_rows_,
                                                   B_cols_,
                                                   B_vals_,
                                                   index_,
                                                   index_,
                                                   base_,
                                                   dataType_));

          rocCheckError(rocsparse_create_csr_descr(&C_descr_,
                                                    m_,
                                                    n_,
                                                    0,
                                                    C_rows_,
                                                    nullptr,
                                                    nullptr,
                                                    index_,
                                                    index_,
                                                    base_,
                                                    dataType_));
          size_t buffer_size = 0;
          void* temp_buffer;

          rocCheckError(rocsparse_spgemm(handle_,
                                         operation_,
                                         operation_,
                                         &alpha,
                                         A_descr_,
                                         B_descr_,
                                         &beta,
                                         C_descr_,
                                         C_descr_,
                                         dataType_,
                                         algorithm_,
                                         rocsparse_spgemm_stage_buffer_size,
                                         &buffer_size,
                                         nullptr));

          hipCheckError(hipMalloc(&temp_buffer, buffer_size));

          rocCheckError(rocsparse_spgemm(handle_,
                                         operation_,
                                         operation_,
                                         &alpha,
                                         A_descr_,
                                         B_descr_,
                                         &beta,
                                         C_descr_,
                                         C_descr_,
                                         dataType_,
                                         algorithm_,
                                         rocsparse_spgemm_stage_nnz,
                                         &buffer_size,
                                         temp_buffer));

          int64_t temp_m, temp_n;
          rocCheckError(rocsparse_spmat_get_size(C_descr_, &temp_m, &temp_n, &C_nnz_));

          hipCheckError(hipMallocManaged((void**)&C_cols_, sizeof(int32_t) * C_nnz_));
          hipCheckError(hipMallocManaged((void**)&C_vals_, sizeof(T) * C_nnz_));

          rocCheckError(rocsparse_csr_set_pointers(C_descr_, C_rows_, C_cols_, C_vals_));

          rocCheckError(rocsparse_spgemm(handle_,
                                         operation_,
                                         operation_,
                                         &alpha,
                                         A_descr_,
                                         B_descr_,
                                         &beta,
                                         C_descr_,
                                         C_descr_,
                                         dataType_,
                                         algorithm_,
                                         rocsparse_spgemm_stage_compute,
                                         &buffer_size,
                                         temp_buffer));

          hipCheckError(hipFree(temp_buffer));
          rocCheckError(rocsparse_destroy_spmat_descr(A_descr_));
          rocCheckError(rocsparse_destroy_spmat_descr(B_descr_));
          rocCheckError(rocsparse_destroy_spmat_descr(C_descr_));
          break;
        }
      }
    }

    void postLoopRequirements() override {
      switch(offload_) {
        case gpuOffloadType::always: {
          break;
        }
        case gpuOffloadType::once: {
          C_vals_ = (T*)malloc(sizeof(T) * C_nnz_);
          C_cols_32_ = (int32_t*)malloc(sizeof(int32_t) * C_nnz_);
          
          hipCheckError(hipMemcpyAsync(C_rows_32_, C_rows_dev_, sizeof(int32_t) * (m_ + 1), hipMemcpyDeviceToHost, s1_));
          hipCheckError(hipMemcpyAsync(C_cols_32_, C_cols_dev_, sizeof(int32_t) * C_nnz_, hipMemcpyDeviceToHost, s2_));
          hipCheckError(hipMemcpyAsync(C_vals_, C_vals_dev_, sizeof(T) * C_nnz_, hipMemcpyDeviceToHost, s3_));
          hipCheckError(hipDeviceSynchronize());

          if (C_allocated_) {
            hipCheckError(hipFree(C_vals_dev_));
            hipCheckError(hipFree(C_cols_dev_));
            C_allocated_ = false;
          }
          break;
        }
        case gpuOffloadType::unified: {
          hipCheckError(hipMemPrefetchAsync(C_vals_, sizeof(T) * C_nnz_, hipCpuDeviceId, s1_));
          hipCheckError(hipMemPrefetchAsync(C_cols_32_, sizeof(int32_t) * C_nnz_, hipCpuDeviceId, s2_));
          hipCheckError(hipMemPrefetchAsync(C_rows_32_, sizeof(int32_t) * (m_ + 1), hipCpuDeviceId, s3_));
          break;
        }
      }
    }

    void postCallKernelCleanup() override {
      switch (offload_) {
        case gpuOffloadType::always: {
          if (C_allocated_) {
            free(C_vals_);
            free(C_cols_32_);
            C_allocated_ = false;
          }
          free(A_vals_);
          free(A_cols_);
          free(A_rows_);
          free(B_vals_);
          free(B_cols_);
          free(B_rows_);
          free(C_rows_32_);

          hipCheckError(hipFree(A_vals_dev_));
          hipCheckError(hipFree(A_cols_dev_));
          hipCheckError(hipFree(A_rows_dev_));
          hipCheckError(hipFree(B_vals_dev_));
          hipCheckError(hipFree(B_cols_dev_));
          hipCheckError(hipFree(B_rows_dev_));
          hipCheckError(hipFree(C_rows_dev_));
          break;
        }
        case gpuOffloadType::once: {
          free(A_vals_);
          free(A_cols_);
          free(A_rows_);
          free(B_vals_);
          free(B_cols_);
          free(B_rows_);
          free(C_vals_);
          free(C_cols_32_);
          free(C_rows_32_);

          hipCheckError(hipFree(A_vals_dev_));
          hipCheckError(hipFree(A_cols_dev_));
          hipCheckError(hipFree(A_rows_dev_));
          hipCheckError(hipFree(B_vals_dev_));
          hipCheckError(hipFree(B_cols_dev_));
          hipCheckError(hipFree(B_rows_dev_));
          hipCheckError(hipFree(C_rows_dev_));
          break;
        }
        case gpuOffloadType::unified: {
          if (C_allocated_) {
            hipCheckError(hipFree(C_vals_dev_));
            hipCheckError(hipFree(C_cols_dev_));
            C_allocated_ = false;
          }
          hipCheckError(hipFree(A_vals_dev_));
          hipCheckError(hipFree(A_cols_dev_));
          hipCheckError(hipFree(A_rows_dev_));
          hipCheckError(hipFree(B_vals_dev_));
          hipCheckError(hipFree(B_cols_dev_));
          hipCheckError(hipFree(B_rows_dev_));
          hipCheckError(hipFree(C_rows_dev_));

          free(A_vals_store_);
          free(A_cols_store_);
          free(A_rows_store_);
          free(B_vals_store_);
          free(B_cols_store_);
          free(B_rows_store_);
          break;
        }
      }
    }

  bool C_allocated_ = false;
  bool initialised_ = false;

  int gpuDevice_;
  hipStream_t s1_;
  hipStream_t s2_;
  hipStream_t s3_;
  hipStream_t s4_;
  hipStream_t s5_;
  hipStream_t s6_;

  rocsparse_handle handle_;
  rocsparse_operation operation_;
  rocsparse_status status_;
  rocsparse_indextype_ index_;
  rocsparse_index_base base_;
  rocsparse_datatype dataType_;
  rocsparse_spgemm_stage stage_;
  rocsparse_spgemm_alg algorithm_;
  
  rocsparse_spmat_descr A_descr_, B_descr_, C_descr_;

  
  const T alpha = ALPHA;
  const T beta = BETA;

  /** Storage for matrices between offload type calls */
  T* A_vals_store_;
  int32_t* A_cols_store_;
  int32_t* A_rows_store_;
  T* B_vals_store_;
  int32_t* B_cols_store_;
  int32_t* B_rows_store_;

	/** CSR format vectors for matrices A, B and C on the host */
	T* A_vals_;
	int32_t* A_cols_;
  int32_t* A_rows_;

  T* B_vals_;
  int32_t* B_cols_;
  int32_t* B_rows_;

  int64_t C_num_rows_;
  int64_t C_num_cols_;

  /** CSR format vectors for matrices A, B and C on the device. */
	T* A_vals_dev_;
  T* B_vals_dev_;
  T* C_vals_dev_;
	int32_t* A_cols_dev_;
  int32_t* A_rows_dev_;
  int32_t* B_cols_dev_;
  int32_t* B_rows_dev_;
  int32_t* C_cols_dev_;
  int32_t* C_rows_dev_;

  int32_t* C_cols_32_;
  int32_t* C_rows_32_;
};
} // namespace gpu

#endif
