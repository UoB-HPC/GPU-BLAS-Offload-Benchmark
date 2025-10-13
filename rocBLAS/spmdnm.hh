#pragma once

#ifdef GPU_ROCBLAS
#include <hip/hip_runtime_api.h>
#include <rocsparse/rocsparse.h>

#include "../include/kernels/GPU/spmdnm.hh"
#include "../include/utilities.hh"
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
  using spmdnm<T>::A_;
  using spmdnm<T>::B_;
  using spmdnm<T>::C_;
  using spmdnm<T>::offload_;
  using spmdnm<T>::sparsity_;

  ~spmdnm_gpu() {
    if (initialised_) {
      rocsparse_destroy_handle(handle_);
      hipCheckError(hipStreamDestroy(s1_));
      hipCheckError(hipStreamDestroy(s2_));
      hipCheckError(hipStreamDestroy(s3_));
    }
  }

  void initialise(gpuOffloadType offload, int m, int n, int k,
                  double sparsity, bool binary = false) override {
      // Set up problem parameters
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
    m_ = m;
    n_ = n;
    k_ = k;
    sparsity_ = sparsity;
    offload_ = offload;
    nnz_ = 1 + (int64_t)((double)m_ * (double)n_ * (1.0 - sparsity_));

    // Set up rocSPARSE metadata
    base_ = rocsparse_index_base_zero;
    type_ = rocsparse_matrix_type_general;
    operation_ = rocsparse_operation_none;
    index_ = rocsparse_indextype_i64;
    order_ = rocsparse_order_column;
    algorithm_ = rocsparse_spmm_alg_csr_nnz_split; // This is the only algo for this one

    if constexpr (std::is_same_v<T, float>) {
      dataType_ = rocsparse_datatype_f32_r;
    } else if constexpr (std::is_same_v<T, double>) {
      dataType_ = rocsparse_datatype_f64_r;
    } else {
      throw std::runtime_error("Unsupported data type for spmdnm_gpu");
    }

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
      hipCheckError(hipMallocManaged(&A_cols_, sizeof(int64_t) * nnz_));
      hipCheckError(hipMallocManaged(&A_vals_, sizeof(T) * nnz_));
      hipCheckError(hipMallocManaged(&B_, sizeof(T) * k_ * n_));
      hipCheckError(hipMallocManaged(&C_, sizeof(T) * m_ * n_));
    } else {
      // Host data structures
      hipCheckError(hipHostMalloc((void**)&A_, sizeof(T) * m_ * k_));
      hipCheckError(hipHostMalloc((void**)&A_rows_, sizeof(int64_t) * (m_ + 1)));
      hipCheckError(hipHostMalloc((void**)&A_cols_, sizeof(int64_t) * nnz_));
      hipCheckError(hipHostMalloc((void**)&A_vals_, sizeof(T) * nnz_));
      hipCheckError(hipHostMalloc((void**)&B_, sizeof(T) * k_ * n_));
      hipCheckError(hipHostMalloc((void**)&C_, sizeof(T) * m_ * n_));
      // GPU data structures
      hipCheckError(hipMalloc((void**)&A_rows_device_, sizeof(int64_t) * (m_ + 1)));
      hipCheckError(hipMalloc((void**)&A_cols_device_, sizeof(int64_t) * nnz_));
      hipCheckError(hipMalloc((void**)&A_vals_device_, sizeof(T) * nnz_));
      hipCheckError(hipMalloc((void**)&B_device_, sizeof(T) * k_ * n_));
      hipCheckError(hipMalloc((void**)&C_device_, sizeof(T) * m_ * n_));
    }

    if (print_) std::cout << "\tInitialising matrices" << std::endl;
    initInputMatrices();
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
                                     s1_));
        hipCheckError(hipMemcpyAsync(A_cols_device_,
                                     A_cols_,
                                     sizeof(int64_t) * nnz_,
                                     hipMemcpyHostToDevice,
                                     s1_));
        hipCheckError(hipMemcpyAsync(A_vals_device_,
                                     A_vals_,
                                     sizeof(T) * nnz_,
                                     hipMemcpyHostToDevice,
                                     s1_));
        hipCheckError(hipMemcpyAsync(B_device_,
                                     B_,
                                     sizeof(T) * k_ * n_,
                                     hipMemcpyHostToDevice,
                                     s2_));
        hipCheckError(hipMemcpyAsync(C_device_,
                                     C_,
                                     sizeof(T) * m_ * n_,
                                     hipMemcpyHostToDevice,
                                     s3_));
        hipCheckError(hipDeviceSynchronize());
        break;
      }
      case gpuOffloadType::unified: {
        if (print_) std::cout << "\tMoving data to GPU" << std::endl;
        hipCheckError(hipMemPrefetchAsync(A_rows_, 
                                          sizeof(int64_t) * (m_ + 1), 
                                          gpuDevice_, 
                                          s1_));
        hipCheckError(hipMemPrefetchAsync(A_cols_, 
                                          sizeof(int64_t) * nnz_, 
                                          gpuDevice_, 
                                          s1_));
        hipCheckError(hipMemPrefetchAsync(A_vals_, 
                                          sizeof(T) * nnz_, 
                                          gpuDevice_, 
                                          s1_));
        hipCheckError(hipMemPrefetchAsync(B_, 
                                          sizeof(T) * k_ * n_, 
                                          gpuDevice_, 
                                          s2_));
        hipCheckError(hipMemPrefetchAsync(C_, 
                                          sizeof(T) * m_ * n_, 
                                          gpuDevice_, 
                                          s3_));
        hipCheckError(hipDeviceSynchronize());
        break;
      }
    }
  }

  void callSpmdnm() override {
    if (print_) std::cout << "callSpmdnm" << std::endl;
    switch (offload_) {
      case gpuOffloadType::always: {
        if (print_) std::cout << "\tMoving data to GPU" << std::endl;
        hipCheckError(hipMemcpyAsync(A_rows_device_,
                                     A_rows_,
                                     sizeof(int64_t) * (m_ + 1),
                                     hipMemcpyHostToDevice,
                                     s1_));
        hipCheckError(hipMemcpyAsync(A_cols_device_,
                                     A_cols_,
                                     sizeof(int64_t) * nnz_,
                                     hipMemcpyHostToDevice,
                                     s1_));
        hipCheckError(hipMemcpyAsync(A_vals_device_,
                                     A_vals_,
                                     sizeof(T) * nnz_,
                                     hipMemcpyHostToDevice,
                                     s1_));
        hipCheckError(hipMemcpyAsync(B_device_,
                                     B_,
                                     sizeof(T) * k_ * n_,
                                     hipMemcpyHostToDevice,
                                     s2_));
        hipCheckError(hipMemcpyAsync(C_device_,
                                     C_,
                                     sizeof(T) * m_ * n_,
                                     hipMemcpyHostToDevice,
                                     s3_));
        hipCheckError(hipDeviceSynchronize());


        if (print_) std::cout << "\tCreating rocSPARSE structures" << std::endl;
        // Set up the rocSPARSE structures for the GEMV
        status_ = rocsparse_create_csr_descr(&A_description_, m_, k_, nnz_, A_rows_device_,
                                             A_cols_device_, A_vals_device_, index_, index_,
                                             base_, dataType_);
        checkStatus("Failed rocsparse_create_csr_descr for A");

        status_ = rocsparse_create_dnmat_descr(&B_description_, k_, n_, k_, B_device_,
                                               dataType_, order_);
        checkStatus("Failed rocsparse_create_dnmat_descr for B");

        status_ = rocsparse_create_dnmat_descr(&C_description_, m_, n_, m_, C_device_,
                                               dataType_, order_);
        checkStatus("Failed rocsparse_create_dnmat_descr for C");
        hipCheckError(hipDeviceSynchronize());

        size_t buffer_size = 0;
        status_ = rocsparse_spmm(handle_,
                                 operation_,
                                 operation_,
                                 &alpha,
                                 A_description_,
                                 B_description_,
                                 &beta,
                                 C_description_,
                                 dataType_,
                                 algorithm_,
                                 rocsparse_spmm_stage_buffer_size,
                                 &buffer_size,
                                 nullptr);
        checkStatus("Failed rocsparse_spmm with stage=rocsparse_spmm_stage_buffer_size");

        void* buffer = nullptr;
        if (print_) std::cout << "\tAllocating buffer with buffer_size = " << buffer_size << std::endl;
        if (buffer_size > 0) hipCheckError(hipMalloc(&buffer, buffer_size));

        status_ = rocsparse_spmm(handle_,
                                 operation_,
                                 operation_,
                                 &alpha,
                                 A_description_,
                                 B_description_,
                                 &beta,
                                 C_description_,
                                 dataType_,
                                 algorithm_,
                                 rocsparse_spmm_stage_preprocess,
                                 &buffer_size,
                                 buffer);
        checkStatus("Failed rocsparse_spmm with stage=rocsparse_spmm_stage_preprocess");

        hipCheckError(hipDeviceSynchronize());
        status_ = rocsparse_spmm(handle_,
                                 operation_,
                                 operation_,
                                 &alpha,
                                 A_description_,
                                 B_description_,
                                 &beta,
                                 C_description_,
                                 dataType_,
                                 algorithm_,
                                 rocsparse_spmm_stage_compute,
                                 &buffer_size,
                                 buffer);
        checkStatus("Failed rocsparse_spmm with stage=rocsparse_spmm_stage_compute");

        hipCheckError(hipDeviceSynchronize());
        if (print_) std::cout << "\tdestroying rocSPARSE structures" << std::endl;
        // Now clean up
        status_ = rocsparse_destroy_spmat_descr(A_description_);
        checkStatus("Failed rocsparse_destroy_spmat_descr for A");
        status_ = rocsparse_destroy_dnmat_descr(B_description_);
        checkStatus("Failed rocsparse_destroy_dnmat_descr for B");
        status_ = rocsparse_destroy_dnmat_descr(C_description_);
        checkStatus("Failed rocsparse_destroy_dnmat_descr for C");
        if (buffer != nullptr) hipCheckError(hipFree(buffer));
        hipCheckError(hipDeviceSynchronize());

        // Move result back to the CPU
        if (print_) std::cout << "\tMoving data to CPU" << std::endl;
        hipCheckError(hipMemcpyAsync(C_, 
                                     C_device_, 
                                     sizeof(T) * m_ * n_, 
                                     hipMemcpyDeviceToHost, 
                                     s3_));
        hipCheckError(hipDeviceSynchronize());
        break;
      }
      case gpuOffloadType::once: {
        if (print_) std::cout << "\tCreating rocSPARSE structures" << std::endl;
        // Set up the rocSPARSE structures for the GEMV
        status_ = rocsparse_create_csr_descr(&A_description_,
                                             m_,
                                             k_,
                                             nnz_,
                                             A_rows_device_,
                                             A_cols_device_,
                                             A_vals_device_,
                                             index_,
                                             index_,
                                             base_,
                                             dataType_);
        checkStatus("Failed rocsparse_create_csr_descr for A");

        status_ = rocsparse_create_dnmat_descr(&B_description_,
                                               k_,
                                               n_,
                                               k_,
                                               B_device_,
                                               dataType_,
                                               order_);
        checkStatus("Failed rocsparse_create_dnmat_descr for B");

        status_ = rocsparse_create_dnmat_descr(&C_description_,
                                               m_,
                                               n_,
                                               m_,
                                               C_device_,
                                               dataType_,
                                               order_);
        checkStatus("Failed rocsparse_create_dnmat_descr for C");
        hipCheckError(hipDeviceSynchronize());

        size_t buffer_size = 0;
        status_ = rocsparse_spmm(handle_,
                                 operation_,
                                 operation_,
                                 &alpha,
                                 A_description_,
                                 B_description_,
                                 &beta,
                                 C_description_,
                                 dataType_,
                                 algorithm_,
                                 rocsparse_spmm_stage_buffer_size,
                                 &buffer_size,
                                 nullptr);
        checkStatus("Failed rocsparse_spmm with stage=rocsparse_spmm_stage_buffer_size");

        void* buffer = nullptr;
        if (print_) std::cout << "\tAllocating buffer with buffer_size = " << buffer_size << std::endl;
        if (buffer_size > 0) hipCheckError(hipMalloc(&buffer, buffer_size));

        status_ = rocsparse_spmm(handle_,
                                 operation_,
                                 operation_,
                                 &alpha,
                                 A_description_,
                                 B_description_,
                                 &beta,
                                 C_description_,
                                 dataType_,
                                 algorithm_,
                                 rocsparse_spmm_stage_preprocess,
                                 &buffer_size,
                                 buffer);
        checkStatus("Failed rocsparse_spmm with stage=rocsparse_spmm_stage_preprocess");

        hipCheckError(hipDeviceSynchronize());
        status_ = rocsparse_spmm(handle_,
                                 operation_,
                                 operation_,
                                 &alpha,
                                 A_description_,
                                 B_description_,
                                 &beta,
                                 C_description_,
                                 dataType_,
                                 algorithm_,
                                 rocsparse_spmm_stage_compute,
                                 &buffer_size,
                                 buffer);
        checkStatus("Failed rocsparse_spmm with stage=rocsparse_spmm_stage_compute");

        hipCheckError(hipDeviceSynchronize());
        if (print_) std::cout << "\tdestroying rocSPARSE structures" << std::endl;
        // Now clean up
        status_ = rocsparse_destroy_spmat_descr(A_description_);
        checkStatus("Failed rocsparse_destroy_spmat_descr for A");
        status_ = rocsparse_destroy_dnmat_descr(B_description_);
        checkStatus("Failed rocsparse_destroy_dnmat_descr for B");
        status_ = rocsparse_destroy_dnmat_descr(C_description_);
        checkStatus("Failed rocsparse_destroy_dnmat_descr for C");
        if (buffer) hipCheckError(hipFree(buffer));
        hipCheckError(hipDeviceSynchronize());
        break;
      }
      case gpuOffloadType::unified: {
        if (print_) std::cout << "\tCreating rocSPARSE structures" << std::endl;
        // Set up the rocSPARSE structures for the GEMV
        status_ = rocsparse_create_csr_descr(&A_description_,
                                             m_,
                                             k_,
                                             nnz_,
                                             A_rows_,
                                             A_cols_,
                                             A_vals_,
                                             index_,
                                             index_,
                                             base_,
                                             dataType_);
        checkStatus("Failed rocsparse_create_csr_descr for A");

        status_ = rocsparse_create_dnmat_descr(&B_description_,
                                               k_,
                                               n_,
                                               k_,
                                               B_,
                                               dataType_,
                                               order_);
        checkStatus("Failed rocsparse_create_dnmat_descr for B");

        status_ = rocsparse_create_dnmat_descr(&C_description_,
                                               m_,
                                               n_,
                                               m_,
                                               C_,
                                               dataType_,
                                               order_);
        checkStatus("Failed rocsparse_create_dnmat_descr for C");
        hipCheckError(hipDeviceSynchronize());

        size_t buffer_size;
        status_ = rocsparse_spmm(handle_,
                                 operation_,
                                 operation_,
                                 &alpha,
                                 A_description_,
                                 B_description_,
                                 &beta,
                                 C_description_,
                                 dataType_,
                                 algorithm_,
                                 rocsparse_spmm_stage_buffer_size,
                                 &buffer_size,
                                 nullptr);
        checkStatus("Failed rocsparse_spmm with stage=rocsparse_spmm_stage_buffer_size");

        void* buffer = nullptr;
        if (print_) std::cout << "\tAllocating buffer with buffer_size = " << buffer_size << std::endl;
        if (buffer_size > 0) hipCheckError(hipMallocManaged(&buffer, buffer_size));

        status_ = rocsparse_spmm(handle_,
                                 operation_,
                                 operation_,
                                 &alpha,
                                 A_description_,
                                 B_description_,
                                 &beta,
                                 C_description_,
                                 dataType_,
                                 algorithm_,
                                 rocsparse_spmm_stage_preprocess,
                                 &buffer_size,
                                 buffer);
        checkStatus("Failed rocsparse_spmm with stage=rocsparse_spmm_stage_preprocess");

        hipCheckError(hipDeviceSynchronize());
        status_ = rocsparse_spmm(handle_,
                                 operation_,
                                 operation_,
                                 &alpha,
                                 A_description_,
                                 B_description_,
                                 &beta,
                                 C_description_,
                                 dataType_,
                                 algorithm_,
                                 rocsparse_spmm_stage_compute,
                                 &buffer_size,
                                 buffer);
        checkStatus("Failed rocsparse_spmm with stage=rocsparse_spmm_stage_compute");

        hipCheckError(hipDeviceSynchronize());
        if (print_) std::cout << "\tdestroying rocSPARSE structures" << std::endl;
        // Now clean up
        status_ = rocsparse_destroy_spmat_descr(A_description_);
        checkStatus("Failed rocsparse_destroy_spmat_descr for A");
        status_ = rocsparse_destroy_dnmat_descr(B_description_);
        checkStatus("Failed rocsparse_destroy_dnmat_descr for B");
        status_ = rocsparse_destroy_dnmat_descr(C_description_);
        checkStatus("Failed rocsparse_destroy_dnmat_descr for C");
        if (buffer) hipCheckError(hipFree(buffer));
        hipCheckError(hipDeviceSynchronize());
        break;
      }
    }
  }

  void postLoopRequirements() override {
    switch (offload_) {
      case gpuOffloadType::always: {
        break;
      }
      case gpuOffloadType::once: {
        // Move result back to the CPU
        if (print_) std::cout << "\tMoving data to CPU" << std::endl;
        hipCheckError(hipMemcpyAsync(C_, 
                                     C_device_, 
                                     sizeof(T) * m_ * n_, 
                                     hipMemcpyDeviceToHost, 
                                     s3_));
        hipCheckError(hipDeviceSynchronize());
        break;
      }
      case gpuOffloadType::unified: {
        // Ensure all output data resides on host once work has completed
        if (print_) std::cout << "\tMoving data to CPU" << std::endl;
        hipCheckError(hipMemPrefetchAsync(C_, 
                                          sizeof(T) * m_ * n_, 
                                          hipCpuDeviceId, 
                                          s3_));
        // Ensure device has finished all work.
        hipCheckError(hipDeviceSynchronize());
        break;
      }
    }
  }

  void postCallKernelCleanup() override {
    if (print_) std::cout << "Post-kernel cleanup" << std::endl;
    if (offload_ == gpuOffloadType::unified) {
      if (print_) std::cout << "\tFreeing unified arrays" << std::endl;
      hipCheckError(hipFree(A_));
      hipCheckError(hipFree(A_rows_));
      hipCheckError(hipFree(A_cols_));
      hipCheckError(hipFree(A_vals_));
      hipCheckError(hipFree(B_));
      hipCheckError(hipFree(C_));
    } else {
      if (print_) std::cout << "\tFreeing CPU arrays" << std::endl;
      hipCheckError(hipHostFree((void*)A_));
      hipCheckError(hipHostFree((void*)A_rows_));
      hipCheckError(hipHostFree((void*)A_cols_));
      hipCheckError(hipHostFree((void*)A_vals_));
      hipCheckError(hipHostFree((void*)B_));
      hipCheckError(hipHostFree((void*)C_));

      if (print_) std::cout << "\tFreeing GPU arrays" << std::endl;
      hipCheckError(hipFree(A_rows_device_));
      hipCheckError(hipFree(A_cols_device_));
      hipCheckError(hipFree(A_vals_device_));
      hipCheckError(hipFree(B_device_));
      hipCheckError(hipFree(C_device_));
    }
  }

  void checkStatus(std::string message) {
    if (status_ != rocsparse_status_success) {
      std::cerr << message << std::endl;
      switch (status_) {
        case rocsparse_status_success: {
          std::cerr << "rocsparse_status_success" << std::endl;
          break;
        }
        case rocsparse_status_invalid_handle: {
          std::cerr << "rocsparse_status_invalid_handle" << std::endl;
          break;
        }
        case rocsparse_status_not_implemented: {
          std::cerr << "rocsparse_status_not_implemented" << std::endl;
          break;
        }
        case rocsparse_status_invalid_pointer: {
          std::cerr << "rocsparse_status_invalid_pointer" << std::endl;
          break;
        }  
        case rocsparse_status_invalid_size: {
          std::cerr << "rocsparse_status_invalid_size" << std::endl;
          break;
        }
        case rocsparse_status_memory_error: {
          std::cerr << "rocsparse_status_memory_error" << std::endl;
          break;
        }
        case rocsparse_status_internal_error: {
          std::cerr << "rocsparse_status_internal_error" << std::endl;
          break;
        }
        case rocsparse_status_invalid_value: {
          std::cerr << "rocsparse_status_invalid_value" << std::endl;
          break;
        }
        case rocsparse_status_arch_mismatch: {
          std::cerr << "rocsparse_status_arch_mismatch" << std::endl;
          break;
        }
        case rocsparse_status_zero_pivot: {
          std::cerr << "rocsparse_status_zero_pivot" << std::endl;
          break;
        }
        case rocsparse_status_not_initialized: {
          std::cerr << "rocsparse_status_not_initialized" << std::endl;
          break;
        }
        case rocsparse_status_type_mismatch: {
          std::cerr << "rocsparse_status_type_mismatch" << std::endl;
          break;
        }
        case rocsparse_status_requires_sorted_storage: {
          std::cerr << "rocsparse_status_requires_sorted_storage" << std::endl;
          break;
        }
        case rocsparse_status_thrown_exception: {
          std::cerr << "rocsparse_status_thrown_exception" << std::endl;
          break;
        }
        default: {
          std::cerr << "Unknown status code: " << status_ << std::endl;
        }
      }
      exit(1);
    }
  }

  bool initialised_ = false;
  bool print_ = false;

  rocsparse_status status_;
  rocsparse_operation operation_;
  rocsparse_handle handle_;
  rocsparse_index_base base_;
  rocsparse_datatype dataType_;
  rocsparse_matrix_type type_;
  rocsparse_indextype index_;
  rocsparse_spmm_alg algorithm_;
  rocsparse_order order_;

  rocsparse_spmat_descr A_description_;
  rocsparse_dnmat_descr B_description_;
  rocsparse_dnmat_descr C_description_;

  int64_t* A_rows_;
  int64_t* A_cols_;
  T* A_vals_;

  int64_t* A_rows_device_;
  int64_t* A_cols_device_;
  T* A_vals_device_;
  T* B_device_;
  T* C_device_;

  int gpuDevice_;
  hipStream_t s1_, s2_, s3_;

  const T alpha = ALPHA;
  const T beta = BETA;
};
}

#endif
