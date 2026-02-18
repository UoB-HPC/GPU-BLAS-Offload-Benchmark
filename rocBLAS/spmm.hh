#pragma once

#ifdef GPU_ROCBLAS
#include <iostream>
#include <hip/hip_runtime_api.h>
#include <rocsparse/rocsparse.h>

#include "../include/kernels/GPU/spmm.hh"
#include "../include/utilities.hh"
#include "common.hh"


namespace gpu {
template <typename T>
class spmm_gpu : public spmm<T> {
public:
  using spmm<T>::spmm;
  using spmm<T>::initInputMatrices;
  using spmm<T>::m_;
  using spmm<T>::n_;
  using spmm<T>::k_;
  using spmm<T>::B_;
  using spmm<T>::C_;
  using spmm<T>::offload_;
  using spmm<T>::nnz_;
  using spmm<T>::sparsity_;
  using spmm<T>::type_;

  ~spmm_gpu() {
    if (initialised_) {
      rocsparse_destroy_handle(handle_);
      hipCheckError(hipStreamDestroy(s1_));
      hipCheckError(hipStreamDestroy(s2_));
      hipCheckError(hipStreamDestroy(s3_));
      hipCheckError(hipStreamDestroy(s4_));
      hipCheckError(hipStreamDestroy(s5_));
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

      rocCheckError(rocsparse_set_stream(handle_, s1_));

      hipCheckError(hipGetDevice(&gpuDevice_));
    }
    
    offload_ = offload;
    sparsity_ = sparsity;
    type_ = type;

    m_ = m;
    n_ = n;
    k_ = k;

    B_ = C_ = B_dev_ = C_dev_ = A_vals_ = A_vals_dev_ = nullptr;
    A_rows_ = A_cols_ = A_rows_dev_ = A_cols_dev_ = nullptr;
    
    if (std::is_same_v<T, float>) {
      dataType_ = rocsparse_datatype_f32_r;
    } else if (std::is_same_v<T, double>) {
      dataType_ = rocsparse_datatype_f64_r;
    } else {
      std::cerr << "INVALID DATA TYPE PASSED TO rocSPARSE" << std::endl;
      exit(1);
    }

    if (offload_ == gpuOffloadType::unified) {
      hipCheckError(hipMallocManaged(&B_, sizeof(T) * k_ * n_));
      hipCheckError(hipMallocManaged(&C_, sizeof(T) * m_ * n_));
    } else {
      // Host data structures
      hipCheckError(hipHostMalloc((void**)&B_, sizeof(T) * k_ * n_));
      hipCheckError(hipHostMalloc((void**)&C_, sizeof(T) * m_ * n_));
      // GPU data structures
      hipCheckError(hipMalloc((void**)&B_dev_, sizeof(T) * k_ * n_));
      hipCheckError(hipMalloc((void**)&C_dev_, sizeof(T) * m_ * n_));
    }
    hipCheckError(hipDeviceSynchronize());

    initInputMatrices();
  }


protected:
  void toSparseFormat() override {
    if (offload_ == gpuOffloadType::always) {
      A_vals_store_ = (T*)malloc(sizeof(T) * nnz_);
      A_cols_store_ = (int64_t*)malloc(sizeof(int64_t) * nnz_);
      A_rows_store_ = (int64_t*)malloc(sizeof(int64_t) * (m_ + 1));

      if (type_ == matrixType::rmat) {
        rMatCSR<T, int64_t>(A_vals_store_, A_cols_store_, A_rows_store_, m_, k_, nnz_);
      } else if (type_ == matrixType::random) {
        randomCSR<T, int64_t>(A_vals_store_, A_cols_store_, A_rows_store_, m_, k_, nnz_);
      } else if (type_ == matrixType::finiteElements) {
        finiteElementCSR<T, int64_t>(A_vals_store_, A_cols_store_, A_rows_store_, m_, k_, nnz_);
      } else {
        exit(1);
      }
    }

    // Allocate CSR arrays
    if (offload_ == gpuOffloadType::unified) {
      hipCheckError(hipMallocManaged(&A_vals_, nnz_ * sizeof(T)));
      hipCheckError(hipMallocManaged(&A_cols_, nnz_ * sizeof(int64_t)));
      hipCheckError(hipMallocManaged(&A_rows_, (m_ + 1) * sizeof(int64_t)));
    } else {
      A_vals_ = (T*)malloc(nnz_ * sizeof(T));
      A_cols_ = (int64_t*)malloc(nnz_ * sizeof(int64_t));
      A_rows_ = (int64_t*)malloc((m_ + 1) * sizeof(int64_t));
      hipCheckError(hipMalloc((void**)&A_vals_dev_, nnz_ * sizeof(T)));
      hipCheckError(hipMalloc((void**)&A_cols_dev_, nnz_ * sizeof(int64_t)));
      hipCheckError(hipMalloc((void**)&A_rows_dev_, (m_ + 1) * sizeof(int64_t)));
    }
    hipCheckError(hipDeviceSynchronize());

    memcpy(A_vals_, A_vals_store_, sizeof(T) * nnz_);
    memcpy(A_cols_, A_cols_store_, sizeof(int64_t) * nnz_);
    memcpy(A_rows_, A_rows_store_, sizeof(int64_t) * (m_ + 1));
    hipCheckError(hipDeviceSynchronize());
  }

private:
  void preLoopRequirements() override {
    switch (offload_) {
      case gpuOffloadType::always: {
        break;
      }
      case gpuOffloadType::once: {
        hipCheckError(hipMemcpyAsync(A_vals_dev_, A_vals_, nnz_ * sizeof(T), hipMemcpyHostToDevice, s1_));
        hipCheckError(hipMemcpyAsync(A_cols_dev_, A_cols_, nnz_ * sizeof(int64_t), hipMemcpyHostToDevice, s2_));
        hipCheckError(hipMemcpyAsync(A_rows_dev_, A_rows_, (m_ + 1) * sizeof(int64_t), hipMemcpyHostToDevice, s3_));
        hipCheckError(hipMemcpyAsync(B_dev_, B_, (k_ * n_) * sizeof(T), hipMemcpyHostToDevice, s4_));
        hipCheckError(hipMemcpyAsync(C_dev_, C_, (m_ * n_) * sizeof(T), hipMemcpyHostToDevice, s5_));
        hipCheckError(hipDeviceSynchronize());
        break;
      }
      case gpuOffloadType::unified: {
        hipCheckError(hipMemPrefetchAsync(A_vals_, nnz_ * sizeof(T), gpuDevice_, s1_));
        hipCheckError(hipMemPrefetchAsync(A_cols_, nnz_ * sizeof(int64_t), gpuDevice_, s2_));
        hipCheckError(hipMemPrefetchAsync(A_rows_, (m_ + 1) * sizeof(int64_t), gpuDevice_, s3_));
        hipCheckError(hipMemPrefetchAsync(B_, (n_ * k_) * sizeof(T), gpuDevice_, s4_));
        hipCheckError(hipMemPrefetchAsync(C_, (m_ * n_) * sizeof(T), gpuDevice_, s5_));
        hipCheckError(hipDeviceSynchronize());
        break;
      }
    }
  }

  void callSpmm() override {
    switch (offload_) {
      case gpuOffloadType::always: {
        hipCheckError(hipMemcpyAsync(A_vals_dev_, A_vals_, nnz_ * sizeof(T), hipMemcpyHostToDevice, s1_));
        hipCheckError(hipMemcpyAsync(A_cols_dev_, A_cols_, nnz_ * sizeof(int64_t), hipMemcpyHostToDevice, s2_));
        hipCheckError(hipMemcpyAsync(A_rows_dev_, A_rows_, (m_ + 1) * sizeof(int64_t), hipMemcpyHostToDevice, s3_));
        hipCheckError(hipMemcpyAsync(B_dev_, B_, (k_ * n_) * sizeof(T), hipMemcpyHostToDevice, s4_));
        hipCheckError(hipMemcpyAsync(C_dev_, C_, (m_ * n_) * sizeof(T), hipMemcpyHostToDevice, s5_));
        // Set up the rocSPARSE structures for the GEMV
        rocCheckError(rocsparse_create_csr_descr(&A_descr_, m_, k_, nnz_, A_rows_dev_,
                                                 A_cols_dev_, A_vals_dev_, index_, index_,
                                                 base_, dataType_));
        
        rocCheckError(rocsparse_create_dnmat_descr(&B_descr_, k_, n_, k_, B_dev_,
                                                   dataType_, order_));

        rocCheckError(rocsparse_create_dnmat_descr(&C_descr_, m_, n_, m_, C_dev_,
                                                   dataType_, order_));
        hipCheckError(hipDeviceSynchronize());

        size_t buffer_size = 0;
        rocCheckError(rocsparse_spmm(handle_,
                                     operation_,
                                     operation_,
                                     &alpha,
                                     A_descr_,
                                     B_descr_,
                                     &beta,
                                     C_descr_,
                                     dataType_,
                                     algorithm_,
                                     rocsparse_spmm_stage_buffer_size,
                                     &buffer_size,
                                     nullptr));
        
        void* buffer = nullptr;
        if (buffer_size > 0) hipCheckError(hipMalloc(&buffer, buffer_size));

        rocCheckError(rocsparse_spmm(handle_,
                                     operation_,
                                     operation_,
                                     &alpha,
                                     A_descr_,
                                     B_descr_,
                                     &beta,
                                     C_descr_,
                                     dataType_,
                                     algorithm_,
                                     rocsparse_spmm_stage_preprocess,
                                     &buffer_size,
                                     buffer));
        
        hipCheckError(hipDeviceSynchronize());
        rocCheckError(rocsparse_spmm(handle_,
                                     operation_,
                                     operation_,
                                     &alpha,
                                     A_descr_,
                                     B_descr_,
                                     &beta,
                                     C_descr_,
                                     dataType_,
                                     algorithm_,
                                     rocsparse_spmm_stage_compute,
                                     &buffer_size,
                                     buffer));
        
        hipCheckError(hipDeviceSynchronize());
        // Now clean up
        rocCheckError(rocsparse_destroy_spmat_descr(A_descr_));
        rocCheckError(rocsparse_destroy_dnmat_descr(B_descr_));
        rocCheckError(rocsparse_destroy_dnmat_descr(C_descr_));
        if (buffer != nullptr) hipCheckError(hipFree(buffer));
        hipCheckError(hipDeviceSynchronize());

        // Move result back to the CPU
        hipCheckError(hipMemcpyAsync(C_, C_dev_, 
                                     sizeof(T) * m_ * n_, 
                                     hipMemcpyDeviceToHost, 
                                     s3_));
        hipCheckError(hipDeviceSynchronize());
        break;
      }
      case gpuOffloadType::once: {
        rocCheckError(rocsparse_create_csr_descr(&A_descr_, m_, k_, nnz_, A_rows_dev_,
                                                 A_cols_dev_, A_vals_dev_, index_, index_,
                                                 base_, dataType_));
        
        rocCheckError(rocsparse_create_dnmat_descr(&B_descr_, k_, n_, k_, B_dev_,
                                                   dataType_, order_));

        rocCheckError(rocsparse_create_dnmat_descr(&C_descr_, m_, n_, m_, C_dev_,
                                                   dataType_, order_));
        hipCheckError(hipDeviceSynchronize());

        size_t buffer_size = 0;
        rocCheckError(rocsparse_spmm(handle_,
                                     operation_,
                                     operation_,
                                     &alpha,
                                     A_descr_,
                                     B_descr_,
                                     &beta,
                                     C_descr_,
                                     dataType_,
                                     algorithm_,
                                     rocsparse_spmm_stage_buffer_size,
                                     &buffer_size,
                                     nullptr));
        
        void* buffer = nullptr;
        if (buffer_size > 0) hipCheckError(hipMalloc(&buffer, buffer_size));

        rocCheckError(rocsparse_spmm(handle_,
                                     operation_,
                                     operation_,
                                     &alpha,
                                     A_descr_,
                                     B_descr_,
                                     &beta,
                                     C_descr_,
                                     dataType_,
                                     algorithm_,
                                     rocsparse_spmm_stage_preprocess,
                                     &buffer_size,
                                     buffer));
        
        hipCheckError(hipDeviceSynchronize());
        rocCheckError(rocsparse_spmm(handle_,
                                     operation_,
                                     operation_,
                                     &alpha,
                                     A_descr_,
                                     B_descr_,
                                     &beta,
                                     C_descr_,
                                     dataType_,
                                     algorithm_,
                                     rocsparse_spmm_stage_compute,
                                     &buffer_size,
                                     buffer));
        
        hipCheckError(hipDeviceSynchronize());
        // Now clean up
        rocCheckError(rocsparse_destroy_spmat_descr(A_descr_));
        rocCheckError(rocsparse_destroy_dnmat_descr(B_descr_));
        rocCheckError(rocsparse_destroy_dnmat_descr(C_descr_));
        if (buffer != nullptr) hipCheckError(hipFree(buffer));
        hipCheckError(hipDeviceSynchronize());
        break;
      }
      case gpuOffloadType::unified: {
        // Set up the rocSPARSE structures for the GEMV
        rocCheckError(rocsparse_create_csr_descr(&A_descr_,
                                                 m_,
                                                 k_,
                                                 nnz_,
                                                 A_rows_,
                                                 A_cols_,
                                                 A_vals_,
                                                 index_,
                                                 index_,
                                                 base_,
                                                 dataType_));

        rocCheckError(rocsparse_create_dnmat_descr(&B_descr_,
                                                   k_,
                                                   n_,
                                                   k_,
                                                   B_,
                                                   dataType_,
                                                   order_));
        
        rocCheckError(rocsparse_create_dnmat_descr(&C_descr_,
                                                   m_,
                                                   n_,
                                                   m_,
                                                   C_,
                                                   dataType_,
                                                   order_));
        hipCheckError(hipDeviceSynchronize());

        size_t buffer_size = 0;
        rocCheckError(rocsparse_spmm(handle_,
                                     operation_,
                                     operation_,
                                     &alpha,
                                     A_descr_,
                                     B_descr_,
                                     &beta,
                                     C_descr_,
                                     dataType_,
                                     algorithm_,
                                     rocsparse_spmm_stage_buffer_size,
                                     &buffer_size,
                                     nullptr));
        
        void* buffer = nullptr;
        if (buffer_size > 0) hipCheckError(hipMalloc(&buffer, buffer_size));

        rocCheckError(rocsparse_spmm(handle_,
                                     operation_,
                                     operation_,
                                     &alpha,
                                     A_descr_,
                                     B_descr_,
                                     &beta,
                                     C_descr_,
                                     dataType_,
                                     algorithm_,
                                     rocsparse_spmm_stage_preprocess,
                                     &buffer_size,
                                     buffer));
        
        hipCheckError(hipDeviceSynchronize());
        rocCheckError(rocsparse_spmm(handle_,
                                     operation_,
                                     operation_,
                                     &alpha,
                                     A_descr_,
                                     B_descr_,
                                     &beta,
                                     C_descr_,
                                     dataType_,
                                     algorithm_,
                                     rocsparse_spmm_stage_compute,
                                     &buffer_size,
                                     buffer));
        
        hipCheckError(hipDeviceSynchronize());
        // Now clean up
        rocCheckError(rocsparse_destroy_spmat_descr(A_descr_));
        rocCheckError(rocsparse_destroy_dnmat_descr(B_descr_));
        rocCheckError(rocsparse_destroy_dnmat_descr(C_descr_));
        if (buffer != nullptr) hipCheckError(hipFree(buffer));
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
        hipCheckError(hipMemcpyAsync(C_, 
                                     C_dev_, 
                                     sizeof(T) * m_ * n_, 
                                     hipMemcpyDeviceToHost, 
                                     s3_));
        hipCheckError(hipDeviceSynchronize());
        break;
      }
      case gpuOffloadType::unified: {
        // Ensure all output data resides on host once work has completed
        hipCheckError(hipMemPrefetchAsync(C_, sizeof(T) * m_ * n_, 
                                          hipCpuDeviceId, s3_));
        // Ensure device has finished all work.
        hipCheckError(hipDeviceSynchronize());
        break;
      }
    }
  }

  void postCallKernelCleanup() override {
    if (offload_ == gpuOffloadType::unified) {
      hipCheckError(hipFree(A_rows_));
      hipCheckError(hipFree(A_cols_));
      hipCheckError(hipFree(A_vals_));
      hipCheckError(hipFree(B_));
      hipCheckError(hipFree(C_));
      free(A_vals_store_);
      free(A_cols_store_);
      free(A_rows_store_);
    } else {
      hipCheckError(hipHostFree((void*)A_rows_));
      hipCheckError(hipHostFree((void*)A_cols_));
      hipCheckError(hipHostFree((void*)A_vals_));
      hipCheckError(hipHostFree((void*)B_));
      hipCheckError(hipHostFree((void*)C_));

      hipCheckError(hipFree(A_rows_dev_));
      hipCheckError(hipFree(A_cols_dev_));
      hipCheckError(hipFree(A_vals_dev_));
      hipCheckError(hipFree(B_dev_));
      hipCheckError(hipFree(C_dev_));
    }
  }

  bool initialised_ = false;

  int gpuDevice_;
  hipStream_t s1_, s2_, s3_, s4_, s5_;

  const T alpha = ALPHA;
  const T beta = BETA;

  rocsparse_status status_;
  rocsparse_operation operation_;
  rocsparse_handle handle_;
  rocsparse_index_base base_;
  rocsparse_datatype dataType_;
  rocsparse_indextype index_;
  rocsparse_spmm_alg algorithm_;
  rocsparse_order order_;

  rocsparse_spmat_descr A_descr_;
  rocsparse_dnmat_descr B_descr_;
  rocsparse_dnmat_descr C_descr_;




  /**
   * ___________ Host data ______________
   */
	/** CSR format vectors for matrix A */
	T* A_vals_;
	int64_t* A_cols_;
  int64_t* A_rows_;
  int64_t A_num_rows_;
  int64_t A_num_cols_;

  /** dense format values for matrices B and C */
  int64_t B_num_rows_;
  int64_t B_num_cols_;

  int64_t C_num_rows_;
  int64_t C_num_cols_;

  /**
   * _____________ Device data ________________
   */
  T* A_vals_dev_;
  int64_t* A_cols_dev_;
  int64_t* A_rows_dev_;

  T* B_dev_;

  T* C_dev_;

  T* A_vals_store_;
  int64_t* A_cols_store_;
  int64_t* A_rows_store_;
};
}

#endif
