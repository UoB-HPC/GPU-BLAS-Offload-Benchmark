#pragma once

#ifdef GPU_ROCBLAS
#include <hip/hip_runtime_api.h>
#include <rocsparse/rocsparse.h>

#include "../include/kernels/GPU/spgemm.hh"
#include "../include/utilities.hh"
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

  ~spgemm_gpu() {
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
    print_ = true;
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
    print_ = false;
    m_ = m;
    n_ = n;
    k_ = k;
    sparsity_ = sparsity;
    offload_ = offload;
    nnz_ = 1 + (uint64_t)((double)m_ * (double)n_ * (1.0 - sparsity_));
    
    // Set up rocSPARSE type parameters
    m_roc_ = m_;
    n_roc_ = n_;
    k_roc_ = k_;
    nnz_roc_ = nnz_;

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
                                     sizeof(rocsparse_int) * (m_ + 1),
                                     hipMemcpyHostToDevice,
                                     s1_));
        hipCheckError(hipMemcpyAsync(A_cols_device_,
                                     A_cols_,
                                     sizeof(rocsparse_int) * nnz_,
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
                                          sizeof(rocsparse_int) * (m_ + 1), 
                                          gpuDevice_, 
                                          s1_));
        hipCheckError(hipMemPrefetchAsync(A_cols_, 
                                          sizeof(rocsparse_int) * nnz_, 
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

  void callSpgemm() override {
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
                                     sizeof(rocsparse_int) * nnz_,
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
        status_ = rocsparse_create_mat_descr(&description_); // The defaults are for base=0, and type=general.  This is okay for us.
        checkStatus("Failed rocsparse_create_mat_descr");

        status_ = rocsparse_create_mat_info(&info_);
        checkStatus("Failed rocsparse_create_mat_info");

        if constexpr (std::is_same_v<T, float>) {
          status_ = rocsparse_scsrmm(handle_, 
                                     operation_, 
                                     operation_,  
                                     m_roc_, 
                                     n_roc_, 
                                     k_roc_, 
                                     nnz_roc_, 
                                     &alpha, 
                                     description_, 
                                     A_vals_device_, 
                                     A_rows_device_, 
                                     A_cols_device_, 
                                     B_device_, 
                                     k_roc_, // csrmm requires column-major format.  Therefore, leading dimensions are numbers of rows 
                                     &beta, 
                                     C_device_, 
                                     m_roc_); // csrmm requires column-major format.  Therefore, leading dimensions are numbers of rows
          checkStatus("Falied rocsparse_scsrmm");
        } else if constexpr (std::is_same_v<T, double>) {
          status_ = rocsparse_dcsrmm(handle_, 
                                     operation_, 
                                     operation_,  
                                     m_roc_, 
                                     n_roc_, 
                                     k_roc_, 
                                     nnz_roc_, 
                                     &alpha, 
                                     description_, 
                                     A_vals_device_, 
                                     A_rows_device_, 
                                     A_cols_device_, 
                                     B_device_, 
                                     k_roc_, // csrmm requires column-major format.  Therefore, leading dimensions are numbers of rows 
                                     &beta, 
                                     C_device_, 
                                     m_roc_); // csrmm requires column-major format.  Therefore, leading dimensions are numbers of rows
          checkStatus("Failed rocsparse_dcsrmm");
        }
        if (print_) std::cout << "\tdestroying rocSPARSE structures" << std::endl;
        // Now clean up
        status_ = rocsparse_destroy_mat_descr(description_);
        checkStatus("Failed rocsparse_destroy_mat_descr");
        status_ = rocsparse_destroy_mat_info(info_);
        checkStatus("Failed rocsparse_destroy_mat_info");

        // Move result back to the CPU
        if (print_) std::cout << "\tMovin data to CPU" << std::endl;
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
        status_ = rocsparse_create_mat_descr(&description_); // The defaults are for base=0, and type=general.  This is okay for us.
        checkStatus("Failed rocsparse_create_mat_descr");

        status_ = rocsparse_create_mat_info(&info_);
        checkStatus("Failed rocsparse_create_mat_info");

        if constexpr (std::is_same_v<T, float>) {
          status_ = rocsparse_scsrmm(handle_, 
                                     operation_, 
                                     operation_,  
                                     m_roc_, 
                                     n_roc_, 
                                     k_roc_, 
                                     nnz_roc_, 
                                     &alpha, 
                                     description_, 
                                     A_vals_device_, 
                                     A_rows_device_, 
                                     A_cols_device_, 
                                     B_device_, 
                                     k_roc_, // csrmm requires column-major format.  Therefore, leading dimensions are numbers of rows 
                                     &beta, 
                                     C_device_, 
                                     m_roc_); // csrmm requires column-major format.  Therefore, leading dimensions are numbers of rows
          checkStatus("Falied rocsparse_scsrmm");
        } else if constexpr (std::is_same_v<T, double>) {
          status_ = rocsparse_dcsrmm(handle_, 
                                     operation_, 
                                     operation_,  
                                     m_roc_, 
                                     n_roc_, 
                                     k_roc_, 
                                     nnz_roc_, 
                                     &alpha, 
                                     description_, 
                                     A_vals_device_, 
                                     A_rows_device_, 
                                     A_cols_device_, 
                                     B_device_, 
                                     k_roc_, // csrmm requires column-major format.  Therefore, leading dimensions are numbers of rows 
                                     &beta, 
                                     C_device_, 
                                     m_roc_); // csrmm requires column-major format.  Therefore, leading dimensions are numbers of rows
          checkStatus("Failed rocsparse_dcsrmm");
        }
        if (print_) std::cout << "\tdestroying rocSPARSE structures" << std::endl;
        // Now clean up
        status_ = rocsparse_destroy_mat_descr(description_);
        checkStatus("Failed rocsparse_destroy_mat_descr");
        status_ = rocsparse_destroy_mat_info(info_);
        checkStatus("Failed rocsparse_destroy_mat_info");
        break;
      }
      case gpuOffloadType::unified: {

        if (print_) std::cout << "\tCreating rocSPARSE structures" << std::endl;
        // Set up the rocSPARSE structures for the GEMV
        status_ = rocsparse_create_mat_descr(&description_); // The defaults are for base=0, and type=general.  This is okay for us.
        checkStatus("Failed rocsparse_create_mat_descr");

        status_ = rocsparse_create_mat_info(&info_);
        checkStatus("Failed rocsparse_create_mat_info");

        if constexpr (std::is_same_v<T, float>) {
          status_ = rocsparse_scsrmm(handle_, 
                                     operation_, 
                                     operation_,  
                                     m_roc_, 
                                     n_roc_, 
                                     k_roc_, 
                                     nnz_roc_, 
                                     &alpha, 
                                     description_, 
                                     A_vals_, 
                                     A_rows_, 
                                     A_cols_, 
                                     B_, 
                                     k_roc_, // csrmm requires column-major format.  Therefore, leading dimensions are numbers of rows 
                                     &beta, 
                                     C_, 
                                     m_roc_); // csrmm requires column-major format.  Therefore, leading dimensions are numbers of rows
          checkStatus("Failed rocsparse_scsrmm");
        } else if constexpr (std::is_same_v<T, double>) {
          status_ = rocsparse_dcsrmm(handle_, 
                                     operation_, 
                                     operation_,  
                                     m_roc_, 
                                     n_roc_, 
                                     k_roc_, 
                                     nnz_roc_, 
                                     &alpha, 
                                     description_, 
                                     A_vals_, 
                                     A_rows_, 
                                     A_cols_, 
                                     B_, 
                                     k_roc_, // csrmm requires column-major format.  Therefore, leading dimensions are numbers of rows 
                                     &beta, 
                                     C_, 
                                     m_roc_); // csrmm requires column-major format.  Therefore, leading dimensions are numbers of rows
          checkStatus("Failed rocsparse_dcsrmm");
        }
        if (print_) std::cout << "\tdestroying rocSPARSE structures" << std::endl;
        // Now clean up
        status_ = rocsparse_destroy_mat_descr(description_);
        checkStatus("Failed rocsparse_destroy_mat_descr");
        status_ = rocsparse_destroy_mat_info(info_);
        checkStatus("Failed rocsparse_destroy_mat_info");
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
        if (print_) std::cout << "\tMovin data to CPU" << std::endl;
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
        if (print_) std::cout << "\tMovin data to CPU" << std::endl;
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
      exit(1);
    }
  }

  bool initialised_ = false;
  bool print_ = false;

  rocsparse_mat_info info_;
  rocsparse_status status_;
  rocsparse_operation operation_;
  rocsparse_handle handle_;
  rocsparse_mat_descr description_;
  rocsparse_index_base index_;
  rocsparse_matrix_type type_;

  rocsparse_int m_roc_, n_roc_, k_roc_, nnz_roc_;

  rocsparse_int* A_rows_;
  rocsparse_int* A_cols_;
  T* A_vals_;

  rocsparse_int* A_rows_device_;
  rocsparse_int* A_cols_device_;
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
