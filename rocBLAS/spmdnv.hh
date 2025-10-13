#pragma once

#ifdef GPU_ROCBLAS
#include <hip/hip_runtime_api.h>
#include <rocsparse/rocsparse.h>

#include "../include/kernels/GPU/spmdnv.hh"
#include "../include/utilities.hh"
#include "common.hh"

namespace gpu {
template <typename T>
class spmdnv_gpu : public spmdnv<T> {
public:
    using spmdnv<T>::spmdnv;
    using spmdnv<T>::initInputMatrixVector;
    using spmdnv<T>::nnz_;
    using spmdnv<T>::m_;
    using spmdnv<T>::n_;
    using spmdnv<T>::A_;
    using spmdnv<T>::x_;
    using spmdnv<T>::y_;
    using spmdnv<T>::offload_;
    using spmdnv<T>::sparsity_;

    ~spmdnv_gpu() {
      if (initialised_) {
        rocsparse_destroy_handle(handle_);
        hipCheckError(hipStreamDestroy(s1_));
        hipCheckError(hipStreamDestroy(s2_));
        hipCheckError(hipStreamDestroy(s3_));
      }
    }

    void initialise(gpuOffloadType offload, int m, int n, double sparsity)
    override {
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
      if (print_) std::cout << "Initialising with matrix of " << m << "x" << n << std::endl;
      m_ = m;
      n_ = n;
      sparsity_ = sparsity;
      offload_ = offload;

      nnz_ = 1 + (uint64_t)((double)m_ * (double)n_ * (1.0 - sparsity_));


      // Set up rocSPARSE metadata
      index_ = rocsparse_indextype_i64;
      type_ = rocsparse_matrix_type_general;
      operation_ = rocsparse_operation_none;
      base_ = rocsparse_index_base_zero;
      algorithm_ = rocsparse_spmv_alg_default; // There are a couple of CSR algorithms -- investigate which is best!
      if constexpr (std::is_same_v<T, float>) {
        dataType_ = rocsparse_datatype_f32_r;
      } else if constexpr (std::is_same_v<T, double>) {
        dataType_ = rocsparse_datatype_f64_r;
      } else {
        throw std::runtime_error("Unsupported data type for spmdnv_gpu");
      }


      if (print_) std::cout << "\tAbout to set up handle and hip streams" << std::endl;
      if (!initialised_) {
        // Get the GPU
        int count;
        hipCheckError(hipGetDeviceCount(&count));
        if (print_) std::cout << "Number of devices: " << count << std::endl;
        if (print_) std::cout << "Getting device ID" << std::endl;
        if (print_) std::cout << "\t\tGetting GPU device" << std::endl;
        hipCheckError(hipGetDevice(&gpuDevice_));
        if (print_) std::cout << "Device ID: " << gpuDevice_ << std::endl;
        
        // Make streams for asynchronous GPU comunication
        if (print_) std::cout << "\t\tCreating GPU streams" << std::endl;
        hipCheckError(hipStreamCreate(&s1_));
        hipCheckError(hipStreamCreate(&s2_));
        hipCheckError(hipStreamCreate(&s3_));

        if (print_) std::cout << "\t\tSetting up GPU handle" << std::endl;
        status_ = rocsparse_create_handle(&handle_);
        checkStatus("Failed rocsparse_create_handle");
      }

      if (print_) std::cout << "\tAbout to malloc arrays" << std::endl;
      if (offload_ == gpuOffloadType::unified) {
        hipCheckError(hipMallocManaged(&A_, sizeof(T) * m_ * n_));
        hipCheckError(hipMallocManaged(&A_rows_, sizeof(int64_t) * (m_ + 1)));
        hipCheckError(hipMallocManaged(&A_cols_, sizeof(int64_t) * nnz_));
        hipCheckError(hipMallocManaged(&A_vals_, sizeof(T) * nnz_));
        hipCheckError(hipMallocManaged(&x_, sizeof(T) * n_));
        hipCheckError(hipMallocManaged(&y_, sizeof(T) * m_));
      } else {
        // Host data structures
        hipCheckError(hipHostMalloc((void**)&A_, sizeof(T) * m_ * n_));
        hipCheckError(hipHostMalloc((void**)&A_rows_, sizeof(int64_t) * (m_ + 1)));
        hipCheckError(hipHostMalloc((void**)&A_cols_, sizeof(int64_t) * nnz_));
        hipCheckError(hipHostMalloc((void**)&A_vals_, sizeof(T) * nnz_));
        hipCheckError(hipHostMalloc((void**)&x_, sizeof(T) * n_));
        hipCheckError(hipHostMalloc((void**)&y_, sizeof(T) * m_));
        // GPU data structures
        hipCheckError(hipMalloc((void**)&A_rows_device_, sizeof(int64_t) * (m_ + 1)));
        hipCheckError(hipMalloc((void**)&A_cols_device_, sizeof(int64_t) * nnz_));
        hipCheckError(hipMalloc((void**)&A_vals_device_, sizeof(T) * nnz_));
        hipCheckError(hipMalloc((void**)&x_device_, sizeof(T) * n_));
        hipCheckError(hipMalloc((void**)&y_device_, sizeof(T) * m_));
      }

      if (print_) std::cout << "\tInitialising matrix and vector" << std::endl;
      initInputMatrixVector();
    }


protected:
    void toSparseFormat() override {
      if (print_) std::cout << "\tTo Sparse" << std::endl;
      int64_t nnz_encountered = 0;

      A_rows_[0] = 0;

      for (int64_t row = 0; row < m_; row++) {
        for (int64_t col = 0; col < n_; col++) {
          if (A_[(row * n_) + col] != 0.0) {
            A_cols_[nnz_encountered] = col;
            A_vals_[nnz_encountered] = static_cast<T>(A_[(row * n_) + col]);
            nnz_encountered++;
          }
        }
        A_rows_[row + 1] = nnz_encountered;
      }
    }

private:
    /**
     * Before we enter the loop of calling the kernel, 
     * we need to move any data we may need.
     */
    void preLoopRequirements() override {
      if (print_) std::cout << "pre-loop stuff" << std::endl;
      switch (offload_) {
        case gpuOffloadType::always: {
          // For Always there is nothing to do here, 
          // as all memory is moved each time the 
          // kernel is called
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
          hipCheckError(hipMemcpyAsync(x_device_, 
                                       x_, 
                                       sizeof(T) * n_, 
                                       hipMemcpyHostToDevice, 
                                       s2_));
          hipCheckError(hipMemcpyAsync(y_device_, 
                                       y_, 
                                       sizeof(T) * m_, 
                                       hipMemcpyHostToDevice, 
                                       s3_));
          hipCheckError(hipDeviceSynchronize());
          break;
        }
        case gpuOffloadType::unified: {
          if (print_) std::cout << "\tMoving data to GPU" << std::endl;
          hipCheckError(hipMemPrefetchAsync(A_rows_, sizeof(int64_t) * (m_ + 1), gpuDevice_, s1_));
          hipCheckError(hipMemPrefetchAsync(A_cols_, sizeof(int64_t) * nnz_, gpuDevice_, s1_));
          hipCheckError(hipMemPrefetchAsync(A_vals_, sizeof(T) * nnz_, gpuDevice_, s1_));
          hipCheckError(hipMemPrefetchAsync(x_, sizeof(T) * n_, gpuDevice_, s2_));
          hipCheckError(hipMemPrefetchAsync(y_, sizeof(T) * m_, gpuDevice_, s3_));
          hipCheckError(hipDeviceSynchronize());
          break;
        }
      }
    }

    void callSpMDnV() override {
      if (print_) std::cout << "callSpMDnV" << std::endl;
      switch (offload_) {
        case gpuOffloadType::always: {
          // Start by moving all the data over to the GPU
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
          hipCheckError(hipMemcpyAsync(x_device_, 
                                       x_, 
                                       sizeof(T) * n_, 
                                       hipMemcpyHostToDevice, 
                                       s2_));
          hipCheckError(hipMemcpyAsync(y_device_, 
                                       y_, 
                                       sizeof(T) * m_, 
                                       hipMemcpyHostToDevice, 
                                       s3_));
          hipCheckError(hipDeviceSynchronize());

          if (print_) std::cout << "\tCreating rocSPARSE structures" << std::endl;
          // Set up the rocSPARSE structures for the SpMDnV
          status_ = rocsparse_create_csr_descr(&description_,
                                               m_,
                                               n_,
                                               nnz_,
                                               A_rows_device_,
                                               A_cols_device_,
                                               A_vals_device_,
                                               index_,
                                               index_,
                                               base_,
                                               dataType_);
          checkStatus("Failed rocsparse_create_csr_descr");

          status_ = rocsparse_create_dnvec_descr(&x_description_,
                                                 n_,
                                                 x_device_,
                                                 dataType_);
          checkStatus("Failed rocsparse_create_dnvec_descr for x");

          status_ = rocsparse_create_dnvec_descr(&y_description_,
                                                 m_,
                                                 y_device_,
                                                 dataType_);
          checkStatus("Failed rocsparse_create_dnvec_descr for y");
          hipCheckError(hipDeviceSynchronize());

          size_t buffer_size = 0;
          status_ = rocsparse_spmv(handle_,
                                      operation_,
                                      &alpha,
                                      description_,
                                      x_description_,
                                      &beta,
                                      y_description_,
                                      dataType_,
                                      algorithm_,
                                      rocsparse_spmv_stage_buffer_size,
                                      &buffer_size,
                                      nullptr);
          checkStatus("Failed rocsparse_spmv_ex with rocsparse_spmv_stage_buffer_size");
          hipCheckError(hipDeviceSynchronize());
          
          void* temp_buffer;
          hipCheckError(hipMalloc(&temp_buffer, buffer_size));
          hipCheckError(hipDeviceSynchronize());

          status_ = rocsparse_spmv(handle_,
                                      operation_,
                                      &alpha,
                                      description_,
                                      x_description_,
                                      &beta,
                                      y_description_,
                                      dataType_,
                                      algorithm_,
                                      rocsparse_spmv_stage_preprocess,
                                      &buffer_size,
                                      temp_buffer);
          checkStatus("Failed rocsparse_spmv_ex with rocsparse_spmv_stage_preprocess");
          hipCheckError(hipDeviceSynchronize());

          status_ = rocsparse_spmv(handle_,
                                      operation_,
                                      &alpha,
                                      description_,
                                      x_description_,
                                      &beta,
                                      y_description_,
                                      dataType_,
                                      algorithm_,
                                      rocsparse_spmv_stage_compute,
                                      &buffer_size,
                                      temp_buffer);
          checkStatus("Failed rocsparse_spmv_ex with rocsparse_spmv_stage_compute");
          hipCheckError(hipDeviceSynchronize());

          if (print_) std::cout << "\tdestroying rocSPARSE structures" << std::endl;
          // Now clean up
          status_ = rocsparse_destroy_spmat_descr(description_);
          checkStatus("Failed rocsparse_destroy_spmat_descr");
          hipCheckError(hipFree(temp_buffer));

          // Move result back to the CPU
          if (print_) std::cout << "\tMoving data to CPU" << std::endl;
          hipCheckError(hipMemcpyAsync(y_, y_device_, sizeof(T) * m_, hipMemcpyDeviceToHost, s3_));
          hipCheckError(hipDeviceSynchronize());
          break;
        }
        case gpuOffloadType::once: {
          // Set up the rocSPARSE structures for the SpMDnV
          if (print_) std::cout << "\tCreating rocSPARSE structures" << std::endl;
          // Set up the rocSPARSE structures for the SpMDnV
          status_ = rocsparse_create_csr_descr(&description_,
                                               m_,
                                               n_,
                                               nnz_,
                                               A_rows_device_,
                                               A_cols_device_,
                                               A_vals_device_,
                                               index_,
                                               index_,
                                               base_,
                                               dataType_);
          checkStatus("Failed rocsparse_create_csr_descr");

          status_ = rocsparse_create_dnvec_descr(&x_description_,
                                                 n_,
                                                 x_device_,
                                                 dataType_);
          checkStatus("Failed rocsparse_create_dnvec_descr for x");

          status_ = rocsparse_create_dnvec_descr(&y_description_,
                                                 m_,
                                                 y_device_,
                                                 dataType_);
          checkStatus("Failed rocsparse_create_dnvec_descr for y");
          hipCheckError(hipDeviceSynchronize());

          size_t buffer_size = 0;
          status_ = rocsparse_spmv(handle_,
                                      operation_,
                                      &alpha,
                                      description_,
                                      x_description_,
                                      &beta,
                                      y_description_,
                                      dataType_,
                                      algorithm_,
                                      rocsparse_spmv_stage_buffer_size,
                                      &buffer_size,
                                      nullptr);
          checkStatus("Failed rocsparse_spmv_ex with rocsparse_spmv_stage_buffer_size");
          hipCheckError(hipDeviceSynchronize());
          
          void* temp_buffer;
          hipCheckError(hipMalloc(&temp_buffer, buffer_size));
          hipCheckError(hipDeviceSynchronize());

          status_ = rocsparse_spmv(handle_,
                                      operation_,
                                      &alpha,
                                      description_,
                                      x_description_,
                                      &beta,
                                      y_description_,
                                      dataType_,
                                      algorithm_,
                                      rocsparse_spmv_stage_preprocess,
                                      &buffer_size,
                                      temp_buffer);
          checkStatus("Failed rocsparse_spmv_ex with rocsparse_spmv_stage_preprocess");
          hipCheckError(hipDeviceSynchronize());

          status_ = rocsparse_spmv(handle_,
                                      operation_,
                                      &alpha,
                                      description_,
                                      x_description_,
                                      &beta,
                                      y_description_,
                                      dataType_,
                                      algorithm_,
                                      rocsparse_spmv_stage_compute,
                                      &buffer_size,
                                      temp_buffer);
          checkStatus("Failed rocsparse_spmv_ex with rocsparse_spmv_stage_compute");
          hipCheckError(hipDeviceSynchronize());

          if (print_) std::cout << "\tdestroying rocSPARSE structures" << std::endl;
          // Now clean up
          status_ = rocsparse_destroy_spmat_descr(description_);
          checkStatus("Failed rocsparse_destroy_spmat_descr");
          hipCheckError(hipFree(temp_buffer));
          break;
        }
        case gpuOffloadType::unified: {
          // Set up the rocSPARSE structures for the SpMDnV
          if (print_) std::cout << "\tCreating rocSPARSE structures" << std::endl;
          // Set up the rocSPARSE structures for the SpMDnV
          status_ = rocsparse_create_csr_descr(&description_,
                                               m_,
                                               n_,
                                               nnz_,
                                               A_rows_,
                                               A_cols_,
                                               A_vals_,
                                               index_,
                                               index_,
                                               base_,
                                               dataType_);
          checkStatus("Failed rocsparse_create_csr_descr");

          status_ = rocsparse_create_dnvec_descr(&x_description_,
                                                 n_,
                                                 x_,
                                                 dataType_);
          checkStatus("Failed rocsparse_create_dnvec_descr for x");

          status_ = rocsparse_create_dnvec_descr(&y_description_,
                                                 m_,
                                                 y_,
                                                 dataType_);
          checkStatus("Failed rocsparse_create_dnvec_descr for y");
          hipCheckError(hipDeviceSynchronize());

          size_t buffer_size = 0;
          status_ = rocsparse_spmv(handle_,
                                      operation_,
                                      &alpha,
                                      description_,
                                      x_description_,
                                      &beta,
                                      y_description_,
                                      dataType_,
                                      algorithm_,
                                      rocsparse_spmv_stage_buffer_size,
                                      &buffer_size,
                                      nullptr);
          checkStatus("Failed rocsparse_spmv_ex with rocsparse_spmv_stage_buffer_size");
          hipCheckError(hipDeviceSynchronize());
          
          void* temp_buffer;
          hipCheckError(hipMalloc(&temp_buffer, buffer_size));
          hipCheckError(hipDeviceSynchronize());

          status_ = rocsparse_spmv(handle_,
                                      operation_,
                                      &alpha,
                                      description_,
                                      x_description_,
                                      &beta,
                                      y_description_,
                                      dataType_,
                                      algorithm_,
                                      rocsparse_spmv_stage_preprocess,
                                      &buffer_size,
                                      temp_buffer);
          checkStatus("Failed rocsparse_spmv_ex with rocsparse_spmv_stage_preprocess");
          hipCheckError(hipDeviceSynchronize());

          status_ = rocsparse_spmv(handle_,
                                      operation_,
                                      &alpha,
                                      description_,
                                      x_description_,
                                      &beta,
                                      y_description_,
                                      dataType_,
                                      algorithm_,
                                      rocsparse_spmv_stage_compute,
                                      &buffer_size,
                                      temp_buffer);
          checkStatus("Failed rocsparse_spmv_ex with rocsparse_spmv_stage_compute");
          hipCheckError(hipDeviceSynchronize());

          if (print_) std::cout << "\tdestroying rocSPARSE structures" << std::endl;
          // Now clean up
          status_ = rocsparse_destroy_spmat_descr(description_);
          checkStatus("Failed rocsparse_destroy_spmat_descr");
          hipCheckError(hipFree(temp_buffer));
          break;
        }
      }
    }

    void postLoopRequirements() override {
      if (print_) std::cout << "Post loop " << std::endl;
      switch (offload_) {
        case gpuOffloadType::always: {
          break;
        }
        case gpuOffloadType::once: {
          // Move result back to the CPU
          if (print_) std::cout << "\tMovin data to CPU" << std::endl;
          hipCheckError(hipMemcpyAsync(y_, y_device_, sizeof(T) * m_, hipMemcpyDeviceToHost, s3_));
          hipCheckError(hipDeviceSynchronize());
          break;
        }
        case gpuOffloadType::unified: {
          // Ensure all output data resides on host once work has completed
          if (print_) std::cout << "\tMovin data to CPU" << std::endl;
          hipCheckError(hipMemPrefetchAsync(y_, sizeof(T) * m_, hipCpuDeviceId, s3_));
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
        hipCheckError(hipFree(x_));
        hipCheckError(hipFree(y_));
      } else {
        if (print_) std::cout << "\tFreeing CPU arrays" << std::endl;
        hipCheckError(hipHostFree((void*)A_));
        hipCheckError(hipHostFree((void*)A_rows_));
        hipCheckError(hipHostFree((void*)A_cols_));
        hipCheckError(hipHostFree((void*)A_vals_));
        hipCheckError(hipHostFree((void*)x_));
        hipCheckError(hipHostFree((void*)y_));

        if (print_) std::cout << "\tFreeing GPU arrays" << std::endl;
        hipCheckError(hipFree(A_rows_device_));
        hipCheckError(hipFree(A_cols_device_));
        hipCheckError(hipFree(A_vals_device_));
        hipCheckError(hipFree(x_device_));
        hipCheckError(hipFree(y_device_));
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

    rocsparse_status status_;
    rocsparse_operation operation_;
    rocsparse_handle handle_;
    rocsparse_indextype index_;
    rocsparse_matrix_type type_;
    rocsparse_index_base base_;
    rocsparse_datatype dataType_;
    rocsparse_spmv_alg algorithm_;

    rocsparse_spmat_descr description_;
    rocsparse_dnvec_descr x_description_;
    rocsparse_dnvec_descr y_description_;


    int64_t* A_rows_;
    int64_t* A_cols_;
    T* A_vals_;

    int64_t* A_rows_device_;
    int64_t* A_cols_device_;
    T* A_vals_device_;
    T* x_device_;
    T* y_device_;

    int gpuDevice_;
    hipStream_t s1_, s2_, s3_;

    const T alpha = ALPHA;
    const T beta = BETA;
};
}

#endif
