#pragma once

#ifdef GPU_ROCBLAS
#include <hip/hip_runtime_api.h>
#include <rocsparse/rocsparse.h>

#include "../include/kernels/GPU/spmv.hh"
#include "../include/utilities.hh"
#include "common.hh"

namespace gpu {
template <typename T>
class spmv_gpu : public spmv<T> {
  public:
  using spmv<T>::spmv;
  using spmv<T>::initInputMatrixVector;
  using spmv<T>::nnz_;
  using spmv<T>::m_;
  using spmv<T>::n_;
  using spmv<T>::x_;
  using spmv<T>::y_;
  using spmv<T>::offload_;
  using spmv<T>::sparsity_;
  using spmv<T>::type_;

  ~spmv_gpu() {
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

  void initialise(gpuOffloadType offload, int m, int n, 
                  double sparsity, matrixType type) override {
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

    if (std::is_same_v<T, float>) {
      dataType_ = rocsparse_datatype_f32_r;
    } else if (std::is_same_v<T, double>) {
      dataType_ = rocsparse_datatype_f64_r;
    } else {
      std::cerr << "Data type not supported" << std::endl;
      exit(1);
    }

    // SET METADATA HERE

    m_ = m;
    n_ = n;
    nnz_ = 1 + (uint64_t)((double)m_ * (double)n_ * (1.0 - sparsity_));

    operation_ = rocsparse_operation_none;
    algorithm_ = rocsparse_spmv_alg_default;
    index_ = rocsparse_indextype_i64;
    base_ = rocsparse_index_base_zero;
    if constexpr (std::is_same_v<T, float>) {
      dataType_ = rocsparse_datatype_f32_r;
    } else if constexpr (std::is_same_v<T, double>) {
      dataType_ = rocsparse_datatype_f64_r;
    } else {
      std::cerr << "INVALID DATA TYPE PASSED TO rocSPARSE" << std::endl;
      exit(1);
    }

    if (offload_ == gpuOffloadType::unified) {
      hipCheckError(hipMallocManaged(&x_, n_ * sizeof(T)));
      hipCheckError(hipMallocManaged(&y_, m_ * sizeof(T)));
      hipCheckError(hipDeviceSynchronize());
    } else {
      x_ = (T*)malloc(n_ * sizeof(T));
      y_ = (T*)malloc(m_ * sizeof(T));

      hipCheckError(hipMalloc((void**)&x_dev_, n_ * sizeof(T)));
      hipCheckError(hipMalloc((void**)&y_dev_, m_ * sizeof(T)));
      hipCheckError(hipDeviceSynchronize());
    }

    initInputMatrixVector();
  }


protected:
  void toSparseFormat() override {
    if (offload_ == gpuOffloadType::always) {
      A_vals_store_ = (T*)malloc(sizeof(T) * nnz_);
      A_cols_store_ = (int64_t*)malloc(sizeof(int64_t) * nnz_);
      A_rows_store_ = (int64_t*)malloc(sizeof(int64_t) * (m_ + 1));

      if (type_ == matrixType::random) {
        randomCSR<T, int64_t>(A_vals_store_, A_cols_store_, A_rows_store_, m_, n_, nnz_);
      } else if (type_ == matrixType::rmat) {
        rMatCSR<T, int64_t>(A_vals_store_, A_cols_store_, A_rows_store_, m_, n_, nnz_);
      } else if (type_ == matrixType::bandedDiagonal) {
        bandedDiagonalCSR<T, int64_t>(A_vals_store_, A_cols_store_, A_rows_store_, m_, n_, nnz_);
      } else {
        std::cerr << "Matrix type not supported" << std::endl;
        exit(1);
      }
    }

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
  /**
    * Before we enter the loop of calling the kernel, 
    * we need to move any data we may need.
    */
  void preLoopRequirements() override {
    switch(offload_) {
      case gpuOffloadType::always: {
        break;
      }
      case gpuOffloadType::once: {
        hipCheckError(hipMemcpyAsync(A_vals_dev_, A_vals_, nnz_ * sizeof(T), hipMemcpyHostToDevice, s1_));
        hipCheckError(hipMemcpyAsync(A_cols_dev_, A_cols_, nnz_ * sizeof(int64_t), hipMemcpyHostToDevice, s2_));
        hipCheckError(hipMemcpyAsync(A_rows_dev_, A_rows_, (m_ + 1) * sizeof(int64_t), hipMemcpyHostToDevice, s3_));
        hipCheckError(hipMemcpyAsync(x_dev_, x_, n_ * sizeof(T), hipMemcpyHostToDevice, s4_));
        hipCheckError(hipMemcpyAsync(y_dev_, y_, m_ * sizeof(T), hipMemcpyHostToDevice, s5_));
        hipCheckError(hipDeviceSynchronize());
        break;
      }
      case gpuOffloadType::unified: {
        hipCheckError(hipMemPrefetchAsync(A_vals_, nnz_ * sizeof(T), gpuDevice_, s1_));
        hipCheckError(hipMemPrefetchAsync(A_cols_, nnz_ * sizeof(int64_t), gpuDevice_, s2_));
        hipCheckError(hipMemPrefetchAsync(A_rows_, (m_ + 1) * sizeof(int64_t), gpuDevice_, s3_));
        hipCheckError(hipMemPrefetchAsync(x_, n_ * sizeof(T), gpuDevice_, s4_));
        hipCheckError(hipMemPrefetchAsync(y_, m_ * sizeof(T), gpuDevice_, s5_));
        hipCheckError(hipDeviceSynchronize());
        break;
      }
    }

  }

  void callSpmv() override {
    rocCheckError(rocsparse_create_spmv_descr(&spmv_descr_));
    rocCheckError(rocsparse_spmv_set_input(handle_,
                                           spmv_descr_,
                                           rocsparse_spmv_input_operation,
                                           &operation_,
                                           sizeof(operation_),
                                           p_error_));
    rocCheckError(rocsparse_spmv_set_input(handle_,
                                           spmv_descr_,
                                           rocsparse_spmv_input_alg,
                                           &algorithm_,
                                           sizeof(algorithm_),
                                           p_error_));

    rocCheckError(rocsparse_spmv_set_input(handle_,
                                           spmv_descr_,
                                           rocsparse_spmv_input_scalar_datatype,
                                           &dataType_,
                                           sizeof(dataType_),
                                           p_error_));

    rocCheckError(rocsparse_spmv_set_input(handle_,
                                           spmv_descr_,
                                           rocsparse_spmv_input_compute_datatype,
                                           &dataType_,
                                           sizeof(dataType_),
                                           p_error_));
    switch(offload_) {
      case gpuOffloadType::always: {
        hipCheckError(hipMemcpyAsync(A_vals_dev_, A_vals_, nnz_ * sizeof(T), hipMemcpyHostToDevice, s1_));
        hipCheckError(hipMemcpyAsync(A_cols_dev_, A_cols_, nnz_ * sizeof(int64_t), hipMemcpyHostToDevice, s2_));
        hipCheckError(hipMemcpyAsync(A_rows_dev_, A_rows_, (m_ + 1) * sizeof(int64_t), hipMemcpyHostToDevice, s3_));
        hipCheckError(hipMemcpyAsync(x_dev_, x_, n_ * sizeof(T), hipMemcpyHostToDevice, s4_));
        hipCheckError(hipMemcpyAsync(y_dev_, y_, m_ * sizeof(T), hipMemcpyHostToDevice, s5_));
        
        hipCheckError(hipDeviceSynchronize());
        rocCheckError(rocsparse_create_csr_descr(&A_descr_, m_, n_, nnz_, A_rows_dev_, A_cols_dev_, A_vals_dev_, index_, index_, base_, dataType_));
        rocCheckError(rocsparse_create_dnvec_descr(&x_descr_, n_, x_dev_, dataType_));
        rocCheckError(rocsparse_create_dnvec_descr(&y_descr_, m_, y_dev_, dataType_));
        hipCheckError(hipDeviceSynchronize());

        size_t buffer_size = 0;
        rocCheckError(rocsparse_v2_spmv_buffer_size(handle_, 
                                                    spmv_descr_, 
                                                    A_descr_, 
                                                    x_descr_, 
                                                    y_descr_, 
                                                    rocsparse_v2_spmv_stage_analysis, 
                                                    &buffer_size, 
                                                    p_error_));
        
        void* temp_buffer = nullptr;
        hipCheckError(hipMalloc(&temp_buffer, buffer_size));

        rocCheckError(rocsparse_v2_spmv(handle_, 
                                        spmv_descr_, 
                                        &alpha, 
                                        A_descr_, 
                                        x_descr_, 
                                        &beta, 
                                        y_descr_, 
                                        rocsparse_v2_spmv_stage_analysis, 
                                        buffer_size, 
                                        temp_buffer,
                                        p_error_));

        hipCheckError(hipFree(temp_buffer));

        rocCheckError(rocsparse_v2_spmv_buffer_size(handle_, 
                                                    spmv_descr_, 
                                                    A_descr_, 
                                                    x_descr_, 
                                                    y_descr_, 
                                                    rocsparse_v2_spmv_stage_compute, 
                                                    &buffer_size, 
                                                    p_error_));
        
        hipCheckError(hipMalloc(&temp_buffer, buffer_size));

        rocCheckError(rocsparse_v2_spmv(handle_, 
                                        spmv_descr_, 
                                        &alpha, 
                                        A_descr_, 
                                        x_descr_, 
                                        &beta, 
                                        y_descr_, 
                                        rocsparse_v2_spmv_stage_compute, 
                                        buffer_size, 
                                        temp_buffer,
                                        p_error_));

        hipCheckError(hipFree(temp_buffer));

        
        hipCheckError(hipDeviceSynchronize());

        hipCheckError(hipMemcpyAsync(y_, y_dev_, m_ * sizeof(T), hipMemcpyDeviceToHost, s2_));

        rocCheckError(rocsparse_destroy_spmat_descr(A_descr_));
        rocCheckError(rocsparse_destroy_dnvec_descr(x_descr_));
        rocCheckError(rocsparse_destroy_dnvec_descr(y_descr_));
        
        hipCheckError(hipDeviceSynchronize());
        break;
      }
      case gpuOffloadType::once: {
        rocCheckError(rocsparse_create_csr_descr(&A_descr_, m_, n_, nnz_, A_rows_dev_, A_cols_dev_, A_vals_dev_, index_, index_, base_, dataType_));
        rocCheckError(rocsparse_create_dnvec_descr(&x_descr_, n_, x_dev_, dataType_));
        rocCheckError(rocsparse_create_dnvec_descr(&y_descr_, m_, y_dev_, dataType_));
        hipCheckError(hipDeviceSynchronize());


        size_t buffer_size;
        rocCheckError(rocsparse_v2_spmv_buffer_size(handle_, 
                                                    spmv_descr_, 
                                                    A_descr_, 
                                                    x_descr_, 
                                                    y_descr_, 
                                                    rocsparse_v2_spmv_stage_analysis, 
                                                    &buffer_size, 
                                                    p_error_));
        
        void* temp_buffer = nullptr;
        hipCheckError(hipMalloc(&temp_buffer, buffer_size));

        rocCheckError(rocsparse_v2_spmv(handle_, 
                                        spmv_descr_, 
                                        &alpha, 
                                        A_descr_, 
                                        x_descr_, 
                                        &beta, 
                                        y_descr_, 
                                        rocsparse_v2_spmv_stage_analysis, 
                                        buffer_size, 
                                        temp_buffer,
                                        p_error_));

        hipCheckError(hipFree(temp_buffer));

        rocCheckError(rocsparse_v2_spmv_buffer_size(handle_, 
                                                    spmv_descr_, 
                                                    A_descr_, 
                                                    x_descr_, 
                                                    y_descr_, 
                                                    rocsparse_v2_spmv_stage_compute, 
                                                    &buffer_size, 
                                                    p_error_));
        
        hipCheckError(hipMalloc(&temp_buffer, buffer_size));

        rocCheckError(rocsparse_v2_spmv(handle_, 
                                        spmv_descr_, 
                                        &alpha, 
                                        A_descr_, 
                                        x_descr_, 
                                        &beta, 
                                        y_descr_, 
                                        rocsparse_v2_spmv_stage_compute, 
                                        buffer_size, 
                                        temp_buffer,
                                        p_error_));

        hipCheckError(hipFree(temp_buffer));

        hipCheckError(hipDeviceSynchronize());

        rocCheckError(rocsparse_destroy_spmat_descr(A_descr_));
        rocCheckError(rocsparse_destroy_dnvec_descr(x_descr_));
        rocCheckError(rocsparse_destroy_dnvec_descr(y_descr_));
        
        if (temp_buffer != nullptr) {
          hipCheckError(hipFree(temp_buffer));
        }
        hipCheckError(hipDeviceSynchronize());
        break;
      }
      case gpuOffloadType::unified: {
        rocCheckError(rocsparse_create_csr_descr(&A_descr_, m_, n_, nnz_, A_rows_, A_cols_, A_vals_, index_, index_, base_, dataType_));
        rocCheckError(rocsparse_create_dnvec_descr(&x_descr_, n_, x_, dataType_));
        rocCheckError(rocsparse_create_dnvec_descr(&y_descr_, m_, y_, dataType_));
        hipCheckError(hipDeviceSynchronize());

        size_t buffer_size;
        rocCheckError(rocsparse_v2_spmv_buffer_size(handle_, 
                                                    spmv_descr_, 
                                                    A_descr_, 
                                                    x_descr_, 
                                                    y_descr_, 
                                                    rocsparse_v2_spmv_stage_analysis, 
                                                    &buffer_size, 
                                                    p_error_));
        
        void* temp_buffer = nullptr;
        hipCheckError(hipMalloc(&temp_buffer, buffer_size));

        rocCheckError(rocsparse_v2_spmv(handle_, 
                                        spmv_descr_, 
                                        &alpha, 
                                        A_descr_, 
                                        x_descr_, 
                                        &beta, 
                                        y_descr_, 
                                        rocsparse_v2_spmv_stage_analysis, 
                                        buffer_size, 
                                        temp_buffer,
                                        p_error_));

        hipCheckError(hipFree(temp_buffer));

        rocCheckError(rocsparse_v2_spmv_buffer_size(handle_, 
                                                    spmv_descr_, 
                                                    A_descr_, 
                                                    x_descr_, 
                                                    y_descr_, 
                                                    rocsparse_v2_spmv_stage_compute, 
                                                    &buffer_size, 
                                                    p_error_));
        
        hipCheckError(hipMalloc(&temp_buffer, buffer_size));

        rocCheckError(rocsparse_v2_spmv(handle_, 
                                        spmv_descr_, 
                                        &alpha, 
                                        A_descr_, 
                                        x_descr_, 
                                        &beta, 
                                        y_descr_, 
                                        rocsparse_v2_spmv_stage_compute, 
                                        buffer_size, 
                                        temp_buffer,
                                        p_error_));

        hipCheckError(hipFree(temp_buffer));

        hipCheckError(hipDeviceSynchronize());

        rocCheckError(rocsparse_destroy_spmat_descr(A_descr_));
        rocCheckError(rocsparse_destroy_dnvec_descr(x_descr_));
        rocCheckError(rocsparse_destroy_dnvec_descr(y_descr_));
        
        if (temp_buffer != nullptr) {
          hipCheckError(hipFree(temp_buffer));
        }
        hipCheckError(hipDeviceSynchronize());
        break;
      }
    }

    rocCheckError(rocsparse_destroy_spmv_descr(spmv_descr_));
  }

  void postLoopRequirements() override {
    rocCheckError(rocsparse_destroy_error(p_error_[0]));
    switch (offload_) {
      case gpuOffloadType::always: {
        break;
      }
      case gpuOffloadType::once: {
        hipCheckError(hipMemcpyAsync(y_, y_dev_, m_ * sizeof(T), hipMemcpyDeviceToHost, s3_));
        break;
      }
      case gpuOffloadType::unified: {
        hipCheckError(hipMemPrefetchAsync(y_, m_ * sizeof(T), hipCpuDeviceId, s3_));
        break;
      }
    }
    hipCheckError(hipDeviceSynchronize());
  }

  void postCallKernelCleanup() override {
    if (offload_ == gpuOffloadType::unified) {
      hipCheckError(hipFree(A_vals_));
      hipCheckError(hipFree(A_cols_));
      hipCheckError(hipFree(A_rows_));
      hipCheckError(hipFree(x_));
      hipCheckError(hipFree(y_));
      free(A_vals_store_);
      free(A_cols_store_);
      free(A_rows_store_);
    } else {
      free(A_vals_);
      free(A_cols_);
      free(A_rows_);
      free(x_);
      free(y_);
      hipCheckError(hipFree(A_vals_dev_));
      hipCheckError(hipFree(A_cols_dev_));
      hipCheckError(hipFree(A_rows_dev_));
      hipCheckError(hipFree(x_dev_));
      hipCheckError(hipFree(y_dev_));
    }
  }

  bool initialised_ = false;

  rocsparse_status status_;
  rocsparse_operation operation_;
  rocsparse_handle handle_;
  rocsparse_indextype index_;
  rocsparse_index_base base_;
  rocsparse_datatype dataType_;
  rocsparse_spmv_alg algorithm_;

  rocsparse_spmat_descr A_descr_;
  rocsparse_dnvec_descr x_descr_;
  rocsparse_dnvec_descr y_descr_;

  rocsparse_spmv_descr spmv_descr_;
  rocsparse_error p_error_[1] = {};

  int gpuDevice_;
  hipStream_t s1_, s2_, s3_, s4_, s5_;

  const T alpha = ALPHA;
  const T beta = BETA;


  /**
   * ################################
   *        Matrix A parameters
   * ################################
   */
  /** CSR format vectors for storage of matrix between offload type runs */
  T* A_vals_store_;
  int64_t* A_cols_store_;
  int64_t* A_rows_store_;

	/** CSR format vectors on the host (also used for USM) */
	T* A_vals_;
	int64_t* A_cols_;
  int64_t* A_rows_;
  /** CSR format vectors on the device. */
	T* A_vals_dev_;
	int64_t* A_cols_dev_;
	int64_t* A_rows_dev_; 


  /**
   * ################################
   *    Vectors x and y parameters
   * ################################
   */
  /** Vectors on the host (also used for USM) */
  T* x_host_;
  T* y_host_;
  /** Vectors on the device */
  T* x_dev_;
  T* y_dev_;
};
}

#endif
