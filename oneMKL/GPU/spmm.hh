#pragma once

#ifdef GPU_ONEMKL

#include <memory>
#include "../../include/kernels/GPU/spmm.hh"
#include "../../include/utilities.hh"
#include "common.hh"

namespace gpu {
template <typename T>
class spmm_gpu : public spmm<T> {
public:
    using spmm<T>::spmm;
    using spmm<T>::initInputMatrices;
    using spmm<T>::A_nnz_;
    using spmm<T>::B_nnz_;
    using spmm<T>::C_nnz_;
    using spmm<T>::m_;
    using spmm<T>::n_;
    using spmm<T>::k_;
    using spmm<T>::C_rows_;
    using spmm<T>::C_cols_;
    using spmm<T>::C_vals_;
    using spmm<T>::offload_;
    using spmm<T>::sparsity_;
    using spmm<T>::type_;

    void initialise(gpuOffloadType offload, int m, int n, int k,
                    double sparsity, matrixType type, 
                    bool binary = false) override {

      if (print_) {
        std::cout << "Initialising ";
        switch (offload) {
          case gpuOffloadType::always:
            std::cout << "ALWAYS" << std::endl;
            break;
          case gpuOffloadType::once:
            std::cout << "ONCE" << std::endl;
            break;
          case gpuOffloadType::unified:
            std::cout << "UNIFIED" << std::endl;
            break;
        }
      }   
      m_ = m;
      n_ = n;
      k_ = k;
      sparsity_ = sparsity;
      offload_ = offload;
      type_ = type;


      // Set up the sycl device for the GPU
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

      A_nnz_ = 1 + (uint64_t)((double)m_ * (double)k_ * (1.0 - sparsity_));
      B_nnz_ = 1 + (uint64_t)((double)k_ * (double)n_ * (1.0 - sparsity_));
      C_rows_ = nullptr;
      C_cols_ = nullptr;
      C_vals_ = nullptr;
      C_nnz_sycl_ = nullptr;

      // Setting MKL metadata
      // Todo 

      if (print_) std::cout << "\tMallocing" << std::endl;
      if (offload_ == gpuOffloadType::unified) {
        A_vals_ = sycl::malloc_shared<T>(A_nnz_, gpuQueue_);
        A_cols_ = sycl::malloc_shared<int64_t>(A_nnz_, gpuQueue_);
        A_rows_ = sycl::malloc_shared<int64_t>(m_ + 1, gpuQueue_);

        B_vals_ = sycl::malloc_shared<T>(B_nnz_, gpuQueue_);
        B_cols_ = sycl::malloc_shared<int64_t>(B_nnz_, gpuQueue_);
        B_rows_ = sycl::malloc_shared<int64_t>(k_ + 1, gpuQueue_);

        gpuQueue_.wait();
      } else {
        A_vals_ = sycl::malloc_host<T>(A_nnz_, gpuQueue_);
        A_cols_ = sycl::malloc_host<int64_t>(A_nnz_, gpuQueue_);
        A_rows_ = sycl::malloc_host<int64_t>(m_ + 1, gpuQueue_);

        B_vals_ = sycl::malloc_host<T>(B_nnz_, gpuQueue_);
        B_cols_ = sycl::malloc_host<int64_t>(B_nnz_, gpuQueue_);
        B_rows_ = sycl::malloc_host<int64_t>(k_ + 1, gpuQueue_);

        A_vals_device_ = sycl::malloc_device<T>(A_nnz_, gpuQueue_);
        A_cols_device_ = sycl::malloc_device<int64_t>(A_nnz_, gpuQueue_);
        A_rows_device_ = sycl::malloc_device<int64_t>(m_ + 1, gpuQueue_);

        B_vals_device_ = sycl::malloc_device<T>(B_nnz_, gpuQueue_);
        B_cols_device_ = sycl::malloc_device<int64_t>(B_nnz_, gpuQueue_);
        B_rows_device_ = sycl::malloc_device<int64_t>(k_ + 1, gpuQueue_);
        gpuQueue_.wait();
      }
    
      if (print_) std::cout << "\tInitialising matrices" << std::endl;
      initInputMatrices();
    }

protected:
    void toSparseFormat() override {
      int seedOffset = 0;
      if (type_ == matrixType::rmat) {
        rMatCSR<T, int64_t>(A_vals_, A_cols_, A_rows_, m_, k_, A_nnz_, SEED + seedOffset++);
        rMatCSR<T, int64_t>(B_vals_, B_cols_, B_rows_, k_, n_, B_nnz_, SEED + seedOffset++);
      } else if (type_ == matrixType::random) {
        randomCSR<T, int64_t>(A_vals_, A_cols_, A_rows_, m_, k_, A_nnz_, SEED + seedOffset++);
        randomCSR<T, int64_t>(B_vals_, B_cols_, B_rows_, k_, n_, B_nnz_, SEED + seedOffset++);
      } else {
        std::cerr << "ERROR - Unrecognized matrix type" << std::endl;
        exit(1);
      }
      while (calcCNNZ<int64_t>(m_, A_nnz_, A_rows_, A_cols_, k_, B_nnz_, B_rows_, B_cols_) == 0) {
        if (type_ == matrixType::rmat) {
          rMatCSR<T, int64_t>(A_vals_, A_cols_, A_rows_, m_, k_, A_nnz_, SEED + seedOffset++);
          rMatCSR<T, int64_t>(B_vals_, B_cols_, B_rows_, k_, n_, B_nnz_, SEED + seedOffset++);
        } else if (type_ == matrixType::random) {
          randomCSR<T, int64_t>(A_vals_, A_cols_, A_rows_, m_, k_, A_nnz_, SEED + seedOffset++);
          randomCSR<T, int64_t>(B_vals_, B_cols_, B_rows_, k_, n_, B_nnz_, SEED + seedOffset++);
        } else {
          std::cerr << "Matrix type not supported" << std::endl;
          exit(1);
        }
      }

      // Ensure synchronization for unified memory
      gpuQueue_.wait();
    }

private:
    void preLoopRequirements() override {
      if (print_) std::cout << "pre-loop stuff" << std::endl;
      switch (offload_) {
        case gpuOffloadType::always: 
          break;
        case gpuOffloadType::once: {
          // Copy A and B over to the GPU
          if (print_) std::cout << "\tCopying data over to GPU" << std::endl;
          gpuQueue_.copy<T>(A_vals_, A_vals_device_, A_nnz_);
          gpuQueue_.copy<int64_t>(A_cols_, A_cols_device_, A_nnz_);
          gpuQueue_.copy<int64_t>(A_rows_, A_rows_device_, m_ + 1);

          gpuQueue_.copy<T>(B_vals_, B_vals_device_, B_nnz_);
          gpuQueue_.copy<int64_t>(B_cols_, B_cols_device_, B_nnz_);
          gpuQueue_.copy<int64_t>(B_rows_, B_rows_device_, k_ + 1);
          gpuQueue_.wait();
          break;
        }
        case gpuOffloadType::unified:
          // Data doesn't need to be moved around, so nothing to do here
          break;
      }
    }

    void callSpmm() override {
      if (print_) std::cout << "callSpmm" << std::endl;
      switch (offload_) {
        case gpuOffloadType::always: {
          // Copy A and B over to the GPU
          if (print_) std::cout << "\tCopying data over to the GPU" << std::endl;
          gpuQueue_.copy<T>(A_vals_, A_vals_device_, A_nnz_);
          gpuQueue_.copy<int64_t>(A_cols_, A_cols_device_, A_nnz_);
          gpuQueue_.copy<int64_t>(A_rows_, A_rows_device_, m_ + 1);

          gpuQueue_.copy<T>(B_vals_, B_vals_device_, B_nnz_);
          gpuQueue_.copy<int64_t>(B_cols_, B_cols_device_, B_nnz_);
          gpuQueue_.copy<int64_t>(B_rows_, B_rows_device_, k_ + 1);
          gpuQueue_.wait();

          if (print_) std::cout << "\tMallocing C rows, and setting up handles etc." << std::endl;
          if (C_rows_ != nullptr) sycl::free(C_rows_, gpuQueue_);
          if (C_cols_ != nullptr) sycl::free(C_cols_, gpuQueue_);
          if (C_vals_ != nullptr) sycl::free(C_vals_, gpuQueue_);
          C_rows_device_ = sycl::malloc_device<int64_t>(m_ + 1, gpuQueue_);
          gpuQueue_.wait();

          oneapi::mkl::sparse::init_matrix_handle(&A_handle_);
          oneapi::mkl::sparse::init_matrix_handle(&B_handle_);
          oneapi::mkl::sparse::init_matrix_handle(&C_handle_);

          auto setA = oneapi::mkl::sparse::set_csr_data(gpuQueue_, 
                                                        A_handle_,
                                                        m_,
                                                        k_,
                                                        base_, 
                                                        A_rows_device_,
                                                        A_cols_device_,
                                                        A_vals_device_);
          auto setB = oneapi::mkl::sparse::set_csr_data(gpuQueue_, 
                                                        B_handle_,
                                                        k_,
                                                        n_,
                                                        base_, 
                                                        B_rows_device_,
                                                        B_cols_device_,
                                                        B_vals_device_);
          auto setC = oneapi::mkl::sparse::set_csr_data(gpuQueue_, 
                                                        C_handle_,
                                                        m_,
                                                        n_,
                                                        base_, 
                                                        C_rows_device_,
                                                        (int64_t*)nullptr,
                                                        (T*)nullptr);

          oneapi::mkl::sparse::init_matmat_descr(&description_);
          oneapi::mkl::sparse::set_matmat_data(description_,
                                               view_, 
                                               operationA_,
                                               view_,
                                               operationB_,
                                               view_);
          
          if (print_) std::cout << "\tGetting buffer size " << std::endl;
          // Query the size of the work-estimation temporary buffer
          request_ = oneapi::mkl::sparse::matmat_request::get_work_estimation_buf_size;
          sizeTempBuffer = sycl::malloc_host<int64_t>(1, gpuQueue_);
          auto matmat1 = oneapi::mkl::sparse::matmat(gpuQueue_,
                                                     A_handle_,
                                                     B_handle_,
                                                     C_handle_,
                                                     request_,
                                                     description_,
                                                     sizeTempBuffer,
                                                     nullptr,
                                                     {setA, setB, setC});
          matmat1.wait();
          
          // Allocate the temporary buffer for work-estimation
          tempBuffer = sycl::malloc_device<uint8_t>(sizeTempBuffer[0], gpuQueue_);
          gpuQueue_.wait();

          if (print_) std::cout << "\tWork estimation" << std::endl;
          // Do work-estimation
          request_ = oneapi::mkl::sparse::matmat_request::work_estimation;
          auto matmat2 = oneapi::mkl::sparse::matmat(gpuQueue_,
                                                     A_handle_,
                                                     B_handle_,
                                                     C_handle_,
                                                     request_,
                                                     description_,
                                                     sizeTempBuffer,
                                                     tempBuffer,
                                                     {matmat1});
          matmat2.wait();

          if (print_) std::cout << "\tGetting buffer size" << std::endl;
          // Query the size of the compute temporary buffer
          request_ = oneapi::mkl::sparse::matmat_request::get_compute_buf_size;
          sizeTempBuffer2 = sycl::malloc_host<int64_t>(1, gpuQueue_);
          auto matmat3 = oneapi::mkl::sparse::matmat(gpuQueue_,
                                                     A_handle_,
                                                     B_handle_,
                                                     C_handle_,
                                                     request_,
                                                     description_,
                                                     sizeTempBuffer2,
                                                     nullptr,
                                                     {matmat2});
          matmat3.wait();

          // Allocate the temporary buffer for compute
          tempBuffer2 = sycl::malloc_device<uint8_t>(sizeTempBuffer2[0], gpuQueue_);
          gpuQueue_.wait();

          if (print_) std::cout << "\tComputing " << std::endl;
          // Do compute
          request_ = oneapi::mkl::sparse::matmat_request::compute;
          auto matmat4 = oneapi::mkl::sparse::matmat(gpuQueue_,
                                                     A_handle_,
                                                     B_handle_,
                                                     C_handle_,
                                                     request_,
                                                     description_,
                                                     sizeTempBuffer2,
                                                     tempBuffer2,
                                                     {matmat3});

          if (print_) std::cout << "\tGetting nnz for C" << std::endl;
          // get NNZ for C
          C_nnz_sycl_ = sycl::malloc_host<int64_t>(1, gpuQueue_);
          request_ = oneapi::mkl::sparse::matmat_request::get_nnz;
          auto matmat5 = oneapi::mkl::sparse::matmat(gpuQueue_,
                                                     A_handle_,
                                                     B_handle_,
                                                     C_handle_,
                                                     request_,
                                                     description_,
                                                     C_nnz_sycl_,
                                                     nullptr,
                                                     {matmat4});
          matmat5.wait();

          // Make sure that C_nnz_ is non-zero.  If it isn't then clean up and return -- no computation to be done
          if (print_) std::cout << "\t\tC_nnz_ = " << *C_nnz_sycl_ << std::endl;
          if (*C_nnz_sycl_ == 0) {
            oneapi::mkl::sparse::release_matmat_descr(&description_);
            oneapi::mkl::sparse::release_matrix_handle(gpuQueue_, &A_handle_);
            oneapi::mkl::sparse::release_matrix_handle(gpuQueue_, &B_handle_);
            oneapi::mkl::sparse::release_matrix_handle(gpuQueue_, &C_handle_);
            gpuQueue_.wait();

            if (sizeTempBuffer != nullptr) sycl::free(sizeTempBuffer, gpuQueue_);
            if (tempBuffer != nullptr) sycl::free(tempBuffer, gpuQueue_);
            if (sizeTempBuffer2 != nullptr) sycl::free(sizeTempBuffer2, gpuQueue_);
            if (tempBuffer2 != nullptr) sycl::free(tempBuffer2, gpuQueue_);
            if (C_rows_device_ != nullptr) sycl::free(C_rows_device_, gpuQueue_);
            if (C_nnz_sycl_ != nullptr) sycl::free(C_nnz_sycl_, gpuQueue_);
            gpuQueue_.wait();
            return;
          }
          
          if (print_) std::cout << "\tAllocating C structures" << std::endl;
          C_cols_device_ = sycl::malloc_device<int64_t>(*C_nnz_sycl_, gpuQueue_);
          C_vals_device_ = sycl::malloc_device<T>(*C_nnz_sycl_, gpuQueue_);
          gpuQueue_.wait();
          setC = oneapi::mkl::sparse::set_csr_data(gpuQueue_, 
                                                   C_handle_,
                                                   m_,
                                                   n_,
                                                   base_, 
                                                   C_rows_device_, 
                                                   C_cols_device_, 
                                                   C_vals_device_,
                                                   {matmat5});

          if (print_) std::cout << "\tFinalising" << std::endl;
          // Finalise into C matrix
          request_ = oneapi::mkl::sparse::matmat_request::finalize;
          auto matmat6 = oneapi::mkl::sparse::matmat(gpuQueue_,
                                                     A_handle_,
                                                     B_handle_,
                                                     C_handle_,
                                                     request_,
                                                     description_,
                                                     nullptr,
                                                     nullptr,
                                                     {setC});
          
                                  
          if (print_) std::cout << "\tSorting output matrix" << std::endl;
          auto sort = oneapi::mkl::sparse::sort_matrix(gpuQueue_,
                                                       C_handle_,
                                                       {matmat6});
          gpuQueue_.wait_and_throw();
          if (print_) std::cout << "\tCopying data back to CPU" << std::endl;
          // Copy data back to host
          if (print_) std::cout << "\t\tMoving C nnz over --- NNZ for C = " << *C_nnz_sycl_ << std::endl;
          C_nnz_ = *C_nnz_sycl_;
          if (print_) std::cout << "\t\tMallocing C_rows_" << std::endl;
          C_rows_ = sycl::malloc_host<int64_t>(m_ + 1, gpuQueue_);
          if (C_rows_ == nullptr) std::cerr << "malloc_host returned nullptr" << std::endl;
          if (print_) std::cout << "\t\tMallocing C_cols_ at " << __LINE__ << std::endl;
          C_cols_ = sycl::malloc_host<int64_t>(C_nnz_, gpuQueue_);
          if (C_cols_ == nullptr) std::cerr << "malloc_host returned nullptr" << std::endl;
          if (print_) std::cout << "\t\tMallocing C_vals_" << std::endl;
          C_vals_ = sycl::malloc_host<T>(C_nnz_, gpuQueue_);
          if (print_) std::cout << "\t\tCopying rows over" << std::endl;
          gpuQueue_.copy<int64_t>(C_rows_device_, C_rows_, m_ + 1);
          if (print_) std::cout << "\t\tCopying cols over" << std::endl;
          gpuQueue_.copy<int64_t>(C_cols_device_, C_cols_, C_nnz_);
          if (print_) std::cout << "\t\tCopying vals over" << std::endl;
          gpuQueue_.copy<T>(C_vals_device_, C_vals_, C_nnz_);
          gpuQueue_.wait();

          if (print_) std::cout << "\tCleaning up temp allocations" << std::endl;
          // Now clean everything up ready for the next spmm call
          oneapi::mkl::sparse::release_matmat_descr(&description_);          
          oneapi::mkl::sparse::release_matrix_handle(gpuQueue_, &A_handle_);
          oneapi::mkl::sparse::release_matrix_handle(gpuQueue_, &B_handle_);
          oneapi::mkl::sparse::release_matrix_handle(gpuQueue_, &C_handle_);
          gpuQueue_.wait();

          if (sizeTempBuffer != nullptr) sycl::free(sizeTempBuffer, gpuQueue_);
          if (tempBuffer != nullptr) sycl::free(tempBuffer, gpuQueue_);
          if (sizeTempBuffer2 != nullptr) sycl::free(sizeTempBuffer2, gpuQueue_);
          if (tempBuffer2 != nullptr) sycl::free(tempBuffer2, gpuQueue_);
          if (C_rows_device_ != nullptr) sycl::free(C_rows_device_, gpuQueue_); 
          if (C_cols_device_ != nullptr) sycl::free(C_cols_device_, gpuQueue_);
          if (C_vals_device_ != nullptr) sycl::free(C_vals_device_, gpuQueue_);
          if (C_nnz_sycl_ != nullptr) sycl::free(C_nnz_sycl_, gpuQueue_);
          gpuQueue_.wait();
          break;
        }
        case gpuOffloadType::once: {
          if (print_) std::cout << "\tFreeing old memory allocations, if present" << std::endl;
          // Check to see if C device arrays already exist.  If they do, get rid
          if (C_rows_device_ != nullptr) sycl::free(C_rows_device_, gpuQueue_); 
          if (C_cols_device_ != nullptr) sycl::free(C_cols_device_, gpuQueue_);
          if (C_vals_device_ != nullptr) sycl::free(C_vals_device_, gpuQueue_);
          if (C_nnz_sycl_ != nullptr) sycl::free(C_nnz_sycl_, gpuQueue_);

          // Allocate space for the C rows array
          C_rows_device_ = sycl::malloc_device<int64_t>(m_ + 1, gpuQueue_);
          gpuQueue_.wait();

          if (print_) std::cout << "\tSetting up MKL structures" << std::endl;
          // Set up MKL structures
          oneapi::mkl::sparse::init_matrix_handle(&A_handle_);
          oneapi::mkl::sparse::init_matrix_handle(&B_handle_);
          oneapi::mkl::sparse::init_matrix_handle(&C_handle_);

          auto setA = oneapi::mkl::sparse::set_csr_data(gpuQueue_, 
                                                        A_handle_,
                                                        m_,
                                                        k_,
                                                        base_, 
                                                        A_rows_device_,
                                                        A_cols_device_,
                                                        A_vals_device_);
          auto setB = oneapi::mkl::sparse::set_csr_data(gpuQueue_, 
                                                        B_handle_,
                                                        k_,
                                                        n_,
                                                        base_, 
                                                        B_rows_device_,
                                                        B_cols_device_,
                                                        B_vals_device_);
          auto setC = oneapi::mkl::sparse::set_csr_data(gpuQueue_, 
                                                        C_handle_,
                                                        m_,
                                                        n_,
                                                        base_, 
                                                        C_rows_device_,
                                                        (int64_t*)nullptr,
                                                        (T*)nullptr);

          oneapi::mkl::sparse::init_matmat_descr(&description_);
          oneapi::mkl::sparse::set_matmat_data(description_,
                                               view_, 
                                               operationA_,
                                               view_,
                                               operationB_,
                                               view_);

          if (print_) std::cout << "\tGetting buffer size" << std::endl;
          // Query the size of the work-estimation temporary buffer
          request_ = oneapi::mkl::sparse::matmat_request::get_work_estimation_buf_size;
          sizeTempBuffer = sycl::malloc_host<int64_t>(1, gpuQueue_);
          auto matmat1 = oneapi::mkl::sparse::matmat(gpuQueue_,
                                                     A_handle_,
                                                     B_handle_,
                                                     C_handle_,
                                                     request_,
                                                     description_,
                                                     sizeTempBuffer,
                                                     nullptr,
                                                     {setA, setB, setC});
          matmat1.wait();
          
          // Allocate the temporary buffer for work-estimation
          tempBuffer = sycl::malloc_device<uint8_t>(sizeTempBuffer[0], gpuQueue_);
          gpuQueue_.wait();

          if (print_) std::cout << "\tWork estimation" << std::endl;
          // Do work-estimation
          request_ = oneapi::mkl::sparse::matmat_request::work_estimation;
          auto matmat2 = oneapi::mkl::sparse::matmat(gpuQueue_,
                                                     A_handle_,
                                                     B_handle_,
                                                     C_handle_,
                                                     request_,
                                                     description_,
                                                     sizeTempBuffer,
                                                     tempBuffer,
                                                     {matmat1});
          matmat2.wait();

          if (print_) std::cout << "\tGetting buffer size" << std::endl;
          // Query the size of the compute temporary buffer
          request_ = oneapi::mkl::sparse::matmat_request::get_compute_buf_size;
          sizeTempBuffer2 = sycl::malloc_host<int64_t>(1, gpuQueue_);
          auto matmat3 = oneapi::mkl::sparse::matmat(gpuQueue_,
                                                     A_handle_,
                                                     B_handle_,
                                                     C_handle_,
                                                     request_,
                                                     description_,
                                                     sizeTempBuffer2,
                                                     nullptr,
                                                     {matmat2});
          matmat3.wait();

          // Allocate the temporary buffer for compute
          tempBuffer2 = sycl::malloc_device<uint8_t>(sizeTempBuffer2[0], gpuQueue_);
          gpuQueue_.wait();

          // Do compute
          if (print_) std::cout << "\tComputing" << std::endl;
          request_ = oneapi::mkl::sparse::matmat_request::compute;
          auto matmat4 = oneapi::mkl::sparse::matmat(gpuQueue_,
                                                     A_handle_,
                                                     B_handle_,
                                                     C_handle_,
                                                     request_,
                                                     description_,
                                                     sizeTempBuffer2,
                                                     tempBuffer2,
                                                     {matmat3});
        

          // get NNZ for C
          if (print_) std::cout << "\tGetting nnz for C" << std::endl;
          C_nnz_sycl_ = sycl::malloc_host<int64_t>(1, gpuQueue_);
          request_ = oneapi::mkl::sparse::matmat_request::get_nnz;
          auto matmat5  =oneapi::mkl::sparse::matmat(gpuQueue_,
                                                     A_handle_,
                                                     B_handle_,
                                                     C_handle_,
                                                     request_,
                                                     description_,
                                                     C_nnz_sycl_,
                                                     nullptr,
                                                     {matmat4});
          matmat5.wait();

          // Exit early if C_nnz_ is zero -- no calculation to be done
          C_nnz_ = *C_nnz_sycl_;
          if (print_) std::cout << "\t\tC_nnz_sycl_ = " << *C_nnz_sycl_ << ", C_nnz_ = " << C_nnz_ << std::endl;
          if (*C_nnz_sycl_ == 0) {
            oneapi::mkl::sparse::release_matmat_descr(&description_);       
            oneapi::mkl::sparse::release_matrix_handle(gpuQueue_, &A_handle_);
            oneapi::mkl::sparse::release_matrix_handle(gpuQueue_, &B_handle_);
            oneapi::mkl::sparse::release_matrix_handle(gpuQueue_, &C_handle_);
            gpuQueue_.wait();
            if (sizeTempBuffer != nullptr) sycl::free(sizeTempBuffer, gpuQueue_);
            if (tempBuffer != nullptr) sycl::free(tempBuffer, gpuQueue_);
            if (sizeTempBuffer2 != nullptr) sycl::free(sizeTempBuffer2, gpuQueue_);
            if (tempBuffer2 != nullptr) sycl::free(tempBuffer2, gpuQueue_);
            if (C_rows_device_ != nullptr) sycl::free(C_rows_device_, gpuQueue_);
            gpuQueue_.wait();
            return;
          }

          if (print_) std::cout << "\tAllocating C structures" << std::endl;
          if (print_) std::cout << "\t\tAllocating C_cols_device_" << std::endl;
          if (C_cols_device_ != nullptr) sycl::free(C_cols_device_, gpuQueue_);
          C_cols_device_ = sycl::malloc_device<int64_t>(*C_nnz_sycl_, gpuQueue_);
          if (print_) std::cout << "\t\tAllocating C_vals_device_" << std::endl;
          if (C_vals_device_ != nullptr) sycl::free(C_vals_device_, gpuQueue_);
          C_vals_device_ = sycl::malloc_device<T>(*C_nnz_sycl_, gpuQueue_);
          gpuQueue_.wait();
          setC = oneapi::mkl::sparse::set_csr_data(gpuQueue_, 
                                                   C_handle_,
                                                   m_,
                                                   n_,
                                                   base_, 
                                                   C_rows_device_, 
                                                   C_cols_device_, 
                                                   C_vals_device_,
                                                   {matmat5});
          gpuQueue_.wait();

          // Finalise into C matrix
          if (print_) std::cout << "\tFinalising" << std::endl;
          request_ = oneapi::mkl::sparse::matmat_request::finalize;
          auto matmat6 = oneapi::mkl::sparse::matmat(gpuQueue_,
                                                     A_handle_,
                                                     B_handle_,
                                                     C_handle_,
                                                     request_,
                                                     description_,
                                                     nullptr,
                                                     nullptr,
                                                     {setC});
                  
          if (print_) std::cout << "\tSorting output matrix" << std::endl;
          auto sort = oneapi::mkl::sparse::sort_matrix(gpuQueue_,
                                                       C_handle_,
                                                       {matmat6});

          // Now clean everything up ready for the next spmm call
          if (print_) std::cout << "\tReleasing description" << std::endl;
          oneapi::mkl::sparse::release_matmat_descr(&description_);       
          if (print_) std::cout << "\tReleasing handle A" << std::endl;   
          oneapi::mkl::sparse::release_matrix_handle(gpuQueue_, &A_handle_);
          if (print_) std::cout << "\tReleasing handle B" << std::endl;
          oneapi::mkl::sparse::release_matrix_handle(gpuQueue_, &B_handle_);
          if (print_) std::cout << "\tReleasing handle C" << std::endl;
          oneapi::mkl::sparse::release_matrix_handle(gpuQueue_, &C_handle_);
          gpuQueue_.wait();
          if (print_) std::cout << "\tFreeing sizeTempBuffer" << std::endl;
          if (sizeTempBuffer != nullptr) sycl::free(sizeTempBuffer, gpuQueue_);
          if (print_) std::cout << "\tFreeing tempBuffer" << std::endl;
          if (tempBuffer != nullptr) sycl::free(tempBuffer, gpuQueue_);
          if (print_) std::cout << "\tFreeing sizeTempBuffer2" << std::endl;
          if (sizeTempBuffer2 != nullptr) sycl::free(sizeTempBuffer2, gpuQueue_);
          if (print_) std::cout << "\tFreeing tempBuffer2" << std::endl;
          if (tempBuffer2 != nullptr) sycl::free(tempBuffer2, gpuQueue_);
          gpuQueue_.wait();
          break;
        }
        case gpuOffloadType::unified: {
          if (C_rows_ != nullptr) sycl::free(C_rows_, gpuQueue_);
          if (C_cols_ != nullptr) sycl::free(C_cols_, gpuQueue_);
          if (C_vals_ != nullptr) sycl::free(C_vals_, gpuQueue_);
          if (C_nnz_sycl_ != nullptr) sycl::free(C_nnz_sycl_, gpuQueue_);

          if (print_) std::cout << "\tSetting up structures" << std::endl;
          C_rows_ = sycl::malloc_shared<int64_t>((m_ + 1), gpuQueue_);
          gpuQueue_.wait();

          oneapi::mkl::sparse::init_matrix_handle(&A_handle_);
          oneapi::mkl::sparse::init_matrix_handle(&B_handle_);
          oneapi::mkl::sparse::init_matrix_handle(&C_handle_);

          auto setA = oneapi::mkl::sparse::set_csr_data(gpuQueue_, 
                                                        A_handle_,
                                                        m_,
                                                        k_,
                                                        base_, 
                                                        A_rows_,
                                                        A_cols_,
                                                        A_vals_);
          auto setB = oneapi::mkl::sparse::set_csr_data(gpuQueue_, 
                                                        B_handle_,
                                                        k_,
                                                        n_,
                                                        base_, 
                                                        B_rows_,
                                                        B_cols_,
                                                        B_vals_);
          auto setC = oneapi::mkl::sparse::set_csr_data(gpuQueue_, 
                                                        C_handle_,
                                                        m_,
                                                        n_,
                                                        base_, 
                                                        C_rows_,
                                                        (int64_t*)nullptr,
                                                        (T*)nullptr);
          
          oneapi::mkl::sparse::init_matmat_descr(&description_);
          oneapi::mkl::sparse::set_matmat_data(description_,
                                               view_, 
                                               operationA_,
                                               view_,
                                               operationB_,
                                               view_);

          if (print_) std::cout << "\tGetting buffer size" << std::endl;
          // Query the size of the work-estimation temporary buffer
          request_ = oneapi::mkl::sparse::matmat_request::get_work_estimation_buf_size;
          sizeTempBuffer = sycl::malloc_host<int64_t>(1, gpuQueue_);
          auto matmat1 = oneapi::mkl::sparse::matmat(gpuQueue_,
                                                     A_handle_,
                                                     B_handle_,
                                                     C_handle_,
                                                     request_,
                                                     description_,
                                                     sizeTempBuffer,
                                                     nullptr,
                                                     {setA, setB, setC});
          matmat1.wait();
          
          if (print_) std::cout << "\tWork estimation" << std::endl;
          // Allocate the temporary buffer for work-estimation
          tempBuffer = sycl::malloc_shared<uint8_t>(sizeTempBuffer[0], gpuQueue_);
          gpuQueue_.wait();

          // Do work-estimation
          request_ = oneapi::mkl::sparse::matmat_request::work_estimation;
          auto matmat2 = oneapi::mkl::sparse::matmat(gpuQueue_,
                                                     A_handle_,
                                                     B_handle_,
                                                     C_handle_,
                                                     request_,
                                                     description_,
                                                     sizeTempBuffer,
                                                     tempBuffer,
                                                     {matmat1});
          matmat2.wait();

          if (print_) std::cout << "\tGetting buffer size" << std::endl;
          // Query the size of the compute temporary buffer
          request_ = oneapi::mkl::sparse::matmat_request::get_compute_buf_size;
          sizeTempBuffer2 = sycl::malloc_host<int64_t>(1, gpuQueue_);
          auto matmat3 = oneapi::mkl::sparse::matmat(gpuQueue_,
                                                     A_handle_,
                                                     B_handle_,
                                                     C_handle_,
                                                     request_,
                                                     description_,
                                                     sizeTempBuffer2,
                                                     nullptr,
                                                     {matmat2});
          matmat3.wait();

          // Allocate the temporary buffer for compute
          tempBuffer2 = sycl::malloc_shared<uint8_t>(sizeTempBuffer2[0], gpuQueue_);

          // Do compute
          if (print_) std::cout << "\tDoing compute" << std::endl;
          request_ = oneapi::mkl::sparse::matmat_request::compute;
          auto matmat4 = oneapi::mkl::sparse::matmat(gpuQueue_,
                                                     A_handle_,
                                                     B_handle_,
                                                     C_handle_,
                                                     request_,
                                                     description_,
                                                     sizeTempBuffer2,
                                                     tempBuffer2,
                                                     {matmat3});
          
          if (print_) std::cout << "\tCalculation of nnz for C" << std::endl;
          // get NNZ for C
          request_ = oneapi::mkl::sparse::matmat_request::get_nnz;
          C_nnz_sycl_ = sycl::malloc_shared<int64_t>(1, gpuQueue_);
          auto matmat5 = oneapi::mkl::sparse::matmat(gpuQueue_,
                                                     A_handle_,
                                                     B_handle_,
                                                     C_handle_,
                                                     request_,
                                                     description_,
                                                     C_nnz_sycl_,
                                                     nullptr,
                                                     {matmat4});
          matmat5.wait();

          // If C_nnz_ is zero, exit early as no calculation to be done
          if (print_) std::cout << "\t\tC_nnz_ = " << *C_nnz_sycl_ << std::endl;
          if (*C_nnz_sycl_ == 0) {
            oneapi::mkl::sparse::release_matmat_descr(&description_);
            oneapi::mkl::sparse::release_matrix_handle(gpuQueue_, &A_handle_);
            oneapi::mkl::sparse::release_matrix_handle(gpuQueue_, &B_handle_);
            oneapi::mkl::sparse::release_matrix_handle(gpuQueue_, &C_handle_);
            gpuQueue_.wait();

            if (sizeTempBuffer != nullptr) sycl::free(sizeTempBuffer, gpuQueue_);
            if (tempBuffer != nullptr) sycl::free(tempBuffer, gpuQueue_);
            if (sizeTempBuffer2 != nullptr) sycl::free(sizeTempBuffer2, gpuQueue_);
            if (tempBuffer2 != nullptr) sycl::free(tempBuffer2, gpuQueue_);
            if (C_rows_ != nullptr) sycl::free(C_rows_, gpuQueue_);
            if (C_nnz_sycl_ != nullptr) sycl::free(C_nnz_sycl_, gpuQueue_);
            gpuQueue_.wait();
            return;
          }

          if (print_) std::cout << "\tAllocating C structures" << std::endl;
          C_cols_ = sycl::malloc_shared<int64_t>(*C_nnz_sycl_, gpuQueue_);
          C_vals_ = sycl::malloc_shared<T>(*C_nnz_sycl_, gpuQueue_);
          gpuQueue_.wait();
          setC = oneapi::mkl::sparse::set_csr_data(gpuQueue_, 
                                                   C_handle_,
                                                   m_,
                                                   n_,
                                                   base_, 
                                                   C_rows_,
                                                   C_cols_,
                                                   C_vals_,
                                                   {matmat5});

          if (print_) std::cout << "\tFinalising" << std::endl;
          // Finalise into C matrix
          request_ = oneapi::mkl::sparse::matmat_request::finalize;
          auto matmat6 = oneapi::mkl::sparse::matmat(gpuQueue_,
                                                     A_handle_,
                                                     B_handle_,
                                                     C_handle_,
                                                     request_,
                                                     description_,
                                                     nullptr,
                                                     nullptr,
                                                     {setC});
                                  
          if (print_) std::cout << "\tSorting output matrix" << std::endl;
          auto sort = oneapi::mkl::sparse::sort_matrix(gpuQueue_,
                                                       C_handle_,
                                                       {matmat6});

          if (print_) std::cout << "\tFreeing up old structures" << std::endl;
          // Now clean everything up ready for the next spmm call
          oneapi::mkl::sparse::release_matmat_descr(&description_);          
          oneapi::mkl::sparse::release_matrix_handle(gpuQueue_, &A_handle_);
          oneapi::mkl::sparse::release_matrix_handle(gpuQueue_, &B_handle_);
          oneapi::mkl::sparse::release_matrix_handle(gpuQueue_, &C_handle_);
          gpuQueue_.wait();

          if (sizeTempBuffer != nullptr) sycl::free(sizeTempBuffer, gpuQueue_);
          if (tempBuffer != nullptr) sycl::free(tempBuffer, gpuQueue_);
          if (sizeTempBuffer2 != nullptr) sycl::free(sizeTempBuffer2, gpuQueue_);
          if (tempBuffer2 != nullptr) sycl::free(tempBuffer2, gpuQueue_);
          gpuQueue_.wait();
          break;
        }
      }
    }

    void postLoopRequirements() override {
      if (print_) std::cout << "Post-loop stuff" << std::endl;
      switch (offload_) {
        case gpuOffloadType::always:{
          // Already dealt with within the callSpmm function.
          break;
        }
        case gpuOffloadType::once:{
          // Copy data back to host
          if (print_) std::cout << "\tMoving data back to the CPU" << std::endl;
          C_rows_ = sycl::malloc_host<int64_t>(m_ + 1, gpuQueue_);
          C_cols_ = sycl::malloc_host<int64_t>(*C_nnz_sycl_, gpuQueue_);
          C_vals_ = sycl::malloc_host<T>(*C_nnz_sycl_, gpuQueue_);
          C_nnz_ = *C_nnz_sycl_;
          gpuQueue_.wait();
          gpuQueue_.copy<int64_t>(C_rows_device_, C_rows_, m_ + 1);
          gpuQueue_.copy<int64_t>(C_cols_device_, C_cols_, *C_nnz_sycl_);
          gpuQueue_.copy<T>(C_vals_device_, C_vals_, *C_nnz_sycl_);
          gpuQueue_.wait();
          // Now free everything
          if (print_) std::cout << "\tCleaning up device memory" << std::endl;
          if (C_rows_device_ != nullptr) sycl::free(C_rows_device_, gpuQueue_);
          if (C_cols_device_ != nullptr) sycl::free(C_cols_device_, gpuQueue_);
          if (C_vals_device_ != nullptr) sycl::free(C_vals_device_, gpuQueue_);
          if (C_nnz_sycl_ != nullptr) sycl::free(C_nnz_sycl_, gpuQueue_);
          gpuQueue_.wait();
          break;
        }
        case gpuOffloadType::unified: {
          C_nnz_ = *C_nnz_sycl_;
          sycl::free(C_nnz_sycl_, gpuQueue_);
          break;
        }
      }
    }

    void postCallKernelCleanup() override {
      if (print_) std::cout << "Kernel cleanup" << std::endl;
      if (offload_ == gpuOffloadType::unified) {
        if (A_vals_ != nullptr) sycl::free(A_vals_, gpuQueue_);
        if (A_cols_ != nullptr) sycl::free(A_cols_, gpuQueue_);
        if (A_rows_ != nullptr) sycl::free(A_rows_, gpuQueue_);
        if (B_vals_ != nullptr) sycl::free(B_vals_, gpuQueue_);
        if (B_cols_ != nullptr) sycl::free(B_cols_, gpuQueue_);
        if (B_rows_ != nullptr) sycl::free(B_rows_, gpuQueue_);
        if (C_vals_ != nullptr) sycl::free(C_vals_, gpuQueue_);
        if (C_cols_ != nullptr) sycl::free(C_cols_, gpuQueue_);
        if (C_rows_ != nullptr) sycl::free(C_rows_, gpuQueue_);
        gpuQueue_.wait();
      } else {
        if (A_vals_ != nullptr) sycl::free(A_vals_, gpuQueue_);
        if (A_cols_ != nullptr) sycl::free(A_cols_, gpuQueue_);
        if (A_rows_ != nullptr) sycl::free(A_rows_, gpuQueue_);
        if (B_vals_ != nullptr) sycl::free(B_vals_, gpuQueue_);
        if (B_cols_ != nullptr) sycl::free(B_cols_, gpuQueue_);
        if (B_rows_ != nullptr) sycl::free(B_rows_, gpuQueue_);
        if (C_vals_ != nullptr) sycl::free(C_vals_, gpuQueue_);
        if (C_cols_ != nullptr) sycl::free(C_cols_, gpuQueue_);
        if (C_rows_ != nullptr) sycl::free(C_rows_, gpuQueue_);

        if (A_vals_device_ != nullptr) sycl::free(A_vals_device_, gpuQueue_);
        if (A_cols_device_ != nullptr) sycl::free(A_cols_device_, gpuQueue_);
        if (A_rows_device_ != nullptr) sycl::free(A_rows_device_, gpuQueue_);
        if (B_vals_device_ != nullptr) sycl::free(B_vals_device_, gpuQueue_);
        if (B_cols_device_ != nullptr) sycl::free(B_cols_device_, gpuQueue_);
        if (B_rows_device_ != nullptr) sycl::free(B_rows_device_, gpuQueue_);
        gpuQueue_.wait();
      }
    }

    bool print_ = true;

    bool firstRun_ = true;

    sycl::device myGpu_;
    sycl::queue gpuQueue_;

    // MKL metadata parameters
    oneapi::mkl::transpose operationA_ = oneapi::mkl::transpose::nontrans;
    oneapi::mkl::transpose operationB_ = oneapi::mkl::transpose::nontrans;
    oneapi::mkl::sparse::matmat_descr_t description_ = nullptr;
    oneapi::mkl::sparse::matrix_view_descr view_ = oneapi::mkl::sparse::matrix_view_descr::general;

    oneapi::mkl::index_base base_ = oneapi::mkl::index_base::zero;

    oneapi::mkl::sparse::matmat_request request_;

    // Matrix data pointers (host memory)
    T* A_vals_ = nullptr;
    int64_t* A_cols_ = nullptr;
    int64_t* A_rows_ = nullptr;

    T* B_vals_ = nullptr;
    int64_t* B_cols_ = nullptr;
    int64_t* B_rows_ = nullptr;

    // Device memory pointers (for 'once' and 'always' modes)
    T* A_vals_device_ = nullptr;
    int64_t* A_cols_device_ = nullptr;
    int64_t* A_rows_device_ = nullptr;

    T* B_vals_device_ = nullptr;
    int64_t* B_cols_device_ = nullptr;
    int64_t* B_rows_device_ = nullptr;

    T* C_vals_device_ = nullptr;
    int64_t* C_cols_device_ = nullptr;
    int64_t* C_rows_device_ = nullptr;
    int64_t* C_nnz_sycl_ = nullptr;

    // Matrix handles
    oneapi::mkl::sparse::matrix_handle_t A_handle_ = nullptr;
    oneapi::mkl::sparse::matrix_handle_t B_handle_ = nullptr;
    oneapi::mkl::sparse::matrix_handle_t C_handle_ = nullptr;

    // Temp buffers
    int64_t* sizeTempBuffer = nullptr;
    int64_t* sizeTempBuffer2 = nullptr;
    void* tempBuffer = nullptr;
    void* tempBuffer2 = nullptr;

    const T alpha = ALPHA;
    const T beta = BETA;
};
}

#endif
