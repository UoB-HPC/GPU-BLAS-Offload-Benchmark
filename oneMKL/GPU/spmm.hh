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

      nnzA_ = 1 + (uint64_t)((double)m_ * (double)k_ * (1.0 - sparsity_));
      nnzB_ = 1 + (uint64_t)((double)k_ * (double)n_ * (1.0 - sparsity_));

      // Setting MKL metadata
      // Todo 

      if (print_) std::cout << "\tMallocing" << std::endl;
      if (offload_ == gpuOffloadType::unified) {
        A_ = (T*)sycl::malloc_shared(sizeof(T) * m_ * k_, gpuQueue_);
        A_vals_ = (T*)sycl::malloc_shared(sizeof(T) * nnzA_, gpuQueue_);
        A_cols_ = (int64_t*)sycl::malloc_shared(sizeof(int64_t) * nnzA_, gpuQueue_);
        A_rows_ = (int64_t*)sycl::malloc_shared(sizeof(int64_t) * (m_ + 1), gpuQueue_);

        B_ = (T*)sycl::malloc_shared(sizeof(T) * k_ * n_, gpuQueue_);
        B_vals_ = (T*)sycl::malloc_shared(sizeof(T) * nnzB_, gpuQueue_);
        B_cols_ = (int64_t*)sycl::malloc_shared(sizeof(int64_t) * nnzB_, gpuQueue_);
        B_rows_ = (int64_t*)sycl::malloc_shared(sizeof(int64_t) * (k_ + 1), gpuQueue_);

        C_ = (T*)sycl::malloc_shared(sizeof(T) * m_ * n_, gpuQueue_);

        gpuQueue_.wait();
      } else {
        A_ = (T*)sycl::malloc_host<T>(m_ * k_, gpuQueue_);
        A_vals_ = (T*)sycl::malloc_host(sizeof(T) * nnzA_, gpuQueue_);
        A_cols_ = (int64_t*)sycl::malloc_host(sizeof(int64_t) * nnzA_, gpuQueue_);
        A_rows_ = (int64_t*)sycl::malloc_host(sizeof(int64_t) * (m_ + 1), gpuQueue_);

        B_ = (T*)sycl::malloc_host(sizeof(T) * k_ * n_, gpuQueue_);
        B_vals_ = (T*)sycl::malloc_host(sizeof(T) * nnzB_, gpuQueue_);
        B_cols_ = (int64_t*)sycl::malloc_host(sizeof(int64_t) * nnzB_, gpuQueue_);
        B_rows_ = (int64_t*)sycl::malloc_host(sizeof(int64_t) * (k_ + 1), gpuQueue_);

        C_ = (T*)sycl::malloc_host(sizeof(T) * m_ * n_, gpuQueue_);

        A_vals_device_ = (T*)sycl::malloc_device(sizeof(T) * nnzA_, gpuQueue_);
        A_cols_device_ = (int64_t*)sycl::malloc_device(sizeof(int64_t) * nnzA_, gpuQueue_);
        A_rows_device_ = (int64_t*)sycl::malloc_device(sizeof(int64_t) * (m_ + 1), gpuQueue_);

        B_vals_device_ = (T*)sycl::malloc_device(sizeof(T) * nnzB_, gpuQueue_);
        B_cols_device_ = (int64_t*)sycl::malloc_device(sizeof(int64_t) * nnzB_, gpuQueue_);
        B_rows_device_ = (int64_t*)sycl::malloc_device(sizeof(int64_t) * (k_ + 1), gpuQueue_);
        gpuQueue_.wait();
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
      gpuQueue_.wait();
    }

private:
    void preLoopRequirements() override {
      if (print_) std::cout << "pre-loop stuff" << std::endl;
      switch (offload_) {
        case gpuOffloadType::always: 
          break;
        case gpuOffloadType::once:
          // Copy A and B over to the GPU
          if (print_) std::cout << "\tCopying data over to GPU" << std::endl;
          gpuQueue_.copy<T>(A_vals_, A_vals_device_, nnzA_);
          gpuQueue_.copy<int64_t>(A_cols_, A_cols_device_, nnzA_);
          gpuQueue_.copy<int64_t>(A_rows_, A_rows_device_, m_ + 1);

          gpuQueue_.copy<T>(B_vals_, B_vals_device_, nnzB_);
          gpuQueue_.copy<int64_t>(B_cols_, B_cols_device_, nnzB_);
          gpuQueue_.copy<int64_t>(B_rows_, B_rows_device_, k_ + 1);
          gpuQueue_.wait();

          onceFirst_ = true;
          break;
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
          gpuQueue_.copy<T>(A_vals_, A_vals_device_, nnzA_);
          gpuQueue_.copy<int64_t>(A_cols_, A_cols_device_, nnzA_);
          gpuQueue_.copy<int64_t>(A_rows_, A_rows_device_, m_ + 1);

          gpuQueue_.copy<T>(B_vals_, B_vals_device_, nnzB_);
          gpuQueue_.copy<int64_t>(B_cols_, B_cols_device_, nnzB_);
          gpuQueue_.copy<int64_t>(B_rows_, B_rows_device_, k_ + 1);
          gpuQueue_.wait();

          if (print_) std::cout << "\tMallocing C rows, and setting up handles etc." << std::endl;
          C_rows_device_ = (int64_t*)sycl::malloc_device<int64_t>(m_ + 1, gpuQueue_);
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
          sizeTempBuffer = (int64_t*)sycl::malloc_host<int64_t>(1, gpuQueue_);
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
          tempBuffer = sycl::malloc_device(sizeTempBuffer[0] * sizeof(uint8_t), gpuQueue_);
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
          sizeTempBuffer2 = (int64_t*)sycl::malloc_host(sizeof(int64_t), gpuQueue_);
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
          tempBuffer2 = sycl::malloc_device(sizeTempBuffer2[0] * sizeof(uint8_t), gpuQueue_);
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
          nnzC_ = (int64_t*)sycl::malloc_host(sizeof(int64_t), gpuQueue_);
          request_ = oneapi::mkl::sparse::matmat_request::get_nnz;
          auto matmat5 = oneapi::mkl::sparse::matmat(gpuQueue_,
                                                     A_handle_,
                                                     B_handle_,
                                                     C_handle_,
                                                     request_,
                                                     description_,
                                                     nnzC_,
                                                     nullptr,
                                                     {matmat4});
          matmat5.wait();

          // Make sure that nnzC_ is non-zero.  If it isn't then clean up and return -- no computation to be done
          if (print_) std::cout << "\t\tnnzC_ = " << *nnzC_ << std::endl;
          if (*nnzC_ == 0) {
            oneapi::mkl::sparse::release_matmat_descr(&description_);          
            oneapi::mkl::sparse::release_matrix_handle(gpuQueue_, &A_handle_);
            oneapi::mkl::sparse::release_matrix_handle(gpuQueue_, &B_handle_);
            oneapi::mkl::sparse::release_matrix_handle(gpuQueue_, &C_handle_);
            gpuQueue_.wait();

            sycl::free(sizeTempBuffer, gpuQueue_);
            sycl::free(tempBuffer, gpuQueue_);
            sycl::free(sizeTempBuffer2, gpuQueue_);
            sycl::free(tempBuffer2, gpuQueue_);
            sycl::free(C_rows_device_, gpuQueue_); 
            sycl::free(nnzC_, gpuQueue_);
            gpuQueue_.wait();
            return;
          }
          
          if (print_) std::cout << "\tAllocating C structures" << std::endl;
          C_cols_device_ = (int64_t*)sycl::malloc_device(*nnzC_ * sizeof(int64_t), gpuQueue_);
          C_vals_device_ = (T*)sycl::malloc_device(*nnzC_ * sizeof(T), gpuQueue_);
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

          if (print_) std::cout << "\tCopying data back to CPU" << std::endl;
          // Copy data back to host
          C_rows_ = (int64_t*)sycl::malloc_host((m_ + 1) * sizeof(int64_t), gpuQueue_);
          C_cols_ = (int64_t*)sycl::malloc_host(*nnzC_ * sizeof(int64_t), gpuQueue_);
          C_vals_ = (T*)sycl::malloc_host(*nnzC_ * sizeof(T), gpuQueue_);
          gpuQueue_.wait();
          gpuQueue_.copy(C_rows_device_, C_rows_, (m_ + 1) * sizeof(int64_t));
          gpuQueue_.copy(C_cols_device_, C_cols_, *nnzC_ * sizeof(int64_t));
          gpuQueue_.copy(C_vals_device_, C_vals_, *nnzC_ * sizeof(int64_t));
          gpuQueue_.wait();

          if (print_) std::cout << "\tCleaning up temp allocations" << std::endl;
          // Now clean everything up ready for the next spmm call
          oneapi::mkl::sparse::release_matmat_descr(&description_);          
          oneapi::mkl::sparse::release_matrix_handle(gpuQueue_, &A_handle_);
          oneapi::mkl::sparse::release_matrix_handle(gpuQueue_, &B_handle_);
          oneapi::mkl::sparse::release_matrix_handle(gpuQueue_, &C_handle_);
          gpuQueue_.wait();

          sycl::free(sizeTempBuffer, gpuQueue_);
          sycl::free(tempBuffer, gpuQueue_);
          sycl::free(sizeTempBuffer2, gpuQueue_);
          sycl::free(tempBuffer2, gpuQueue_);
          sycl::free(C_rows_device_, gpuQueue_); 
          sycl::free(C_cols_device_, gpuQueue_);
          sycl::free(C_vals_device_, gpuQueue_);
          sycl::free(C_rows_, gpuQueue_);
          sycl::free(C_cols_, gpuQueue_);
          sycl::free(C_vals_, gpuQueue_);
          sycl::free(nnzC_, gpuQueue_);
          gpuQueue_.wait();
          break;
        }
        case gpuOffloadType::once: {
          if (print_) std::cout << "\tFreeing old memory allocations, if present" << std::endl;
          // Check to see if C device arrays already exist.  If they do, get rid
          if (!onceFirst_) sycl::free(C_rows_device_, gpuQueue_); 
          if (!onceFirst_) sycl::free(C_cols_device_, gpuQueue_);
          if (!onceFirst_) sycl::free(C_vals_device_, gpuQueue_);
          if (!onceFirst_) sycl::free(nnzC_, gpuQueue_);

          // Allocate space for the C rows array
          C_rows_device_ = (int64_t*)sycl::malloc_device((m_ + 1) * sizeof(int64_t), gpuQueue_);
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
          sizeTempBuffer = (int64_t*)sycl::malloc_host(sizeof(int64_t), gpuQueue_);
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
          tempBuffer = sycl::malloc_device(sizeTempBuffer[0] * sizeof(uint8_t), gpuQueue_);
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
          sizeTempBuffer2 = (int64_t*)sycl::malloc_host(sizeof(int64_t), gpuQueue_);
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
          tempBuffer2 = sycl::malloc_device(sizeTempBuffer2[0] * sizeof(uint8_t), gpuQueue_);
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
          nnzC_ = (int64_t*)sycl::malloc_host(sizeof(int64_t), gpuQueue_);
          request_ = oneapi::mkl::sparse::matmat_request::get_nnz;
          auto matmat5  =oneapi::mkl::sparse::matmat(gpuQueue_,
                                                     A_handle_,
                                                     B_handle_,
                                                     C_handle_,
                                                     request_,
                                                     description_,
                                                     nnzC_,
                                                     nullptr,
                                                     {matmat4});
          matmat5.wait();

          // Exit early if nnzC_ is zero -- no calculation to be done
          if (print_) std::cout << "\t\tnnzC_ = " << *nnzC_ << std::endl;
          if (*nnzC_ == 0) {
            oneapi::mkl::sparse::release_matmat_descr(&description_);       
            oneapi::mkl::sparse::release_matrix_handle(gpuQueue_, &A_handle_);
            oneapi::mkl::sparse::release_matrix_handle(gpuQueue_, &B_handle_);
            oneapi::mkl::sparse::release_matrix_handle(gpuQueue_, &C_handle_);
            gpuQueue_.wait();
            sycl::free(sizeTempBuffer, gpuQueue_);
            sycl::free(tempBuffer, gpuQueue_);
            sycl::free(sizeTempBuffer2, gpuQueue_);
            sycl::free(tempBuffer2, gpuQueue_);
            if (onceFirst_) sycl::free(C_rows_device_, gpuQueue_);
            gpuQueue_.wait();
            return;
          }

          if (print_) std::cout << "\tAllocating C structures" << std::endl;
          C_cols_device_ = (int64_t*)sycl::malloc_device(*nnzC_ * sizeof(int64_t), gpuQueue_);
          C_vals_device_ = (T*)sycl::malloc_device(*nnzC_ * sizeof(T), gpuQueue_);
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
          sycl::free(sizeTempBuffer, gpuQueue_);
          if (print_) std::cout << "\tFreeing tempBuffer" << std::endl;
          sycl::free(tempBuffer, gpuQueue_);
          if (print_) std::cout << "\tFreeing sizeTempBuffer2" << std::endl;
          sycl::free(sizeTempBuffer2, gpuQueue_);
          if (print_) std::cout << "\tFreeing tempBuffer2" << std::endl;
          sycl::free(tempBuffer2, gpuQueue_);
          gpuQueue_.wait();
          onceFirst_ = false;
          break;
        }
        case gpuOffloadType::unified: {
          if (print_) std::cout << "\tSetting up structures" << std::endl;
          C_rows_ = (int64_t*)sycl::malloc_shared((m_ + 1) * sizeof(int64_t), gpuQueue_);
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
          sizeTempBuffer = (int64_t*)sycl::malloc_host(sizeof(int64_t), gpuQueue_);
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
          tempBuffer = sycl::malloc_shared(sizeTempBuffer[0] * sizeof(uint8_t), gpuQueue_);
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
          sizeTempBuffer2 = (int64_t*)sycl::malloc_host(sizeof(int64_t), gpuQueue_);
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
          tempBuffer2 = sycl::malloc_shared(sizeTempBuffer2[0] * sizeof(uint8_t), gpuQueue_);
          
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
          nnzC_ = (int64_t*)sycl::malloc_shared(sizeof(int64_t), gpuQueue_);
          auto matmat5 = oneapi::mkl::sparse::matmat(gpuQueue_,
                                                     A_handle_,
                                                     B_handle_,
                                                     C_handle_,
                                                     request_,
                                                     description_,
                                                     nnzC_,
                                                     nullptr,
                                                     {matmat4});
          matmat5.wait();

          // If nnzC_ is zero, exit early as no calculation to be done
          if (print_) std::cout << "\t\tnnzC_ = " << *nnzC_ << std::endl;
          if (*nnzC_ == 0) {
            oneapi::mkl::sparse::release_matmat_descr(&description_);          
            oneapi::mkl::sparse::release_matrix_handle(gpuQueue_, &A_handle_);
            oneapi::mkl::sparse::release_matrix_handle(gpuQueue_, &B_handle_);
            oneapi::mkl::sparse::release_matrix_handle(gpuQueue_, &C_handle_);
            gpuQueue_.wait();

            sycl::free(sizeTempBuffer, gpuQueue_);
            sycl::free(tempBuffer, gpuQueue_);
            sycl::free(sizeTempBuffer2, gpuQueue_);
            sycl::free(tempBuffer2, gpuQueue_);
            sycl::free(C_rows_, gpuQueue_);
            sycl::free(nnzC_, gpuQueue_);
            gpuQueue_.wait();
            return;
          }

          if (print_) std::cout << "\tAllocating C structures" << std::endl;
          C_cols_ = (int64_t*)sycl::malloc_shared(*nnzC_ * sizeof(int64_t), gpuQueue_);
          C_vals_ = (T*)sycl::malloc_shared(*nnzC_ * sizeof(T), gpuQueue_);
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

          sycl::free(sizeTempBuffer, gpuQueue_);
          sycl::free(tempBuffer, gpuQueue_);
          sycl::free(sizeTempBuffer2, gpuQueue_);
          sycl::free(tempBuffer2, gpuQueue_);
          sycl::free(C_rows_, gpuQueue_);
          sycl::free(C_cols_, gpuQueue_);
          sycl::free(C_vals_, gpuQueue_);
          sycl::free(nnzC_, gpuQueue_);
          gpuQueue_.wait();
          break;
        }
      }
    }

    void postLoopRequirements() override {
      if (print_) std::cout << "Post-loop stuff" << std::endl;
      switch (offload_) {
        case gpuOffloadType::always:
          // Already dealt with within the callSpmm function.
          break;
        case gpuOffloadType::once:
          // Copy data back to host
          if (print_) std::cout << "\tMoving data back to the CPU" << std::endl;
          C_rows_ = (int64_t*)sycl::malloc_host((m_ + 1) * sizeof(int64_t), gpuQueue_);
          C_cols_ = (int64_t*)sycl::malloc_host(*nnzC_ * sizeof(int64_t), gpuQueue_);
          C_vals_ = (T*)sycl::malloc_host(*nnzC_ * sizeof(T), gpuQueue_);
          gpuQueue_.wait();
          gpuQueue_.copy(C_rows_device_, C_rows_, (m_ + 1) * sizeof(int64_t));
          gpuQueue_.copy(C_cols_device_, C_cols_, *nnzC_ * sizeof(int64_t));
          gpuQueue_.copy(C_vals_device_, C_vals_, *nnzC_ * sizeof(T));
          gpuQueue_.wait();
          // Now free everything
          if (print_) std::cout << "\tCleaning up memory" << std::endl;
          sycl::free(C_rows_device_, gpuQueue_); 
          sycl::free(C_cols_device_, gpuQueue_);
          sycl::free(C_vals_device_, gpuQueue_);
          sycl::free(C_rows_, gpuQueue_);
          sycl::free(C_cols_, gpuQueue_);
          sycl::free(C_vals_, gpuQueue_);
          gpuQueue_.wait();
          break;
        case gpuOffloadType::unified:
          break;
      }
    }

    void postCallKernelCleanup() override {
      if (print_) std::cout << "Kernel cleanup" << std::endl;
      if (offload_ == gpuOffloadType::unified) {
        sycl::free(A_, gpuQueue_);
        sycl::free(A_vals_, gpuQueue_);
        sycl::free(A_cols_, gpuQueue_);
        sycl::free(A_rows_, gpuQueue_);
        sycl::free(B_, gpuQueue_);
        sycl::free(B_vals_, gpuQueue_);
        sycl::free(B_cols_, gpuQueue_);
        sycl::free(B_rows_, gpuQueue_);
        gpuQueue_.wait();
      } else {
        sycl::free(A_, gpuQueue_);
        sycl::free(A_vals_, gpuQueue_);
        sycl::free(A_cols_, gpuQueue_);
        sycl::free(A_rows_, gpuQueue_);
        sycl::free(B_, gpuQueue_);
        sycl::free(B_vals_, gpuQueue_);
        sycl::free(B_cols_, gpuQueue_);
        sycl::free(B_rows_, gpuQueue_);

        sycl::free(A_vals_device_, gpuQueue_);
        sycl::free(A_cols_device_, gpuQueue_);
        sycl::free(A_rows_device_, gpuQueue_);
        sycl::free(B_vals_device_, gpuQueue_);
        sycl::free(B_cols_device_, gpuQueue_);
        sycl::free(B_rows_device_, gpuQueue_);
        gpuQueue_.wait();
      }
    }

    bool print_ = true;

    bool firstRun_ = true;
    bool onceFirst_ = true;

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

    T* C_vals_ = nullptr;
    int64_t* C_cols_ = nullptr;
    int64_t* C_rows_ = nullptr;

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

    int64_t* nnzC_;

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
