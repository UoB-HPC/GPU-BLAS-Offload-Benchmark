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
            std::cout << "========== ALWAYS ===========" << std::endl;
            break;
          case gpuOffloadType::once:
            std::cout << "=========== ONCE ============" << std::endl;
            break;
          case gpuOffloadType::unified:
            std::cout << "========== UNIFIED ==========" << std::endl;
            break;
        }
      } 

      if (print_) std::cout << "Initialising SPMM: " << m << " " << n << " " << k << std::endl;

      // Storing initialise parameters into global variables
      m_ = m;
      n_ = n;
      k_ = k;
      sparsity_ = sparsity;
      type_ = type;
      offload_ = offload;

      // Calculating starting matrix NNZ values
      A_nnz_ = 1 + (uint64_t)((double)m_ * (double)k_ * (1.0 - sparsity_));
      B_nnz_ = 1 + (uint64_t)((double)k_ * (double)n_ * (1.0 - sparsity_));

      // Set up the sycl parameters
      queue_ = sycl::queue(sycl::gpu_selector_v);
      context_ = queue_.get_context();
      device_ = queue_.get_device();

      if (print_) std::cout << "\tAllocating CSR arrays" << std::endl;
      switch (offload_) {
        case gpuOffloadType::always: {
          A_rows_ = sycl::malloc_host<int64_t>(m_ + 1, queue_);
          A_cols_ = sycl::malloc_host<int64_t>(A_nnz_, queue_);
          A_vals_ = sycl::malloc_host<T>(A_nnz_, queue_);
          A_rows_device_ = sycl::malloc_device<int64_t>(m_ + 1, queue_);
          A_cols_device_ = sycl::malloc_device<int64_t>(A_nnz_, queue_);
          A_vals_device_ = sycl::malloc_device<T>(A_nnz_, queue_);

          B_rows_ = sycl::malloc_host<int64_t>(k_ + 1, queue_);
          B_cols_ = sycl::malloc_host<int64_t>(B_nnz_, queue_);
          B_vals_ = sycl::malloc_host<T>(B_nnz_, queue_);
          B_rows_device_ = sycl::malloc_device<int64_t>(k_ + 1, queue_);
          B_cols_device_ = sycl::malloc_device<int64_t>(B_nnz_, queue_);
          B_vals_device_ = sycl::malloc_device<T>(B_nnz_, queue_);

          C_rows_ = nullptr;
          C_cols_ = nullptr;
          C_vals_ = nullptr;
          break;
        }
        case gpuOffloadType::once: {
          A_rows_ = sycl::malloc_host<int64_t>(m_ + 1, queue_);
          A_cols_ = sycl::malloc_host<int64_t>(A_nnz_, queue_);
          A_vals_ = sycl::malloc_host<T>(A_nnz_, queue_);
          A_rows_device_ = sycl::malloc_device<int64_t>(m_ + 1, queue_);
          A_cols_device_ = sycl::malloc_device<int64_t>(A_nnz_, queue_);
          A_vals_device_ = sycl::malloc_device<T>(A_nnz_, queue_);

          B_rows_ = sycl::malloc_host<int64_t>(k_ + 1, queue_);
          B_cols_ = sycl::malloc_host<int64_t>(B_nnz_, queue_);
          B_vals_ = sycl::malloc_host<T>(B_nnz_, queue_);
          B_rows_device_ = sycl::malloc_device<int64_t>(k_ + 1, queue_);
          B_cols_device_ = sycl::malloc_device<int64_t>(B_nnz_, queue_);
          B_vals_device_ = sycl::malloc_device<T>(B_nnz_, queue_);

          C_rows_ = nullptr;
          C_cols_ = nullptr;
          C_vals_ = nullptr;
          break;
        }
        case gpuOffloadType::unified: {
          A_rows_ = sycl::malloc_shared<int64_t>(m_ + 1, queue_);
          A_cols_ = sycl::malloc_shared<int64_t>(A_nnz_, queue_);
          A_vals_ = sycl::malloc_shared<T>(A_nnz_, queue_);

          B_rows_ = sycl::malloc_shared<int64_t>(k_ + 1, queue_);
          B_cols_ = sycl::malloc_shared<int64_t>(B_nnz_, queue_);
          B_vals_ = sycl::malloc_shared<T>(B_nnz_, queue_);

          C_rows_ = nullptr;
          C_cols_ = nullptr;
          C_vals_ = nullptr;
          break;
        }
      }

      initInputMatrices();
      if (print_) printInputMatrices();
    }

protected:
    void toSparseFormat() override {
      if (print_) std::cout << "toSparse" << std::endl;
      int seedOffset = 0;
      if (type_ == matrixType::rmat) {
        do {
          if (print_) std::cout << "\tGenerating rMAT matrices" << std::endl;
          rMatCSR<T, int64_t>(A_vals_, A_cols_, A_rows_, m_, k_, A_nnz_, SEED + seedOffset++);
          rMatCSR<T, int64_t>(B_vals_, B_cols_, B_rows_, k_, n_, B_nnz_, SEED + seedOffset++);
        } while (calcCNNZ<int64_t>(m_, A_nnz_, A_rows_, A_cols_, k_, B_nnz_, B_rows_, B_cols_) == 0);
      } else if (type_ == matrixType::random) {
        do {
          if (print_) std::cout << "\tGenerating random matrices" << std::endl;
          randomCSR<T, int64_t>(A_vals_, A_cols_, A_rows_, m_, k_, A_nnz_, SEED + seedOffset++);
          randomCSR<T, int64_t>(B_vals_, B_cols_, B_rows_, k_, n_, B_nnz_, SEED + seedOffset++);
        } while (calcCNNZ<int64_t>(m_, A_nnz_, A_rows_, A_cols_, k_, B_nnz_, B_rows_, B_cols_) == 0);
      } else {
        std::cerr << "Unknown matrix type" << std::endl;
        exit(1);
      }
    }

private:
    void preLoopRequirements() override {
      if (print_) std::cout << "preLoopRequirements" << std::endl;
      switch (offload_) {
        case gpuOffloadType::always: {
          // Nothing to do, does it all in the callSpmm loop
          break;
        }
        case gpuOffloadType::once: {
          if (print_) std::cout << "\tCopying A to device" << std::endl;
          auto ARows = queue_.copy<int64_t>(A_rows_, A_rows_device_, m_ + 1);
          auto ACols = queue_.copy<int64_t>(A_cols_, A_cols_device_, A_nnz_);
          auto AVals = queue_.copy<T>(A_vals_, A_vals_device_, A_nnz_);

          if (print_) std::cout << "\tCopying B to device" << std::endl;
          auto BRows = queue_.copy<int64_t>(B_rows_, B_rows_device_, k_ + 1);
          auto BCols = queue_.copy<int64_t>(B_cols_, B_cols_device_, B_nnz_);
          auto BVals = queue_.copy<T>(B_vals_, B_vals_device_, B_nnz_);

          ARows.wait();
          ACols.wait();
          AVals.wait();
          BRows.wait();
          BCols.wait();
          BVals.wait();
          break;
        }
        case gpuOffloadType::unified: {
          // Nothing to do here as shared memory
          break;
        }
      }
    }

    void callSpmm() override {
      if (print_) std::cout << "callSpmm" << std::endl;
      switch (offload_) {
        case gpuOffloadType::always: {
          if (print_) std::cout << "\tCopying A to device" << std::endl;
          auto ARows = queue_.copy<int64_t>(A_rows_, A_rows_device_, m_ + 1);
          auto ACols = queue_.copy<int64_t>(A_cols_, A_cols_device_, A_nnz_);
          auto AVals = queue_.copy<T>(A_vals_, A_vals_device_, A_nnz_);

          if (print_) std::cout << "\tCopying B to device" << std::endl;
          auto BRows = queue_.copy<int64_t>(B_rows_, B_rows_device_, k_ + 1);
          auto BCols = queue_.copy<int64_t>(B_cols_, B_cols_device_, B_nnz_);
          auto BVals = queue_.copy<T>(B_vals_, B_vals_device_, B_nnz_);

          if (print_) std::cout << "\tAllocating device memory for C rows" << std::endl;
          if (C_rows_device_ != nullptr) {
            if (print_) std::cout << "\t\tFreeing old C rows" << std::endl;
            sycl::free(C_rows_device_, queue_);
            C_rows_device_ = nullptr;
          }
          C_rows_device_ = sycl::malloc_device<int64_t>(m_ + 1, queue_);

          if (print_) std::cout << "\tMaking handles for matrices" << std::endl;
          oneapi::mkl::sparse::init_matrix_handle(&A_handle_);
          oneapi::mkl::sparse::init_matrix_handle(&B_handle_);
          oneapi::mkl::sparse::init_matrix_handle(&C_handle_);

          if (print_) std::cout << "\tSeting CSR arrays for matrix handles" << std::endl;
          auto setA = oneapi::mkl::sparse::set_csr_data(queue_,
                                                        A_handle_,
                                                        m_,
                                                        k_,
                                                        AIndex_,
                                                        A_rows_device_,
                                                        A_cols_device_,
                                                        A_vals_device_,
                                                        {ARows, ACols, AVals});
          auto setB = oneapi::mkl::sparse::set_csr_data(queue_,
                                                        B_handle_,
                                                        k_,
                                                        n_,
                                                        BIndex_,
                                                        B_rows_device_,
                                                        B_cols_device_,
                                                        B_vals_device_,
                                                        {BRows, BCols, BVals});
          auto setC = oneapi::mkl::sparse::set_csr_data(queue_,
                                                        C_handle_,
                                                        m_,
                                                        n_,
                                                        CIndex_,
                                                        C_rows_device_,
                                                        (int64_t*)nullptr,
                                                        (T*)nullptr,
                                                        {});

          if (print_) std::cout << "\tInitialising descriptor" << std::endl;
          oneapi::mkl::sparse::init_matmat_descr(&description_);

          if (print_) std::cout << "\tSetting descriptor metadata" << std::endl;
          oneapi::mkl::sparse::set_matmat_data(description_,
                                               viewA_,
                                               opA_,
                                               viewB_,
                                               opB_,
                                               viewC_);
          
          if (print_) std::cout << "\tQuerying size of work estimation buffer" << std::endl;
          request_ = oneapi::mkl::sparse::matmat_request::get_work_estimation_buf_size;
          sizeTempBuffer = sycl::malloc_host<int64_t>(1, queue_);
          if (!sizeTempBuffer) throw std::runtime_error("Could not allocate memory");
          auto ev1_1 = oneapi::mkl::sparse::matmat(queue_,
                                                   A_handle_,
                                                   B_handle_,
                                                   C_handle_,
                                                   request_,
                                                   description_,
                                                   sizeTempBuffer,
                                                   nullptr,
                                                   {setA, setB, setC});
          ev1_1.wait();

          if (print_) std::cout << "\tAllocating work estimation buffer" << std::endl;
          tempBuffer = sycl::malloc_device<uint8_t>(sizeTempBuffer[0], queue_);
          if (!tempBuffer) throw std::runtime_error("Could not allocate memory");

          if (print_) std::cout << "\tDo work estimation" << std::endl;
          request_ = oneapi::mkl::sparse::matmat_request::work_estimation;
          auto ev1_3 = oneapi::mkl::sparse::matmat(queue_,
                                                   A_handle_,
                                                   B_handle_,
                                                   C_handle_,
                                                   request_,
                                                   description_,
                                                   sizeTempBuffer,
                                                   tempBuffer,
                                                   {ev1_1});

          if (print_) std::cout << "\tQuerying size of compute buffer" << std::endl;
          request_ = oneapi::mkl::sparse::matmat_request::get_compute_buf_size;
          sizeTempBuffer = sycl::malloc_host<int64_t>(1, queue_);
          if (!sizeTempBuffer) throw std::runtime_error("Could not allocate memory");
          auto ev2_1 = oneapi::mkl::sparse::matmat(queue_,
                                                   A_handle_,
                                                   B_handle_,
                                                   C_handle_,
                                                   request_,
                                                   description_,
                                                   sizeTempBuffer,
                                                   nullptr,
                                                   {ev1_3});
          ev2_1.wait();

          if (print_) std::cout << "\tAllocating compute buffer" << std::endl;
          tempBuffer2 = sycl::malloc_device<uint8_t>(sizeTempBuffer[0], queue_);
          if (!tempBuffer2) throw std::runtime_error("Could not allocate memory");

          if (print_) std::cout << "\tDo compute" << std::endl;
          request_ = oneapi::mkl::sparse::matmat_request::compute;
          auto ev2_3 = oneapi::mkl::sparse::matmat(queue_,
                                                   A_handle_,
                                                   B_handle_,
                                                   C_handle_,
                                                   request_,
                                                   description_,
                                                   sizeTempBuffer,
                                                   tempBuffer2,
                                                   {ev2_1});

          if (print_) std::cout << "\tGetting nnz" << std::endl;
          request_ = oneapi::mkl::sparse::matmat_request::get_nnz;
          cNnzBuffer = sycl::malloc_host<int64_t>(1, queue_);
          if (!cNnzBuffer) throw std::runtime_error("Could not allocate memory");
          auto ev3_1 = oneapi::mkl::sparse::matmat(queue_,
                                                   A_handle_,
                                                   B_handle_,
                                                   C_handle_,
                                                   request_,
                                                   description_,
                                                   cNnzBuffer,
                                                   nullptr,
                                                   {ev2_3});
          ev3_1.wait();

          if (print_) std::cout << "\tCopying C_nnz_ and allocating cols and vals for C on device" << std::endl;
          C_nnz_ = cNnzBuffer[0];
          if (C_cols_device_) sycl::free(C_cols_device_, queue_);
          C_cols_device_ = sycl::malloc_device<int64_t>(C_nnz_, queue_);
          if (!C_cols_device_) throw std::runtime_error("Could not allocate memory");
          if (C_vals_device_) sycl::free(C_vals_device_, queue_);
          C_vals_device_ = sycl::malloc_device<T>(C_nnz_, queue_);
          if (!C_vals_device_) throw std::runtime_error("Could not allocate memory");

          if (print_) std::cout << "\tSetting C csr arrays" << std::endl;
          setC = oneapi::mkl::sparse::set_csr_data(queue_,
                                                        C_handle_,
                                                        m_,
                                                        n_,
                                                        CIndex_,
                                                        C_rows_device_,
                                                        C_cols_device_,
                                                        C_vals_device_,
                                                        {ev3_1});

          if (print_) std::cout << "\tFinalising" << std::endl;
          request_ = oneapi::mkl::sparse::matmat_request::finalize;
          auto ev3_3 = oneapi::mkl::sparse::matmat(queue_,
                                                   A_handle_,
                                                   B_handle_,
                                                   C_handle_,
                                                   request_,
                                                   description_,
                                                   nullptr,
                                                   nullptr,
                                                   {setC});

          if (print_) std::cout << "\tSorting C" << std::endl;
          auto ev_sort = oneapi::mkl::sparse::sort_matrix(queue_, C_handle_, {ev3_3});

          if (print_) std::cout << "\tAllocate host CSR arrays for C" << std::endl;
          if (C_rows_ != nullptr) {
            if (print_) std::cout << "\t\tFreeing old C rows" << std::endl;
            sycl::free(C_rows_, queue_);
          }
          if (C_cols_ != nullptr) {
            if (print_) std::cout << "\t\tFreeing old C cols" << std::endl;
            sycl::free(C_cols_, queue_);
          }
          if (C_vals_ != nullptr) {
            if (print_) std::cout << "\t\tFreeing old C vals" << std::endl;
            sycl::free(C_vals_, queue_);
          }
          if (print_) std::cout << "\t\tAllocating C rows" << std::endl;
          C_rows_ = sycl::malloc_host<int64_t>(m_ + 1, queue_);
          if (print_) std::cout << "\t\tAllocating C cols" << std::endl;
          C_cols_ = sycl::malloc_host<int64_t>(C_nnz_, queue_);
          if (print_) std::cout << "\t\tAllocating C vals" << std::endl;
          C_vals_ = sycl::malloc_host<T>(C_nnz_, queue_);

          if (print_) std::cout << "\tCopying C back to host" << std::endl;
          auto CRows = queue_.copy<int64_t>(C_rows_device_, C_rows_, m_ + 1);
          auto CCols = queue_.copy<int64_t>(C_cols_device_, C_cols_, C_nnz_);
          auto CVals = queue_.copy<T>(C_vals_device_, C_vals_, C_nnz_);
          CRows.wait();
          CCols.wait();
          CVals.wait();

          if (print_) std::cout << "\tRelease handles" << std::endl;
          oneapi::mkl::sparse::release_matmat_descr(&description_);
          oneapi::mkl::sparse::release_matrix_handle(queue_, &A_handle_).wait();
          oneapi::mkl::sparse::release_matrix_handle(queue_, &B_handle_).wait();
          oneapi::mkl::sparse::release_matrix_handle(queue_, &C_handle_).wait();

          if (C_rows_device_ != nullptr) sycl::free(C_rows_device_, queue_);
          if (C_cols_device_ != nullptr) sycl::free(C_cols_device_, queue_);
          if (C_vals_device_ != nullptr) sycl::free(C_vals_device_, queue_);
          if (sizeTempBuffer != nullptr) sycl::free(sizeTempBuffer, queue_);
          if (sizeTempBuffer2 != nullptr) sycl::free(sizeTempBuffer2, queue_);
          if (tempBuffer != nullptr) sycl::free(tempBuffer, queue_);
          if (tempBuffer2 != nullptr) sycl::free(tempBuffer2, queue_);
          if (cNnzBuffer != nullptr) sycl::free(cNnzBuffer, queue_);
          break;
        }
        case gpuOffloadType::once: {
          // If already allocated, free the device C arrays
          if (C_rows_device_ != nullptr) sycl::free(C_rows_device_, queue_);
          if (C_cols_device_ != nullptr) sycl::free(C_cols_device_, queue_);
          if (C_vals_device_ != nullptr) sycl::free(C_vals_device_, queue_);
          
          if (print_) std::cout << "\tAllocating device memory for C rows" << std::endl;
          C_rows_device_ = sycl::malloc_device<int64_t>(m_ + 1, queue_);

          if (print_) std::cout << "\tMaking handles for matrices" << std::endl;
          oneapi::mkl::sparse::init_matrix_handle(&A_handle_);
          oneapi::mkl::sparse::init_matrix_handle(&B_handle_);
          oneapi::mkl::sparse::init_matrix_handle(&C_handle_);

          if (print_) std::cout << "\tSeting CSR arrays for matrix handles" << std::endl;
          auto setA = oneapi::mkl::sparse::set_csr_data(queue_,
                                                        A_handle_,
                                                        m_,
                                                        k_,
                                                        AIndex_,
                                                        A_rows_device_,
                                                        A_cols_device_,
                                                        A_vals_device_,
                                                        {});
          auto setB = oneapi::mkl::sparse::set_csr_data(queue_,
                                                        B_handle_,
                                                        k_,
                                                        n_,
                                                        BIndex_,
                                                        B_rows_device_,
                                                        B_cols_device_,
                                                        B_vals_device_,
                                                        {});
          auto setC = oneapi::mkl::sparse::set_csr_data(queue_,
                                                        C_handle_,
                                                        m_,
                                                        n_,
                                                        CIndex_,
                                                        C_rows_device_,
                                                        (int64_t*)nullptr,
                                                        (T*)nullptr,
                                                        {});

          if (print_) std::cout << "\tInitialising descriptor" << std::endl;
          oneapi::mkl::sparse::init_matmat_descr(&description_);

          if (print_) std::cout << "\tSetting descriptor metadata" << std::endl;
          oneapi::mkl::sparse::set_matmat_data(description_,
                                               viewA_,
                                               opA_,
                                               viewB_,
                                               opB_,
                                               viewC_);
          
          if (print_) std::cout << "\tQuerying size of work estimation buffer" << std::endl;
          request_ = oneapi::mkl::sparse::matmat_request::get_work_estimation_buf_size;
          sizeTempBuffer = sycl::malloc_host<int64_t>(1, queue_);
          if (!sizeTempBuffer) throw std::runtime_error("Could not allocate memory");
          auto ev1_1 = oneapi::mkl::sparse::matmat(queue_,
                                                   A_handle_,
                                                   B_handle_,
                                                   C_handle_,
                                                   request_,
                                                   description_,
                                                   sizeTempBuffer,
                                                   nullptr,
                                                   {setA, setB, setC});
          ev1_1.wait();

          if (print_) std::cout << "\tAllocating work estimation buffer" << std::endl;
          tempBuffer = sycl::malloc_device<uint8_t>(sizeTempBuffer[0], queue_);
          if (!tempBuffer) throw std::runtime_error("Could not allocate memory");

          if (print_) std::cout << "\tDo work estimation" << std::endl;
          request_ = oneapi::mkl::sparse::matmat_request::work_estimation;
          auto ev1_3 = oneapi::mkl::sparse::matmat(queue_,
                                                   A_handle_,
                                                   B_handle_,
                                                   C_handle_,
                                                   request_,
                                                   description_,
                                                   sizeTempBuffer,
                                                   tempBuffer,
                                                   {ev1_1});

          if (print_) std::cout << "\tQuerying size of compute buffer" << std::endl;
          request_ = oneapi::mkl::sparse::matmat_request::get_compute_buf_size;
          if (print_) std::cout << "\t\tAllocating temp buffer" << std::endl;
          sizeTempBuffer = sycl::malloc_host<int64_t>(1, queue_);
          if (!sizeTempBuffer) throw std::runtime_error("Could not allocate memory");
          if (print_) std::cout << "\t\tCalling matmat" << std::endl;
          auto ev2_1 = oneapi::mkl::sparse::matmat(queue_,
                                                   A_handle_,
                                                   B_handle_,
                                                   C_handle_,
                                                   request_,
                                                   description_,
                                                   sizeTempBuffer,
                                                   nullptr,
                                                   {ev1_3});
          ev2_1.wait();

          if (print_) std::cout << "\tAllocating compute buffer" << std::endl;
          tempBuffer2 = sycl::malloc_device<uint8_t>(sizeTempBuffer[0], queue_);
          if (!tempBuffer2) throw std::runtime_error("Could not allocate memory");

          if (print_) std::cout << "\tDo compute" << std::endl;
          request_ = oneapi::mkl::sparse::matmat_request::compute;
          auto ev2_3 = oneapi::mkl::sparse::matmat(queue_,
                                                   A_handle_,
                                                   B_handle_,
                                                   C_handle_,
                                                   request_,
                                                   description_,
                                                   sizeTempBuffer,
                                                   tempBuffer2,
                                                   {ev2_1});

          if (print_) std::cout << "\tGetting nnz" << std::endl;
          request_ = oneapi::mkl::sparse::matmat_request::get_nnz;
          cNnzBuffer = sycl::malloc_host<int64_t>(1, queue_);
          if (!cNnzBuffer) throw std::runtime_error("Could not allocate memory");
          auto ev3_1 = oneapi::mkl::sparse::matmat(queue_,
                                                   A_handle_,
                                                   B_handle_,
                                                   C_handle_,
                                                   request_,
                                                   description_,
                                                   cNnzBuffer,
                                                   nullptr,
                                                   {ev2_3});
          ev3_1.wait();

          if (print_) std::cout << "\tCopying C_nnz_ and allocating cols and vals for C on device" << std::endl;
          C_nnz_ = cNnzBuffer[0];
          if (C_cols_device_) sycl::free(C_cols_device_, queue_);
          C_cols_device_ = sycl::malloc_device<int64_t>(C_nnz_, queue_);
          if (!C_cols_device_) throw std::runtime_error("Could not allocate memory");
          if (C_vals_device_) sycl::free(C_vals_device_, queue_);
          C_vals_device_ = sycl::malloc_device<T>(C_nnz_, queue_);
          if (!C_vals_device_) throw std::runtime_error("Could not allocate memory");

          if (print_) std::cout << "\tSetting C csr arrays" << std::endl;
          setC = oneapi::mkl::sparse::set_csr_data(queue_,
                                                        C_handle_,
                                                        m_,
                                                        n_,
                                                        CIndex_,
                                                        C_rows_device_,
                                                        C_cols_device_,
                                                        C_vals_device_,
                                                        {ev3_1});

          if (print_) std::cout << "\tFinalising" << std::endl;
          request_ = oneapi::mkl::sparse::matmat_request::finalize;
          auto ev3_3 = oneapi::mkl::sparse::matmat(queue_,
                                                   A_handle_,
                                                   B_handle_,
                                                   C_handle_,
                                                   request_,
                                                   description_,
                                                   nullptr,
                                                   nullptr,
                                                   {setC});

          if (print_) std::cout << "\tSorting C" << std::endl;
          auto ev_sort = oneapi::mkl::sparse::sort_matrix(queue_, C_handle_, {ev3_3});

          if (print_) std::cout << "\tRelease handles" << std::endl;
          oneapi::mkl::sparse::release_matmat_descr(&description_);
          oneapi::mkl::sparse::release_matrix_handle(queue_, &A_handle_).wait();
          oneapi::mkl::sparse::release_matrix_handle(queue_, &B_handle_).wait();
          oneapi::mkl::sparse::release_matrix_handle(queue_, &C_handle_).wait();
          if (sizeTempBuffer != nullptr) sycl::free(sizeTempBuffer, queue_);
          if (sizeTempBuffer2 != nullptr) sycl::free(sizeTempBuffer2, queue_);
          if (tempBuffer != nullptr) sycl::free(tempBuffer, queue_);
          if (tempBuffer2 != nullptr) sycl::free(tempBuffer2, queue_);
          if (cNnzBuffer != nullptr) sycl::free(cNnzBuffer, queue_);
          break;
        }
        case gpuOffloadType::unified: {
          break;
        }
      }
    }

    void postLoopRequirements() override {
      if (print_) std::cout << "postLoopRequirements" << std::endl;
      switch (offload_) {
        case gpuOffloadType::always: {
          break;
        }
        case gpuOffloadType::once: {
          if (print_) std::cout << "\tAllocate host CSR arrays for C" << std::endl;
          if (C_rows_ != nullptr) {
            if (print_) std::cout << "\t\tFreeing old C rows" << std::endl;
            sycl::free(C_rows_, queue_);
          }
          if (C_cols_ != nullptr) {
            if (print_) std::cout << "\t\tFreeing old C cols" << std::endl;
            sycl::free(C_cols_, queue_);
          }
          if (C_vals_ != nullptr) {
            if (print_) std::cout << "\t\tFreeing old C vals" << std::endl;
            sycl::free(C_vals_, queue_);
          }
          if (print_) std::cout << "\t\tAllocating C rows" << std::endl;
          C_rows_ = sycl::malloc_host<int64_t>(m_ + 1, queue_);
          if (print_) std::cout << "\t\tAllocating C cols" << std::endl;
          C_cols_ = sycl::malloc_host<int64_t>(C_nnz_, queue_);
          if (print_) std::cout << "\t\tAllocating C vals" << std::endl;
          C_vals_ = sycl::malloc_host<T>(C_nnz_, queue_);

          if (print_) std::cout << "\tCopying C back to host" << std::endl;
          auto CRows = queue_.copy<int64_t>(C_rows_device_, C_rows_, m_ + 1);
          auto CCols = queue_.copy<int64_t>(C_cols_device_, C_cols_, C_nnz_);
          auto CVals = queue_.copy<T>(C_vals_device_, C_vals_, C_nnz_);
          CRows.wait();
          CCols.wait();
          CVals.wait();

          if (C_rows_device_ != nullptr) sycl::free(C_rows_device_, queue_);
          if (C_cols_device_ != nullptr) sycl::free(C_cols_device_, queue_);
          if (C_vals_device_ != nullptr) sycl::free(C_vals_device_, queue_);
          break;
        }
        case gpuOffloadType::unified: {
          if (C_rows_ != nullptr) sycl::free(C_rows_, queue_);
          if (C_cols_ != nullptr) sycl::free(C_cols_, queue_);
          if (C_vals_ != nullptr) sycl::free(C_vals_, queue_);

          if (print_) std::cout << "\tAllocating C rows array" << std::endl;
          C_rows_ = sycl::malloc_shared<int64_t>(m_ + 1, queue_);

          if (print_) std::cout << "\tMaking handles for matrices" << std::endl;
          oneapi::mkl::sparse::init_matrix_handle(&A_handle_);
          oneapi::mkl::sparse::init_matrix_handle(&B_handle_);
          oneapi::mkl::sparse::init_matrix_handle(&C_handle_);

          if (print_) std::cout << "\tSeting CSR arrays for matrix handles" << std::endl;
          auto setA = oneapi::mkl::sparse::set_csr_data(queue_,
                                                        A_handle_,
                                                        m_,
                                                        k_,
                                                        AIndex_,
                                                        A_rows_,
                                                        A_cols_,
                                                        A_vals_,
                                                        {});
          auto setB = oneapi::mkl::sparse::set_csr_data(queue_,
                                                        B_handle_,
                                                        k_,
                                                        n_,
                                                        BIndex_,
                                                        B_rows_,
                                                        B_cols_,
                                                        B_vals_,
                                                        {});
          auto setC = oneapi::mkl::sparse::set_csr_data(queue_,
                                                        C_handle_,
                                                        m_,
                                                        n_,
                                                        CIndex_,
                                                        C_rows_,
                                                        (int64_t*)nullptr,
                                                        (T*)nullptr,
                                                        {});

          if (print_) std::cout << "\tInitialising descriptor" << std::endl;
          oneapi::mkl::sparse::init_matmat_descr(&description_);

          if (print_) std::cout << "\tSetting descriptor metadata" << std::endl;
          oneapi::mkl::sparse::set_matmat_data(description_,
                                               viewA_,
                                               opA_,
                                               viewB_,
                                               opB_,
                                               viewC_);
          
          if (print_) std::cout << "\tQuerying size of work estimation buffer" << std::endl;
          request_ = oneapi::mkl::sparse::matmat_request::get_work_estimation_buf_size;
          sizeTempBuffer = sycl::malloc_host<int64_t>(1, queue_);
          if (!sizeTempBuffer) throw std::runtime_error("Could not allocate memory");
          auto ev1_1 = oneapi::mkl::sparse::matmat(queue_,
                                                   A_handle_,
                                                   B_handle_,
                                                   C_handle_,
                                                   request_,
                                                   description_,
                                                   sizeTempBuffer,
                                                   nullptr,
                                                   {setA, setB, setC});
          ev1_1.wait();

          if (print_) std::cout << "\tAllocating work estimation buffer" << std::endl;
          tempBuffer = sycl::malloc_device<uint8_t>(sizeTempBuffer[0], queue_);
          if (!tempBuffer) throw std::runtime_error("Could not allocate memory");

          if (print_) std::cout << "\tDo work estimation" << std::endl;
          request_ = oneapi::mkl::sparse::matmat_request::work_estimation;
          auto ev1_3 = oneapi::mkl::sparse::matmat(queue_,
                                                   A_handle_,
                                                   B_handle_,
                                                   C_handle_,
                                                   request_,
                                                   description_,
                                                   sizeTempBuffer,
                                                   tempBuffer,
                                                   {ev1_1});

          if (print_) std::cout << "\tQuerying size of compute buffer" << std::endl;
          request_ = oneapi::mkl::sparse::matmat_request::get_compute_buf_size;
          sizeTempBuffer = sycl::malloc_host<int64_t>(1, queue_);
          if (!sizeTempBuffer) throw std::runtime_error("Could not allocate memory");
          auto ev2_1 = oneapi::mkl::sparse::matmat(queue_,
                                                   A_handle_,
                                                   B_handle_,
                                                   C_handle_,
                                                   request_,
                                                   description_,
                                                   sizeTempBuffer,
                                                   nullptr,
                                                   {ev1_3});
          ev2_1.wait();

          if (print_) std::cout << "\tAllocating compute buffer" << std::endl;
          tempBuffer2 = sycl::malloc_device<uint8_t>(sizeTempBuffer[0], queue_);
          if (!tempBuffer2) throw std::runtime_error("Could not allocate memory");

          if (print_) std::cout << "\tDo compute" << std::endl;
          request_ = oneapi::mkl::sparse::matmat_request::compute;
          auto ev2_3 = oneapi::mkl::sparse::matmat(queue_,
                                                   A_handle_,
                                                   B_handle_,
                                                   C_handle_,
                                                   request_,
                                                   description_,
                                                   sizeTempBuffer,
                                                   tempBuffer2,
                                                   {ev2_1});

          if (print_) std::cout << "\tGetting nnz" << std::endl;
          request_ = oneapi::mkl::sparse::matmat_request::get_nnz;
          cNnzBuffer = sycl::malloc_host<int64_t>(1, queue_);
          if (!cNnzBuffer) throw std::runtime_error("Could not allocate memory");
          auto ev3_1 = oneapi::mkl::sparse::matmat(queue_,
                                                   A_handle_,
                                                   B_handle_,
                                                   C_handle_,
                                                   request_,
                                                   description_,
                                                   cNnzBuffer,
                                                   nullptr,
                                                   {ev2_3});
          ev3_1.wait();

          if (print_) std::cout << "\tCopying C_nnz_ and allocating cols and vals for C on device" << std::endl;
          C_nnz_ = cNnzBuffer[0];
          C_cols_ = sycl::malloc_shared<int64_t>(C_nnz_, queue_);
          C_vals_ = sycl::malloc_shared<T>(C_nnz_, queue_);
          if (!C_cols_ || !C_vals_) throw std::runtime_error("Could not allocate memory");

          if (print_) std::cout << "\tSetting C csr arrays" << std::endl;
          setC = oneapi::mkl::sparse::set_csr_data(queue_,
                                                        C_handle_,
                                                        m_,
                                                        n_,
                                                        CIndex_,
                                                        C_rows_,
                                                        C_cols_,
                                                        C_vals_,
                                                        {ev3_1});

          if (print_) std::cout << "\tFinalising" << std::endl;
          request_ = oneapi::mkl::sparse::matmat_request::finalize;
          auto ev3_3 = oneapi::mkl::sparse::matmat(queue_,
                                                   A_handle_,
                                                   B_handle_,
                                                   C_handle_,
                                                   request_,
                                                   description_,
                                                   nullptr,
                                                   nullptr,
                                                   {setC});

          if (print_) std::cout << "\tSorting C" << std::endl;
          auto ev_sort = oneapi::mkl::sparse::sort_matrix(queue_, C_handle_, {ev3_3});

          if (print_) std::cout << "\tRelease handles" << std::endl;
          oneapi::mkl::sparse::release_matmat_descr(&description_);
          oneapi::mkl::sparse::release_matrix_handle(queue_, &A_handle_).wait();
          oneapi::mkl::sparse::release_matrix_handle(queue_, &B_handle_).wait();
          oneapi::mkl::sparse::release_matrix_handle(queue_, &C_handle_).wait();
          if (sizeTempBuffer != nullptr) sycl::free(sizeTempBuffer, queue_);
          if (sizeTempBuffer2 != nullptr) sycl::free(sizeTempBuffer2, queue_);
          if (tempBuffer != nullptr) sycl::free(tempBuffer, queue_);
          if (tempBuffer2 != nullptr) sycl::free(tempBuffer2, queue_);
          if (cNnzBuffer != nullptr) sycl::free(cNnzBuffer, queue_);
          break;
        }
      }
    }

    void postCallKernelCleanup() override {
      if (print_) std::cout << "postCallKernelCleanup" << std::endl;
      switch (offload_) {
        case gpuOffloadType::always: {
          if (A_rows_ != nullptr) sycl::free(A_rows_, queue_);
          if (A_cols_ != nullptr) sycl::free(A_cols_, queue_);
          if (A_vals_ != nullptr) sycl::free(A_vals_, queue_);
          if (A_rows_device_ != nullptr) sycl::free(A_rows_device_, queue_);
          if (A_cols_device_ != nullptr) sycl::free(A_cols_device_, queue_);
          if (A_vals_device_ != nullptr) sycl::free(A_vals_device_, queue_);

          if (B_rows_ != nullptr) sycl::free(B_rows_, queue_);
          if (B_cols_ != nullptr) sycl::free(B_cols_, queue_);
          if (B_vals_ != nullptr) sycl::free(B_vals_, queue_);
          if (B_rows_device_ != nullptr) sycl::free(B_rows_device_, queue_);
          if (B_cols_device_ != nullptr) sycl::free(B_cols_device_, queue_);
          if (B_vals_device_ != nullptr) sycl::free(B_vals_device_, queue_);

          if (C_rows_ != nullptr) sycl::free(C_rows_, queue_);
          if (C_cols_ != nullptr) sycl::free(C_cols_, queue_);
          if (C_vals_ != nullptr) sycl::free(C_vals_, queue_);
          break;
        }
        case gpuOffloadType::once: {
          if (A_rows_ != nullptr) sycl::free(A_rows_, queue_);
          if (A_cols_ != nullptr) sycl::free(A_cols_, queue_);
          if (A_vals_ != nullptr) sycl::free(A_vals_, queue_);
          if (A_rows_device_ != nullptr) sycl::free(A_rows_device_, queue_);
          if (A_cols_device_ != nullptr) sycl::free(A_cols_device_, queue_);
          if (A_vals_device_ != nullptr) sycl::free(A_vals_device_, queue_);

          if (B_rows_ != nullptr) sycl::free(B_rows_, queue_);
          if (B_cols_ != nullptr) sycl::free(B_cols_, queue_);
          if (B_vals_ != nullptr) sycl::free(B_vals_, queue_);
          if (B_rows_device_ != nullptr) sycl::free(B_rows_device_, queue_);
          if (B_cols_device_ != nullptr) sycl::free(B_cols_device_, queue_);
          if (B_vals_device_ != nullptr) sycl::free(B_vals_device_, queue_);

          if (C_rows_ != nullptr) sycl::free(C_rows_, queue_);
          if (C_cols_ != nullptr) sycl::free(C_cols_, queue_);
          if (C_vals_ != nullptr) sycl::free(C_vals_, queue_);
          break;
        }
        case gpuOffloadType::unified: {
          if (A_rows_ != nullptr) sycl::free(A_rows_, queue_);
          if (A_cols_ != nullptr) sycl::free(A_cols_, queue_);
          if (A_vals_ != nullptr) sycl::free(A_vals_, queue_);

          if (B_rows_ != nullptr) sycl::free(B_rows_, queue_);
          if (B_cols_ != nullptr) sycl::free(B_cols_, queue_);
          if (B_vals_ != nullptr) sycl::free(B_vals_, queue_);
          
          if (C_rows_ != nullptr) sycl::free(C_rows_, queue_);
          if (C_cols_ != nullptr) sycl::free(C_cols_, queue_);
          if (C_vals_ != nullptr) sycl::free(C_vals_, queue_);
          break;
        }
      }
    }

    void printInputMatrices() {
      std::cout << "---------------------------------------------" << std::endl;
      std::cout << "Matrix A" << std::endl;
      std::cout << "\tRows: [";
      for (int64_t i = 0; i < m_ + 1; i++) {
        std::cout << A_rows_[i];
        if (i < m_) std::cout << ", ";
      }
      std::cout << "]" << std::endl;
      std::cout << "\tCols: [";
      for (int64_t i = 0; i < A_nnz_; i++) {
        std::cout << A_cols_[i];
        if (i < A_nnz_ - 1) std::cout << ", ";
      }
      std::cout << "]" << std::endl;
      std::cout << "\tVals: [";
      for (int64_t i = 0; i < A_nnz_; i++) {
        std::cout << A_vals_[i];
        if (i < A_nnz_ - 1) std::cout << ", ";
      }
      std::cout << "]" << std::endl;
      std::cout << "Matrix B" << std::endl;
      std::cout << "\tRows: [";
      for (int64_t i = 0; i < k_ + 1; i++) {
        std::cout << B_rows_[i];
        if (i < k_) std::cout << ", ";
      }
      std::cout << "]" << std::endl;
      std::cout << "\tCols: [";
      for (int64_t i = 0; i < B_nnz_; i++) {
        std::cout << B_cols_[i];
        if (i < B_nnz_ - 1) std::cout << ", ";
      }
      std::cout << "]" << std::endl;
      std::cout << "\tVals: [";
      for (int64_t i = 0; i < B_nnz_; i++) {
        std::cout << B_vals_[i];
        if (i < B_nnz_ - 1) std::cout << ", ";
      }
      std::cout << "]" << std::endl;
      std::cout << "---------------------------------------------" << std::endl;
    }

    // Debugging output switch
    bool print_ = true;

    // Sycl parameters
    sycl::queue queue_;
    sycl::device device_;
    sycl::context context_;

    // oneMKL parameters
    oneapi::mkl::transpose opA_ = oneapi::mkl::transpose::nontrans;
    oneapi::mkl::transpose opB_ = oneapi::mkl::transpose::nontrans;

    oneapi::mkl::sparse::matrix_view_descr viewA_ = oneapi::mkl::sparse::matrix_view_descr::general;
    oneapi::mkl::sparse::matrix_view_descr viewB_ = oneapi::mkl::sparse::matrix_view_descr::general;
    oneapi::mkl::sparse::matrix_view_descr viewC_ = oneapi::mkl::sparse::matrix_view_descr::general;

    oneapi::mkl::index_base AIndex_ = oneapi::mkl::index_base::zero;
    oneapi::mkl::index_base BIndex_ = oneapi::mkl::index_base::zero;
    oneapi::mkl::index_base CIndex_ = oneapi::mkl::index_base::zero;

    oneapi::mkl::sparse::matrix_handle_t A_handle_ = nullptr;
    oneapi::mkl::sparse::matrix_handle_t B_handle_ = nullptr;
    oneapi::mkl::sparse::matrix_handle_t C_handle_ = nullptr;

    oneapi::mkl::sparse::matmat_descr_t description_ = nullptr;
    oneapi::mkl::sparse::matmat_request request_;

    // A CSR arrays
    //    LOCAL
    int64_t* A_rows_ = nullptr;
    int64_t* A_cols_ = nullptr;
    T* A_vals_ = nullptr;
    //    DEVICE
    int64_t* A_rows_device_ = nullptr;
    int64_t* A_cols_device_ = nullptr;
    T* A_vals_device_ = nullptr;

    // B CSR arrays
    //    LOCAL
    int64_t* B_rows_ = nullptr;
    int64_t* B_cols_ = nullptr;
    T* B_vals_ = nullptr;
    //    DEVICE
    int64_t* B_rows_device_ = nullptr;
    int64_t* B_cols_device_ = nullptr;
    T* B_vals_device_ = nullptr;

    // C CSR arrays
    //    LOCAL -- carried through from parent class -- needed externally for checksum
    //    DEVICE
    int64_t* C_rows_device_ = nullptr;
    int64_t* C_cols_device_ = nullptr;
    T* C_vals_device_ = nullptr;

    // Temporary buffers
    int64_t* sizeTempBuffer = nullptr;
    int64_t* sizeTempBuffer2 = nullptr;
    int64_t* cNnzBuffer = nullptr;
    void* tempBuffer = nullptr;
    void* tempBuffer2 = nullptr;

    const T alpha = ALPHA;
    const T beta = BETA;
};
}

#endif
