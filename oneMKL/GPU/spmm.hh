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
      firstRun_ = true;
      if (!initialised_) {
        // Set up the sycl parameters
        device_ = sycl::device(sycl::gpu_selector_v);
        queue_ = sycl::queue(device_, exception_handler);
        context_ = queue_.get_context();
        auto dev = queue_.get_device();
        initialised_ = true;
      }

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

      switch (offload_) {
        case gpuOffloadType::always: {
          A_rows_store_ = (int64_t*)malloc(static_cast<size_t>(m_ + 1) * sizeof(int64_t));
          A_cols_store_ = (int64_t*)malloc(static_cast<size_t>(A_nnz_) * sizeof(int64_t));
          A_vals_store_ = (T*)malloc(static_cast<size_t>(A_nnz_) * sizeof(T));
          B_rows_store_ = (int64_t*)malloc(static_cast<size_t>(k_ + 1) * sizeof(int64_t));
          B_cols_store_ = (int64_t*)malloc(static_cast<size_t>(B_nnz_) * sizeof(int64_t));
          B_vals_store_ = (T*)malloc(static_cast<size_t>(B_nnz_) * sizeof(T));

          A_rows_ = sycl::malloc_host<int64_t>(static_cast<size_t>(m_ + 1), queue_);
          A_cols_ = sycl::malloc_host<int64_t>(static_cast<size_t>(A_nnz_), queue_);
          A_vals_ = sycl::malloc_host<T>(static_cast<size_t>(A_nnz_), queue_);
          A_rows_device_ = sycl::malloc_device<int64_t>(static_cast<size_t>(m_ + 1), queue_);
          A_cols_device_ = sycl::malloc_device<int64_t>(static_cast<size_t>(A_nnz_), queue_);
          A_vals_device_ = sycl::malloc_device<T>(static_cast<size_t>(A_nnz_), queue_);

          B_rows_ = sycl::malloc_host<int64_t>(static_cast<size_t>(k_ + 1), queue_);
          B_cols_ = sycl::malloc_host<int64_t>(static_cast<size_t>(B_nnz_), queue_);
          B_vals_ = sycl::malloc_host<T>(static_cast<size_t>(B_nnz_), queue_);
          B_rows_device_ = sycl::malloc_device<int64_t>(static_cast<size_t>(k_ + 1), queue_);
          B_cols_device_ = sycl::malloc_device<int64_t>(static_cast<size_t>(B_nnz_), queue_);
          B_vals_device_ = sycl::malloc_device<T>(static_cast<size_t>(B_nnz_), queue_);

          C_rows_ = nullptr;
          C_cols_ = nullptr;
          C_vals_ = nullptr;
          C_rows_device_ = nullptr;
          C_cols_device_ = nullptr;
          C_vals_device_ = nullptr;
          break;
        }
        case gpuOffloadType::once: {
          A_rows_ = sycl::malloc_host<int64_t>(static_cast<size_t>(m_ + 1), queue_);
          A_cols_ = sycl::malloc_host<int64_t>(static_cast<size_t>(A_nnz_), queue_);
          A_vals_ = sycl::malloc_host<T>(static_cast<size_t>(A_nnz_), queue_);
          A_rows_device_ = sycl::malloc_device<int64_t>(static_cast<size_t>(m_ + 1), queue_);
          A_cols_device_ = sycl::malloc_device<int64_t>(static_cast<size_t>(A_nnz_), queue_);
          A_vals_device_ = sycl::malloc_device<T>(static_cast<size_t>(A_nnz_), queue_);

          B_rows_ = sycl::malloc_host<int64_t>(static_cast<size_t>(k_ + 1), queue_);
          B_cols_ = sycl::malloc_host<int64_t>(static_cast<size_t>(B_nnz_), queue_);
          B_vals_ = sycl::malloc_host<T>(static_cast<size_t>(B_nnz_), queue_);
          B_rows_device_ = sycl::malloc_device<int64_t>(static_cast<size_t>(k_ + 1), queue_);
          B_cols_device_ = sycl::malloc_device<int64_t>(static_cast<size_t>(B_nnz_), queue_);
          B_vals_device_ = sycl::malloc_device<T>(static_cast<size_t>(B_nnz_), queue_);

          C_rows_ = nullptr;
          C_cols_ = nullptr;
          C_vals_ = nullptr;
          C_rows_device_ = nullptr;
          C_cols_device_ = nullptr;
          C_vals_device_ = nullptr;
          break;
        }
        case gpuOffloadType::unified: {
          A_rows_ = sycl::malloc_shared<int64_t>(static_cast<size_t>(m_ + 1), queue_);
          A_cols_ = sycl::malloc_shared<int64_t>(static_cast<size_t>(A_nnz_), queue_);
          A_vals_ = sycl::malloc_shared<T>(static_cast<size_t>(A_nnz_), queue_);

          B_rows_ = sycl::malloc_shared<int64_t>(static_cast<size_t>(k_ + 1), queue_);
          B_cols_ = sycl::malloc_shared<int64_t>(static_cast<size_t>(B_nnz_), queue_);
          B_vals_ = sycl::malloc_shared<T>(static_cast<size_t>(B_nnz_), queue_);

          C_rows_ = nullptr;
          C_cols_ = nullptr;
          C_vals_ = nullptr;
          break;
        }
      }
      queue_.wait_and_throw();
      initInputMatrices();
    }

protected:
    void toSparseFormat() override {
      if (offload_ == gpuOffloadType::always) {
        int seedOffset = 0;
        if (type_ == matrixType::rmat) {
          do {
            rMatCSR<T, int64_t>(A_vals_store_, A_cols_store_, A_rows_store_, m_, k_, A_nnz_, SEED + seedOffset++);
            rMatCSR<T, int64_t>(B_vals_store_, B_cols_store_, B_rows_store_, k_, n_, B_nnz_, SEED + seedOffset++);
          } while (calcCNNZ<int64_t>(m_, A_nnz_, A_rows_store_, A_cols_store_, k_, B_nnz_, B_rows_store_, B_cols_store_) == 0);
        } else if (type_ == matrixType::random) {
          do {
            randomCSR<T, int64_t>(A_vals_store_, A_cols_store_, A_rows_store_, m_, k_, A_nnz_, SEED + seedOffset++);
            randomCSR<T, int64_t>(B_vals_store_, B_cols_store_, B_rows_store_, k_, n_, B_nnz_, SEED + seedOffset++);
          } while (calcCNNZ<int64_t>(m_, A_nnz_, A_rows_store_, A_cols_store_, k_, B_nnz_, B_rows_store_, B_cols_store_) == 0);
        } else {
          std::cerr << "Unknown matrix type" << std::endl;
          exit(1);
        }
      }

      memcpy(A_rows_, A_rows_store_, static_cast<size_t>(m_ + 1) * sizeof(int64_t));
      memcpy(A_cols_, A_cols_store_, static_cast<size_t>(A_nnz_) * sizeof(int64_t));
      memcpy(A_vals_, A_vals_store_, static_cast<size_t>(A_nnz_) * sizeof(T));
      memcpy(B_rows_, B_rows_store_, static_cast<size_t>(k_ + 1) * sizeof(int64_t));
      memcpy(B_cols_, B_cols_store_, static_cast<size_t>(B_nnz_) * sizeof(int64_t));
      memcpy(B_vals_, B_vals_store_, static_cast<size_t>(B_nnz_) * sizeof(T));
    }

private:
    void preLoopRequirements() override {
      switch (offload_) {
        case gpuOffloadType::always: {
          // Nothing to do, does it all in the callSpmm loop
          break;
        }
        case gpuOffloadType::once: {
          auto ARows = queue_.copy<int64_t>(A_rows_, A_rows_device_, static_cast<size_t>(m_ + 1));
          auto ACols = queue_.copy<int64_t>(A_cols_, A_cols_device_, static_cast<size_t>(A_nnz_));
          auto AVals = queue_.copy<T>(A_vals_, A_vals_device_, static_cast<size_t>(A_nnz_));

          auto BRows = queue_.copy<int64_t>(B_rows_, B_rows_device_, static_cast<size_t>(k_ + 1));
          auto BCols = queue_.copy<int64_t>(B_cols_, B_cols_device_, static_cast<size_t>(B_nnz_));
          auto BVals = queue_.copy<T>(B_vals_, B_vals_device_, static_cast<size_t>(B_nnz_));

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
      switch (offload_) {
        case gpuOffloadType::always: {
          if (!firstRun_) {
            sycl::free(C_rows_, queue_);
            sycl::free(C_cols_, queue_);
            sycl::free(C_vals_, queue_);
          }

          auto ARows = queue_.copy<int64_t>(A_rows_, A_rows_device_, static_cast<size_t>(m_ + 1));
          auto ACols = queue_.copy<int64_t>(A_cols_, A_cols_device_, static_cast<size_t>(A_nnz_));
          auto AVals = queue_.copy<T>(A_vals_, A_vals_device_, static_cast<size_t>(A_nnz_));

          auto BRows = queue_.copy<int64_t>(B_rows_, B_rows_device_, static_cast<size_t>(k_ + 1));
          auto BCols = queue_.copy<int64_t>(B_cols_, B_cols_device_, static_cast<size_t>(B_nnz_));
          auto BVals = queue_.copy<T>(B_vals_, B_vals_device_, static_cast<size_t>(B_nnz_));

          C_rows_device_ = sycl::malloc_device<int64_t>(static_cast<size_t>(m_ + 1), queue_);

          try {
            oneapi::mkl::sparse::init_matrix_handle(&A_handle_);
            oneapi::mkl::sparse::init_matrix_handle(&B_handle_);
            oneapi::mkl::sparse::init_matrix_handle(&C_handle_);

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

            oneapi::mkl::sparse::init_matmat_descr(&description_);

            oneapi::mkl::sparse::set_matmat_data(description_,
                                                viewA_,
                                                opA_,
                                                viewB_,
                                                opB_,
                                                viewC_);
            
            request_ = oneapi::mkl::sparse::matmat_request::get_work_estimation_buf_size;
            sizeTempBuffer = sycl::malloc_host<int64_t>(1, queue_);

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

            tempBuffer = sycl::malloc_device<uint8_t>(sizeTempBuffer[0], queue_);

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

            request_ = oneapi::mkl::sparse::matmat_request::get_compute_buf_size;
            
            sizeTempBuffer2 = sycl::malloc_host<int64_t>(1, queue_);

            auto ev2_1 = oneapi::mkl::sparse::matmat(queue_,
                                                    A_handle_,
                                                    B_handle_,
                                                    C_handle_,
                                                    request_,
                                                    description_,
                                                    sizeTempBuffer2,
                                                    nullptr,
                                                    {ev1_3});
            ev2_1.wait();

            tempBuffer2 = sycl::malloc_device<uint8_t>(sizeTempBuffer2[0], queue_);

            request_ = oneapi::mkl::sparse::matmat_request::compute;
            auto ev2_3 = oneapi::mkl::sparse::matmat(queue_,
                                                    A_handle_,
                                                    B_handle_,
                                                    C_handle_,
                                                    request_,
                                                    description_,
                                                    sizeTempBuffer2,
                                                    tempBuffer2,
                                                    {ev2_1});

            request_ = oneapi::mkl::sparse::matmat_request::get_nnz;
            
            cNnzBuffer = sycl::malloc_host<int64_t>(1, queue_);

            auto ev3_1 = oneapi::mkl::sparse::matmat(queue_,
                                                     A_handle_,
                                                     B_handle_,
                                                     C_handle_,
                                                     request_,
                                                     description_,
                                                     cNnzBuffer,
                                                     nullptr,
                                                     {ev2_3});
            ev3_1.wait_and_throw();

            C_nnz_ = cNnzBuffer[0];
            C_cols_device_ = sycl::malloc_device<int64_t>(static_cast<size_t>(C_nnz_), queue_);
            C_vals_device_ = sycl::malloc_device<T>(static_cast<size_t>(C_nnz_), queue_);

            setC = oneapi::mkl::sparse::set_csr_data(queue_,
                                                    C_handle_,
                                                    m_,
                                                    n_,
                                                    CIndex_,
                                                    C_rows_device_,
                                                    C_cols_device_,
                                                    C_vals_device_,
                                                    {ev3_1});

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

            auto ev_sort = oneapi::mkl::sparse::sort_matrix(queue_, C_handle_, {ev3_3});

            C_rows_ = sycl::malloc_host<int64_t>(static_cast<size_t>(m_ + 1), queue_);
            C_cols_ = sycl::malloc_host<int64_t>(static_cast<size_t>(C_nnz_), queue_);
            C_vals_ = sycl::malloc_host<T>(static_cast<size_t>(C_nnz_), queue_);

            auto CRows = queue_.copy<int64_t>(C_rows_device_, C_rows_, static_cast<size_t>(m_ + 1));
            auto CCols = queue_.copy<int64_t>(C_cols_device_, C_cols_, static_cast<size_t>(C_nnz_));
            auto CVals = queue_.copy<T>(C_vals_device_, C_vals_, static_cast<size_t>(C_nnz_));
            CRows.wait();
            CCols.wait();
            CVals.wait();

            oneapi::mkl::sparse::release_matmat_descr(&description_);
            oneapi::mkl::sparse::release_matrix_handle(queue_, &A_handle_).wait();
            oneapi::mkl::sparse::release_matrix_handle(queue_, &B_handle_).wait();
            oneapi::mkl::sparse::release_matrix_handle(queue_, &C_handle_).wait();
          } catch (sycl::exception const &e) {
            std::cerr << "\t\tCaught synchronous SYCL exception:\n" << e.what() << std::endl;
            queue_.wait();
            oneapi::mkl::sparse::release_matmat_descr(&description_);
            oneapi::mkl::sparse::release_matrix_handle(queue_, &A_handle_).wait();
            oneapi::mkl::sparse::release_matrix_handle(queue_, &B_handle_).wait();
            oneapi::mkl::sparse::release_matrix_handle(queue_, &C_handle_).wait();
          }
          sycl::free(sizeTempBuffer, queue_);
          sycl::free(sizeTempBuffer2, queue_);
          sycl::free(tempBuffer, queue_);
          sycl::free(tempBuffer2, queue_);
          sycl::free(cNnzBuffer, queue_);
          sycl::free(C_rows_device_, queue_);
          sycl::free(C_cols_device_, queue_);
          sycl::free(C_vals_device_, queue_);
          break;
        }
        case gpuOffloadType::once: {
          // If already allocated, free the device C arrays
          if (!firstRun_) {
            sycl::free(C_rows_device_, queue_);
            sycl::free(C_cols_device_, queue_);
            sycl::free(C_vals_device_, queue_);
          }
          
          C_rows_device_ = sycl::malloc_device<int64_t>(static_cast<size_t>(m_ + 1), queue_);

          oneapi::mkl::sparse::init_matrix_handle(&A_handle_);
          oneapi::mkl::sparse::init_matrix_handle(&B_handle_);
          oneapi::mkl::sparse::init_matrix_handle(&C_handle_);

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

          oneapi::mkl::sparse::init_matmat_descr(&description_);

          oneapi::mkl::sparse::set_matmat_data(description_,
                                               viewA_,
                                               opA_,
                                               viewB_,
                                               opB_,
                                               viewC_);
          
          request_ = oneapi::mkl::sparse::matmat_request::get_work_estimation_buf_size;
          sizeTempBuffer = sycl::malloc_host<int64_t>(1, queue_);
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

          tempBuffer = sycl::malloc_device<uint8_t>(sizeTempBuffer[0], queue_);
          if (!tempBuffer) throw std::runtime_error("Could not allocate memory");

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

          request_ = oneapi::mkl::sparse::matmat_request::get_compute_buf_size;
          sizeTempBuffer2 = sycl::malloc_host<int64_t>(1, queue_);
          if (!sizeTempBuffer2) throw std::runtime_error("Could not allocate memory");
          auto ev2_1 = oneapi::mkl::sparse::matmat(queue_,
                                                   A_handle_,
                                                   B_handle_,
                                                   C_handle_,
                                                   request_,
                                                   description_,
                                                   sizeTempBuffer2,
                                                   nullptr,
                                                   {ev1_3});
          ev2_1.wait();

          tempBuffer2 = sycl::malloc_device<uint8_t>(sizeTempBuffer2[0], queue_);
          if (!tempBuffer2) throw std::runtime_error("Could not allocate memory");

          request_ = oneapi::mkl::sparse::matmat_request::compute;
          auto ev2_3 = oneapi::mkl::sparse::matmat(queue_,
                                                   A_handle_,
                                                   B_handle_,
                                                   C_handle_,
                                                   request_,
                                                   description_,
                                                   sizeTempBuffer2,
                                                   tempBuffer2,
                                                   {ev2_1});

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

          C_nnz_ = cNnzBuffer[0];
          C_cols_device_ = sycl::malloc_device<int64_t>(static_cast<size_t>(C_nnz_), queue_);
          C_vals_device_ = sycl::malloc_device<T>(static_cast<size_t>(C_nnz_), queue_);

          setC = oneapi::mkl::sparse::set_csr_data(queue_,
                                                        C_handle_,
                                                        m_,
                                                        n_,
                                                        CIndex_,
                                                        C_rows_device_,
                                                        C_cols_device_,
                                                        C_vals_device_,
                                                        {ev3_1});

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

          auto ev_sort = oneapi::mkl::sparse::sort_matrix(queue_, C_handle_, {ev3_3});

          oneapi::mkl::sparse::release_matmat_descr(&description_);
          oneapi::mkl::sparse::release_matrix_handle(queue_, &A_handle_).wait();
          oneapi::mkl::sparse::release_matrix_handle(queue_, &B_handle_).wait();
          oneapi::mkl::sparse::release_matrix_handle(queue_, &C_handle_).wait();
          sycl::free(sizeTempBuffer, queue_);
          sycl::free(sizeTempBuffer2, queue_);
          sycl::free(tempBuffer, queue_);
          sycl::free(tempBuffer2, queue_);
          sycl::free(cNnzBuffer, queue_);
          break;
        }
        case gpuOffloadType::unified: {
          // If already allocated, free the device C arrays
          if (!firstRun_) {
            sycl::free(C_rows_, queue_);
            sycl::free(C_cols_, queue_);
            sycl::free(C_vals_, queue_);
          }

          C_rows_ = sycl::malloc_shared<int64_t>(static_cast<size_t>(m_ + 1), queue_);

          oneapi::mkl::sparse::init_matrix_handle(&A_handle_);
          oneapi::mkl::sparse::init_matrix_handle(&B_handle_);
          oneapi::mkl::sparse::init_matrix_handle(&C_handle_);

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

          oneapi::mkl::sparse::init_matmat_descr(&description_);

          oneapi::mkl::sparse::set_matmat_data(description_,
                                               viewA_,
                                               opA_,
                                               viewB_,
                                               opB_,
                                               viewC_);
          
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

          tempBuffer = sycl::malloc_device<uint8_t>(sizeTempBuffer[0], queue_);
          if (!tempBuffer) throw std::runtime_error("Could not allocate memory");

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

          request_ = oneapi::mkl::sparse::matmat_request::get_compute_buf_size;
          sizeTempBuffer2 = sycl::malloc_host<int64_t>(1, queue_);
          if (!sizeTempBuffer2) throw std::runtime_error("Could not allocate memory");
          auto ev2_1 = oneapi::mkl::sparse::matmat(queue_,
                                                   A_handle_,
                                                   B_handle_,
                                                   C_handle_,
                                                   request_,
                                                   description_,
                                                   sizeTempBuffer2,
                                                   nullptr,
                                                   {ev1_3});
          ev2_1.wait();

          tempBuffer2 = sycl::malloc_device<uint8_t>(sizeTempBuffer2[0], queue_);
          if (!tempBuffer2) throw std::runtime_error("Could not allocate memory");

          request_ = oneapi::mkl::sparse::matmat_request::compute;
          auto ev2_3 = oneapi::mkl::sparse::matmat(queue_,
                                                   A_handle_,
                                                   B_handle_,
                                                   C_handle_,
                                                   request_,
                                                   description_,
                                                   sizeTempBuffer2,
                                                   tempBuffer2,
                                                   {ev2_1});

          request_ = oneapi::mkl::sparse::matmat_request::get_nnz;
          cNnzBuffer = sycl::malloc_shared<int64_t>(1, queue_);
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

          C_nnz_ = cNnzBuffer[0];
          C_cols_ = sycl::malloc_shared<int64_t>(static_cast<size_t>(C_nnz_), queue_);
          C_vals_ = sycl::malloc_shared<T>(static_cast<size_t>(C_nnz_), queue_);
          if (!C_vals_) throw std::runtime_error("Could not allocate memory");

          setC = oneapi::mkl::sparse::set_csr_data(queue_,
                                                        C_handle_,
                                                        m_,
                                                        n_,
                                                        CIndex_,
                                                        C_rows_,
                                                        C_cols_,
                                                        C_vals_,
                                                        {ev3_1});

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

          auto ev_sort = oneapi::mkl::sparse::sort_matrix(queue_, C_handle_, {ev3_3});

          oneapi::mkl::sparse::release_matmat_descr(&description_);
          oneapi::mkl::sparse::release_matrix_handle(queue_, &A_handle_).wait();
          oneapi::mkl::sparse::release_matrix_handle(queue_, &B_handle_).wait();
          oneapi::mkl::sparse::release_matrix_handle(queue_, &C_handle_).wait();
          sycl::free(sizeTempBuffer, queue_);
          sycl::free(sizeTempBuffer2, queue_);
          sycl::free(tempBuffer, queue_);
          sycl::free(tempBuffer2, queue_);
          sycl::free(cNnzBuffer, queue_);
          break;
        }
      }
      firstRun_ = false;
    }

    void postLoopRequirements() override {
      switch (offload_) {
        case gpuOffloadType::always: {
          break;
        }
        case gpuOffloadType::once: {
          C_rows_ = sycl::malloc_host<int64_t>(static_cast<size_t>(m_ + 1), queue_);

          C_cols_ = sycl::malloc_host<int64_t>(static_cast<size_t>(C_nnz_), queue_);

          C_vals_ = sycl::malloc_host<T>(static_cast<size_t>(C_nnz_), queue_);

          auto CRows = queue_.copy<int64_t>(C_rows_device_, C_rows_, static_cast<size_t>(m_ + 1));
          auto CCols = queue_.copy<int64_t>(C_cols_device_, C_cols_, static_cast<size_t>(C_nnz_));
          auto CVals = queue_.copy<T>(C_vals_device_, C_vals_, static_cast<size_t>(C_nnz_));
          CRows.wait();
          CCols.wait();
          CVals.wait();

          sycl::free(C_rows_device_, queue_);
          sycl::free(C_cols_device_, queue_);
          sycl::free(C_vals_device_, queue_);
          break;
        }
        case gpuOffloadType::unified: {
          break;
        }
      }
    }

    void postCallKernelCleanup() override {
      switch (offload_) {
        case gpuOffloadType::always: {
          sycl::free(A_rows_, queue_);
          sycl::free(A_cols_, queue_);
          sycl::free(A_vals_, queue_);
          sycl::free(A_rows_device_, queue_);
          sycl::free(A_cols_device_, queue_);
          sycl::free(A_vals_device_, queue_);

          sycl::free(B_rows_, queue_);
          sycl::free(B_cols_, queue_);
          sycl::free(B_vals_, queue_);
          sycl::free(B_rows_device_, queue_);
          sycl::free(B_cols_device_, queue_);
          sycl::free(B_vals_device_, queue_);

          sycl::free(C_rows_, queue_);
          sycl::free(C_cols_, queue_);
          sycl::free(C_vals_, queue_);
          break;
        }
        case gpuOffloadType::once: {
          sycl::free(A_rows_, queue_);
          sycl::free(A_cols_, queue_);
          sycl::free(A_vals_, queue_);
          sycl::free(A_rows_device_, queue_);
          sycl::free(A_cols_device_, queue_);
          sycl::free(A_vals_device_, queue_);

          sycl::free(B_rows_, queue_);
          sycl::free(B_cols_, queue_);
          sycl::free(B_vals_, queue_);
          sycl::free(B_rows_device_, queue_);
          sycl::free(B_cols_device_, queue_);
          sycl::free(B_vals_device_, queue_);

          sycl::free(C_rows_, queue_);
          sycl::free(C_cols_, queue_);
          sycl::free(C_vals_, queue_);
          break;
        }
        case gpuOffloadType::unified: {
          sycl::free(A_rows_, queue_);
          sycl::free(A_cols_, queue_);
          sycl::free(A_vals_, queue_);
          sycl::free(B_rows_, queue_);
          sycl::free(B_cols_, queue_);
          sycl::free(B_vals_, queue_);
          sycl::free(C_rows_, queue_);
          sycl::free(C_cols_, queue_);
          sycl::free(C_vals_, queue_);

          free(A_rows_store_);
          free(A_cols_store_);
          free(A_vals_store_);
          free(B_rows_store_);
          free(B_cols_store_);
          free(B_vals_store_);
          break;
        }
      }
    }

    // First-run check to confirm whether to clean up old arrays or not
    bool firstRun_ = true;

    bool initialised_ = false;

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

    size_t alloc_sz = 0;

    // A CSR arrays
    int64_t* A_rows_store_ = nullptr;
    int64_t* A_cols_store_ = nullptr;
    T* A_vals_store_ = nullptr;
    //    LOCAL
    int64_t* A_rows_ = nullptr;
    int64_t* A_cols_ = nullptr;
    T* A_vals_ = nullptr;
    //    DEVICE
    int64_t* A_rows_device_ = nullptr;
    int64_t* A_cols_device_ = nullptr;
    T* A_vals_device_ = nullptr;

    // B CSR arrays
    int64_t* B_rows_store_ = nullptr;
    int64_t* B_cols_store_ = nullptr;
    T* B_vals_store_ = nullptr;
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
