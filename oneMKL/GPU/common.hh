#pragma once

#ifdef GPU_ONEMKL

#include <mkl.h>
#include <memory>
#include <oneapi/mkl/blas.hpp>
#include <oneapi/mkl/spblas.hpp>
#include <sycl/sycl.hpp>

// Create an exception handler for asynchronous SYCL exceptions
static const std::function<void(sycl::exception_list)> exception_handler =
    [](sycl::exception_list e_list) {
      for (std::exception_ptr const& e : e_list) {
        try {
          std::rethrow_exception(e);
        } catch (std::exception const& e) {
          std::cerr << "ERROR -  Caught asynchronous SYCL exception : " << e.what() << std::endl;
        }
      }
    };

#endif