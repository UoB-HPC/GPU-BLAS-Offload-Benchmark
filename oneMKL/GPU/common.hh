#pragma once

#ifdef GPU_ONEMKL

#include <mkl.h>

#include <oneapi/mkl/blas.hpp>
#include <sycl/sycl.hpp>

// Create an exception handler for asynchronous SYCL exceptions
static auto exception_handler = [](sycl::exception_list e_list,
                                   std::string kernel) {
  for (std::exception_ptr const& e : e_list) {
    try {
      std::rethrow_exception(e);
    } catch (std::exception const& e) {
      std::cout << "ERROR -  Caught asynchronous SYCL exception during "
                << kernel << ": " << e.what() << std::endl;
    }
  }
};

#endif