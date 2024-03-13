#pragma once

#ifdef GPU_ONEMKL

#include <mkl.h>

#include <oneapi/mkl/blas.hpp>
#include <sycl/sycl.hpp>

// Create an exception handler for asynchronous SYCL exceptions
static const std::function<void(sycl::exception_list)> exception_handler =
    [](sycl::exception_list e_list) {
      for (std::exception_ptr const& e : e_list) {
        try {
          std::rethrow_exception(e);
        } catch (std::exception const& e) {
          std::cout << "ERROR -  Caught asynchronous SYCL exception : "
                    << e.what() << std::endl;
        }
      }
    };

// Functions to extract an error code from a sycl::exception.
// Taken from the oneMKL common_for_examples.hpp file that is included in the
// oneAPI Base Toolkit
template <typename T, typename std::enable_if<
                          has_member_code_meta<T>::value>::type* = nullptr>
auto get_error_code(T x) {
  return x.code().value();
};

template <typename T, typename std::enable_if<
                          !has_member_code_meta<T>::value>::type* = nullptr>
auto get_error_code(T x) {
  return x.get_cl_code();
};

#endif