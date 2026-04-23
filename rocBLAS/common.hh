#pragma once

#if defined GPU_ROCBLAS

/** Macro function to check if error occurred when calling cuBLAS. */
#define hipCheckError(f)                                                \
  do {                                                                  \
    if (hipError_t e = (f); e != hipSuccess) {                          \
      std::cout << "HIP error: " << __FILE__ << ":" << __LINE__ << ": " \
                << hipGetErrorString(e) << std::endl;                   \
      std::cout << "[DEBUG] -- " << #f << std::endl;                    \
      exit(1);                                                          \
    }                                                                   \
  } while (false)

#define rocCheckError(f)                                                               \
  do {                                                                                 \
    rocsparse_status _status = (f);                                                    \
    if (_status != rocsparse_status_success) {                                         \
      const char* _err_str;                                                            \
      switch (_status) {                                                               \
        case rocsparse_status_invalid_handle:          _err_str = "Invalid handle"; break; \
        case rocsparse_status_not_implemented:         _err_str = "Not implemented"; break; \
        case rocsparse_status_invalid_pointer:         _err_str = "Invalid pointer"; break; \
        case rocsparse_status_invalid_size:            _err_str = "Invalid size"; break; \
        case rocsparse_status_memory_error:            _err_str = "Memory error"; break; \
        case rocsparse_status_internal_error:          _err_str = "Internal error"; break; \
        case rocsparse_status_invalid_value:           _err_str = "Invalid value"; break; \
        case rocsparse_status_arch_mismatch:           _err_str = "Architecture mismatch"; break; \
        case rocsparse_status_zero_pivot:              _err_str = "Zero pivot encountered"; break; \
        case rocsparse_status_not_initialized:         _err_str = "Not initialized"; break; \
        case rocsparse_status_type_mismatch:           _err_str = "Type mismatch"; break; \
        case rocsparse_status_requires_sorted_storage: _err_str = "Requires sorted storage"; break; \
        case rocsparse_status_thrown_exception:        _err_str = "Exception thrown"; break; \
        case rocsparse_status_continue:                _err_str = "Continue"; break; \
        default:                                       _err_str = "Unknown error code"; break; \
      }                                                                                \
      std::cerr << "\n=== ROCSPARSE ERROR ===\n"                                       \
                << "File:       " << __FILE__ << "\n"                                  \
                << "Line:       " << __LINE__ << "\n"                                  \
                << "Call:       " << #f << "\n"                                        \
                << "Status:     " << _err_str << " (" << _status << ")\n"              \
                << "=======================\n" << std::endl;                           \
      exit(EXIT_FAILURE);                                                              \
    }                                                                                  \
  } while(0)

#endif 