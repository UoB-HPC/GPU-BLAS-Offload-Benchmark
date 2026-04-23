#pragma once

#if defined CPU_AOCL

#include <string>

#define aoclCheckError(f)                                                                                                                                                                                         \
  do {                                                                                                                                                                                                            \
    aoclsparse_status _status = (f);                                                                                                                                                                              \
    if (_status != aoclsparse_status_success) {                                                                                                                                                                   \
      std::string _err_str;                                                                                                                                                                                       \
      switch (_status) {                                                                                                                                                                                          \
        case aoclsparse_status_not_implemented:       _err_str = "NOT_IMPLEMENTED - The requested functionality is not yet implemented in this version"; break;                                                   \
        case aoclsparse_status_invalid_pointer:       _err_str = "INVALID_POINTER - One or more pointer parameters are NULL or otherwise invalid"; break;                                                         \
        case aoclsparse_status_invalid_size:          _err_str = "INVALID_SIZE - One or more size parameters (m, n, nnz, etc.) contain an invalid value (e.g., negative or zero where positive required)"; break; \
        case aoclsparse_status_internal_error:        _err_str = "INTERNAL_ERROR - Internal library failure"; break;                                                                                              \
        case aoclsparse_status_invalid_value:         _err_str = "INVALID_VALUE - Input parameters contain an invalid value (e.g., invalid enum value, base index neither 0 nor 1)"; break;                       \
        case aoclsparse_status_invalid_index_value:   _err_str = "INVALID_INDEX_VALUE - At least one index value is invalid (e.g., negative or out of bounds)"; break;                                            \
        case aoclsparse_status_maxit:                 _err_str = "MAXIT - function stopped after reaching number of iteration limit"; break;                                                                      \
        case aoclsparse_status_user_stop:             _err_str = "USER_STOP - user requested termination"; break;                                                                                                 \
        case aoclsparse_status_wrong_type:            _err_str = "WRONG_TYPE - Data type mismatch (e.g., matrix datatypes don't match between operations)"; break;                                                \
        case aoclsparse_status_memory_error:          _err_str = "MEMORY_ERROR - memory allocation failure"; break;                                                                                               \
        case aoclsparse_status_numerical_error:       _err_str = "NUMERICAL_ERROR - numerical error, e.g., matrix is not positive definite, devide-by-zero error"; break;                                         \
        case aoclsparse_status_invalid_operation:     _err_str = "INVALID_OPERATION - cannot proceed with the request at this point"; break;                                                                      \
        case aoclsparse_status_unsorted_input:        _err_str = "UNSORTED_INPUT - the input matrices are not sorted"; break;                                                                                     \
        case aoclsparse_status_invalid_kid:           _err_str = "INVALID_KID - user requested kernel id was not available"; break;                                                                               \
        default:                                      _err_str = "UNKNOWN_STATUS - Unrecognized status code (" + std::to_string(_status) + ")"; break;                                                            \
      }                                                                                                                                                                                                           \
      std::cerr << std::endl << "=== ARMPL ERROR ===" << std::endl                                                                                                                                                \
                << "File:         " << __FILE__ << std::endl                                                                                                                                                      \
                << "Line:         " << __LINE__ << std::endl                                                                                                                                                      \
                << "Call:         " << #f << std::endl                                                                                                                                                            \
                << "Status:       " << _err_str << std::endl                                                                                                                                                      \
                << "===================" << std::endl;                                                                                                                                                            \
      exit(EXIT_FAILURE);                                                                                                                                                                                         \
    }                                                                                                                                                                                                             \
  } while(0)




#endif 