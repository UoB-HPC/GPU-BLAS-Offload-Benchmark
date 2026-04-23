#pragma once

#if defined CPU_ARMPL

#define armplCheckError(f)                                                                                                                                         \
  do {                                                                                                                                                            \
    armpl_status_t _status = (f);                                                                                                                                 \
    if (_status != ARMPL_STATUS_SUCCESS) {                                                                                                                        \
      const char* _err_str;                                                                                                                                       \
      switch (_status) {                                                                                                                                          \
        case ARMPL_STATUS_INPUT_PARAMETER_ERROR:       _err_str = "INPUT_PARAMETER_ERROR - An error was found with the supplied input parameters."; break;        \
        case ARMPL_STATUS_EXECUTION_FAILURE:           _err_str = "EXECUTION_FAILURE - The function failed for an alternative reason."; break;                    \
      }                                                                                                                                                           \
      std::cerr << std::endl << "=== ARMPL ERROR ===" << std::endl                                                                                                \
                << "File:         " << __FILE__ << std::endl                                                                                                      \
                << "Line:         " << __LINE__ << std::endl                                                                                                      \
                << "Call:         " << #f << std::endl                                                                                                            \
                << "Status:       " << _err_str << std::endl                                                                                                      \
                << "===================" << std::endl;                                                                                                            \
      exit(EXIT_FAILURE);                                                                                                                                         \
    }                                                                                                                                                             \
  } while(0)




#endif 