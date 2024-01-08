#include "utilities.h"

// Select which CPU Library to use
#if defined CPU_DEFAULT
#include "DefaultCPU/default.h"
#elif defined CPU_ARMPL
#include "ArmPL/armpl.h"
#elif defined CPU_ONEMKL
// #include "OneMKL/onemkl.h"
#elif defined CPU_OPENBLAS
// #include "OpenBLAS/openblas.h"
#elif defined CPU_AOCL
// #include "AOCL/aocl.h"
#endif

// Select which GPU Library to use
#if defined GPU_CUBLAS
// #include "cuBLAS/cublas.h"
#elif defined GPU_ONEMKL
// #include "OneMKL/onemkl.h"
#elif defined GPU_ROCBLAS
// #include "rocBLAS/rocblas.h"
#endif