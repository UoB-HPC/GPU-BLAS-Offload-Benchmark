#include "../include/main.hh"

int iters = 10;
int startDim = 1;
int upperLimit = 128;
int step = 1;
double sparsity = 0.99;
// GEMV kernels
bool doSgemv = true;
bool doDgemv = true;
// Sparse GEMV kernels
bool doSspmv = true;
bool doDspmv = true;
// GEMM kernels
bool doSgemm = true;
bool doDgemm = true;
// Sparse MM kernels
bool doSspmm = true;
bool doDspmm = true;
// Sparse GEMM kernels
bool doSspgemm = true;
bool doDspgemm = true;

bool doCpu = CPU_ENABLED;
bool doGpu = GPU_ENABLED;

matrixType type = matrixType::random;

std::string CSV_DIR = "CSV_Results";

int main(int argc, char** argv) {
  getParameters(argc, argv);
  printBenchmarkConfig(iters, upperLimit);

#ifdef CPU_ARMPL
  if (doSspmm || doDspmm) {
    std::cout << "WARNING - ArmPL does not currently provide a Sparse Matrix-Dense Matrix kernel. Disabling Sparse Matrix-Dense Matrix tests." << std::endl;
    doSspmm = false;
    doDspmm = false;
  }
#endif

#ifdef CPU_NVPL
  if (doSspmm || doDspmm) {
    std::cout << "WARNING - NVPL does not currently provide a Sparse Matrix-Dense Matrix kernel. Disabling Sparse Matrix-Dense Matrix tests." << std::endl;
    doSspmm = false;
    doDspmm = false;
  }
  if (doSspgemm || doDspgemm) {
    std::cout << "WARNING - NVPL does not currently provide a Sparse Matrix-Sparse Matrix kernel. Disabling Sparse Matrix-Sparse Matrix tests." << std::endl;
    doSspgemm = false;
    doDspgemm = false;
  }
#endif

  if (!doCpu && !doGpu) {
    std::cout << "Finished!" << std::endl;
    exit(0);
  }

  // Ensure CSV file directory exists.
  struct stat st = {0};
  if (stat(CSV_DIR.c_str(), &st) == -1) {
    mkdir(CSV_DIR.c_str(), 0700);
  }

  char* absPath = realpath(CSV_DIR.c_str(), nullptr);
  std::cout << "All results will be saved in CSV files at '" << absPath << "'"
            << std::endl
            << std::endl;
// -------- GEMV --------
  // Single-Precision GEMV
  if (doSgemv) {
    std::cout << std::endl << "Comparing SGEMV Kernels:" << std::endl;
    doGemv<float> sgemv(std::string(absPath), iters, startDim, upperLimit,
                        step, doCpu, doGpu);
    sgemv.collectData();
    std::cout << "Finished!" << std::endl;
  }

  // Double-Precision GEMV
  if (doDgemv) {
    std::cout << std::endl << "Comparing DGEMV Kernels:" << std::endl;
    doGemv<double> dgemv(std::string(absPath), iters, startDim, upperLimit,
                         step, doCpu, doGpu);
    dgemv.collectData();
    std::cout << "Finished!" << std::endl;
  }

//  // -------- GEMM --------
//  // Single-Precision GEMM
 if (doSgemm) {
   std::cout << std::endl << "Comparing SGEMM Kernels:" << std::endl;
   doGemm<float> sgemm(std::string(absPath), iters, startDim, upperLimit,
                       step, doCpu, doGpu);
   sgemm.collectData();
   std::cout << "Finished!" << std::endl;
 }

 // Double-Precision GEMM
 if (doDgemm) {
   std::cout << std::endl << "Comparing DGEMM Kernels:" << std::endl;
   doGemm<double> dgemm(std::string(absPath), iters, startDim, upperLimit,
                        step, doCpu, doGpu);
   dgemm.collectData();
   std::cout << "Finished!" << std::endl;
 }


  // -------- SGEMV --------
  // Single-Precision Sparse Matrix-Dense Vector
  if (doSspmv) {
    std::cout << std::endl << "Comparing SSpMV Kernels:" << std::endl;
    doSpmv<float> sspmv(std::string(absPath), iters, startDim, upperLimit,
                            step, sparsity, type, doCpu, doGpu);
    sspmv.collectData();
    std::cout << "Finished!" << std::endl;
  }

  // Double-Precision Sparse Matrix-Dense Vector
  if (doDspmv) {
    std::cout << std::endl << "Comparing DSpMV Kernels:" << std::endl;
    doSpmv<double> dspmv(std::string(absPath), iters, startDim, upperLimit,
                             step, sparsity, type, doCpu, doGpu);
    dspmv.collectData();
    std::cout << "Finished!" << std::endl;
  }

  // // -------- SpMM --------
  // // Single-Precision Sparse Matrix-Dense Matrix
  if (doSspmm) {
    std::cout << std::endl << "Comparing SSpMM Kernels:" << std::endl;
    doSpmm<float> sspmm(std::string(absPath), iters, startDim, upperLimit,
                        step, sparsity, type, doCpu, doGpu);
    sspmm.collectData();
    std::cout << "Finished!" << std::endl;
  }

  // Double-Precision Sparse Matrix-Dense Matrix
  if (doDspmm) {
    std::cout << std::endl << "Comparing DSpMM Kernels:" << std::endl;
    doSpmm<double> dspmm(std::string(absPath), iters, startDim, upperLimit,
                         step, sparsity, type, doCpu, doGpu);
    dspmm.collectData();
    std::cout << "Finished!" << std::endl;
  }

  // -------- SpGEMM --------
  // Single-Precision Sparse Matrix-Sparse Matrix
  if (doSspgemm) {
    std::cout << std::endl << "Comparing SSpGEMM Kernels:" << std::endl;
    doSpgemm<float> sspgemm(std::string(absPath), iters, startDim, upperLimit,
                            step, sparsity, type, doCpu, doGpu);
    sspgemm.collectData();
    std::cout << "Finished!" << std::endl;
  }

  // Double-Precision Sparse Matrix-Sparse Matrix
  if (doDspgemm) {
    std::cout << std::endl << "Comparing DSpGEMM Kernels:" << std::endl;
    doSpgemm<double> dspgemm(std::string(absPath), iters, startDim, upperLimit,
                             step, sparsity, type, doCpu, doGpu);
    dspgemm.collectData();
    std::cout << "Finished!" << std::endl;
  }
  free(absPath);
  return 0;
}


void printBenchmarkConfig(const int iters, const int upperLimit) {
  std::string cpuEnabledStr = (doCpu) ? "True" : "False";
  std::string gpuEnabledStr = (doGpu) ? "True" : "False";
  unsigned int ompThreads =
#if defined CPU_AOCL
      (getenv("BLIS_NUM_THREADS") != NULL) ? atoi(getenv("BLIS_NUM_THREADS"))
                                           : 1;
#else
      (getenv("OMP_NUM_THREADS") != nullptr) ? atoi(getenv("OMP_NUM_THREADS")) : 1;
#endif
  const char* ompProcBind =
      (getenv("OMP_PROC_BIND") != nullptr) ? getenv("OMP_PROC_BIND") : "Not "
                                                                       "Set";
  const char* ompPlaces =
      (getenv("OMP_PLACES") != nullptr) ? getenv("OMP_PLACES") : "Not Set";
  const char* matrixType;
  switch (type) {
  case matrixType::rmat:
    matrixType = "rMAT";
    break;
  case matrixType::random:
    matrixType = "random";
    break;
  case matrixType::bandedDiagonal:
    matrixType = "bandedDiagonal";
    break;
  default:
    matrixType = "Unknown";
    break;  
  }
  std::cout << "GPU BLAS Offload Benchmark:" << std::endl;
  std::cout << "\tIterations per Kernel: " << iters << std::endl;
  std::cout << "\tStarting Problem Dimension: " << startDim << std::endl;
  std::cout << "\tMaximum Problem Dimension: " << upperLimit << std::endl;
  std::cout << "\tSparse Matrix Type: " << matrixType << std::endl;
  std::cout << "\tCPU Kernels Enabled: " << cpuEnabledStr << std::endl;
  std::cout << "\tCPU Library: " << CPU_LIB_NAME << std::endl;
  std::cout << "\tGPU Kernels Enabled: " << gpuEnabledStr << std::endl;
  std::cout << "\tGPU Library: " << GPU_LIB_NAME << std::endl;
#if defined CPU_AOCL
  std::cout << "\tBLIS_NUM_THREADS: " << ompThreads << std::endl;
#else
  std::cout << "\tOMP_NUM_THREADS: " << ompThreads << std::endl;
#endif
  std::cout << "\tOMP_PROC_BIND: " << ompProcBind << std::endl;
  std::cout << "\tOMP_PLACES: " << ompPlaces << std::endl;
  std::cout << std::endl;
#ifdef CPU_DEFAULT
  std::cout << "WARNING - No CPU BLAS library selected. No CPU BLAS Kernels "
               "will be run."
            << std::endl;
#endif
#ifdef GPU_DEFAULT
  std::cout << "WARNING - No GPU BLAS Library selected. No GPU BLAS kernels "
               "will be run."
            << std::endl;
#endif
  std::cout << std::endl;
}

int parseInt(const char* str) {
  char* next;
  int value = strtoul(str, &next, 10);
  return strlen(next) ? -1 : value;
}

void getParameters(int argc, char** argv) {
  for (int i = 1; i < argc; i++) {
    if (!strcmp(argv[i], "--iterations") || !strcmp(argv[i], "-i")) {
      if (++i >= argc || (iters = parseInt(argv[i])) < 0) {
        std::cout << "ERROR - Invalid number of iterations" << std::endl;
        exit(1);
      }
    } else if (!strcmp(argv[i], "--start_dimension") ||
               !strcmp(argv[i], "-s")) {
      if (++i >= argc || (startDim = parseInt(argv[i])) < 0) {
        std::cout << "ERROR - Invalid start dimension" << std::endl;
        exit(1);
      }
    } else if (!strcmp(argv[i], "--dimension_limit") ||
               !strcmp(argv[i], "-d")) {
      if (++i >= argc || (upperLimit = parseInt(argv[i])) < 0) {
        std::cout << "ERROR - Invalid dimension limit" << std::endl;
        exit(1);
      }
      if (startDim > upperLimit) {
        std::cout
            << "ERROR - Start dimension cannot be greater than dimension limit"
            << std::endl;
        exit(1);
      }
    } else if (!strcmp(argv[i], "--step")) {
      if (++i >= argc || (step = parseInt(argv[i])) < 0) {
        std::cout << "ERROR - Invalid dimension step size" << std::endl;
        exit(1);
      }
    } else if (!strcmp(argv[i], "--no_cpu")) {
      doCpu = false;
    } else if (!strcmp(argv[i], "--no_gpu")) {
      doGpu = false;
    } else if (!strcmp(argv[i], "--kernels") || !strcmp(argv[i], "-k")) {
      std::string kernelList = argv[++i];
      doSgemm = (kernelList.find("sgemm") != std::string::npos);
      doDgemm = (kernelList.find("dgemm") != std::string::npos);
      doSspmm = (kernelList.find("sspmm") != std::string::npos);
      doDspmm = (kernelList.find("dspmm") != std::string::npos);
      doSspgemm = (kernelList.find("sspgemm") != std::string::npos);
      doDspgemm = (kernelList.find("dspgemm") != std::string::npos);
      doSgemv = (kernelList.find("sgemv") != std::string::npos);
      doDgemv = (kernelList.find("dgemv") != std::string::npos);
      doSspmv = (kernelList.find("sspmv") != std::string::npos);
      doDspmv = (kernelList.find("dspmv") != std::string::npos);

      if (!doSgemv && !doSspmv && !doSgemm && !doSspmm && !doSspgemm &&
          !doDgemv && !doDspmv && !doDgemm && !doDspmm && !doDspgemm) {
        std::cout << "ERROR - no implemented kernels in list" << std::endl;
        exit(1);
      } else {
        CSV_DIR = argv[i];
      }
    } else if (!strcmp(argv[i], "--sparsity")) {
      if (++i >= argc || (sparsity = std::stod(argv[i])) < 0 ||
          sparsity >= 1.00)  {
        std::cout << "ERROR - Invalid sparsity value" << std::endl;
        exit(1);
      }
    } else if (!strcmp(argv[i], "--matrix_type") || !strcmp(argv[i], "-t")) {
      if (++i >= argc) {
        std::cout << "ERROR - No matrix type specified" << std::endl;
        exit(1);
      } else if (!strcmp(argv[i], "rmat")) {
        type = matrixType::rmat;
      } else if (!strcmp(argv[i], "random")) {
        type = matrixType::random;
      } else if (!strcmp(argv[i], "bandedDiagonal")) {
        type = matrixType::bandedDiagonal;
      } else {
        std::cout << "ERROR - Unrecognized matrix type '" << argv[i]
                  << "'" << std::endl;
        exit(1);
      }
    } else if (!strcmp(argv[i], "--output_dir") || !strcmp(argv[i], "-o")) {
      if (++i >= argc) {
        std::cout << "ERROR - No output directory specified" << std::endl;
        exit(1);
      }
      CSV_DIR = argv[i];
    } else if (!strcmp(argv[i], "--help") || !strcmp(argv[i], "-h")) {
      std::cout << std::endl;
      std::cout << "Usage: ./gpu-blob [OPTIONS]" << std::endl << std::endl;
      std::cout << "Options:" << std::endl;
      std::cout << "  -h  --help                   Print this message"
                << std::endl;
      std::cout << "  --no_cpu                     Disable all CPU kernel Runs"
                << std::endl;
      std::cout << "  --no_gpu                     Disable all GPU kernel Runs"
                << std::endl;
      std::cout << "  -o  --output_dir             The CSV file output directory"
                << std::endl;
      std::cout << "  -i  --iterations I           Repeat each kernel I times "
                   "(default: " << iters << ")" 
                << std::endl;
      std::cout << "  -s  --start_dimension S      First value of M, N, K is S "
                   "(default: " << startDim << ")" 
                << std::endl;
      std::cout << "  --step St                    Step size between values of M, N, K"
                   "(default: " << step << ")" 
                << std::endl;
      std::cout << "  -d  --dimension_limit D      Max value of M, N, K is D "
                   "(default: " << upperLimit << ")" 
                << std::endl;
      std::cout << "  -k  --kernels <kernels>      Comma-separated list of "
                   "kernels to be run.  Options are sgemm, dgemm, sspmm, "
                   "dspmm, sspgemm, dspgemm, sgemv, dgemv, sspmv, dspmv "
                   "(default: `-k sgemm,dgemm,sspmm,dspmm,sspgemm,dspgemm,"
                   "sgemv,dgemv,sspmv,dspmv`)" 
                << std::endl;
      std::cout << "  --sparsity Sp                Sparsity value, between 0 "
                   "and 1 (double), to be used by the sparse BLAS kernels.  "
                   "Matrices with be generated with this sparsity value.  "
                   "Defaults to 0.99" 
                << std::endl;
      std::cout << "  -t  --matrix_type M          Type of sparse matrix to use."
                   ".  Only applies to sparse kernels.  Options are rmat, random"
                   ", bandedDiagonal (default -t random)" 
                << std::endl;
      exit(0);
    } else {
      std::cout << "Unrecognized argument '" << argv[i] << "' (try '--help')"
                << std::endl;
      exit(1);
    }
  }
}
