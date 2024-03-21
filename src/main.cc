#include "../include/main.hh"

int iters = 10;
int startDim = 1;
int upperLimit = 128;

int main(int argc, char** argv) {
  getParameters(argc, argv);
  printBenchmarkConfig(iters, upperLimit);

  // Ensure CSV file directory exists.
  struct stat st = {0};
  if (stat(CSV_DIR, &st) == -1) {
    mkdir(CSV_DIR, 0700);
  }
  char absPath[4096];
  realpath(CSV_DIR, absPath);
  std::cout << "All results will be saved in CSV files at '" << absPath << "'"
            << std::endl
            << std::endl;

  // SGEMM Comparison
  std::cout << std::endl << "Comparing SGEMM Kernels:" << std::endl;
  doGemm<float> sgemm(iters, startDim, upperLimit);
  sgemm.collectData();
  std::cout << "Finished!" << std::endl;

  // DGEMM Comparison
  std::cout << std::endl << "Comparing DGEMM Kernels:" << std::endl;
  doGemm<double> dgemm(iters, startDim, upperLimit);
  dgemm.collectData();
  std::cout << "Finished!" << std::endl;
  return 0;
}

void printBenchmarkConfig(const int iters, const int upperLimit) {
  std::string gpuEnabledStr = (GPU_ENABLED) ? "True" : "False";
  std::string cpuEnabledStr = (CPU_ENABLED) ? "True" : "False";
  unsigned int ompThreads =
      (getenv("OMP_NUM_THREADS") != NULL) ? atoi(getenv("OMP_NUM_THREADS")) : 1;
  const char* ompProcBind =
      (getenv("OMP_PROC_BIND") != NULL) ? getenv("OMP_PROC_BIND") : "Not Set";
  const char* ompPlaces =
      (getenv("OMP_PLACES") != NULL) ? getenv("OMP_PLACES") : "Not Set";
  std::cout << "GPU BLAS Offload Benchmark:" << std::endl;
  std::cout << "\tIterations per Kernel: " << iters << std::endl;
  std::cout << "\tStarting Problem Dimension: " << startDim << std::endl;
  std::cout << "\tMaximum Problem Dimension: " << upperLimit << std::endl;
  std::cout << "\tCPU Kernels Enabled: " << cpuEnabledStr << std::endl;
  std::cout << "\tCPU Library: " << CPU_LIB_NAME << std::endl;
  std::cout << "\tGPU Kernels Enabled: " << gpuEnabledStr << std::endl;
  std::cout << "\tGPU Library: " << GPU_LIB_NAME << std::endl;
  std::cout << "\tOMP_NUM_THREADS: " << ompThreads << std::endl;
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

void getParameters(int argc, char* argv[]) {
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
    } else if (!strcmp(argv[i], "--help") || !strcmp(argv[i], "-h")) {
      std::cout << std::endl;
      std::cout << "Usage: ./gpu-blob [OPTIONS]" << std::endl << std::endl;
      std::cout << "Options:" << std::endl;
      std::cout << "  -h  --help                   Print this message"
                << std::endl;
      std::cout << "  -i  --iterations I           Repeat each kernel I times "
                   "(default: "
                << iters << ")" << std::endl;
      std::cout << "  -s  --start_dimension S      First value of M, N, K is S "
                   "(default: "
                << startDim << ")" << std::endl;
      std::cout << "  -d  --dimension_limit D      Max value of M, N, K is D "
                   "(default: "
                << upperLimit << ")" << std::endl;
      std::cout << std::endl;
      exit(0);
    } else {
      std::cout << "Unrecognized argument '" << argv[i] << "' (try '--help')"
                << std::endl;
      exit(1);
    }
  }
}