#include "../include/main.hh"

int iters = 10;
int upperLimit = 128;

bool doCpu = CPU_ENABLED;
bool doGpu = GPU_ENABLED;

std::string CSV_DIR = "CSV_Results";

int main(int argc, char** argv) {
  getParameters(argc, argv);
  printBenchmarkConfig(iters, upperLimit);

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

  // -------- GEMM --------
  // SGEMM Comparison
  std::cout << std::endl << "Comparing SGEMM Kernels:" << std::endl;
  doGemm<float> sgemm(std::string(absPath), iters, startDim, upperLimit, doCpu,
                      doGpu);
  sgemm.collectData();
  std::cout << "Finished!" << std::endl;

  // DGEMM Comparison
  std::cout << std::endl << "Comparing DGEMM Kernels:" << std::endl;
  doGemm<double> dgemm(std::string(absPath), iters, startDim, upperLimit, doCpu,
                       doGpu);
  dgemm.collectData();
  std::cout << "Finished!" << std::endl;

  // -------- GEMV --------
  // SGEMV Comparison
  std::cout << std::endl << "Comparing SGEMV Kernels:" << std::endl;
  doGemv<float> sgemv(std::string(absPath), iters, startDim, upperLimit, doCpu,
                      doGpu);
  sgemv.collectData();
  std::cout << "Finished!" << std::endl;

  // DGEMV Comparison
  std::cout << std::endl << "Comparing DGEMV Kernels:" << std::endl;
  doGemv<double> dgemv(std::string(absPath), iters, startDim, upperLimit, doCpu,
                       doGpu);
  dgemv.collectData();
  std::cout << "Finished!" << std::endl;

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
      (getenv("OMP_NUM_THREADS") != NULL) ? atoi(getenv("OMP_NUM_THREADS")) : 1;
#endif
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
    } else if (!strcmp(argv[i], "--no_cpu")) {
      doCpu = false;
    } else if (!strcmp(argv[i], "--no_gpu")) {
      doGpu = false;
    } else if (!strcmp(argv[i], "--kernels") || !strcmp(argv[i], "-k")) {
	    sgemm = dgemm = sp_sgemm = sp_dgemm = false;
	    std::string kernelList = argv[++i];
	    if (kernelList.find("sp-sgemm") != std::string::npos) {
		    sp_sgemm = true;
		    if (kernelList.find("sgemm") != std::string::npos &&
						kernelList.find("sgemm") != kernelList.find("sp-sgemm") + 3) {
			    sgemm = true;
		    }
	    } else if (kernelList.find("sgemm") != std::string::npos) {
			    sgemm = true;
			}
	    if (kernelList.find("sp-dgemm") != std::string::npos) {
		    sp_dgemm = true;
		    if (kernelList.find("dgemm") != std::string::npos &&
		        kernelList.find("dgemm") != kernelList.find("sp-dgemm") + 3) {
			    dgemm = true;
		    }
	    } else if (kernelList.find("dgemm") != std::string::npos) {
		    dgemm = true;
	    }

	    if (!sgemm && !dgemm && !sp_sgemm && !sp_dgemm) {
		    std::cout << "ERROR - no implemented kernels in list" << std::endl;
		    exit(1);
	    }
    } else if (!strcmp(argv[i], "--output_dir") || !strcmp(argv[i], "-o")) {
      if (++i >= argc) {
        std::cout << "ERROR - Invalid output directory" << std::endl;
        exit(1);
      } else {
        CSV_DIR = argv[i];
      }
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
      std::cout
          << "  -o  --output_dir             The CSV file output directory"
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