#pragma once
#include <type_traits>

#include "helpers.hh"
#include "utilities.hh"

#if defined CPU_DEFAULT
#include "../DefaultCPU/gemm.hh"
#elif defined CPU_ARMPL
#include "../ArmPL/gemm.hh"
#endif

#if defined GPU_DEFAULT
#include "../DefaultGPU/gemm.hh"
#elif defined GPU_CUBLAS
#include "../cuBLAS/gemm.hh"
#endif

template <typename T>
class doGemm {
 public:
  doGemm(const int iters, const int upperLimit)
      : iterations_(iters),
        upperLimit_(upperLimit),
        gemmCpu_(iterations_),
        gemmGpu_(iterations_) {
    static_assert((std::is_same_v<T, float> || std::is_same_v<T, double>) &&
                  "ERROR - doGemm can only be constructed using one of the "
                  "following types: [float, double].");
  }

  /** Run all problem types and write data to CSV files. */
  void collectData() {
    // Square Problem Sizes...
    std::ofstream csvFile = initCSVFile(std::string(CSV_DIR) + "/" +
                                        getKernelName() + "_square.csv");
    for (int dim = 1; dim <= upperLimit_; dim++) {
      const int M = dim, N = dim, K = dim;
      callKernels(csvFile, M, N, K);
    }
    // Close file
    csvFile.close();

    // Rectangular Problem Sizes:
    // Tall and thin (16M x K)...
    csvFile = initCSVFile(std::string(CSV_DIR) + "/" + getKernelName() +
                          "_rectangular_16MxK.csv");
    for (int dim = 16; dim <= upperLimit_; dim += 16) {
      const int M = dim, N = dim, K = (dim / 16);
      callKernels(csvFile, M, N, K);
    }
    // Close file
    csvFile.close();

    // Tall and thin (M x 32)...
    csvFile = initCSVFile(std::string(CSV_DIR) + "/" + getKernelName() +
                          "_rectangular_Mx32.csv");
    if (upperLimit_ >= 32) {
      for (int dim = 1; dim <= upperLimit_; dim++) {
        const int M = dim, N = dim, K = 32;
        callKernels(csvFile, M, N, K);
      }
    }
    // Close file
    csvFile.close();

    // Short and wide (M x 16K)...
    csvFile = initCSVFile(std::string(CSV_DIR) + "/" + getKernelName() +
                          "_rectangular_Mx16K.csv");
    for (int dim = 16; dim <= upperLimit_; dim += 16) {
      const int M = (dim / 16), N = (dim / 16), K = dim;
      callKernels(csvFile, M, N, K);
    }
    // Close file
    csvFile.close();

    // Short and wide (32 x K)...
    csvFile = initCSVFile(std::string(CSV_DIR) + "/" + getKernelName() +
                          "_rectangular_32xK.csv");
    if (upperLimit_ >= 32) {
      for (int dim = 1; dim <= upperLimit_; dim++) {
        const int M = 32, N = 32, K = dim;
        callKernels(csvFile, M, N, K);
      }
    }
    // Close file
    csvFile.close();
  }

 private:
  /** Call the appropriate CPU and GPU GEMM kernels. */
  void callKernels(std::ofstream& csvFile, const int M, const int N,
                   const int K) {
    const double probSize = calcKib(M, N, K);
    std::string kernelName = getKernelName();

    // Perform CPU kernel
    gemmCpu_.initialise(M, N, K);
    time_checksum cpuTime = gemmCpu_.compute();

    // Perform the GPU kernels
    // - ONCE : Offload to/from GPU once before all iterations and once after
    gemmGpu_.initialise(gpuOffloadType::once, M, N, K);
    time_checksum gpuTime_once = gemmGpu_.compute();

    // - ALWAYS: Offload to/from GPU every iteration
    gemmGpu_.initialise(gpuOffloadType::always, M, N, K);
    time_checksum gpuTime_always = gemmGpu_.compute();

    // - UNIFIED : data passed from host to device (and device to host) as
    //             needed
    gemmGpu_.initialise(gpuOffloadType::unified, M, N, K);
    time_checksum gpuTime_unified = gemmGpu_.compute();

// Make sure all checksums match if default GPU kernel not run
#if !defined GPU_DEFAULT
    if (!((std::fabs(cpuTime.checksum - gpuTime_once.checksum) < 0.05) &&
          (std::fabs(cpuTime.checksum - gpuTime_always.checksum) < 0.05) &&
          (std::fabs(cpuTime.checksum - gpuTime_unified.checksum) < 0.05))) {
      std::cerr << "ERROR - " << getKernelName()
                << " kernel checksums do not match:\n\tInput "
                   "dimensions: M="
                << M << ", N=" << N << ", K=" << K << std::endl;
      std::cerr << std::setprecision(10)
                << "\tCPU Checksum = " << cpuTime.checksum << std::endl;
      std::cerr << std::setprecision(10)
                << "\tGPU (Once) Checksum = " << gpuTime_once.checksum
                << std::endl;
      std::cerr << std::setprecision(10)
                << "\tGPU (Always) Checksum = " << gpuTime_always.checksum
                << std::endl;
      std::cerr << std::setprecision(10)
                << "\tGPU (Unified) Checksum = " << gpuTime_unified.checksum
                << std::endl;
      exit(1);
    }
#endif

    // Write lines to CSV file
    writeLineToCsv(
        csvFile, "cpu", kernelName, M, N, K, probSize, iterations_,
        cpuTime.runtime,
        calcGflops(calcFlops(M, N, K), iterations_, cpuTime.runtime));
    writeLineToCsv(
        csvFile, "gpu_offloadOnce", kernelName, M, N, K, probSize, iterations_,
        gpuTime_once.runtime,
        calcGflops(calcFlops(M, N, K), iterations_, gpuTime_once.runtime));
    writeLineToCsv(
        csvFile, "gpu_offloadAlways", kernelName, M, N, K, probSize,
        iterations_, gpuTime_always.runtime,
        calcGflops(calcFlops(M, N, K), iterations_, gpuTime_always.runtime));
    writeLineToCsv(
        csvFile, "gpu_unified", kernelName, M, N, K, probSize, iterations_,
        gpuTime_unified.runtime,
        calcGflops(calcFlops(M, N, K), iterations_, gpuTime_unified.runtime));
  }

  /** A function for calculating FLOPs performed by a GEMM.
   * C = alpha*AB + beta*C */
  uint64_t calcFlops(const int M, const int N, const int K) const {
    return ((ALPHA * (2 * (uint64_t)M * (uint64_t)N * (uint64_t)K)) +
            (BETA * (uint64_t)M * (uint64_t)N));
  }

  /** A function for calculating the total GEMM problem size in KiB. */
  double calcKib(const int M, const int N, const int K) const {
    uint64_t M_ = (uint64_t)M, N_ = (uint64_t)N, K_ = (uint64_t)K;
    uint64_t probSize = (M_ * K_) + (K_ * N_) + (M_ * N_);
    return ((double)(probSize * (sizeof(T))) / 1024);
  }

  /** Get the name of the kernel being run. */
  std::string getKernelName() const {
    switch (sizeof(T)) {
      case 4:
        return "sgemm";
      case 8:
        return "dgemm";
      default:
        return "unknown";
    }
  }

  /** The number of iterations to perform per problem size. */
  const int iterations_;

  /** The maximum value of the largest problem size dimention. */
  const int upperLimit_;

  cpu::gemm_cpu<T> gemmCpu_;
  gpu::gemm_gpu<T> gemmGpu_;
};