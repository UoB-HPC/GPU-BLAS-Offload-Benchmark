#pragma once
#include <type_traits>

#include "helpers.hh"
#include "utilities.hh"

#if defined CPU_DEFAULT
#include "../DefaultCPU/gemm.hh"
#include "../DefaultCPU/spGemm.hh"
#elif defined CPU_ARMPL
#include "../ArmPL/gemm.hh"
#endif

#if defined GPU_DEFAULT
#include "../DefaultGPU/gemm.hh"
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
        callDenseKernels(csvFile, M, N, K);
    }
    // Close file
    csvFile.close();

    // Rectangular Problem Sizes:
    // Tall and thin (16M x K)...
    csvFile = initCSVFile(std::string(CSV_DIR) + "/" + getKernelName() +
                          "_rectangular_16MxK.csv");
    for (int dim = 16; dim <= upperLimit_; dim += 16) {
      const int M = dim, N = dim, K = (dim / 16);
        callDenseKernels(csvFile, M, N, K);
    }
    // Close file
    csvFile.close();

    // Tall and thin (M x 32)...
    csvFile = initCSVFile(std::string(CSV_DIR) + "/" + getKernelName() +
                          "_rectangular_Mx32.csv");
    if (upperLimit_ >= 32) {
      for (int dim = 1; dim <= upperLimit_; dim++) {
        const int M = dim, N = dim, K = 32;
          callDenseKernels(csvFile, M, N, K);
      }
    }
    // Close file
    csvFile.close();

    // Short and wide (M x 16K)...
    csvFile = initCSVFile(std::string(CSV_DIR) + "/" + getKernelName() +
                          "_rectangular_Mx16K.csv");
    for (int dim = 16; dim <= upperLimit_; dim += 16) {
      const int M = (dim / 16), N = (dim / 16), K = dim;
        callDenseKernels(csvFile, M, N, K);
    }
    // Close file
    csvFile.close();

    // Short and wide (32 x K)...
    csvFile = initCSVFile(std::string(CSV_DIR) + "/" + getKernelName() +
                          "_rectangular_32xK.csv");
    if (upperLimit_ >= 32) {
      for (int dim = 1; dim <= upperLimit_; dim++) {
        const int M = 32, N = 32, K = dim;
          callDenseKernels(csvFile, M, N, K);
      }
    }

    // Sparse graph matrix (N x N)
    csvFile = initCSVFile(std::string(CSV_DIR) + "/" + getKernelName() +
                         "_sparse_graph.csv");
    for (int dim = 1; dim <= upperLimit_; dim++) {
        callSparseKernels(csvFile, dim);
    }
    // Close file
    csvFile.close();
  }

 private:
  /** Call the appropriate CPU and GPU GEMM kernels. */
  void callDenseKernels(std::ofstream& csvFile, const int M, const int N,
                        const int K) {
    const double probSize = calcKib(M, N, K);
    std::string kernelName = getKernelName();
    // Perform CPU
    gemmCpu_.initialise(M, N, K);
    double cpuTime = gemmCpu_.compute();
    writeLineToCsv(csvFile, "cpu", kernelName, M, N, K, 0, probSize,
                   iterations_, cpuTime,
                   calcGflops(calcFlops(M, N, K), iterations_, cpuTime));

      // Perform GPU
    // - Offload to/from GPU once before all iterations and once after
    gemmGpu_.initialise(true, M, N, K);
    double gpuTime_once = gemmGpu_.compute();
    writeLineToCsv(csvFile, "gpu_offloadOnce", kernelName, M, N, K, 0, probSize,
                   iterations_, gpuTime_once,
                   calcGflops(calcFlops(M, N, K), iterations_, gpuTime_once));
    // - Offload to/from GPU every iteration
    gemmGpu_.initialise(false, M, N, K);
    double gpuTime_every = gemmGpu_.compute();
    writeLineToCsv(csvFile, "gpu_offloadAlways", kernelName, M, N, K, 0, probSize,
                   iterations_, gpuTime_every,
                   calcGflops(calcFlops(M, N, K), iterations_, gpuTime_every));
  }

  void callSparseKernels(std::ofstream& csvFile, const int N, const double sparsity) {
      const double probSize = calcKib(N, N, N);
      std::string kernelName = getKernelName();

      // Perform CPU
      spGemmCpu_.initialise(N, sparsity);
      double cpuTime = spGemmCpu_.compute();
      writeLineToCsv(csvFile, "cpu", kernelName, N, N, N, sparsity, probSize,
                     iterations_, cpuTime,
                     calcGflops(calcFlops(N, N, N), iterations_, cpuTime));
      /** ToDo -- GPU stuff
      // Perform GPU
      // - Offload to/from GPU once before all iterations and once after
      spGemmGpu_.initialise(true, N, N, N);
      double gpuTime_once = spGemmGpu_.compute();
      writeLineToCsv(csvFile, "gpu_offloadOnce", kernelName, N, N, N, sparsity, probSize,
                     iterations_, gpuTime_once,
                     calcGflops(calcFlops(N, N, N), iterations_, gpuTime_once));
      // - Offload to/from GPU every iteration
      spGemmGpu_.initialise(false, N, N, N);
      double gpuTime_every = spGemmGpu_.compute();
      writeLineToCsv(csvFile, "gpu_offloadAlways", kernelName, N, N, N, sparsity, probSize,
                     iterations_, gpuTime_every,
                     calcGflops(calcFlops(N, N, N), iterations_, gpuTime_every));
                     */
  }

  /** A function for calculating FLOPs performed by a GEMM. */
  uint64_t calcFlops(const int M, const int N, const int K) const {
    return ((uint64_t)M * (uint64_t)N * (uint64_t)K * 2);
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
  cpu::spGemm_cpu<T> spGemmCpu_;
  gpu::gemm_gpu<T> gemmGpu_;
};