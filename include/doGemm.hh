#pragma once
#include <sstream>
#include <type_traits>

#include "helpers.hh"
#include "tablePrinter.hh"
#include "utilities.hh"

#if defined CPU_DEFAULT
#include "../DefaultCPU/gemm.hh"
#elif defined CPU_ARMPL
#include "../ArmPL/gemm.hh"
#elif defined CPU_ONEMKL
#include "../oneMKL/CPU/gemm.hh"
#endif

#if defined GPU_DEFAULT
#include "../DefaultGPU/gemm.hh"
#elif defined GPU_CUBLAS
#include "../cuBLAS/gemm.hh"
#endif

/** Struct to hold key values at the point at which offloading to GPU becomes
 * worthwhile. */
struct cpuGpu_offloadThreshold {
  double cpuGflops = 0.0;
  double gpuGflops = 0.0;
  double probSize_kib = 0.0;
  int M = 0;
  int N = 0;
  int K = 0;
};

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
    // Re-initialise offload threshold structures
    cpuGpu_always_ = cpuGpu_offloadThreshold();
    cpuGpu_once_ = cpuGpu_offloadThreshold();
    cpuGpu_unified_ = cpuGpu_offloadThreshold();
    std::ofstream csvFile = initCSVFile(std::string(CSV_DIR) + "/" +
                                        getKernelName() + "_square.csv");
    for (int dim = 1; dim <= upperLimit_; dim++) {
      const int M = dim, N = dim, K = dim;
      callKernels(csvFile, M, N, K);
    }
    // Close file
    csvFile.close();
    // Print offload results to stdout
    printOffloadThreshold("Square");

    // Rectangular Problem Sizes:
    // Tall and thin (16M x K)...
    // Re-initialise offload threshold structures
    cpuGpu_always_ = cpuGpu_offloadThreshold();
    cpuGpu_once_ = cpuGpu_offloadThreshold();
    cpuGpu_unified_ = cpuGpu_offloadThreshold();
    csvFile = initCSVFile(std::string(CSV_DIR) + "/" + getKernelName() +
                          "_rectangular_16MxK.csv");
    for (int dim = 16; dim <= upperLimit_; dim += 16) {
      const int M = dim, N = dim, K = (dim / 16);
      callKernels(csvFile, M, N, K);
    }
    // Close file
    csvFile.close();
    // Print offload results to stdout
    printOffloadThreshold("Tall and Thin (16M x K)");

    // Tall and thin (M x 32)...
    // Re-initialise offload threshold structures
    cpuGpu_always_ = cpuGpu_offloadThreshold();
    cpuGpu_once_ = cpuGpu_offloadThreshold();
    cpuGpu_unified_ = cpuGpu_offloadThreshold();
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
    // Print offload results to stdout
    printOffloadThreshold("Tall and Thin (M x 32)");

    // Short and wide (M x 16K)...
    // Re-initialise offload threshold structures
    cpuGpu_always_ = cpuGpu_offloadThreshold();
    cpuGpu_once_ = cpuGpu_offloadThreshold();
    cpuGpu_unified_ = cpuGpu_offloadThreshold();
    csvFile = initCSVFile(std::string(CSV_DIR) + "/" + getKernelName() +
                          "_rectangular_Mx16K.csv");
    for (int dim = 16; dim <= upperLimit_; dim += 16) {
      const int M = (dim / 16), N = (dim / 16), K = dim;
      callKernels(csvFile, M, N, K);
    }
    // Close file
    csvFile.close();
    // Print offload results to stdout
    printOffloadThreshold("Short and Wide (M x 16K)");

    // Short and wide (32 x K)...
    // Re-initialise offload threshold structures
    cpuGpu_always_ = cpuGpu_offloadThreshold();
    cpuGpu_once_ = cpuGpu_offloadThreshold();
    cpuGpu_unified_ = cpuGpu_offloadThreshold();
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
    // Print offload results to stdout
    printOffloadThreshold("Short and Wide (32 x K)");
  }

 private:
  /** Call the appropriate CPU and GPU GEMM kernels. */
  void callKernels(std::ofstream& csvFile, const int M, const int N,
                   const int K) {
    const double probSize = calcKib(M, N, K);
    const uint64_t flops = calcFlops(M, N, K);
    std::string kernelName = getKernelName();

    // Perform CPU kernel
    gemmCpu_.initialise(M, N, K);
    time_checksum_gflop cpuResult = gemmCpu_.compute();
    cpuResult.gflops = calcGflops(flops, iterations_, cpuResult.runtime);

    // Perform the GPU kernels
    // - ONCE : Offload to/from GPU once before all iterations and once
    // after
    gemmGpu_.initialise(gpuOffloadType::once, M, N, K);
    time_checksum_gflop gpuResult_once = gemmGpu_.compute();
    gpuResult_once.gflops =
        calcGflops(flops, iterations_, gpuResult_once.runtime);

    // - ALWAYS: Offload to/from GPU every iteration
    gemmGpu_.initialise(gpuOffloadType::always, M, N, K);
    time_checksum_gflop gpuResult_always = gemmGpu_.compute();
    gpuResult_always.gflops =
        calcGflops(flops, iterations_, gpuResult_always.runtime);

    // - UNIFIED : data passed from host to device (and device to host) as
    //             needed
    gemmGpu_.initialise(gpuOffloadType::unified, M, N, K);
    time_checksum_gflop gpuResult_unified = gemmGpu_.compute();
    gpuResult_unified.gflops =
        calcGflops(flops, iterations_, gpuResult_unified.runtime);

// Make sure all checksums match if default GPU kernel not run
#if !defined GPU_DEFAULT
    if (!((std::fabs(cpuResult.checksum - gpuResult_once.checksum) <
           CHECK_ERROR) &&
          (std::fabs(cpuResult.checksum - gpuResult_always.checksum) <
           CHECK_ERROR) &&
          (std::fabs(cpuResult.checksum - gpuResult_unified.checksum) <
           CHECK_ERROR))) {
      std::cerr << "ERROR - " << getKernelName()
                << " kernel checksums do not match:\n\tInput "
                   "dimensions: M="
                << M << ", N=" << N << ", K=" << K << std::endl;
      std::cerr << std::setprecision(10)
                << "\tCPU Checksum = " << cpuResult.checksum << std::endl;
      std::cerr << std::setprecision(10)
                << "\tGPU (Once) Checksum = " << gpuResult_once.checksum
                << std::endl;
      std::cerr << std::setprecision(10)
                << "\tGPU (Always) Checksum = " << gpuResult_always.checksum
                << std::endl;
      std::cerr << std::setprecision(10)
                << "\tGPU (Unified) Checksum = " << gpuResult_unified.checksum
                << std::endl;
      exit(1);
    }

    // TODO: clean up the below logic
    // If CPU.gflops > GPU.gflops, reset offload structures
    if ((cpuGpu_once_.M != 0) && cpuResult.gflops >= gpuResult_once.gflops) {
      cpuGpu_once_.cpuGflops = 0.0;
      cpuGpu_once_.gpuGflops = 0.0;
      cpuGpu_once_.probSize_kib = 0.0;
      cpuGpu_once_.M = 0;
      cpuGpu_once_.N = 0;
      cpuGpu_once_.K = 0;
    }
    if ((cpuGpu_always_.M != 0) &&
        cpuResult.gflops >= gpuResult_always.gflops) {
      cpuGpu_always_.cpuGflops = 0.0;
      cpuGpu_always_.gpuGflops = 0.0;
      cpuGpu_always_.probSize_kib = 0.0;
      cpuGpu_always_.M = 0;
      cpuGpu_always_.N = 0;
      cpuGpu_always_.K = 0;
    }
    if ((cpuGpu_unified_.M != 0) &&
        cpuResult.gflops >= gpuResult_unified.gflops) {
      cpuGpu_unified_.cpuGflops = 0.0;
      cpuGpu_unified_.gpuGflops = 0.0;
      cpuGpu_unified_.probSize_kib = 0.0;
      cpuGpu_unified_.M = 0;
      cpuGpu_unified_.N = 0;
      cpuGpu_unified_.K = 0;
    }

    // Check if offload threshold has been achieved for each GPU offload type.
    if ((cpuGpu_once_.M == 0) && cpuResult.gflops < gpuResult_once.gflops) {
      cpuGpu_once_.cpuGflops = cpuResult.gflops;
      cpuGpu_once_.gpuGflops = gpuResult_once.gflops;
      cpuGpu_once_.probSize_kib = probSize;
      cpuGpu_once_.M = M;
      cpuGpu_once_.N = N;
      cpuGpu_once_.K = K;
    }
    if ((cpuGpu_always_.M == 0) && cpuResult.gflops < gpuResult_always.gflops) {
      cpuGpu_always_.cpuGflops = cpuResult.gflops;
      cpuGpu_always_.gpuGflops = gpuResult_always.gflops;
      cpuGpu_always_.probSize_kib = probSize;
      cpuGpu_always_.M = M;
      cpuGpu_always_.N = N;
      cpuGpu_always_.K = K;
    }
    if ((cpuGpu_unified_.M == 0) &&
        cpuResult.gflops < gpuResult_unified.gflops) {
      cpuGpu_unified_.cpuGflops = cpuResult.gflops;
      cpuGpu_unified_.gpuGflops = gpuResult_unified.gflops;
      cpuGpu_unified_.probSize_kib = probSize;
      cpuGpu_unified_.M = M;
      cpuGpu_unified_.N = N;
      cpuGpu_unified_.K = K;
    }
#endif

    // Write lines to CSV file
    writeLineToCsv(csvFile, "cpu", kernelName, M, N, K, probSize, iterations_,
                   cpuResult.runtime, cpuResult.gflops);
    writeLineToCsv(csvFile, "gpu_offloadOnce", kernelName, M, N, K, probSize,
                   iterations_, gpuResult_once.runtime, gpuResult_once.gflops);
    writeLineToCsv(csvFile, "gpu_offloadAlways", kernelName, M, N, K, probSize,
                   iterations_, gpuResult_always.runtime,
                   gpuResult_always.gflops);
    writeLineToCsv(csvFile, "gpu_unified", kernelName, M, N, K, probSize,
                   iterations_, gpuResult_unified.runtime,
                   gpuResult_unified.gflops);
  }

  /** A function for calculating FLOPs performed by a GEMM.
   * C = alpha*AB + beta*C */
  constexpr uint64_t calcFlops(const int M, const int N, const int K) const {
    return ((ALPHA * (2 * (uint64_t)M * (uint64_t)N * (uint64_t)K)) +
            (BETA * (uint64_t)M * (uint64_t)N));
  }

  /** A function for calculating the total GEMM problem size in KiB. */
  constexpr double calcKib(const int M, const int N, const int K) const {
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

  /** Print to stdout the offload thresholds. */
  void printOffloadThreshold(std::string problemName) const {
    std::vector<std::string> header = {
        "Device",  "M",          "N", "K", "Total Prob. Size (KiB)",
        "GFLOP/s", "CPU GFLOP/s"};

    std::vector<std::vector<std::string>> rows;
    // Initialise GPU_Once row
    std::stringstream probSize_o;
    std::stringstream gpuGflops_o;
    std::stringstream cpuGflops_o;
    probSize_o << std::fixed << std::setprecision(2)
               << cpuGpu_once_.probSize_kib;
    gpuGflops_o << std::fixed << std::setprecision(2) << cpuGpu_once_.gpuGflops;
    cpuGflops_o << std::fixed << std::setprecision(2) << cpuGpu_once_.cpuGflops;
    if (cpuGpu_once_.M == 0) {
      // No offload threshold found
      rows.push_back({"GPU (Offload Once)", std::to_string(0),
                      std::to_string(0), std::to_string(0), probSize_o.str(),
                      "N/A", "N/A"});
    } else {
      rows.push_back({"GPU (Offload Once)", std::to_string(cpuGpu_once_.M),
                      std::to_string(cpuGpu_once_.N),
                      std::to_string(cpuGpu_once_.K), probSize_o.str(),
                      gpuGflops_o.str(), cpuGflops_o.str()});
    }

    // Initialise GPU_always row
    std::stringstream probSize_a;
    std::stringstream gpuGflops_a;
    std::stringstream cpuGflops_a;
    probSize_a << std::fixed << std::setprecision(2)
               << cpuGpu_always_.probSize_kib;
    gpuGflops_a << std::fixed << std::setprecision(2)
                << cpuGpu_always_.gpuGflops;
    cpuGflops_a << std::fixed << std::setprecision(2)
                << cpuGpu_always_.cpuGflops;
    if (cpuGpu_always_.M == 0) {
      // No offload threshold found
      rows.push_back({"GPU (Offload Always)", std::to_string(0),
                      std::to_string(0), std::to_string(0), probSize_a.str(),
                      "N/A", "N/A"});
    } else {
      rows.push_back({"GPU (Offload Always)", std::to_string(cpuGpu_always_.M),
                      std::to_string(cpuGpu_always_.N),
                      std::to_string(cpuGpu_always_.K), probSize_a.str(),
                      gpuGflops_a.str(), cpuGflops_a.str()});
    }

    // Initialise GPU_unified row
    std::stringstream probSize_u;
    std::stringstream gpuGflops_u;
    std::stringstream cpuGflops_u;
    probSize_u << std::fixed << std::setprecision(2)
               << cpuGpu_unified_.probSize_kib;
    gpuGflops_u << std::fixed << std::setprecision(2)
                << cpuGpu_unified_.gpuGflops;
    cpuGflops_u << std::fixed << std::setprecision(2)
                << cpuGpu_unified_.cpuGflops;
    if (cpuGpu_unified_.M == 0) {
      // No offload threshold found
      rows.push_back({"GPU (Unified Memory)", std::to_string(0),
                      std::to_string(0), std::to_string(0), probSize_u.str(),
                      "N/A", "N/A"});
    } else {
      rows.push_back({"GPU (Unified Memory)", std::to_string(cpuGpu_unified_.M),
                      std::to_string(cpuGpu_unified_.N),
                      std::to_string(cpuGpu_unified_.K), probSize_u.str(),
                      gpuGflops_u.str(), cpuGflops_u.str()});
    }

    // Print table
    tablePrinter tPrinter(
        problemName + " Problem Domian GPU Offload Thresholds:", header, rows);
    tPrinter.print(1);
  }

  /** The number of iterations to perform per problem size. */
  const int iterations_;

  /** The maximum value of the largest problem size dimention. */
  const int upperLimit_;

  /** The GEMM CPU kernel. */
  cpu::gemm_cpu<T> gemmCpu_;

  /** The GEMM GPU kernel. */
  gpu::gemm_gpu<T> gemmGpu_;

  /** The point at which offloading to GPU (offload once) becomes worthwhile. */
  cpuGpu_offloadThreshold cpuGpu_once_;

  /** The point at which offloading to GPU (offload always) becomes worthwhile.
   */
  cpuGpu_offloadThreshold cpuGpu_always_;

  /** The point at which offloading to GPU (unified memory) becomes worthwhile.
   */
  cpuGpu_offloadThreshold cpuGpu_unified_;
};