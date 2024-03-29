#pragma once
#include <sstream>
#include <type_traits>

#include "helpers.hh"
#include "tablePrinter.hh"
#include "utilities.hh"

#if defined CPU_ARMPL
#include "../ArmPL/gemm.hh"
#elif defined CPU_ONEMKL
#include "../oneMKL/CPU/gemm.hh"
#endif

#if defined GPU_CUBLAS
#include "../cuBLAS/gemm.hh"
#elif defined GPU_ONEMKL
#include "../oneMKL/GPU/gemm.hh"
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

/** `T` represents the type of kernel that will be run - i.e. T=float is for
 *      SGEMM. */
template <typename T_CPU, typename T_GPU = T_CPU>
class doGemm {
 public:
  doGemm(const int iters, const int startDim, const int upperLimit,
         const bool cpuEnabled = true, const bool gpuEnabled = true)
      : iterations_(iters),
        startDimention_(startDim),
        upperLimit_(upperLimit),
        doCPU_(cpuEnabled),
        doGPU_(gpuEnabled)
#if CPU_ENABLED
        ,
        gemmCpu_(iterations_)
#endif
#if GPU_ENABLED
        ,
        gemmGpu_(iterations_)
#endif
  {
#if CPU_ENABLED && GPU_ENABLED
    static_assert((sizeof(T_CPU) == sizeof(T_GPU)) &&
                  "ERROR - The data size of T_CPU and T_GPU must be the same.");
#endif

#if CPU_ENABLED
    static_assert((
        std::is_same_v<T_CPU, float> || std::is_same_v<T_CPU, double>
        || std::is_same_v<T_CPU, CPU_FP16>) && "ERROR - doGemm can only be constructed using one of the "
          "following types: [half, float, double].");
#endif
#if GPU_ENABLED
    static_assert((
        std::is_same_v<T_GPU, float> || std::is_same_v<T_GPU, double>
        || std::is_same_v<T_GPU, GPU_FP16> ) && "ERROR - doGemm can only be constructed using one of the "
           "following types: [half, float, double].");
#endif
  }

  /** Run all problem types and write data to CSV files. */
  void collectData() {
    // Square Problem Sizes...
    // Re-initialise offload threshold structures
    cpuGpu_always_ = cpuGpu_offloadThreshold();
    cpuGpu_once_ = cpuGpu_offloadThreshold();
    cpuGpu_unified_ = cpuGpu_offloadThreshold();
    std::ofstream csvFile =
        initCSVFile(std::string(CSV_DIR) + "/" + getKernelName() +
                    "_square_square_M=N=K.csv");
    for (int dim = startDimention_; dim <= upperLimit_; dim++) {
      // M = dim, N = dim, K = dim;
      callKernels(csvFile, dim, dim, dim);
    }
    // Close file
    csvFile.close();
#if CPU_ENABLED && GPU_ENABLED
    if (doCPU_ && doGPU_) {
      // Print offload results to stdout
      printOffloadThreshold("Square x Square (M=N=K)");
    }
#endif

    // Rectangular Problem Sizes:
    // Tall and thin x Short and wide
    // Re-initialise offload threshold structures
    cpuGpu_always_ = cpuGpu_offloadThreshold();
    cpuGpu_once_ = cpuGpu_offloadThreshold();
    cpuGpu_unified_ = cpuGpu_offloadThreshold();
    csvFile = initCSVFile(std::string(CSV_DIR) + "/" + getKernelName() +
                          "_tall-thin_short-wide_M=N_M=16K.csv");
    int K = startDimention_;
    int M = 16 * K;
    int N = 16 * K;
    while (M <= upperLimit_) {
      callKernels(csvFile, M, N, K);
      M += 16;
      N += 16;
      K++;
    }
    // Close file
    csvFile.close();
#if CPU_ENABLED && GPU_ENABLED
    if (doCPU_ && doGPU_) {
      // Print offload results to stdout
      printOffloadThreshold("Tall-and-Thin x Short-and-Wide (M=N, M=16K)");
    }
#endif

    // Tall and thin x Short and wide
    // Re-initialise offload threshold structures
    cpuGpu_always_ = cpuGpu_offloadThreshold();
    cpuGpu_once_ = cpuGpu_offloadThreshold();
    cpuGpu_unified_ = cpuGpu_offloadThreshold();
    csvFile = initCSVFile(std::string(CSV_DIR) + "/" + getKernelName() +
                          "_tall-thin_short-wide_M=N_K=32.csv");
    if (upperLimit_ >= 32) {
      for (int dim = startDimention_; dim <= upperLimit_; dim++) {
        // M = dim, N = dim, K = 32;
        callKernels(csvFile, dim, dim, 32);
      }
    }
    // Close file
    csvFile.close();
#if CPU_ENABLED && GPU_ENABLED
    if (doCPU_ && doGPU_) {
      // Print offload results to stdout
      printOffloadThreshold("Tall-and-Thin x Short-and-Wide (M=N, K=32)");
    }
#endif

    // Short and wide x Tall and thin
    // Re-initialise offload threshold structures
    cpuGpu_always_ = cpuGpu_offloadThreshold();
    cpuGpu_once_ = cpuGpu_offloadThreshold();
    cpuGpu_unified_ = cpuGpu_offloadThreshold();
    csvFile = initCSVFile(std::string(CSV_DIR) + "/" + getKernelName() +
                          "_short-wide_tall-thin_M=N_K=16M.csv");
    M = startDimention_;
    N = startDimention_;
    K = 16 * M;
    while (K <= upperLimit_) {
      callKernels(csvFile, M, N, K);
      M++;
      N++;
      K += 16;
    }
    // Close file
    csvFile.close();
#if CPU_ENABLED && GPU_ENABLED
    if (doCPU_ && doGPU_) {
      // Print offload results to stdout
      printOffloadThreshold("Short-and-Wide x Tall-and-Thin (M=N, K=16M)");
    }
#endif

    // Short and wide x Tall and thin
    // Re-initialise offload threshold structures
    cpuGpu_always_ = cpuGpu_offloadThreshold();
    cpuGpu_once_ = cpuGpu_offloadThreshold();
    cpuGpu_unified_ = cpuGpu_offloadThreshold();
    csvFile = initCSVFile(std::string(CSV_DIR) + "/" + getKernelName() +
                          "_short-wide_tall-thin_M=N=32_K.csv");
    if (upperLimit_ >= 32) {
      for (int dim = startDimention_; dim <= upperLimit_; dim++) {
        // M = 32, N = 32, K = dim;
        callKernels(csvFile, 32, 32, dim);
      }
    }
    // Close file
    csvFile.close();
#if CPU_ENABLED && GPU_ENABLED
    if (doCPU_ && doGPU_) {
      // Print offload results to stdout
      printOffloadThreshold("Short-and-Wide x Tall-and-Thin (M=N=32, K)");
    }
#endif

    // Tall and Thin x Square
    // Re-initialise offload threshold structures
    cpuGpu_always_ = cpuGpu_offloadThreshold();
    cpuGpu_once_ = cpuGpu_offloadThreshold();
    cpuGpu_unified_ = cpuGpu_offloadThreshold();
    csvFile = initCSVFile(std::string(CSV_DIR) + "/" + getKernelName() +
                          "_tall-thin_square_K=N_M=16K.csv");
    K = startDimention_;
    N = startDimention_;
    M = 16 * K;
    while (M <= upperLimit_) {
      callKernels(csvFile, M, N, K);
      M += 16;
      N++;
      K++;
    }
    // Close file
    csvFile.close();
#if CPU_ENABLED && GPU_ENABLED
    if (doCPU_ && doGPU_) {
      // Print offload results to stdout
      printOffloadThreshold("Tall-and-Thin x Square (K=N, M=16K)");
    }
#endif

    // Tall and Thin x Square
    // Re-initialise offload threshold structures
    cpuGpu_always_ = cpuGpu_offloadThreshold();
    cpuGpu_once_ = cpuGpu_offloadThreshold();
    cpuGpu_unified_ = cpuGpu_offloadThreshold();
    csvFile = initCSVFile(std::string(CSV_DIR) + "/" + getKernelName() +
                          "_tall-thin_square_K=N=32_M.csv");
    if (upperLimit_ >= 32) {
      for (int dim = startDimention_; dim <= upperLimit_; dim++) {
        // M = dim, N = 32, K = 32;
        callKernels(csvFile, dim, 32, 32);
      }
    }
    // Close file
    csvFile.close();
#if CPU_ENABLED && GPU_ENABLED
    if (doCPU_ && doGPU_) {
      // Print offload results to stdout
      printOffloadThreshold("Tall-and-Thin x Square (M, K=N=32)");
    }
#endif

    // Square x Short and Wide
    // Re-initialise offload threshold structures
    cpuGpu_always_ = cpuGpu_offloadThreshold();
    cpuGpu_once_ = cpuGpu_offloadThreshold();
    cpuGpu_unified_ = cpuGpu_offloadThreshold();
    csvFile = initCSVFile(std::string(CSV_DIR) + "/" + getKernelName() +
                          "_square_short-wide_M=K_N=16K.csv");
    M = startDimention_;
    K = startDimention_;
    N = 16 * K;
    while (N <= upperLimit_) {
      callKernels(csvFile, M, N, K);
      M++;
      N += 16;
      K++;
    }
    // Close file
    csvFile.close();
#if CPU_ENABLED && GPU_ENABLED
    if (doCPU_ && doGPU_) {
      // Print offload results to stdout
      printOffloadThreshold("Square x Short-and-Wide (M=K, N=16K)");
    }
#endif

    // Square x Short and Wide
    // Re-initialise offload threshold structures
    cpuGpu_always_ = cpuGpu_offloadThreshold();
    cpuGpu_once_ = cpuGpu_offloadThreshold();
    cpuGpu_unified_ = cpuGpu_offloadThreshold();
    csvFile = initCSVFile(std::string(CSV_DIR) + "/" + getKernelName() +
                          "_square_short-wide_M=K=32_N.csv");
    if (upperLimit_ >= 32) {
      for (int dim = startDimention_; dim <= upperLimit_; dim++) {
        // M = 32, N = dim, K = 32;
        callKernels(csvFile, 32, dim, 32);
      }
    }
    // Close file
    csvFile.close();
#if CPU_ENABLED && GPU_ENABLED
    if (doCPU_ && doGPU_) {
      // Print offload results to stdout
      printOffloadThreshold("Square x Short-and-Wide (M=K=32, N)");
    }
#endif
  }

 private:
  /** Call the appropriate CPU and GPU GEMM kernels. */
  void callKernels(std::ofstream& csvFile, const int M, const int N,
                   const int K) {
    const double probSize = calcKib(M, N, K);
    const uint64_t flops = calcFlops(M, N, K);
    std::string kernelName = getKernelName();

    time_checksum_gflop cpuResult;
    time_checksum_gflop gpuResult_once;
    time_checksum_gflop gpuResult_always;
    time_checksum_gflop gpuResult_unified;

// Perform CPU kernel
#if CPU_ENABLED
    if (doCPU_) {
      gemmCpu_.initialise(M, N, K);
      cpuResult = gemmCpu_.compute();
      cpuResult.gflops = calcGflops(flops, iterations_, cpuResult.runtime);
      // Write result to CSV file
      writeLineToCsv(csvFile, "cpu", kernelName, M, N, K, probSize, iterations_,
                     cpuResult.runtime, cpuResult.gflops);
    }
#endif

// Perform the GPU kernels
#if GPU_ENABLED
    if (doGPU_) {
      // - ONCE : Offload to/from GPU once before all iterations and once
      // after
      gemmGpu_.initialise(gpuOffloadType::once, M, N, K);
      gpuResult_once = gemmGpu_.compute();
      gpuResult_once.gflops =
          calcGflops(flops, iterations_, gpuResult_once.runtime);

      // - ALWAYS: Offload to/from GPU every iteration
      gemmGpu_.initialise(gpuOffloadType::always, M, N, K);
      gpuResult_always = gemmGpu_.compute();
      gpuResult_always.gflops =
          calcGflops(flops, iterations_, gpuResult_always.runtime);

      // - UNIFIED : data passed from host to device (and device to host) as
      //             needed
      gemmGpu_.initialise(gpuOffloadType::unified, M, N, K);
      gpuResult_unified = gemmGpu_.compute();
      gpuResult_unified.gflops =
          calcGflops(flops, iterations_, gpuResult_unified.runtime);

      // Write results to CSV file
      writeLineToCsv(csvFile, "gpu_offloadOnce", kernelName, M, N, K, probSize,
                     iterations_, gpuResult_once.runtime,
                     gpuResult_once.gflops);
      writeLineToCsv(csvFile, "gpu_offloadAlways", kernelName, M, N, K,
                     probSize, iterations_, gpuResult_always.runtime,
                     gpuResult_always.gflops);
      writeLineToCsv(csvFile, "gpu_unified", kernelName, M, N, K, probSize,
                     iterations_, gpuResult_unified.runtime,
                     gpuResult_unified.gflops);
    }
#endif

#if CPU_ENABLED && GPU_ENABLED
    if (doCPU_ && doGPU_) {
      // Make sure all checksums match if CPU and GPU kernels are run.
      //  - The majority of BLAS Libraries guarentee the same result if a
      //  function
      //    is called multiple times. Given all input matrices are identical for
      //    each GPU offload type, we need only to compare the CPU and GPU
      //    checksums.
      checkChecksums(cpuResult, gpuResult_once, gpuResult_always,
                     gpuResult_unified, M, N, K);

      // Check if offload structs should be reset
      checkOffloadStructReset(cpuResult, gpuResult_once, gpuResult_always,
                              gpuResult_unified);

      // Check if offload threshold has been achieved for each GPU offload type.
      updateOffloadStructs(cpuResult, gpuResult_once, gpuResult_always,
                           gpuResult_unified, M, N, K, probSize);
    }
#endif
  }

  /** Ensure all CPU and GPU checksums are within the permitted limit of
   * eachother. */
  void checkChecksums(time_checksum_gflop cpuResult,
                      time_checksum_gflop gpuResult_once,
                      time_checksum_gflop gpuResult_always,
                      time_checksum_gflop gpuResult_unified, const int M,
                      const int N, const int K) {
    // Ensure that each checksum difference is less than 0.1%
    const double hundredOverChecksum = 100 / std::fabs(cpuResult.checksum);
    if (((std::fabs(cpuResult.checksum - gpuResult_once.checksum) *
          hundredOverChecksum)) > 0.1 &&
        ((std::fabs(cpuResult.checksum - gpuResult_always.checksum) *
          hundredOverChecksum)) > 0.1 &&
        ((std::fabs(cpuResult.checksum - gpuResult_unified.checksum) *
          hundredOverChecksum)) > 0.1) {
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
  }

  /** Check whether the offload structures need to be reset; and doing so if
   * required.
   *   - If CPU.gflops >= GPU.gflops, then reset offload structures as GPU may
   *     not necessarily have reached the offload threshold. */
  void checkOffloadStructReset(time_checksum_gflop cpuResult,
                               time_checksum_gflop gpuResult_once,
                               time_checksum_gflop gpuResult_always,
                               time_checksum_gflop gpuResult_unified) {
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
  }

  /** Update the offload threshold structs if GPU.gflops > CPU.gflops. */
  void updateOffloadStructs(time_checksum_gflop cpuResult,
                            time_checksum_gflop gpuResult_once,
                            time_checksum_gflop gpuResult_always,
                            time_checksum_gflop gpuResult_unified, const int M,
                            const int N, const int K, const double probSize) {
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
    return ((double)(probSize * (sizeof(T_CPU))) / 1024);
  }

  /** Get the name of the kernel being run. */
  std::string getKernelName() const {
#if !CPU_ENABLED
    switch (sizeof(T_GPU))
#else
    switch (sizeof(T_CPU))
#endif
    {
      case 2:
        return "hgemm";
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

  /** The value of the first probelm size dimention run. */
  const int startDimention_;

  /** The maximum value of the largest problem size dimention. */
  const int upperLimit_;

  /** Whether the CPU kernels should be run. */
  const bool doCPU_ = true;

  /** Whether the GPU kernels should be run. */
  const bool doGPU_ = true;

#if CPU_ENABLED
  /** The GEMM CPU kernel. */
  cpu::gemm_cpu<T_CPU> gemmCpu_;
#endif

#if GPU_ENABLED
  /** The GEMM GPU kernel. */
  gpu::gemm_gpu<T_GPU> gemmGpu_;
#endif

  /** The point at which offloading to GPU (offload once) becomes worthwhile. */
  cpuGpu_offloadThreshold cpuGpu_once_;

  /** The point at which offloading to GPU (offload always) becomes worthwhile.
   */
  cpuGpu_offloadThreshold cpuGpu_always_;

  /** The point at which offloading to GPU (unified memory) becomes worthwhile.
   */
  cpuGpu_offloadThreshold cpuGpu_unified_;
};