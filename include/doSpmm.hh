#pragma once
#include <sstream>
#include <type_traits>
#include <cstdint>

#include "helpers.hh"
#include "tablePrinter.hh"
#include "utilities.hh"

#if defined CPU_ARMPL
#include "../ArmPL/spmm.hh"
#elif defined CPU_ONEMKL
#include "../oneMKL/CPU/spmm.hh"
#elif defined CPU_AOCL
#include "../AOCL/spmm.hh"
#endif

#if defined GPU_CUBLAS
#include "../cuBLAS/spmm.hh"
#elif defined GPU_ONEMKL
#include "../oneMKL/GPU/spmm.hh"
#elif defined GPU_ROCBLAS
#include "../rocBLAS/spmm.hh"
#endif

/** `T` represents the type of kernel that will be run - i.e. T=float is for
 *      SGEMM. */
template <typename T>
class doSpmm {
public:
    doSpmm(const std::string csvDir, const int iters, const int startDim,
           const int upperLimit, const int step, const double sparsity, const matrixType type,
           const bool cpuEnabled = true, const bool gpuEnabled = true)
            : CSV_DIR(csvDir),
              iterations_(iters),
              startDimention_(startDim),
              upperLimit_(upperLimit),
              step_(step),
              sparsity_(sparsity),
              type_(type),
              doCPU_(cpuEnabled),
              doGPU_(gpuEnabled)
#if CPU_ENABLED
    ,
        cpu_(iterations_)
#endif
#if GPU_ENABLED
    ,
        gpu_(iterations_)
#endif
    {
      static_assert((std::is_same_v<T, float> || std::is_same_v<T, double>) &&
                    "ERROR - doSpmm can only be constructed using one of the "
                    "following types: [float, double].");                   
    }

    /** Run all problem types and write data to CSV files. */
    void collectData() {
      // Square Problem Sizes...
      // Re-initialise offload threshold structures
      cpuGpu_always_ = cpuGpu_offloadThreshold();
      cpuGpu_once_ = cpuGpu_offloadThreshold();
      cpuGpu_unified_ = cpuGpu_offloadThreshold();
      prev_gpuResult_always = time_checksum_gflop();
      prev_gpuResult_once = time_checksum_gflop();
      prev_gpuResult_unified = time_checksum_gflop();
      std::ofstream csvFile = initCSVFile(CSV_DIR + "/" + getKernelName() +
                                          "_square_square_M=N=K.csv");
      for (int dim = startDimention_; dim <= upperLimit_; dim += step_) {
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
      // Re-initialise offload threshold structures & previous results
      cpuGpu_always_ = cpuGpu_offloadThreshold();
      cpuGpu_once_ = cpuGpu_offloadThreshold();
      cpuGpu_unified_ = cpuGpu_offloadThreshold();
      prev_gpuResult_always = time_checksum_gflop();
      prev_gpuResult_once = time_checksum_gflop();
      prev_gpuResult_unified = time_checksum_gflop();
      csvFile = initCSVFile(CSV_DIR + "/" + getKernelName() +
                            "_tall-thin_short-wide_M=N_M=16K.csv");
      int K = startDimention_;
      int M = 16 * K;
      int N = 16 * K;
      while (M <= upperLimit_) {
        callKernels(csvFile, M, N, K);
        M += 16 * step_;
        N += 16 * step_;
        K += step_;
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
      // Re-initialise offload threshold structures & previous results
      cpuGpu_always_ = cpuGpu_offloadThreshold();
      cpuGpu_once_ = cpuGpu_offloadThreshold();
      cpuGpu_unified_ = cpuGpu_offloadThreshold();
      prev_gpuResult_always = time_checksum_gflop();
      prev_gpuResult_once = time_checksum_gflop();
      prev_gpuResult_unified = time_checksum_gflop();
      csvFile = initCSVFile(CSV_DIR + "/" + getKernelName() +
                            "_tall-thin_short-wide_M=N_K=32.csv");
      if (upperLimit_ >= 32) {
        for (int dim = startDimention_; dim <= upperLimit_; dim += step_) {
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
      // Re-initialise offload threshold structures & previous results
      cpuGpu_always_ = cpuGpu_offloadThreshold();
      cpuGpu_once_ = cpuGpu_offloadThreshold();
      cpuGpu_unified_ = cpuGpu_offloadThreshold();
      prev_gpuResult_always = time_checksum_gflop();
      prev_gpuResult_once = time_checksum_gflop();
      prev_gpuResult_unified = time_checksum_gflop();
      csvFile = initCSVFile(CSV_DIR + "/" + getKernelName() +
                            "_short-wide_tall-thin_M=N_K=16M.csv");
      M = startDimention_;
      N = startDimention_;
      K = 16 * M;
      while (K <= upperLimit_) {
        callKernels(csvFile, M, N, K);
        M += step_;
        N += step_;
        K += 16 * step_;
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
      // Re-initialise offload threshold structures & previous results
      cpuGpu_always_ = cpuGpu_offloadThreshold();
      cpuGpu_once_ = cpuGpu_offloadThreshold();
      cpuGpu_unified_ = cpuGpu_offloadThreshold();
      prev_gpuResult_always = time_checksum_gflop();
      prev_gpuResult_once = time_checksum_gflop();
      prev_gpuResult_unified = time_checksum_gflop();
      csvFile = initCSVFile(CSV_DIR + "/" + getKernelName() +
                            "_short-wide_tall-thin_M=N=32_K.csv");
      if (upperLimit_ >= 32) {
        for (int dim = startDimention_; dim <= upperLimit_; dim += step_) {
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
      // Re-initialise offload threshold structures & previous results
      cpuGpu_always_ = cpuGpu_offloadThreshold();
      cpuGpu_once_ = cpuGpu_offloadThreshold();
      cpuGpu_unified_ = cpuGpu_offloadThreshold();
      prev_gpuResult_always = time_checksum_gflop();
      prev_gpuResult_once = time_checksum_gflop();
      prev_gpuResult_unified = time_checksum_gflop();
      csvFile = initCSVFile(CSV_DIR + "/" + getKernelName() +
                            "_tall-thin_square_K=N_M=16K.csv");
      K = startDimention_;
      N = startDimention_;
      M = 16 * K;
      while (M <= upperLimit_) {
        callKernels(csvFile, M, N, K);
        M += 16 * step_;
        N += step_;
        K += step_;
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
      // Re-initialise offload threshold structures & previous results
      cpuGpu_always_ = cpuGpu_offloadThreshold();
      cpuGpu_once_ = cpuGpu_offloadThreshold();
      cpuGpu_unified_ = cpuGpu_offloadThreshold();
      prev_gpuResult_always = time_checksum_gflop();
      prev_gpuResult_once = time_checksum_gflop();
      prev_gpuResult_unified = time_checksum_gflop();
      csvFile = initCSVFile(CSV_DIR + "/" + getKernelName() +
                            "_tall-thin_square_K=N=32_M.csv");
      if (upperLimit_ >= 32) {
        for (int dim = startDimention_; dim <= upperLimit_; dim += step_) {
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
      // Re-initialise offload threshold structures & previous results
      cpuGpu_always_ = cpuGpu_offloadThreshold();
      cpuGpu_once_ = cpuGpu_offloadThreshold();
      cpuGpu_unified_ = cpuGpu_offloadThreshold();
      prev_gpuResult_always = time_checksum_gflop();
      prev_gpuResult_once = time_checksum_gflop();
      prev_gpuResult_unified = time_checksum_gflop();
      csvFile = initCSVFile(CSV_DIR + "/" + getKernelName() +
                            "_square_short-wide_M=K_N=16K.csv");
      M = startDimention_;
      K = startDimention_;
      N = 16 * K;
      while (N <= upperLimit_) {
        callKernels(csvFile, M, N, K);
        M += step_;
        N += 16 * step_;
        K += step_;
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
      // Re-initialise offload threshold structures & previous results
      cpuGpu_always_ = cpuGpu_offloadThreshold();
      cpuGpu_once_ = cpuGpu_offloadThreshold();
      cpuGpu_unified_ = cpuGpu_offloadThreshold();
      prev_gpuResult_always = time_checksum_gflop();
      prev_gpuResult_once = time_checksum_gflop();
      prev_gpuResult_unified = time_checksum_gflop();
      csvFile = initCSVFile(CSV_DIR + "/" + getKernelName() +
                            "_square_short-wide_M=K=32_N.csv");
      if (upperLimit_ >= 32) {
        for (int dim = startDimention_; dim <= upperLimit_; dim += step_) {
          // M = 32, N = dim, K = 32;
          callKernels(csvFile, 32, dim, 32);
        }
      }
#if CPU_ENABLED && GPU_ENABLED
      if (doCPU_ && doGPU_) {
    // Print offload results to stdout
    printOffloadThreshold("Square x Short-and-Wide (M=K=32, N)");
  }
#endif
      // Close file
      csvFile.close();
    }

private:
    /** Ensure all CPU and GPU checksums are within the permitted limit of
     * eachother. */
    void checkChecksums(time_checksum_gflop cpuResult,
                        time_checksum_gflop gpuResult_once,
                        time_checksum_gflop gpuResult_always,
                        time_checksum_gflop gpuResult_unified, const int M,
                        const int N, const int K) {
      // Ensure that each checksum difference is less than 0.1%
      double hundredOverChecksum = 100 / std::fabs(cpuResult.checksum);
      if (((std::fabs(cpuResult.checksum - gpuResult_once.checksum) *
            hundredOverChecksum)) > 0.1 &&
          ((std::fabs(cpuResult.checksum - gpuResult_always.checksum) *
            hundredOverChecksum)) > 0.1 &&
          ((std::fabs(cpuResult.checksum - gpuResult_unified.checksum) *
            hundredOverChecksum)) > 0.1) {
        std::cerr << "ERROR - " << getKernelName() << " kernel checksums do not match:\n\tInput "
                     "dimensions: M=" << M << ", N=" << N << ", K=" << K << std::endl;
        std::cerr << std::setprecision(10) << "\tCPU Checksum = " << cpuResult.checksum << std::endl;
        std::cerr << std::setprecision(10) << "\tGPU (Once) Checksum = " << gpuResult_once.checksum << std::endl;
        std::cerr << std::setprecision(10) << "\tGPU (Always) Checksum = " << gpuResult_always.checksum << std::endl;
        std::cerr << std::setprecision(10) << "\tGPU (Unified) Checksum = " << gpuResult_unified.checksum << std::endl;
        exit(1);
      }
    }

    /** Check whether the offload structures need to be reset; and doing so if
     * required.
     *   - If CPU.gflops >= GPU.gflops for last two problem sizes, then reset
     * offload structures as GPU may not necessarily have reached the offload
     * threshold. */
    void checkOffloadStructReset(time_checksum_gflop cpuResult,
                                 time_checksum_gflop gpuResult_once,
                                 time_checksum_gflop gpuResult_always,
                                 time_checksum_gflop gpuResult_unified) {
      if ((cpuGpu_once_.M != 0) && (cpuResult.gflops >= gpuResult_once.gflops) &&
          (cpuResult.gflops >= prev_gpuResult_once.gflops)) {
        cpuGpu_once_.cpuGflops = 0.0;
        cpuGpu_once_.gpuGflops = 0.0;
        cpuGpu_once_.probSize_kib = 0.0;
        cpuGpu_once_.M = 0;
        cpuGpu_once_.N = 0;
        cpuGpu_once_.K = 0;
      }
      if ((cpuGpu_always_.M != 0) &&
          (cpuResult.gflops >= gpuResult_always.gflops) &&
          (cpuResult.gflops >= prev_gpuResult_always.gflops)) {
        cpuGpu_always_.cpuGflops = 0.0;
        cpuGpu_always_.gpuGflops = 0.0;
        cpuGpu_always_.probSize_kib = 0.0;
        cpuGpu_always_.M = 0;
        cpuGpu_always_.N = 0;
        cpuGpu_always_.K = 0;
      }
      if ((cpuGpu_unified_.M != 0) &&
          (cpuResult.gflops >= gpuResult_unified.gflops) &&
          (cpuResult.gflops >= prev_gpuResult_unified.gflops)) {
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

    void callKernels(std::ofstream& csvFile, const int N, const int M,
                     const int K) {
      const double probSize = calcKib(N, N, N, sparsity_);
      const uint64_t flops = calcFlops(N, N, N, sparsity_);
      std::string kernelName = getKernelName();

#if CPU_ENABLED
      time_checksum_gflop cpuResult;
      if (doCPU_) {
        cpu_.initialise(N, M, K, sparsity_, type_);
        cpuResult = cpu_.compute();
        cpuResult.gflops = calcGflops(flops, iterations_, cpuResult.runtime);
        writeLineToCsv(csvFile, "cpu", kernelName, N, M, K, probSize,
                       sparsity_, iterations_, cpuResult.runtime,
                       cpuResult.gflops);
      }
#endif
#if GPU_ENABLED
      // Perform the GPU kernels
      time_checksum_gflop gpuResult_always;
      time_checksum_gflop gpuResult_once;
      time_checksum_gflop gpuResult_unified;
      /*
        * We run three different offload types:
        *  - ALWAYS: Offload to/from GPU every iteration
        *  - ONCE : Offload to/from GPU once before all iterations and once after
        *  - UNIFIED : data passed from host to device (and device to host) as needed 
        * THE ORDER OF THESE IS IMPORTANT -- To reduce time spent generating matrices, we 
        * generate once during the ALWAYS offload, and then re-use the same matrices for
        * the ONCE and UNIFIED offload tests.  Deleting them after UNIFIED.  Therefore, 
        * changing the order here will require this logic within the spmm GPU classes to 
        * be updated. 
      */
      if (doGPU_) {
        // - ALWAYS: Offload to/from GPU every iteration
        gpu_.initialise(gpuOffloadType::always, N, M, K, sparsity_, type_);
        gpuResult_always = gpu_.compute();
        gpuResult_always.gflops =
              calcGflops(flops, iterations_, gpuResult_always.runtime);
        writeLineToCsv(csvFile, "gpu_offloadAlways", kernelName, N, M, K,
                       probSize, sparsity_, iterations_, gpuResult_always.runtime,
                       gpuResult_always.gflops);

        // - ONCE : Offload to/from GPU once before all iterations and once
        // after
        gpu_.initialise(gpuOffloadType::once, N, M, K, sparsity_, type_);
        gpuResult_once = gpu_.compute();
        gpuResult_once.gflops =
              calcGflops(flops, iterations_, gpuResult_once.runtime);
        writeLineToCsv(csvFile, "gpu_offloadOnce", kernelName, N, M, K, probSize,
                       sparsity_, iterations_, gpuResult_once.runtime,
                       gpuResult_once.gflops);
        
        // - UNIFIED : data passed from host to device (and device to host) as
        //             needed
        gpu_.initialise(gpuOffloadType::unified, N, M, K, sparsity_, type_);
        gpuResult_unified = gpu_.compute();
        gpuResult_unified.gflops =
        calcGflops(flops, iterations_, gpuResult_unified.runtime);
        writeLineToCsv(csvFile, "gpu_unified", kernelName, N, M, K, probSize,
                       sparsity_, iterations_, gpuResult_unified.runtime,
                       gpuResult_unified.gflops);
      }
#endif
#if CPU_ENABLED && GPU_ENABLED
      if (doCPU_ && doGPU_) {
        // Check that all checksums are within the permitted limit
        checkChecksums(cpuResult, gpuResult_once, gpuResult_always,
                       gpuResult_unified, N, M, K);
        // Check whether offload structs need to be reset
        checkOffloadStructReset(cpuResult, gpuResult_once, gpuResult_always,
                                gpuResult_unified);
        // Update offload structs if required
        updateOffloadStructs(cpuResult, gpuResult_once, gpuResult_always,
                             gpuResult_unified, N, M, K, probSize);
        // Update previous GPU results
        prev_gpuResult_once = gpuResult_once;
        prev_gpuResult_always = gpuResult_always;
        prev_gpuResult_unified = gpuResult_unified;
      }
#endif
    }

    /** A function for calculating FLOPs performed by a GEMM.
     * C = alpha*AB + beta*C */
    constexpr uint64_t calcFlops(const int M, const int N, const int K, const double SPARSITY) const {
      // The number of scalar multiplications is nnz(Ak)*nnz(Bk) for each inner index k
      // Therefore, the expectation is to have NNZA * NNZB / K, as each K index would 
      // on average have NNZA/K * NNZB/K.  This assumes a uniform distribution of non-zero elements
      uint64_t NNZA = 1 + (uint64_t)((double)M * (double)K * (1.0 - SPARSITY));
      uint64_t NNZB = 1 + (uint64_t)((double)K * (double)N * (1.0 - SPARSITY));
      return (NNZA * NNZB) / K;
    }

    /** A function for calculating the total GEMM problem size in KiB. 
      Each matrix is stored in CSR, and so needs (nRows + 1) + 2NNZ space.
      For A and B, this is easy, but for C we do not know its size ahead of time 
      (we know nRows but not NNZ).  However, we can estimate the NNZ, on average.

      Each value of C is the sum of the products of the corresponding row of A 
      and column of B.
      As each value of A and B has a probability of (1 - SPARSITY) if being non-zero, 
      the probability that both A and B are non-zero (and thus that the product is 
      non-zero)is (1 - SPARSITY)^2.
      There are K products that are summed together.  If any one of these products is 
      non-zero, so too shall the sum be.  Therefore, the estimated sparsity of C is
      (1 - (1 - SPARSITY)^2)^K
      */
    constexpr double calcKib(const int M, const int N, const int K, const double SPARSITY) const {
      uint64_t M_ = (uint64_t)M, K_ = (uint64_t)K;
      uint64_t NNZA = 1 + (uint64_t)((double)M * (double)K * (1.0 - SPARSITY));
      uint64_t NNZB = 1 + (uint64_t)((double)K * (double)N * (1.0 - SPARSITY));
      double CSPARSITY = 1 - pow(pow(1.0 - SPARSITY, 2), K);
      uint64_t NNZC = 1 + (uint64_t)((double)M * (double)N * CSPARSITY);

      uint64_t probSize = (M_ + 1) + (2 * NNZA) + (K_ + 1) + (2 * NNZB) + (M_ + 1) + (2 * NNZC);
      return ((double)(probSize * (sizeof(T))) / 1024);
    }

    /** Get the name of the kernel being run. */
    std::string getKernelName() const {
      switch (sizeof(T)) {
        case 4:
          return "sspmm";
        case 8:
          return "dspmm";
        default:
          return "unknown";
      }
    }

    /** Print to stdout the offload thresholds. */
    void printOffloadThreshold(const std::string& problemName) const {
      std::vector<std::string> header = {
              "Device",  "M",          "N", "K", "Total Prob. Size (KiB)",
              "GFLOP/s", "CPU GFLOP/s"};

      std::vector<std::vector<std::string>> rows;
      // Initialise GPU_Once row
      std::stringstream probSize_o;
      std::stringstream gpuGflops_o;
      std::stringstream cpuGflops_o;
      probSize_o << std::fixed << std::setprecision(2) << cpuGpu_once_.probSize_kib;
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
      probSize_a << std::fixed << std::setprecision(2) << cpuGpu_always_.probSize_kib;
      gpuGflops_a << std::fixed << std::setprecision(2) << cpuGpu_always_.gpuGflops;
      cpuGflops_a << std::fixed << std::setprecision(2) << cpuGpu_always_.cpuGflops;
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
      probSize_u << std::fixed << std::setprecision(2) << cpuGpu_unified_.probSize_kib;
      gpuGflops_u << std::fixed << std::setprecision(2) << cpuGpu_unified_.gpuGflops;
      cpuGflops_u << std::fixed << std::setprecision(2) << cpuGpu_unified_.cpuGflops;
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

    /** The output directory where CSV files should be saved to. */
    const std::string CSV_DIR;

    /** The number of iterations to perform per problem size. */
    const int iterations_;

    /** The value of the first probelm size dimention run. */
    const int startDimention_;

    /** The maximum value of the largest problem size dimention. */
    const int upperLimit_;

    /** The step size between problem sizes. */
    const int step_;

    /** The sparsity value of the sparse matrices. */
    const double sparsity_;

    const matrixType type_;

    /** Whether the CPU kernels should be run. */
    const bool doCPU_ = true;

    /** Whether the GPU kernels should be run. */
    const bool doGPU_ = true;

#if CPU_ENABLED
    /** The CPU kernel. */
  cpu::spmm_cpu<T> cpu_;
#endif

#if GPU_ENABLED
    /** The GPU kernel. */
	gpu::spmm_gpu<T> gpu_;
#endif

    /** The point at which offloading to GPU (offload once) becomes worthwhile. */
    cpuGpu_offloadThreshold cpuGpu_once_;

    /** The point at which offloading to GPU (offload always) becomes worthwhile.
     */
    cpuGpu_offloadThreshold cpuGpu_always_;

    /** The point at which offloading to GPU (unified memory) becomes worthwhile.
     */
    cpuGpu_offloadThreshold cpuGpu_unified_;

    /** The previous problem size's GPU (offload once) performance results. */
    time_checksum_gflop prev_gpuResult_once;

    /** The previous problem size's GPU (offload always) performance results. */
    time_checksum_gflop prev_gpuResult_always;

    /** The previous problem size's GPU (unified memory) performance results. */
    time_checksum_gflop prev_gpuResult_unified;
};