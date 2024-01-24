#pragma once

#include <cmath>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <string>

/** Create a new csv file and initialise the column headers.
 * Returns the ofstream to the open file. */
std::ofstream initCSVFile(const std::string filename) {
  if (filename.find(".csv") != filename.size() - 4) {
    std::cout << "ERROR - filename must end with '.csv'" << std::endl;
    exit(1);
  }

  std::ofstream newFile(filename);

  newFile << "Device,Kernel,M,N,K,Total Problem Size (KiB),Iterations,Total "
             "Seconds,GFLOP/s"
          << std::endl;

  return newFile;
}

/** Write a new line to an open CSV file.
 * Function does not close the file. */
void writeLineToCsv(std::ofstream& file, const std::string device,
                    const std::string kernel, const int M, const int N,
                    const int K, const double totalProbSize, const int iters,
                    const double totalTime, const double gflops) {
  if (!file.is_open()) {
    std::cout << "ERROR - Attempted to write line to a closed CSV file."
              << std::endl;
    exit(1);
  }
  file << device << "," << kernel << "," << M << "," << N << "," << K << ","
       << std::fixed << std::setprecision(3) << totalProbSize << "," << iters
       << "," << std::fixed << std::setprecision(5) << totalTime << ","
       << std::fixed << std::setprecision(3) << gflops << std::endl;
}

/** Calculate average GFLOPs. */
double calcGflops(const uint64_t flops, const int iters, const double seconds) {
  return (seconds == 0.0 || seconds == INFINITY)
             ? 0.0
             : ((double)(flops * iters) / seconds) * 1e-9;
}