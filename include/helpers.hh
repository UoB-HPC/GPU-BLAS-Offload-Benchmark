#pragma once

#include <cmath>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

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

/** Gets all problem types from a file in include/Problems.
 * Each problem type is parsed as a vector of strings, with each string
 * representing one of the problem's dimensions or problem type name. */
std::vector<std::vector<std::string>> getProblemTypes(
    const std::string probFilename) {
  std::ifstream probFile(probFilename);
  std::string line;
  std::vector<std::vector<std::string>> outVec;
  while (std::getline(probFile, line)) {
    // Ignore all lines that do not start with `(` and end with `)`
    // (i.e. ignore comments and incorrect problem definitions)
    if (!(line.front() == '(' && line.back() == ')')) continue;

    std::vector<std::string> parsedLine;
    // Remove 0th and last characters (removing the parentheses)
    line = line.substr(1, line.length() - 2);
    // Extract all entries from problem (substrings delimited by ',')
    size_t commaPos = line.find(',');
    size_t front = 0;
    while (commaPos != std::string::npos) {
      parsedLine.push_back(line.substr(front, (commaPos - front)));
      front = commaPos + 1;
      commaPos = line.find(',', front);
    }
    // Add final entry to parsed line
    parsedLine.push_back(line.substr(front));
    outVec.push_back(parsedLine);
  }
  return outVec;
}