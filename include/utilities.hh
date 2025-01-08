#pragma once

#include <random>

// Define CPU related macros
#if defined CPU_ARMPL
#define CPU_LIB_NAME "Arm Performance Libraries"
#elif defined CPU_ONEMKL
#define CPU_LIB_NAME "Intel OneMKL"
#elif defined CPU_AOCL
#define CPU_LIB_NAME "AMD Optimized CPU Libraries"
#elif defined CPU_NVPL
#define CPU_LIB_NAME "NVIDIA Performance Libraries"
#elif defined CPU_OPENBLAS
#define CPU_LIB_NAME "OpenBLAS"
#else
#define CPU_DEFAULT
#define CPU_LIB_NAME "None"
#define CPU_ENABLED false
#endif

#ifndef CPU_ENABLED
#define CPU_ENABLED true
#endif

// Define GPU related macros
#if defined GPU_CUBLAS
#define GPU_LIB_NAME "NVIDIA cuBLAS"
#elif defined GPU_ONEMKL
#define GPU_LIB_NAME "Intel OneMKL"
#elif defined GPU_ROCBLAS
#define GPU_LIB_NAME "AMD rocBLAS"
#else
#define GPU_DEFAULT
#define GPU_LIB_NAME "None"
#define GPU_ENABLED false
#endif

#ifndef GPU_ENABLED
#define GPU_ENABLED true
#endif

// Define macros for alpha and beta
#define ALPHA 1
#define BETA 0

// Define seed for random number generation - use seeded srand() to ensure
// inputs across libraries are consistent & comparable
const unsigned int SEED = 19123005;

// Define enum class for GPU offload type
enum class gpuOffloadType : uint8_t {
  always = 0,
  once,
  unified,
};

// Define struct which contains a runtime, checksum value, and gflop/s value
struct time_checksum_gflop {
  double runtime = 0.0;
  double checksum = 0.0;
  double gflops = 0.0;
};

// Struct to hold key values at the point at which offloading to GPU becomes
// worthwhile.
struct cpuGpu_offloadThreshold {
  double cpuGflops = 0.0;
  double gpuGflops = 0.0;
  double probSize_kib = 0.0;
  int M = 0;
  int N = 0;
  int K = 0;
};

// External consume function used to ensure naive code is performed and not
// optimised away, and that all iterations of any library BLAS call are
// performed.
extern "C" {
int consume(void* a, void* b, void* c);
}


/**
 * RMAT is a recursive function used to generate sparse matrices.  It is
 * needed for both single and double precision so I've simply overloaded this
 * function to have M as both float and double types.  Ugly, but works for
 * now.
 * Todo -- Consider different approach if other data types are supported in the
 * future.
 */

/**
 * @param M input matrix
 * @param n number of columns in the full matrix (i.e. full range of the x axis)
 * @param x1 beginning x coordinate of the submatrix
 * @param x2 ending x coordinate of the submatrix
 * @param y1 starting y coordinate of the submatrix
 * @param y2 ending y coordinate of the submatrix
 * @param a probability of tile a being chosen
 * @param b probability of tile b being chosen
 * @param c probability of tile c being chosen
 * @param gen random number generator
 * @param dist random number distribution
 * @param bin bool to decide whether values added are binary of float/double
 * @return
 */
bool rMat(float* M, int n, int x1, int x2, int y1, int y2, float a, float b,
          float c, std::default_random_engine* gen,
          std::uniform_real_distribution<double> dist, bool bin) {
  // If a 1x1 submatrix, then add an edge and return out
  if (x1 >= x2 && y1 >= y2) {
    // Needed to avoid overflow segfaults with large problem sizes
    uint64_t index = (((uint64_t)y1 * (uint64_t)n) + (uint64_t)x1);
    if (abs(M[index]) > 0.1) {
      return false;
    } else {
      // Add 1.0 if this is a binary graph, and a random real number otherwise
      M[index] = (bin) ? 1.0 : (((rand() % 10000) / 100.0) - 50.0);
      return true;
    }
  } else {
    // Divide up the matrix
    int xMidPoint = x1 + floor((x2 - x1) / 2);
    int yMidPoint = y1 + floor((y2 - y1) / 2);

    // Work out which quarter to recurse into
    // There are some ugly ternary operators here to avoid going out of bounds
    // in the edge case that we are already at 1 width or 1 height
    float randomNum = dist(*gen);
    if (randomNum < a) {
      return rMat(M, n, x1, xMidPoint, y1, yMidPoint,
                  a, b, c, gen, dist, bin);
    } else if (randomNum < (a + b)) {
      return rMat(M, n, ((xMidPoint < x2) ? xMidPoint + 1 : xMidPoint), x2, y1, yMidPoint,
                  a, b, c, gen, dist, bin);
    } else if (randomNum < (a + b + c)) {
      return rMat(M, n, x1, xMidPoint, ((yMidPoint < y2) ? yMidPoint + 1 : yMidPoint), y2,
                  a, b, c, gen, dist, bin);
    } else {
      return rMat(M, n, ((xMidPoint < x2) ? xMidPoint + 1 : xMidPoint), x2,
                  ((yMidPoint < y2) ? yMidPoint + 1 : yMidPoint), y2, a,
                  b, c, gen, dist, bin);
    }
  }
  return true;
}
bool rMat(double* M, int n, int x1, int x2, int y1, int y2, float a, float b,
          float c, std::default_random_engine* gen,
          std::uniform_real_distribution<double> dist, bool bin) {
  // If a 1x1 submatrix, then add an edge and return out
  if (x1 >= x2 && y1 >= y2) {
    // Needed to avoid overflow segfaults with large problem sizes
    uint64_t index = (((uint64_t)y1 * (uint64_t)n) + (uint64_t)x1);
    if (abs(M[index]) > 0.1) {
      return false;
    } else {
      // Add 1.0 if this is a binary graph, and a random real number otherwise
      M[index] = (bin) ? 1.0 : (((rand() % 10000) / 100.0) - 50.0);
      return true;
    }
  } else {
    // Divide up the matrix
    int xMidPoint = x1 + floor((x2 - x1) / 2);
    int yMidPoint = y1 + floor((y2 - y1) / 2);

    // Work out which quarter to recurse into
    // There are some ugly ternary operators here to avoid going out of bounds in the edge case
    // that we are already at 1 width or 1 height
    float randomNum = dist(*gen);
    if (randomNum < a) {
      return rMat(M, n, x1, xMidPoint, y1, yMidPoint,
                  a, b, c, gen, dist, bin);
    } else if (randomNum < (a + b)) {
      return rMat(M, n, ((xMidPoint < x2) ? xMidPoint + 1 : xMidPoint), x2, y1, yMidPoint,
                  a, b, c, gen, dist, bin);
    } else if (randomNum < (a + b + c)) {
      return rMat(M, n, x1, xMidPoint, ((yMidPoint < y2) ? yMidPoint + 1 : yMidPoint), y2,
                  a, b, c, gen, dist, bin);
    } else {
      return rMat(M, n, ((xMidPoint < x2) ? xMidPoint + 1 : xMidPoint), x2,
                  ((yMidPoint < y2) ? yMidPoint + 1 : yMidPoint), y2, a,
                  b, c, gen, dist, bin);
    }
  }
  return true;
}
