#pragma once

#include <random>
#include <iostream>

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

/**
 * R-MAT (Recursive MATrix) Graph Generator - Single Edge Addition
 *
 * Implements the R-MAT model for generating scale-free graphs with realistic
 * structural properties. R-MAT recursively subdivides the adjacency matrix into
 * four quadrants and probabilistically selects which quadrant to place each edge,
 * creating graphs with power-law degree distributions and community structure
 * similar to real-world networks.
 *
 * The algorithm works by:
 * 1. Dividing the current matrix region into 4 quadrants
 * 2. Using probabilities (a,b,c,d) where d = 1-(a+b+c) to select a quadrant
 * 3. Recursively descending until reaching a 1x1 cell
 * 4. Attempting to place a non-zero value at that position
 *
 * This implementation is particularly relevant for sparse linear algebra benchmarks
 * as R-MAT graphs exhibit:
 * - High sparsity (typical density < 0.1%)
 * - Irregular structure that challenges cache efficiency
 * - Realistic non-uniform sparsity patterns found in real applications
 * - Scalable generation for large problem sizes
 *
 * @param M         Pointer to flattened n×n adjacency matrix (row-major order)
 * @param n         Matrix dimension (number of vertices in the graph)
 * @param x1        Left boundary of current matrix subregion (inclusive)
 * @param x2        Right boundary of current matrix subregion (inclusive)
 * @param y1        Top boundary of current matrix subregion (inclusive)
 * @param y2        Bottom boundary of current matrix subregion (inclusive)
 * @param a         Probability of selecting top-left quadrant [0,1]
 * @param b         Probability of selecting top-right quadrant [0,1]
 * @param c         Probability of selecting bottom-left quadrant [0,1]
 *                  Note: bottom-right probability d = 1-(a+b+c)
 * @param gen       Pointer to random number generator for reproducible results
 * @param dist      Uniform real distribution [0,1) for quadrant selection
 * @param bin       If true, creates binary matrix (edges = 1.0);
 *                  if false, assigns random weights in range [-50, 50)
 *
 * @return true if successfully added non-zero value to an empty position,
 *         false if selected position already contains non-zero value
 *
 * @note Typical R-MAT parameters for realistic graphs:
 *       - a=0.45, b=0.15, c=0.15, d=0.25 (Kronecker-like)
 *       - a=0.57, b=0.19, c=0.19, d=0.05 (more skewed)
 *
 * @note For sparse linear algebra benchmarks, this generates matrices with:
 *       - Irregular sparsity patterns (not banded/block-structured)
 *       - Variable row/column densities challenging load balancing
 *       - Realistic cache behavior representative of graph applications
 *
 * @warning Uses 64-bit indexing to prevent overflow for large matrices (n > 46k)
 * @warning Non-thread-safe due to shared random number generator
 *
 * References:
 * - Chakrabarti, D., Zhan, Y., & Faloutsos, C. (2004). R-MAT: A recursive model
 *   for graph mining. SIAM International Conference on Data Mining.
 * - Leskovec, J., et al. (2010). Kronecker graphs: An approach to modeling networks.
 *   Journal of Machine Learning Research, 11, 985-1042.
 */
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
