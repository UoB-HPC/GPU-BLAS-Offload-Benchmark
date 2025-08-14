#pragma once

#include <random>
#include <queue>
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
 * R-MAT (Recursive MATrix) Graph Generator - Single Edge Addition
 *
 * Iterative Implementation of the R-MAT model for generating scale-free graphs
 * with  realistic
 * structural properties. R-MAT subdivides the adjacency matrix into
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
 * @param n         Matrix dimension (number of columns in the matrix).
 * Needed for calcluating index
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
template <typename T>
void rMat(T* M, int rows, int cols, int nnz,
          double a = 0.57,
          double b = 0.19,
          double c = 0.19,
          double d = 0.05,
          double noise = 0.0,
          bool no_self_loops = false,
          bool undirected = false) {
  // Determine number of bits to cover rows and cols
  int row_bits = static_cast<int>(std::ceil(std::log2(rows)));
  int col_bits = static_cast<int>(std::ceil(std::log2(cols)));

  // Random number generator objects for use in descent
  std::default_random_engine gen;
  std::uniform_real_distribution<double> dist(0.0, 1.0);
  // Set the seed to allow checksum to work
  gen.seed(SEED);

  int edge_idx = 0;
  while (edge_idx < nnz) {
    int u = 0;
    int v = 0;

    double A = a, B = b, C = c, D = d;

    // For each bit position (MSB to LSB)
    for (int bit = 0; bit < std::max(row_bits, col_bits); ++bit) {
      // Optional noise
      if (noise > 0.0) {
        auto jitter = [&](double val) {
          return std::max(0.0, val + (dist(gen) * 2.0 - 1.0) * noise);
        };
        A = jitter(a);
        B = jitter(b);
        C = jitter(c);
        D = jitter(d);
        double sum = A + B + C + D;
        A /= sum; B /= sum; C /= sum; D /= sum;
      }

      double r = dist(gen);
      double t1 = A;
      double t2 = A + B;
      double t3 = A + B + C;

      int row_bit = 0;
      int col_bit = 0;
      if (r < t1) {
        row_bit = 0;  col_bit = 0;
      } else if (r < t2) {
        row_bit = 0; col_bit = 1;
      } else if (r < t3) {
        row_bit = 1; col_bit = 0;
      } else {
        row_bit = 1; col_bit = 1;
      }

      if (bit < row_bits)
        u = (u << 1) | row_bit;
      if (bit < col_bits)
        v = (v << 1) | col_bit;
    }

    if (u >= rows || v >= cols)
      continue; // Out of bounds due to non-power-of-two dims

    if (no_self_loops && u == v)
      continue;

    if (undirected && u > v)
      std::swap(u, v);

    M[2 * edge_idx]     = (T)u;
    M[2 * edge_idx + 1] = (T)v;
    ++edge_idx;
  }
}

