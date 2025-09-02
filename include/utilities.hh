#pragma once

#include <random>
#include <cmath>
#include <algorithm>
#include <vector>
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
const unsigned int SEED2 = 19123027;

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


template <typename fp_type, typename int_type> 
void printCSR(uint64_t nRows,
              uint64_t nnz,
              const int_type* rows,
              const int_type* cols,
              const fp_type* vals) {
  std::cout << "ROWS:" << std::endl;
  std::cout << "\t[";
  for (uint64_t i = 0; i <= nRows; i++) {
    std::cout << rows[i];
    if (i < nRows) {
      std::cout << ", ";
    }
  }
  std::cout << "]" << std::endl;

  std::cout << "COLS:" << std::endl;
  std::cout << "\t[";
  for (uint64_t i = 0; i < nnz; i++) {
    std::cout << cols[i];
    if (i < nnz - 1) {
      std::cout << ", ";
    }
  }
  std::cout << "]" << std::endl;

  std::cout << "VALS:" << std::endl;
  std::cout << "\t[";
  for (uint64_t i = 0; i < nnz; i++) {
    std::cout << vals[i];
    if (i < nnz - 1) {
      std::cout << ", ";
    }
  }
  std::cout << "]" << std::endl;
}

template <typename fp_type, typename int_type>
void checkCSRValid(uint64_t nRows,
                   uint64_t nCols,
                   uint64_t nnz,
                   const int_type* rows,
                   const int_type* cols,
                   const fp_type* vals) {
  if (rows[0] != 0) {
    std::cerr << "[ERROR]: CSR INVALID - row_pointer[0] is not 0" << std::endl;
    printCSR(nRows, nnz, rows, cols, vals);
    exit(1);
  }

  for (uint64_t r = 0; r < nRows; r++) {
    if (rows[r] > rows[r + 1]) {
      std::cerr << "[ERROR]: CSR INVALID - row_pointer[" << r << "] > row_pointer[" << (r + 1) << "]" << std::endl;
    printCSR(nRows, nnz, rows, cols, vals);
      exit(1);
    }
  }

  if (rows[nRows] != (int_type)nnz) {
    std::cerr << "[ERROR]: CSR INVALID - row_pointer[nRows] != nnz" << std::endl;
    printCSR(nRows, nnz, rows, cols, vals);
    exit(1);
  }

  for (uint64_t i = 0; i < nnz; i++) {
    if (cols[i] < 0 || cols[i] >= (int_type)nCols) {
      std::cerr << "[ERROR]: CSR INVALID - column index out of bounds" << std::endl;
      printCSR(nRows, nnz, rows, cols, vals);
      exit(1);
    }
  }

  for (uint64_t r = 0; r < nRows; r++) {
    for (int_type j = rows[r]; j < (rows[r + 1] - 1); j++) {
      if (cols[j] > cols[j + 1]) {
        std::cerr << "[ERROR]: CSR INVALID - column indices not sorted in row " << r << std::endl;
        printCSR(nRows, nnz, rows, cols, vals);
        exit(1);
      }
      if (cols[j] == cols[j + 1]) {
        std::cerr << "[ERROR]: CSR INVALID - duplicate column indices in row " << r << std::endl;
        printCSR(nRows, nnz, rows, cols, vals);
        exit(1);
      }
    }
  }
}


/**
 * @brief Generate an R-MAT matrix directly in CSR format.
 *
 * This function samples `nnz` edges (nonzeros) from the R-MAT distribution and
 * writes the result directly into CSR arrays:
 *   - vals[k] : value of the k-th nonzero (here set to 1 by default)
 *   - cols[k] : column index of the k-th nonzero
 *   - rows[i] : starting offset in (vals, cols) for row i
 *               (classic CSR row pointer of length nrows+1)
 *
 * Memory usage is O(nnz) (plus a temporary edge list), avoiding any dense
 * matrix construction.
 *
 * IMPORTANT BEHAVIOR:
 *  - Undirected: When `undirected == true`, this code only enforces u <= v
 *    during sampling (so edges are oriented consistently). It does NOT insert
 *    the symmetric counterpart (v,u). If you want a symmetric matrix, you must
 *    explicitly duplicate edges (except diagonal) before CSR conversion.
 *  - Seeding: Uses a global or external `SEED` to make the generator
 *    deterministic/reproducible. Ensure `SEED` is defined in your translation unit.
 *
 * Complexity:
 *  - Sampling:    O(nnz * log(max(nrows,ncols))) bit-decisions per edge
 *  - Sorting:     O(nnz log nnz) (by row, then col)
 *  - CSR build:   O(nnz + nrows)
 *
 * @tparam T         Numeric type for values (e.g., float, double, int)
 * @tparam int_type  Integer type for indices (e.g., int, int32_t, int64_t)
 *
 * @param vals   Output array of length nnz (nonzero values)
 * @param cols   Output array of length nnz (column indices)
 * @param rows   Output array of length nrows+1 (row pointer)
 * @param nrows  Number of rows in the matrix
 * @param ncols  Number of columns in the matrix
 * @param nnz    Number of nonzeros to generate
 * @param a,b,c,d R-MAT quadrant probabilities (must sum to 1; typical: 0.57,0.19,0.19,0.05)
 * @param noise  Optional jitter in probabilities each bit step (0.0 = none)
 * @param no_self_loops If true, edges with u == v are discarded and resampled
 * @param undirected    If true, enforce u <= v in the sampled edge; does NOT mirror edges
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
template <typename T, typename int_type>
void rMatCSR(T* vals, int_type* cols, int_type* rows,
             int nrows, int ncols, int nnz, 
             bool isB = false,
             double a = 0.57,
             double b = 0.19,
             double c = 0.19,
             double d = 0.05,
             double noise = 0.0,
             bool no_self_loops = false,
             bool undirected = false) {
  // Number of bits needed to index into the row/col ranges.
  // R-MAT decides each bit from MSB→LSB by picking a quadrant.
  int row_bits = static_cast<int>(std::ceil(std::log2(nrows)));
  int col_bits = static_cast<int>(std::ceil(std::log2(ncols)));

  // Set up RNG.  Uses srand for value generation, and uniform[0,1)
  // for quadrant selection
  srand((isB ? SEED2 : SEED));
  std::default_random_engine gen;
  std::uniform_real_distribution<double> dist(0.0, 1.0);
  gen.seed((isB ? SEED2 : SEED));


  // Temporary storage of sampled edges as (row, col) pairs.
  // We reserve exactly nnz slots and will push_back exactly nnz valid edges.
  std::vector<std::pair<int_type, int_type>> edges;
  edges.reserve(nnz);

  // Keep sampling until we have nnz valid edges.
  // Invalid candidates (out-of-bounds due to non-powers-of-two, self-loops, etc.)
  // are discarded by continuing the loop without incrementing the edge count.
  int edge_idx = 0;
  while (edge_idx < nnz) {
    int u = 0; // Sampled row index (as int, cast to int_type later)
    int v = 0; // Sampled column index

    // Base quadrant probabilities (A,B,C,D). We optionally jitter these
    // at each bit decision if 'noise' > 0.
    double A = a, B = b, C = c, D = d;

    // For each bit (from most-significant to least), decide which quadrant
    // the edge falls into and set the corresponding bit of (u,v).
    for (int bit = 0; bit < std::max(row_bits, col_bits); ++bit) {
      // Optional noise: perturb A,B,C,D slightly, then renormalize.
      if (noise > 0.0) {
        auto jitter = [&](double val) {
          // Perturb within ±noise, clamp to [0,1] lower bound via max(0,•)
          return std::max(0.0, val + (dist(gen) * 2.0 - 1.0) * noise);
        };
        A = jitter(a);
        B = jitter(b);
        C = jitter(c);
        D = jitter(d);
        double sum = A + B + C + D;
        // Guard against degenerate total (shouldn’t happen unless noise is extreme)
        A = (sum > 0) ? (A / sum) : 0.25;
        B = (sum > 0) ? (B / sum) : 0.25;
        C = (sum > 0) ? (C / sum) : 0.25;
        D = (sum > 0) ? (D / sum) : 0.25;
      }

      // Draw r ~ U(0,1) and select quadrant by cumulative thresholds.
      double r = dist(gen);
      double t1 = A;
      double t2 = A + B;
      double t3 = A + B + C;

      int row_bit = 0, col_bit = 0;
      if (r < t1) {
      // Quadrant 00
        row_bit = 0; col_bit = 0;
      } else if (r < t2) {
      // Quadrant 01
        row_bit = 0; col_bit = 1;
      } else if (r < t3) {
      // Quadrant 10
        row_bit = 1; col_bit = 0;
      } else {
      // Quadrant 11
        row_bit = 1; col_bit = 1;
      }

      // Only set bits that are within the bit-width of rows/cols respectively.
      if (bit < row_bits) u = (u << 1) | row_bit;
      if (bit < col_bits) v = (v << 1) | col_bit;
    }

    // If dimensions are not powers of two, some combinations will exceed bounds.
    if (u >= nrows || v >= ncols) continue;
    // Optionally filter self-loops
    if (no_self_loops && u == v) continue;
    // If undirected, orient edges consistently (store the "upper-triangular" orientation).
    // NOTE: This does NOT create symmetric pairs; it only enforces a canonical ordering.
    if (undirected && u > v) std::swap(u, v);
    // If a duplicate, do not commit edge
    if (std::find(edges.begin(), edges.end(), std::make_pair((int_type)u, (int_type)v)) != edges.end()) continue;

    // Commit the sampled edge.
    edges.emplace_back((int_type)u, (int_type)v);
    ++edge_idx;
  }

  // Sort edges primarily by row, and secondarily by column.
  // CSR expects nonzeros grouped by row; sorting also makes columns within
  // each row non-decreasing, which is often desirable.
  std::sort(edges.begin(), edges.end(),
            [](auto& a, auto& b) {
                return (a.first < b.first) ||
                        (a.first == b.first && a.second < b.second);
            });

  // Initialize row pointer array with zeros.
  // rows[i] will eventually hold the starting index in (vals, cols) of row i.
  // rows[nrows] will equal nnz after prefix-sum (the total number of nonzeros).
  std::fill(rows, rows + nrows + 1, 0);

  // Linear pass over sorted edges to fill cols/vals and count entries per row.
  // We write the k-th edge's column into cols[k] and its value into vals[k].
  // Simultaneously, we increment a per-row count into rows[r+1].
  for (int i = 0; i < nnz; ++i) {
    const int_type r = edges[static_cast<size_t>(i)].first;
    const int_type c = edges[static_cast<size_t>(i)].second;

    cols[static_cast<size_t>(i)] = c;
    vals[static_cast<size_t>(i)] = (T)((double)(rand() % 100) / 3.0);

    // Count one nonzero in row r by bumping rows[r+1].
    // After this loop, rows[k+1] holds the count of nonzeros in row k.
    ++rows[static_cast<size_t>(r) + 1];
  }


    // Convert per-row counts into exclusive prefix sums:
    // rows[0] = 0
    // rows[i+1] = rows[i] + (count of row i)
    // After this, rows[i] is the starting offset of row i in (vals, cols),
    // and rows[nrows] == nnz.
    for (int i = 0; i < nrows; i++) {
        rows[static_cast<size_t>(i) + 1] += rows[static_cast<size_t>(i)];
    }

    // At this point:
    //  - For each row r, nonzeros occupy indices [rows[r], rows[r+1]) in (vals, cols).
    //  - cols in each row are sorted non-decreasingly (due to the global sort above).
    //  - vals are all 1 by default (modify above if you want random or specific weights).
    checkCSRValid(nrows, ncols, nnz, rows, cols, vals);
}
