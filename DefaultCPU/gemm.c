#include "../cpuKernels.h"
#include "commonHeaders.h"

void naiveSgemm(const int m, const int n, const int k, const float *restrict A,
                const float *restrict B, float *restrict C);

void naiveDgemm(const int m, const int n, const int k, const double *restrict A,
                const double *restrict B, double *restrict C);

double gemm_cpu(const dataTypes dType, const int iters, const int m,
                const int n, const int k) {
  // Define timer variables
  struct timeval tv, start_tv;
  switch (dType) {
  case _fp32_: {
    float *A;
    float *B;
    float *C;
    A = (float *)malloc(sizeof(float) * m * k);
    B = (float *)malloc(sizeof(float) * k * n);
    C = (float *)malloc(sizeof(float) * m * n);
    // Initialise the matricies
    for (int y = 0; y < m; y++) {
      for (int x = 0; x < k; x++) {
        A[y * k + x] = (((float)(rand() % 10000) / 100.0f) - 30.0);
      }
    }
    for (int y = 0; y < k; y++) {
      for (int x = 0; x < n; x++) {
        B[y * n + x] = (((float)(rand() % 10000) / 100.0f) - 30.0);
      }
    }
    // Warmup run
    naiveSgemm(m, n, k, A, B, C);
    // Start timer
    gettimeofday(&start_tv, NULL);
    // Perform all SGEMM iterations
    for (int i = 0; i < iters; i++) {
      naiveSgemm(m, n, k, A, B, C);
    }
    // Post run consume - ensures naive kernel isn't optimised away
    consume((void *)A, (void *)B, (void *)C);
    break;
  }
  case _fp64_: {
    double *A;
    double *B;
    double *C;
    A = (double *)malloc(sizeof(double) * m * k);
    B = (double *)malloc(sizeof(double) * k * n);
    C = (double *)malloc(sizeof(double) * m * n);
    // Initialise the matricies
    for (int y = 0; y < m; y++) {
      for (int x = 0; x < k; x++) {
        A[y * k + x] = (((double)(rand() % 10000) / 100.0) - 30.0);
      }
    }
    for (int y = 0; y < k; y++) {
      for (int x = 0; x < n; x++) {
        B[y * n + x] = (((double)(rand() % 10000) / 100.0) - 30.0);
      }
    }
    // Warmup run
    naiveDgemm(m, n, k, A, B, C);
    // Start timer
    gettimeofday(&start_tv, NULL);
    // Perform all SGEMM iterations
    for (int i = 0; i < iters; i++) {
      naiveDgemm(m, n, k, A, B, C);
    }
    // Post run consume - ensures naive kernel isn't optimised away
    consume((void *)A, (void *)B, (void *)C);
    break;
  }
  default:
    printf("GEMM_CPU - Unsuported dataType\n");
  }
  // Stop timer
  gettimeofday(&tv, NULL);
  return ((tv.tv_sec - start_tv.tv_sec) +
          (tv.tv_usec - start_tv.tv_usec) / 1000000.0);
}

/** A naive implementation of a FP32 GEMM. Alpha and Beta are always 1 and 0
 * respectively.
 * Operation takes the form of C[M,N] = A[M,K] * B[K,N].
 * A return value is required to ensure that the compiler does not optimise away
 * this function. */
void naiveSgemm(const int m, const int n, const int k, const float *restrict A,
                const float *restrict B, float *restrict C) {
  int x, y, z;
  float acc;
  for (x = 0; x < m; x++) {
    for (y = 0; y < n; y++) {
      acc = 0.0f;
      for (z = 0; z < k; z++) {
        acc += A[x * k + z] * B[z * n + y];
      }
      C[x * n + y] = acc;
    }
  }
}

/** A naive implementation of a FP64 GEMM. Alpha and Beta are always 1 and 0
 * respectively.
 * Operation takes the form of C[M,N] = A[M,K] * B[K,N].
 * A return value is required to ensure that the compiler does not optimise away
 * this function. */
void naiveDgemm(const int m, const int n, const int k, const double *restrict A,
                const double *restrict B, double *restrict C) {
  int x, y, z;
  double acc;
  for (x = 0; x < m; x++) {
    for (y = 0; y < n; y++) {
      acc = 0.0;
      for (z = 0; z < k; z++) {
        acc += A[x * k + z] * B[z * n + y];
      }
      C[x * n + y] = acc;
    }
  }
}