#include "../include/cpuKernels.h"
#include "commonHeaders.h"

#ifdef CPU_ARMPL
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
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, m, n, k, ALPHA, A,
                MAX(1, k), B, MAX(1, n), BETA, C, MAX(1, n));
    consume((void *)A, (void *)B, (void *)C);
    // Start timer
    gettimeofday(&start_tv, NULL);
    // Perform all SGEMM iterations
    for (int i = 0; i < iters; i++) {
      cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, m, n, k, ALPHA, A,
                  MAX(1, k), B, MAX(1, n), BETA, C, MAX(1, n));
      // Call to `consume()` function to ensure all CPU BLAS Library
      // implementations account for function call overhead (only required for
      // DefaultCPU)
      consume((void *)A, (void *)B, (void *)C);
    }
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
    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, m, n, k, ALPHA, A,
                MAX(1, m), B, MAX(1, k), BETA, C, MAX(1, m));
    consume((void *)A, (void *)B, (void *)C);
    // Start timer
    gettimeofday(&start_tv, NULL);
    // Perform all SGEMM iterations
    for (int i = 0; i < iters; i++) {
      cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, m, n, k, ALPHA, A,
                  MAX(1, m), B, MAX(1, k), BETA, C, MAX(1, m));
      // Call to `consume()` function to ensure all CPU BLAS Library
      // implementations account for function call overhead (only required for
      // DefaultCPU)
      consume((void *)A, (void *)B, (void *)C);
    }
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
#endif
