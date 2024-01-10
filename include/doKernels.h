#pragma once

/** Perform all SGEMM kernels on CPU and GPU, and write results to CSV. */
void doSgemm(const int iters, const int upperLimit);

/** Perform all SGEMM kernels on CPU and GPU, and write results to CSV. */
void doDgemm(const int iters, const int upperLimit);