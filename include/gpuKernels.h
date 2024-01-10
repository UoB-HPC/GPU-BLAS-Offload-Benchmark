#pragma once

#include "dataTypes.h"
#include <stdbool.h>

/** Performs GEMM operations of type `dType` on host GPU for `iters` iterations.
 * Returns the time taken to perform the operation in seconds.
 *  - `offloadOnce` refers to whether the matrix data should be offloaded to the
 *    device once before computation and then copied back at the end of all
 *    iterations, or if the matrcies should be offloaded to/from the device
 *    every iteration. */
double gemm_gpu(const dataTypes dType, const int iters, const int m,
                const int n, const int k, const bool offloadOnce);