#pragma once

#include "utilities.h"

/** Performs GEMM operations of type `dType` on host CPU for `iters` iterations.
 * Returns the time taken to perform the operation in seconds. */
double gemm_cpu(const dataTypes dType, const int iters, const int m,
                const int n, const int k);