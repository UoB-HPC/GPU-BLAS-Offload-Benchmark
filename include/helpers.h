#pragma once

#include "../include/dataTypes.h"

#include <stdint.h>
#include <stdio.h>

/** A function to open a new csv file in WRITE mode and write the standard
 * headers to the file.
 * Returns the file pointer. */
FILE *newCSV(const char *filename);

/** A function to write a new line to an open CSV file. */
void writeLineToCsv(FILE *fptr, const char *device, const char *kernel,
                    const int M, const int N, const int K,
                    const double totalProbSize, const int iters,
                    const double totalTime, const double gflops);

/** A function to calculate GFLOPs. */
double calcGflops(const uint64_t flops, const int iters, const double seconds);

/** A function to calculate KiB from a data-structur's dimensions. */
double calcKib(const uint64_t probSize, const uint64_t bytesPerElem);

/* ------------------------------ GEMM -------------------------------------- */

/** A function for calculating FLOPs performed by a GEMM. */
uint64_t gemmFlops(const int M, const int N, const int K);

/** A function for calculating the total GEMM problem size in KiB. */
double gemmKib(const dataTypes type, const int M, const int N, const int K);

/* -------------------------------------------------------------------------- */