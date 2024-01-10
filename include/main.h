#include "doKernels.h"
#include "utilities.h"

#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>

/** A function which prints standard configuration information to stdout. */
void printBenchmarkConfig(const int iters, const int upperLimit);

/** A function to parse a string to integer. */
int parseInt(const char *str);

/** A function which parsen the runtime arguments. */
void getParameters(int argc, char *argv[]);