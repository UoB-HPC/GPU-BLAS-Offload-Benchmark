#include <sys/stat.h>

#include <iostream>
#include <string>

#include "doGemm.hh"
#include "utilities.hh"

/** A function which prints standard configuration information to stdout. */
void printBenchmarkConfig(const int iters, const int upperLimit);

/** A function to parse a string to integer. */
int parseInt(const char* str);

/** A function which parsen the runtime arguments. */
void getParameters(int argc, char* argv[]);