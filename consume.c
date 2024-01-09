/** This file contains `consume()` function who's purpose is to ensure that
 * any naive implementation of a BLAS kernel in `DefaultCPU` is not optimised
 * away by the compiler.
 * That is, `consume()` ensures that the resulting data structures appear to be
 * "used" after computation.
 *
 * `consume.c` is compiled as a shared object and linked at compile time.*/

#include <stdlib.h>

int consume(void *a, void *b, void *c) {
  // Do nothing
  return 0;
}