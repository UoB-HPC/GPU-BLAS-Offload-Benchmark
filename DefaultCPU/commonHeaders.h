#pragma once

#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>

// External consume function used to ensure naive code is performed and not
// optimised away.
extern int consume(void *a, void *b, void *c);