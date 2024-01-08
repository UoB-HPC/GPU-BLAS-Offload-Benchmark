SHELL := bash
.SHELLFLAGS := -eu -o pipefail -c
# .ONESHELL: # This doesn't work on all systems. Disabling for now.
.DELETE_ON_ERROR:

MAKEFLAGS += --warn-undefined-variables --no-builtin-rules

# -------

MACHINE = $(shell uname -m)
ifeq ($(MACHINE), x86_64)
ARCHFLAG = march
else
# The target CPU is specificed differently on x86 and on aarch64
# https://community.arm.com/developer/tools-software/tools/b/tools-software-ides-blog/posts/compiler-flags-across-architectures-march-mtune-and-mcpu
ARCHFLAG = mcpu
endif

# -------

ifndef COMPILER
$(warning COMPILER not set (use ARM, CLANG, GNU, or INTEL))
COMPILER=GNU
endif

CC_ARM     = armclang
CC_CLANG   = clang
CC_GNU     = gcc
CC_INTEL   = icc
CC = $(CC_$(COMPILER))

CFLAGS_ARM     = -std=c99 -Wall -Ofast -$(ARCHFLAG)=native
CFLAGS_CLANG   = -std=c99 -Wall -Ofast -$(ARCHFLAG)=native
CFLAGS_GNU     = -std=c99 -Wall -Ofast -$(ARCHFLAG)=native
CFLAGS_INTEL   = -std=c99 -Wall -Ofast -$(ARCHFLAG)=native
CFLAGS = $(CFLAGS_$(COMPILER))

# -------

ifndef CPU_LIBRARY
$(warning CPU_LIBRARY not set (use ARMPL, ONEMKL, AOCL, OPENBLAS). Naive solutions being used.)
else ifeq ($(CPU_LIBRARY), ARMPL)
# Add ARM compiler options
ifeq ($(COMPILER), ARM)
CFLAGS += -armpl=ilp64,parallel -fopenmp
# Error to select ArmPL otherwise
else
$(error Selected compiler $(COMPILER) is not currently compatible with ArmPL)
endif
else ifeq ($(CPU_LIBRARY), ONEMKL)
# Do OneMKL stuff
$(error The CPU_LIBRARY $(CPU_LIBRARY) is currently not supported.)
else ifeq ($(CPU_LIBRARY), AOCL)
# Do AOCL stuff
$(error The CPU_LIBRARY $(CPU_LIBRARY) is currently not supported.)
else ifeq ($(CPU_LIBRARY), OPENBLAS)
# Do OpenBLAS stuff
$(error The CPU_LIBRARY $(CPU_LIBRARY) is currently not supported.)
else
$(warning Provided CPU_LIBRARY not valid (use ARMPL, ONEMKL, AOCL, OPENBLAS). Naive solutions being used.)
endif

# -------

ifndef GPU_LIBRARY
$(warning GPU_LIBRARY not set (use CUBLAS, ONEMKL, ROCBLAS). No GPU kernels will be run.)
else ifeq ($(GPU_LIBRARY), CUBLAS)
# Do cuBLAS stuff
$(error The GPU_LIBRARY $(GPU_LIBRARY) is currently not supported.)
else ifeq ($(GPU_LIBRARY), ONEMKL)
# Do OneMKL stuff
$(error The GPU_LIBRARY $(GPU_LIBRARY) is currently not supported.)
else ifeq ($(GPU_LIBRARY), ROCBLAS)
# Do rocBLAS stuff
$(error The GPU_LIBRARY $(GPU_LIBRARY) is currently not supported.)
else
$(warning Provided GPU_LIBRARY not valid (use CUBLAS, ONEMKL, ROCBLAS). No GPU kernels will be run.)
endif

# -------

ifdef CPU_LIBRARY
CFLAGS += -DCPU_$(CPU_LIBRARY)
endif
ifdef GPU_LIBRARY
CFLAGS += -DGPU_$(GPU_LIBRARY)
endif
ifdef ITERATIONS
CFLAGS += -DITERATIONS=$(ITERATIONS)
endif
ifdef UPPER_LIMIT
CFLAGS += -DUPPER_LIMIT=$(UPPER_LIMIT)
endif

LDFLAGS = -lm 

# -------

EXE = gpu-blob
HEADER_FILES = $(wildcard DefaultCPU/* DefaultGPU/* ArmPL/*)

.PHONY: all $(EXE) clean

all: $(EXE)

$(EXE): main.c main.h utilities.h $(HEADER_FILES)
	$(CC) $(CFLAGS) main.c -o $@ $(LDFLAGS)

clean:
	rm -f $(EXE)