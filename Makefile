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

SRC_FILES = $(wildcard src/*.c)
HEADER_FILES = $(wildcard include/*.h)

# -------

ifndef CPU_LIBRARY
$(warning CPU_LIBRARY not set (use ARMPL, ONEMKL, AOCL, OPENBLAS). Naive, single threaded solutions being used.)
SRC_FILES += $(wildcard DefaultCPU/*.c)
HEADER_FILES += $(wildcard DefaultCPU/*.h)
else ifeq ($(CPU_LIBRARY), ARMPL)
# Add ARM compiler options
ifeq ($(COMPILER), ARM)
CFLAGS += -armpl=ilp64,parallel -fopenmp
# Error to select ArmPL otherwise
else
$(error Selected compiler $(COMPILER) is not currently compatible with ArmPL)
endif
SRC_FILES += $(wildcard ArmPL/*.c)
HEADER_FILES += $(wildcard ArmPL/*.h)
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
$(warning Provided CPU_LIBRARY not valid (use ARMPL, ONEMKL, AOCL, OPENBLAS). Naive, single threaded solutions being used.)
SRC_FILES += $(wildcard DefaultCPU/*.c)
HEADER_FILES += $(wildcard DefaultCPU/*.h)
endif

# -------

ifndef GPU_LIBRARY
$(warning GPU_LIBRARY not set (use CUBLAS, ONEMKL, ROCBLAS). No GPU kernels will be run.)
SRC_FILES += $(wildcard DefaultGPU/*.c)
HEADER_FILES += $(wildcard DefaultGPU/*.h)
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
SRC_FILES += $(wildcard DefaultGPU/*.c)
HEADER_FILES += $(wildcard DefaultGPU/*.h)
endif

# -------

ifdef CPU_LIBRARY
CFLAGS += -DCPU_$(CPU_LIBRARY)
endif
ifdef GPU_LIBRARY
CFLAGS += -DGPU_$(GPU_LIBRARY)
endif

LDFLAGS = -lm 

# -------

EXE = gpu-blob

.PHONY: all $(EXE) clean

all: $(EXE)

$(EXE): src/Consume/consume.c $(SRC_FILES) $(HEADER_FILES)
	gcc src/Consume/consume.c -fpic -O0 -shared -o src/Consume/consume.so
	$(CC) $(CFLAGS) $(SRC_FILES) src/Consume/consume.so $(LDFLAGS) -o $@

clean:
	rm -f $(EXE) src/Consume/consume.so