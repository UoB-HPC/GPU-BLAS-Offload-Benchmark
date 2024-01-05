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

CFLAGS_ARM     = -std=c99 -Wall -Ofast -fopenmp -$(ARCHFLAG)=native
CFLAGS_CLANG   = -std=c99 -Wall -Ofast -fopenmp -$(ARCHFLAG)=native
CFLAGS_GNU     = -std=c99 -Wall -Ofast -fopenmp -$(ARCHFLAG)=native
CFLAGS_INTEL   = -std=c99 -Wall -Ofast -fopenmp -$(ARCHFLAG)=native
CFLAGS = $(CFLAGS_$(COMPILER))

# -------

ifndef CPU_LIBRARY
$(warning CPU_LIBRARY not set (use ARMPL, ONEMKL, AOCL, OPENBLAS). Naive solutions being used.)
else ifeq ($(CPU_LIBRARY), ARMPL)
# Test for compatible compiler
valid_compiler = no
ifeq ($(COMPILER), ARM)
valid_compiler = yes
else ifeq ($(COMPILER), GNU)
valid_compiler = yes
endif
ifneq ($(valid_compiler), yes)
$(error Selected compiler $(COMPILER) is not compatible with ArmPL)
endif
# Add aditional flags needed
CFLAGS += -armpl=ilp64,parallel
else ifeq ($(CPU_LIBRARY), ONEMKL)
# Do OneMKL stuff
else ifeq ($(CPU_LIBRARY), AOCL)
# Do AOCL stuff
else ifeq ($(CPU_LIBRARY), OPENBLAS)
# Do OpenBLAS stuff
else
$(warning Provided CPU_LIBRARY not valid (use ARMPL, ONEMKL, AOCL, OPENBLAS). Naive solutions being used.)
endif

# -------

ifndef GPU_LIBRARY
$(warning GPU_LIBRARY not set (use CUBLAS, ONEMKL, ROCBLAS). No GPU kernels will be run.)
else ifeq ($(GPU_LIBRARY), CUBLAS)
# Do cuBLAS stuff
else ifeq ($(GPU_LIBRARY), ONEMKL)
# Do OneMKL stuff
else ifeq ($(GPU_LIBRARY), ROCBLAS)
# Do rocBLAS stuff
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

.PHONY: all $(EXE) clean

all: $(EXE)

$(EXE): main.c include/flags.h include/gemm.h include/gemv.h include/spmm.h include/spmv.h
	$(CC) $(CFLAGS) main.c -o $@ $(LDFLAGS)

clean:
	rm -f $(EXE)