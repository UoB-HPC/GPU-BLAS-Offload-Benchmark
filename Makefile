SHELL := bash
.SHELLFLAGS := -eu -o pipefail -c
# .ONESHELL: # This doesn't work on all systems. Disabling for now.
.DELETE_ON_ERROR:

MAKEFLAGS += --warn-undefined-variables --no-builtin-rules

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

# -------

ifndef CPU_LIBRARY
$(warning CPU_LIBRARY not set (use ARMPL, ONEMKL, AOCL, OPENBLAS). Naive solutions being used.)
else ifeq ($(CPU_LIBRARY), ARMPL)
# Do ArmPL stuff
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

CFLAGS_ARM     = -std=c99 -Wall -Ofast
CFLAGS_CLANG   = -std=c99 -Wall -Ofast
CFLAGS_GNU     = -std=c99 -Wall -Ofast
CFLAGS_INTEL   = -std=c99 -Wall -Ofast
CFLAGS = $(CFLAGS_$(COMPILER))

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