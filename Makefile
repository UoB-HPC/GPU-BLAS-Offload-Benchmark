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
$(warning COMPILER not set (use ARM, CLANG, GNU, INTEL, or NVIDIA))
COMPILER=GNU
endif

CXX_ARM     = armclang++
CXX_CLANG   = clang++
CXX_GNU     = g++
CXX_INTEL   = icc
CXX_NVIDIA  = nvc++
CXX = $(CXX_$(COMPILER))

CXXFLAGS_ARM     = -std=c++20 -Wall -Ofast -$(ARCHFLAG)=native
CXXFLAGS_CLANG   = -std=c++20 -Wall -Ofast -$(ARCHFLAG)=native
CXXFLAGS_GNU     = -std=c++20 -Wall -Ofast -$(ARCHFLAG)=native
CXXFLAGS_INTEL   = -std=c++20 -Wall -Ofast -$(ARCHFLAG)=native
CXXFLAGS_NVIDIA   = -std=c++20 -Wall -Ofast -$(ARCHFLAG)=native
CXXFLAGS = $(CXXFLAGS_$(COMPILER))

SRC_FILES = $(wildcard src/*.cc)
HEADER_FILES = $(wildcard include/*.hh)

# -------

ifndef CPU_LIBRARY
$(warning CPU_LIBRARY not set (use ARMPL, ONEMKL, AOCL, OPENBLAS). Naive, single threaded solutions being used.)
SRC_FILES += $(wildcard DefaultCPU/*.cc)
HEADER_FILES += $(wildcard DefaultCPU/*.hh)
else ifeq ($(CPU_LIBRARY), ARMPL)
# Add ARM compiler options
ifeq ($(COMPILER), ARM)
CXXFLAGS += -armpl=ilp64,parallel -fopenmp
# Error to select ArmPL otherwise
else
$(error Selected compiler $(COMPILER) is not currently compatible with ArmPL)
endif
SRC_FILES += $(wildcard ArmPL/*.cc)
HEADER_FILES += $(wildcard ArmPL/*.hh)
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
SRC_FILES += $(wildcard DefaultCPU/*.cc)
HEADER_FILES += $(wildcard DefaultCPU/*.hh)
endif

# -------

ifndef GPU_LIBRARY
$(warning GPU_LIBRARY not set (use CUBLAS, ONEMKL, ROCBLAS). No GPU kernels will be run.)
SRC_FILES += $(wildcard DefaultGPU/*.cc)
HEADER_FILES += $(wildcard DefaultGPU/*.hh)
else ifeq ($(GPU_LIBRARY), CUBLAS)
# Do cuBLAS stuff
ifeq ($(COMPILER), NVIDIA)
CXXFLAGS += -lcublas
# Error to select ArmPL otherwise
else
$(error Selected compiler $(COMPILER) is not currently compatible with cuBLAS)
endif
else ifeq ($(GPU_LIBRARY), ONEMKL)
# Do OneMKL stuff
$(error The GPU_LIBRARY $(GPU_LIBRARY) is currently not supported.)
else ifeq ($(GPU_LIBRARY), ROCBLAS)
# Do rocBLAS stuff
$(error The GPU_LIBRARY $(GPU_LIBRARY) is currently not supported.)
else
$(warning Provided GPU_LIBRARY not valid (use CUBLAS, ONEMKL, ROCBLAS). No GPU kernels will be run.)
SRC_FILES += $(wildcard DefaultGPU/*.cc)
HEADER_FILES += $(wildcard DefaultGPU/*.hh)
endif

# -------

ifdef CPU_LIBRARY
CXXFLAGS += -DCPU_$(CPU_LIBRARY)
endif
ifdef GPU_LIBRARY
CXXFLAGS += -DGPU_$(GPU_LIBRARY)
endif

LDFLAGS = -lm 

# -------

EXE = gpu-blob

.PHONY: all $(EXE) clean

all: $(EXE)

$(EXE): src/Consume/consume.c $(SRC_FILES) $(HEADER_FILES)
	gcc src/Consume/consume.c -fpic -O0 -shared -o src/Consume/consume.so
	$(CXX) $(CXXFLAGS) $(SRC_FILES) src/Consume/consume.so $(LDFLAGS) -o $@

clean:
	rm -f $(EXE) src/Consume/consume.so