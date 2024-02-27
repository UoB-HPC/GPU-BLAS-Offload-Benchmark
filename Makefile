SHELL := bash
.SHELLFLAGS := -eu -o pipefail -c
# .ONESHELL: # This doesn't work on all systems. Disabling for now.
.DELETE_ON_ERROR:

MAKEFLAGS += --warn-undefined-variables --no-builtin-rules

NULL :=
TAB  := $(NULL)          $(NULL)

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
$(warning COMPILER not set (use ARM, CLANG, GNU, INTEL, or NVIDIA). Using GNU as default)
COMPILER=GNU
endif

CXX_ARM     = armclang++
CXX_CLANG   = clang++
CXX_GNU     = g++
CXX_INTEL   = icc
CXX_NVIDIA  = nvc++
CXX = $(CXX_$(COMPILER))

CXXFLAGS_ARM     = -std=c++17 -Wall -Ofast -$(ARCHFLAG)=native
CXXFLAGS_CLANG   = -std=c++17 -Wall -Ofast -$(ARCHFLAG)=native
CXXFLAGS_GNU     = -std=c++17 -Wall -Ofast -$(ARCHFLAG)=native
CXXFLAGS_INTEL   = -std=c++17 -Wall -Ofast -$(ARCHFLAG)=native
CXXFLAGS_NVIDIA  = -std=c++17 -Wall -Ofast -$(ARCHFLAG)=native

ifndef CXXFLAGS
CXXFLAGS = $(CXXFLAGS_$(COMPILER))
else
override CXXFLAGS += $(CXXFLAGS_$(COMPILER))
endif



SRC_FILES = $(wildcard src/*.cc)
HEADER_FILES = $(wildcard include/*.hh)

# -------

ifndef CPU_LIB
$(warning CPU_LIB not set (use ARMPL, ONEMKL, AOCL, OPENBLAS). Naive, single threaded solutions being used.)
HEADER_FILES += $(wildcard DefaultCPU/*.hh)

else ifeq ($(CPU_LIB), ARMPL)
# Add ARM compiler options
ifeq ($(COMPILER), ARM)
override CXXFLAGS += -armpl=parallel -fopenmp
# For all other compilers, require additional input flags for linking
else ifneq($(COMPILER), INTEL)
override CXXFLAGS += -larmpl_lp64_mp -fopenmp
$(warning Users may be required to do the following to use $(COMPILER) with $(CPU_LIB):)
$(info $(TAB)$(TAB)Add `CXXFLAGS="-L<ARMPL_DIR>/lib -I<ARMPL_DIR>/include_lp64_mp"` to make command)
$(info $(TAB)$(TAB)Add `<ARMPL_DIR>/lib` to LD_LIBRARY_PATH via `export LD_LIBRARY_PATH=$$LD_LIBRARY_PATH:<ARMPL_DIR>/lib`)
$(info )
# INTEL compiler not compatible with ArmPL
else
$(error Selected compiler $(COMPILER) is not currently compatible with ArmPL)
endif
HEADER_FILES += $(wildcard ArmPL/*.hh)

else ifeq ($(CPU_LIB), ONEMKL)
# Do OneMKL stuff
$(error The CPU_LIB $(CPU_LIB) is currently not supported.)

else ifeq ($(CPU_LIB), AOCL)
# Do AOCL stuff
$(error The CPU_LIB $(CPU_LIB) is currently not supported.)

else ifeq ($(CPU_LIB), OPENBLAS)
# Do OpenBLAS stuff
$(error The CPU_LIB $(CPU_LIB) is currently not supported.)

else
$(warning Provided CPU_LIB not valid (use ARMPL, ONEMKL, AOCL, OPENBLAS). Naive, single threaded solutions being used.)
HEADER_FILES += $(wildcard DefaultCPU/*.hh)
endif

# -------

ifndef GPU_LIB
$(warning GPU_LIB not set (use CUBLAS, ONEMKL, ROCBLAS). No GPU kernels will be run.)
HEADER_FILES += $(wildcard DefaultGPU/*.hh)

else ifeq ($(GPU_LIB), CUBLAS)
# Do cuBLAS stuff
ifeq ($(COMPILER), NVIDIA)
override CXXFLAGS += -cudalib=cublas
else ifeq ($(COMPILER), GNU)
$(info )
$(info Users may be required to add `CXXFLAGS=-I<dir containing CUBLAS header>` to make command if cuBLAS is installed at non-standard location (e.g /opt/nvidia/hpc_sdk/Linux_x86_64/24.1/math_libs/include))
$(info )
override CXXFLAGS += -lcublas -lcudart
# Error to select cuBLAS otherwise
else
$(error Selected compiler $(COMPILER) is not currently compatible with cuBLAS)
endif
HEADER_FILES += $(wildcard cuBLAS/*.hh)

else ifeq ($(GPU_LIB), ONEMKL)
# Do OneMKL stuff
$(error The GPU_LIB $(GPU_LIB) is currently not supported.)

else ifeq ($(GPU_LIB), ROCBLAS)
# Do rocBLAS stuff
$(error The GPU_LIB $(GPU_LIB) is currently not supported.)

else
$(warning Provided GPU_LIB not valid (use CUBLAS, ONEMKL, ROCBLAS). No GPU kernels will be run.)
HEADER_FILES += $(wildcard DefaultGPU/*.hh)
endif

# -------

ifdef CPU_LIB
override CXXFLAGS += -DCPU_$(CPU_LIB)
endif
ifdef GPU_LIB
override CXXFLAGS += -DGPU_$(GPU_LIB)
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