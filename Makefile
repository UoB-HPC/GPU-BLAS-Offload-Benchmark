SHELL := bash
.SHELLFLAGS := -eu -o pipefail -c
# .ONESHELL: # This doesn't work on all systems. Disabling for now.
.DELETE_ON_ERROR:

MAKEFLAGS += --warn-undefined-variables --no-builtin-rules

# Defining TAB variable for easy indenting of warning messages
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
$(warning COMPILER not set (use ARM, CLANG, GNU, INTEL, NVIDIA, or HIP). Using GNU as default)
COMPILER=GNU
endif

ifneq ($(COMPILER), ARM)
ifneq ($(COMPILER), CLANG)
ifneq ($(COMPILER), GNU)
ifneq ($(COMPILER), INTEL)
ifneq ($(COMPILER), NVIDIA)
ifneq ($(COMPILER), HIP)
$(error Given compiler $(COMPILER) not valid. Please choose from ARM, CLANG, GNU, INTEL, NVIDIA, or HIP)
endif
endif
endif
endif
endif
endif

CXX_ARM     = armclang++
CXX_CLANG   = clang++
CXX_GNU     = g++
CXX_INTEL   = icpx
CXX_NVIDIA  = nvc++
CXX_HIP     = hipcc
CXX = $(CXX_$(COMPILER))

CXXFLAGS_ARM     = -std=c++17 -Wall -Ofast -$(ARCHFLAG)=native
CXXFLAGS_CLANG   = -std=c++17 -Wall -Ofast -$(ARCHFLAG)=native
CXXFLAGS_GNU     = -std=c++17 -Wall -Wno-deprecated-declarations -Ofast -$(ARCHFLAG)=native
CXXFLAGS_INTEL   = -std=c++17 -Wall -Ofast -$(ARCHFLAG)=native -Wno-tautological-constant-compare
CXXFLAGS_NVIDIA  = -std=c++17 -Wall -O3 -fast -$(ARCHFLAG)=native
CXXFLAGS_HIP     = -std=c++17 -Wall -Ofast -$(ARCHFLAG)=native

ifndef CXXFLAGS
CXXFLAGS = $(CXXFLAGS_$(COMPILER))
else
override CXXFLAGS += $(CXXFLAGS_$(COMPILER))
endif



SRC_FILES = $(wildcard src/*.cc)
HEADER_FILES = $(wildcard include/*.hh)

# -------

ifndef CPU_LIB
$(warning CPU_LIB not set (use ARMPL, ONEMKL, AOCL, NVPL, OPENBLAS). No CPU kernels will be run.)

else ifeq ($(CPU_LIB), ARMPL)
# Add ARM compiler options
ifeq ($(COMPILER), ARM)
override CXXFLAGS += -armpl=parallel -fopenmp
else ifeq ($(COMPILER), INTEL)
# INTEL compiler not compatible with ArmPL
$(error Selected compiler $(COMPILER) is not currently compatible with ArmPL)
else ifeq ($(COMPILER), HIP)
# HIP compiler not compatible with ArmPL
$(error Selected compiler $(COMPILER) is not currently compatible with ArmPL)
else 
# For all other compilers, require additional input flags for linking
override CXXFLAGS += -larmpl_lp64_mp -fopenmp
$(warning Users may be required to do the following to use $(COMPILER) with $(CPU_LIB):)
$(info $(TAB)$(TAB)Add `CXXFLAGS="-L<ARMPL_DIR>/lib -I<ARMPL_DIR>/include_lp64_mp -Wl,-rpath,<ARMPL_DIR>/lib"` to make command)
$(info )
endif
HEADER_FILES += $(wildcard ArmPL/*.hh)

else ifeq ($(CPU_LIB), ONEMKL)
# Ensure MKLROOT is defined
ifndef MKLROOT
$(error Must add `MKLROOT=/path/to/mkl/` to make command to use OneMKL CPU Library)
endif
# Add INTEL compiler options
ifeq ($(COMPILER), INTEL)
override CXXFLAGS += -L$(MKLROOT)/lib -lmkl_intel_lp64 -lmkl_intel_thread -lmkl_core -liomp5 -lpthread -lm -ldl -qmkl=parallel -DMKL_INT=int
# Add GNU compiler options
else ifeq ($(COMPILER), GNU)
override CXXFLAGS += -m64 -L$(MKLROOT)/lib -Wl,--no-as-needed -lmkl_intel_lp64 -lmkl_gnu_thread -lmkl_core -lgomp -lpthread -lm -ldl -I"${MKLROOT}/include" -DMKL_INT=int
$(warning Users may be required to do the following to use $(COMPILER) with $(CPU_LIB):)
$(info $(TAB)$(TAB)Add `<MKLROOT>/lib` to `$$LD_LIBRARY_PATH`)
$(info )
# Add CLANG compiler options
else ifeq ($(COMPILER), CLANG)
override CXXFLAGS += -L$(MKLROOT)/lib -Wl,--no-as-needed -lmkl_intel_lp64 -lmkl_gnu_thread -lmkl_core -lgomp -lpthread -lm -ldl -m64 -I"${MKLROOT}/include" -DMKL_INT=int
$(warning Users may be required to do the following to use $(COMPILER) with $(CPU_LIB):)
$(info $(TAB)$(TAB)Add `<MKLROOT>/lib` to `$$LD_LIBRARY_PATH`)
$(info )
# Other compilers not compatible with ONEMKL
else
$(error Selected compiler $(COMPILER) is not currently compatible with oneMKL CPU Library)
endif
HEADER_FILES+= $(wildcard oneMKL/CPU/*.hh)

else ifeq ($(CPU_LIB), AOCL)
ifeq ($(COMPILER), INTEL)
override CXXFLAGS += -lblis-mt -qopenmp
else
override CXXFLAGS += -lblis-mt -fopenmp
endif
$(warning Users may be required to do the following to use $(COMPILER) with $(CPU_LIB):)
$(info $(TAB)$(TAB)Add `CXXFLAGS="-L<AOCL_DIR>/lib -I<AOCL_DIR>/include/blis -Wl,-rpath,<AOCL_DIR>/lib"` to make command)
$(info )
HEADER_FILES+= $(wildcard AOCL/*.hh)

else ifeq ($(CPU_LIB), NVPL)
ifeq ($(COMPILER), INTEL)
# INTEL compiler not compatible with NVPL
$(error Selected compiler $(COMPILER) is not currently compatible with NVPL)
else ifeq ($(COMPILER), HIP)
# HIP compiler not compatible with NVPL
$(error Selected compiler $(COMPILER) is not currently compatible with NVPL)
else
override CXXFLAGS += -lnvpl_blas_lp64_gomp
$(warning Users may be required to do the following to use $(COMPILER) with $(CPU_LIB):)
$(info $(TAB)$(TAB)Add `CXXFLAGS="-L<NVPL_DIR>/lib -I<NVPL_DIR>/include -Wl,-rpath,<NVPL_DIR>/lib"` to make command)
$(info )
ifeq ($(COMPILER), GNU)
override CXXFLAGS += -lgomp
else ifeq ($(COMPILER), NVIDIA)
override CXXFLAGS += -lnvomp
else
# LLVM based compilers (CLANG, ARMCLANG)
override CXXFLAGS += -lomp
endif
endif
HEADER_FILES+= $(wildcard NVPL/*.hh)


else ifeq ($(CPU_LIB), OPENBLAS)
override CXXFLAGS += -lopenblas -lpthread -lgfortran
$(warning Users may be required to do the following to use $(COMPILER) with $(CPU_LIB):)
$(info $(TAB)$(TAB)Add `CXXFLAGS="-L<OPENBLAS_DIR>/lib -I<OPENBLAS_DIR>/include -Wl,-rpath,<OPENBLAD_DIR>/lib"` to make command)
$(info )

else
$(warning Provided CPU_LIB not valid (use ARMPL, ONEMKL, AOCL, NVPL, OPENBLAS). No CPU kernels will be run.)
endif

# -------

ifndef GPU_LIB
$(warning GPU_LIB not set (use CUBLAS, ONEMKL, ROCBLAS). No GPU kernels will be run.)

else ifeq ($(GPU_LIB), CUBLAS)
# Do cuBLAS stuff
ifeq ($(COMPILER), NVIDIA)
override CXXFLAGS += -cudalib=cublas -lcusparse_static
else
$(warning Users may be required to do the following to use $(COMPILER) with $(GPU_LIB):)
$(info $(TAB)$(TAB)Add `CXXFLAGS=-L<NVHPC_DIR>/.../math_libs/lib64 -L<NVHPC_DIR>/.../cuda/lib64` to make command)
$(info $(TAB)$(TAB)Add `CXXFLAGS=-I<NVHPC_DIR>/.../math_libs/include -I<NVHPC_DIR>/.../cuda/include` to make command)
$(info $(TAB)$(TAB)Add `CXXFLAGS=-Wl,-rpath,<NVHPC_DIR>/.../math_libs/lib64 -Wl,-rpath,<NVHPC_DIR>/.../cuda/lib64` to make command)
$(info )
override CXXFLAGS += -lcublas -lcudart -lcusparse
endif
HEADER_FILES += $(wildcard cuBLAS/*.hh)

else ifeq ($(GPU_LIB), ONEMKL)
ifeq ($(COMPILER), INTEL)
# Ensure MKLROOT is defined
ifndef MKLROOT
$(error Must add `MKLROOT=/path/to/mkl/` to make command to use OneMKL CPU Library)
endif
# Add compiler and link options
override CXXFLAGS += -fsycl -L$(MKLROOT)/lib -lmkl_sycl_blas -lmkl_intel_ilp64 -lmkl_tbb_thread -lmkl_core -lsycl -lpthread -lm -ldl  -fsycl -DMKL_ILP64  -I"$(MKLROOT)/include"
# `lmkl_tbb_thread` can replace `lmkl_sequential`
$(warning Users may be required to do the following to use $(COMPILER) with $(GPU_LIB):)
$(info $(TAB)$(TAB)Add `<MKLROOT>/lib` to `$$LD_LIBRARY_PATH`)
$(info )
else
# Only Intel DPC++ compiler is supported for OneMKL GPU implementation.
$(error Selected compiler $(COMPILER) is not currently compatible with oneMKL GPU Library)
endif

else ifeq ($(GPU_LIB), ROCBLAS)
ifeq ($(COMPILER), HIP)
# Do rocBLAS stuff
override CXXFLAGS += -lrocblas -lm -lpthread -D__HIP_PLATFORM_AMD__
$(warning Users may be required to do the following to use $(COMPILER) with $(GPU_LIB):)
$(info $(TAB)$(TAB)Add `CXXFLAGS=-L<ROCM_PATH>/lib -L<ROCBLAS_PATH>/lib` to make command)
$(info $(TAB)$(TAB)Add `CXXFLAGS=-I<ROCM_PATH>/include -I<ROCBLAS_PATH>/include` to make command)
$(info $(TAB)$(TAB)Add `CXXFLAGS=-Wl,-rpath,<ROCM_PATH>/lib -Wl,-rpath,<ROCBLAS_PATH>/lib` to make command)
HEADER_FILES += $(wildcard rocBLAS/*.hh)
else
$(error Selected compiler $(COMPILER) is not currently compatible with rocBLAS GPU Library)
endif


else
$(warning Provided GPU_LIB not valid (use CUBLAS, ONEMKL, ROCBLAS). No GPU kernels will be run.)
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
	gcc src/Consume/consume.c -fpic -O0 -shared -o src/Consume/libconsume.so
	$(CXX) $(SRC_FILES) $(CXXFLAGS) -Lsrc/Consume -Wl,-rpath,src/Consume -lconsume $(LDFLAGS) -o $@

clean:
	rm -f $(EXE) src/Consume/libconsume.so