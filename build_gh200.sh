#!/bin/bash

set -eu

function run(){
    local name=$1
    local out=$2
    local job=$3
    : >"$out"
    set +e # don't fail on non-zero exit
    bash "$job" &> >(tee -a "$out")
    set -e # restore
    echo "$name complete."
}

# NVPL 
export CPATH="$HOME/nvpl_blas-linux-sbsa-0.2.0.1-archive/include:${CPATH:-}"
export LIBRARY_PATH="$HOME/nvpl_blas-linux-sbsa-0.2.0.1-archive/lib:${LIBRARY_PATH:-}"
export LD_LIBRARY_PATH="$HOME/nvpl_blas-linux-sbsa-0.2.0.1-archive/lib:${LD_LIBRARY_PATH:-}"

case "$1" in
build)

    # BLIS_DIR=$(spack location -i amdblis threads=openmp)
    # echo "Using BLIS at ${BLIS_DIR}"
    # make COMPILER=GNU CPU_LIB=AOCL CXXFLAGS="-L${BLIS_DIR}/lib -I${BLIS_DIR}/include/blis -Wl,-rpath,${BLIS_DIR}/lib"
    # mv gpu-blob cpu-blob
    # ldd ./cpu-blob
    module purge
    # module load nvhpc/23.9
    spack load nvhpc@24.3


    # echo "Using OpenBLAS at ${OPENBLAS_DIR}"
    make COMPILER=NVIDIA CPU_LIB=NVPL
    mv gpu-blob cpu-blob
    ldd ./cpu-blob

    # ===================================

    echo "Using nvcc at $(which nvcc)"
    make COMPILER=NVIDIA GPU_LIB=CUBLAS
    ldd ./gpu-blob
    ;;

run | submit)
    export SIZE=4096
    for iter in 64 128; do
        export ITER=$iter
        export NPROC=$(nproc)

        run "gpu-blob_cpu_${ITER}i_${SIZE}d_${NPROC}t_nvpl" \
            "gpu-blob_cpu_${ITER}i_${SIZE}d_${NPROC}t_nvpl.out" \
            jobs/gh200_cpu_nvpl_template.job
    done

    for iter in 64 128; do
        export ITER=$iter
        run "gpu-blob_gpu_${ITER}i_${SIZE}d_cublas" \
            "gpu-blob_gpu_${ITER}i_${SIZE}d_cublas.out" \
            jobs/gh200_gpu_cublas_template.job
    done
    ;;
*)
    echo "Unknown action: $1, should be build|run"
    exit 1
    ;;
esac
