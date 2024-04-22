#!/bin/bash

set -eu

case "$1" in
build)

    # BLIS_DIR=$(spack location -i amdblis threads=openmp)
    # echo "Using BLIS at ${BLIS_DIR}"
    # make COMPILER=GNU CPU_LIB=AOCL CXXFLAGS="-L${BLIS_DIR}/lib -I${BLIS_DIR}/include/blis -Wl,-rpath,${BLIS_DIR}/lib"
    # mv gpu-blob cpu-blob
    # ldd ./cpu-blob

    OPENBLAS_DIR=/users/tomlin/23.09/0.21.0/openblas-0.3.24-6ki7joz
    echo "Using OpenBLAS at ${OPENBLAS_DIR}"
    make COMPILER=GNU CPU_LIB=OPENBLAS CXXFLAGS="-L${OPENBLAS_DIR}/lib -I${OPENBLAS_DIR}/include -Wl,-rpath,${OPENBLAS_DIR}/lib"
    mv gpu-blob cpu-blob
    ldd ./cpu-blob

    # ===================================

    echo "Using hipcc at $(which hipcc)"
    module load rocm/5.2.3
    make COMPILER=HIP GPU_LIB=ROCBLAS
    ldd ./gpu-blob
    ;;

run | submit)
    export SIZE=4096
    for iter in 128; do
    # for iter in 1 8 32 64 128; do
        # for iter in 1; do
        export ITER=$iter
        export NPROC=56
        
        sbatch -J "gpu-blob_cpu_${ITER}i_${SIZE}d_${NPROC}t_openblas" \
            --output "gpu-blob_cpu_${ITER}i_${SIZE}d_${NPROC}t_openblas.out" \
            jobs/lumi_cpu_openblas_template.job
    done

    # for iter in 1 8 32 64 128; do
    #     # for iter in 1; do
    #     export ITER=$iter
    #     sbatch -J "gpu-blob_gpu_${ITER}i_${SIZE}d_rocBLAS" \
    #         --output "gpu-blob_gpu_${ITER}i_${SIZE}d_rocBLAS.out" \
    #         jobs/lumi_gpu_rocblas_template.job
    # done
    ;;
*)
    echo "Unknown action: $1, should be build|run"
    exit 1
    ;;
esac
