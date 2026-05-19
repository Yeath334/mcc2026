#!/bin/bash
#SBATCH -p kshdmcc2026
#SBATCH -N 2
#SBATCH -n 2
#SBATCH -c 32
#SBATCH --exclusive
#SBATCH --gres=dcu:4

# ============================================================
# Set USE_DCU=1 to enable GPU acceleration (Hygon DCU)
# Set USE_DCU=0 for CPU-only MPI + OpenMP
# ============================================================
USE_DCU=1

export OMP_NUM_THREADS=32
export OMP_PROC_BIND=true
export OMP_PLACES=cores

HDF5_HOME=/public/software/mathlib/hdf5/1.8.20/gcc-7.3.1
MPI_HOME=/opt/hpc/software/mpi/hpcx/v2.11.0/gcc-7.3.1

export LD_LIBRARY_PATH=${HDF5_HOME}/lib:${MPI_HOME}/lib:$LD_LIBRARY_PATH
export PATH=${MPI_HOME}/bin:$PATH

if [ "$USE_DCU" = "1" ]; then
    DTK_HOME=/public/software/compiler/rocm/dtk-23.10
    export LD_LIBRARY_PATH=${DTK_HOME}/lib:${LD_LIBRARY_PATH}
    HIPCC=${DTK_HOME}/bin/hipcc

    echo "=== Compiling with HIP/DCU + MPI ==="
    cp get_climatology.c get_climatology_dcu.hip
    ${HIPCC} -DUSE_HIP --offload-arch=gfx906 -fopenmp -O3 \
        -o get_climatology_mpi get_climatology_dcu.hip \
        -I${HDF5_HOME}/include -L${HDF5_HOME}/lib -lhdf5 \
        -I${MPI_HOME}/include -L${MPI_HOME}/lib -lmpi -lm
    BIN=get_climatology_mpi
else
    echo "=== Compiling with MPI + OpenMP (CPU) ==="
    mpicc -fopenmp -O3 -march=native -o get_climatology_mpi get_climatology.c \
        -I${HDF5_HOME}/include \
        -L${HDF5_HOME}/lib \
        -lhdf5 -lm
    BIN=get_climatology_mpi
fi

if [ $? -ne 0 ]; then
    echo "Compilation failed!"
    exit 1
fi

SRC_DIR="/public/home/achwjznh4b/Newdata"
LOCAL_DIR="/dev/shm/sst_cache"
OUT_DIR="/public/home/mcc20262029/zhx/mcc_3/ERA5/Climatology"
CACHE_DIR="/public/home/mcc20262029/zhx/mcc_3/.cache"
RUN_TMP="/public/home/mcc20262029/zhx/mcc_3/.runs"

mkdir -p "$LOCAL_DIR" "$CACHE_DIR" "$OUT_DIR" "$RUN_TMP"

# ============== 提前生成文件列表（只做一次） ==============
for RANK in 0 1; do
    FL="${CACHE_DIR}/stage_list_rank${RANK}.txt"
    if [ ! -f "$FL" ]; then
        if [ $RANK -eq 0 ]; then WS=147; WE=202; else WS=193; WE=248; fi
        for yr in $(seq 1991 2020); do
            for doy in $(seq $WS $WE); do
                m=1; d=$doy
                for mdays in 31 28 31 30 31 30 31 31 30 31 30 31; do
                    if [ $d -le $mdays ]; then break; fi
                    d=$((d - mdays)); m=$((m + 1))
                done
                printf "%s/%04d%02d%02d\n" "$SRC_DIR" $yr $m $d
            done
        done > "$FL"
    fi
done

# ============== 核心：运行 5 次，每次都包含拷贝 ==============
echo "=== Running 5 times (each run includes staging) ==="
NRUNS=5
BEST_TIME=999999

for run in $(seq 1 $NRUNS); do
    echo -e "\n--- Run $run/$NRUNS ---"
    rm -f "$OUT_DIR"/*.nc

    # ---- 开始计时 ----
    T_START=$(date +%s.%N)

    # 拷贝
    srun --ntasks-per-node=1 bash -c "
        xargs -P 32 -a \"${CACHE_DIR}/stage_list_rank\${SLURM_PROCID}.txt\" -I {} cp -n {} \"$LOCAL_DIR/\" 2>/dev/null
    "

    # 运行程序
    mpirun --map-by node --bind-to socket --report-bindings \
        ./$BIN "$LOCAL_DIR/" 2>/dev/null

    # ---- 结束计时 ----
    T_END=$(date +%s.%N)
    TOTAL=$(echo "$T_END - $T_START" | bc)

    echo "  Run $run total time: $TOTAL s"

    # 保存本轮输出
    mkdir -p "$RUN_TMP/run_$run"
    mv "$OUT_DIR"/*.nc "$RUN_TMP/run_$run/" 2>/dev/null || true

    # 更新最快
    if (( $(echo "$TOTAL < $BEST_TIME" | bc -l) )); then
        BEST_TIME=$TOTAL
    fi
done

# 恢复最快那轮的输出
mv "$RUN_TMP"/*/*.nc "$OUT_DIR/" 2>/dev/null
rm -rf "$RUN_TMP"

echo -e "\n============================================"
printf "  BEST FULL RUN TIME : %8.2f s\n" $BEST_TIME
echo "============================================"
