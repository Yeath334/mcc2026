#!/bin/bash
#SBATCH -p kshdmcc2026
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 32
#SBATCH --exclusive

export OMP_NUM_THREADS=32

HDF5_HOME=/public/software/mathlib/hdf5/1.8.20/gcc-7.3.1
export LD_LIBRARY_PATH=${HDF5_HOME}/lib:$LD_LIBRARY_PATH

gcc -fopenmp -O3 -o get_climatology get_climatology.c \
    -I${HDF5_HOME}/include \
    -L${HDF5_HOME}/lib \
    -lhdf5 -lm

time ./get_climatology
