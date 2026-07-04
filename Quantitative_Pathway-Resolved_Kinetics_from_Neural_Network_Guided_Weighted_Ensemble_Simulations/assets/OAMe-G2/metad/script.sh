#!/bin/bash
#PBS -N SB-MFEP_2D
#PBS -q gpu1
#PBS -j oe
#PBS -l nodes=1:ppn=32
#PBS -l walltime=720:00:00

source ~/Dibyendu/Soft/GMXPLMD/bin/GMXRC
cd /home/suman/Sayari/OAeM_G2_NOGSPATH_OPES
export OMP_NUM_THREADS=1

#mpirun -np 8 gmx_mpi mdrun \
#    -deffnm process \
#    -plumed plumed_process.dat \
#    -cpi process.cpt \
#    -nb cpu \
#    -ntomp 4
#wait
