#!/bin/sh
#PBS -lwalltime=10:00
#PBS -lnodes=4:ppn=24
#PBS -N lc_suffix_array
#PBS -q phi

cd $PBS_O_WORKDIR

cat $PBS_NODEFILE | uniq > hostfile
export NPROCS=`wc -l hostfile | gawk '//{print $1}'`
#export OMP_NUM_THREADS=8
/usr/lib64/openmpi/bin/mpirun --verbose --display-allocation -machinefile hostfile --mca btl tcp,sm,self --mca btl_tcp_if_include em1 -np $NPROCS --map-by ppr:1:node -bind-to none ~/ParallelSuffixArrays/src/suffixArray $1 
