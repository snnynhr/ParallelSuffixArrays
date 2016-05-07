#!/bin/sh
#PBS -lwalltime=10:00
#PBS -lnodes=4:ppn=12
#PBS -N small
#PBS -q phi

cd $PBS_O_WORKDIR

#source ~/ParallelSuffixArrays/jobs/openmpi-setup.sh


/usr/lib64/openmpi/bin/mpirun --verbose --display-allocation -machinefile $PBS_NODEFILE --mca btl tcp,sm,self --mca btl_tcp_if_include em1 -bind-to core:overload-allowed ~/ParallelSuffixArrays/src/suffixArray $1 
