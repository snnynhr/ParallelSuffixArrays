#ifndef __SA_SAMPLESORT
#define __SA_SAMPLESORT

#include <mpi.h>

template <typename _RandomAccessIter, typename _Compare>
void samplesort(_RandomAccessIter begin, _RandomAccessIter end, _Compare comp,
                MPI_Datatype mpi_dtype, int numprocs, int myid);

#include "sort.hpp"

#endif
