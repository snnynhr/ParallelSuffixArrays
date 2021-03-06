#ifndef __SUFFIX_ARRAY__
#define __SUFFIX_ARRAY__

#include <stdio.h>
#include <stdlib.h>
#include "mpi.h"

class SuffixArray {
 public:
  SuffixArray();
  int32_t build(const char* data, uint32_t size, uint32_t file_size,
                uint32_t offset, int numprocs, int myid,
                uint32_t* suffix_array);

 private:
  MPI_Datatype mpi_dc3_elem;
  MPI_Datatype mpi_dc3_tuple_elem;
};

#endif
