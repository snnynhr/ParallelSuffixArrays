#ifndef __SUFFIX_ARRAY__
#define __SUFFIX_ARRAY__

#include <stdio.h>
#include <stdlib.h>
#include "mpi.h"

class SuffixArray {
 public:
  SuffixArray();
  void build(const char* data, uint32_t size, int numprocs, int myid);

 private:
  MPI_Datatype mpi_dc3_elem;
};

#endif