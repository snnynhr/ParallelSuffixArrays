#ifndef __SUFFIX_ARRAY__
#define __SUFFIX_ARRAY__

#include <stdio.h>
#include <stdlib.h>
#include "mpi.h"

typedef struct css_elem {
  uint64_t word;
  uint64_t index;
} css_elem;

class SuffixArray {
 public:
  SuffixArray();
  int32_t build(const char* data, uint32_t size, uint64_t offset, int numprocs,
                int myid, uint64_t* suffix_array, MPI_Comm comm);

 private:
  MPI_Datatype mpi_css_elem;
  uint64_t _size;
  const char* _data;
};

#endif