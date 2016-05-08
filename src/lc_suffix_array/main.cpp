#include <stdio.h>
#include <stdlib.h>
#include <fstream>
#include <streambuf>
#include <sstream>
#include <iostream>
#include "mpi.h"
#include "../lc_suffix_array/suffix_array.h"

using namespace std;

size_t get_filesize(const char* filename) {
  std::ifstream in(filename, std::ifstream::ate | std::ifstream::binary);
  return in.tellg();
}

int main(int argc, char* argv[]) {
  if (argc > 2) {
    fprintf(stdout, "<input file> <output file>\n");
    exit(1);
  }
  int numprocs;
  int rank;
  int namelen;
  char processor_name[MPI_MAX_PROCESSOR_NAME];

  MPI_Init(&argc, &argv);
  MPI_Comm_size(MPI_COMM_WORLD, &numprocs);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Get_processor_name(processor_name, &namelen);

  MPI_Comm MPI_SPLIT_COMM;

  if (rank % 2 == 0) {
    MPI_Comm_split(MPI_COMM_WORLD, 0, numprocs, &MPI_SPLIT_COMM);
  } else {
    MPI_Comm_split(MPI_COMM_WORLD, MPI_UNDEFINED, numprocs, &MPI_SPLIT_COMM);
  }

  uint64_t file_size = 0;
  char* data = NULL;
  if (rank % 2 == 1) {
    MPI_Finalize();
    exit(0);
  }
  MPI_Comm_size(MPI_SPLIT_COMM, &numprocs);
  MPI_Comm_rank(MPI_SPLIT_COMM, &rank);
  // fprintf(stdout, "Process %d on %s\n", rank, processor_name);

  if (!rank) fprintf(stdout, "Reading file from disk onto node.\n");

  if (rank == 0) {
    fprintf(stdout, "There are %d total nodes.\n", numprocs);
    ifstream f(argv[1]);
    if (!f.good()) {
      f.close();
      fprintf(stdout, "File doesn't exist\n");
      MPI_Finalize();
      exit(-1);
    } else {
      f.close();
    }
  }

  file_size = get_filesize(argv[1]);
  std::ifstream t(argv[1]);
  if (!rank) fprintf(stdout, "Reading file of size %lu\n", file_size);
  try {
    data = new char[file_size + 7];
  } catch (std::bad_alloc& ba) {
    fprintf(stdout, "File allocation on node %d failed.\n", rank);
    MPI_Finalize();
    exit(-1);
  }

  t.readsome(data, file_size);
  for (int i = 0; i < 7; i++) {
    data[file_size + i] = 0;
  }

  if (!rank) fprintf(stdout, "Starting suffix array construction\n");

  MPI_Barrier(MPI_SPLIT_COMM);  // test only

  // Decompose file
  uint64_t offset = 0;
  const uint32_t mod = file_size % numprocs;
  const uint64_t size = file_size / static_cast<uint64_t>(numprocs) +
                        (static_cast<uint64_t>(rank) < mod);
  if (static_cast<uint32_t>(rank) < mod) {
    offset = rank * size;
  } else {
    offset = mod * (size + 1) + (rank - mod) * size;
  }
  // fprintf(stdout, "Node %d has size %lu offset %lu\n", rank, size, offset);
  MPI_Barrier(MPI_SPLIT_COMM);  // test only

  /*
   *  Begin construction!
   */

  if (!rank) fprintf(stdout, "Begin suffix array construction\n");

  uint64_t* suffixarray = NULL;
  try {
    suffixarray = new uint64_t[size]();
  } catch (std::bad_alloc& ba) {
    fprintf(stderr, "Bad alloc \n");
    MPI_Finalize();
    exit(-1);
  }

  MPI_Barrier(MPI_SPLIT_COMM);  // test only
  double construction_time = MPI::Wtime();

  SuffixArray st;
  if (st.build(data, size, file_size, offset, numprocs, rank, suffixarray,
               MPI_SPLIT_COMM) < 0) {
    fprintf(stderr, "Error in process %d, terminating.\n", rank);
    MPI_Finalize();
    exit(-1);
  }

  // fprintf(stdout, "Done building at rank %d\n", rank);
  MPI_Barrier(MPI_SPLIT_COMM);
  if (!rank)
    fprintf(stdout, "Building time: %f\n", MPI::Wtime() - construction_time);

  // printf("Result: %d: ", rank);
  // for(uint64_t i = 0; i < size; i++) {
  //   printf("%lu ", suffixarray[i]);
  // }
  // printf("\n");

  // Done
  free(data);
  free(suffixarray);
  MPI_Finalize();
  exit(0);
}
