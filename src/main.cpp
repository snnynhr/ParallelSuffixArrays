#include <stdio.h>
#include <stdlib.h>
#include "mpi.h"
#include "io/fileio.h"
#include "suffix_array/suffix_array.h"

using namespace std;

int main(int argc, char* argv[]) {
  int numprocs;
  int myid;
  int namelen;
  char processor_name[MPI_MAX_PROCESSOR_NAME];

  MPI_Init(&argc, &argv);
  MPI_Comm_size(MPI_COMM_WORLD, &numprocs);
  MPI_Comm_rank(MPI_COMM_WORLD, &myid);
  MPI_Get_processor_name(processor_name, &namelen);

  // fprintf(stdout, "Process %d on %s\n", myid, processor_name);
  if (myid == 0) {
    fprintf(stdout, "There are %d total processors.\n", numprocs);
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

  if(!myid) fprintf(stdout, "Starting suffix array construction\n");

  MPI_Barrier(MPI_COMM_WORLD);  // test only

  // Decompose file
  uint64_t size = 0;
  uint64_t file_size = 0;
  uint64_t offset = 0;

  // Read chunk from file.
  // IMPORTANT: this reads size + 2 characters to remove communication.
  // Hacky ... be careful with data
  char* data =
      file_block_decompose(argv[1], size, file_size, offset, MPI_COMM_WORLD, 1);
  if (data == NULL) {
    fprintf(stdout, "File allocation on processor %d failed.\n", myid);
    MPI_Finalize();
    exit(-1);
  }
  //fprintf(stdout, "%d: %.*s\n", myid, static_cast<uint32_t>(size), data);

  MPI_Barrier(MPI_COMM_WORLD);  // test only

  /*
   *  Begin construction!
   */

  if(!myid) fprintf(stdout,"Begin suffix array construction\n");

  uint64_t* suffixarray = NULL;
  try {
    suffixarray = new uint64_t[size]();
  } catch (std::bad_alloc& ba) {
    fprintf(stderr, "Bad alloc \n");
    MPI_Finalize();
    exit(-1);
  }

  MPI_Barrier(MPI_COMM_WORLD);  // test only
  double construction_time = MPI::Wtime();

  SuffixArray st;
  if (st.build(data, size, file_size, offset, numprocs, myid, suffixarray) <
      0) {
    fprintf(stderr, "Error in process %d, terminating.\n", myid);
    MPI_Finalize();
    exit(-1);
  }

  // fprintf(stdout, "Done building at rank %d\n", myid);
  MPI_Barrier(MPI_COMM_WORLD);
  if(!myid)
  fprintf(stdout, "Building time: %f\n", MPI::Wtime() - construction_time);

  // printf("Result: %d: ", myid);
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
