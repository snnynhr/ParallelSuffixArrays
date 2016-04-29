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

  fprintf(stderr, "Process %d on %s\n", myid, processor_name);
  if (myid == 0) {
    fprintf(stderr, "There are %d total processors.\n", numprocs);
    ifstream f(argv[1]);
    if (!f.good()) {
      f.close();
      MPI_Finalize();
      fprintf(stderr, "File doesn't exist\n");
      exit(-1);
    } else {
      f.close();
    }
  }

  // MPI_Barrier(MPI_COMM_WORLD);

  // Decompose file
  uint64_t size = 0;

  // Read chunk from file.
  // IMPORTANT: this reads size + 2 characters to remove communication.
  char* data = file_block_decompose(argv[1], size, MPI_COMM_WORLD, 1);
  if (data == NULL) {
    fprintf(stderr, "File allocation on processor %d failed.\n", myid);
    MPI_Finalize();
    exit(-1);
  }
  fprintf(stdout, "%d: %.*s", myid, static_cast<uint32_t>(size), data);

  /*
   *  Begin construction!
   */
   SuffixArray st;
   st.build(data, size, numprocs, myid);

  // MPI_Barrier(MPI_COMM_WORLD);

  // Done
  free(data);
  MPI_Finalize();
  exit(0);
}
