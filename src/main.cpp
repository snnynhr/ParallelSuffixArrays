#include <stdio.h>
#include <stdlib.h>
#include "mpi.h"
#include "fileio.h"

using namespace std;

int main(int argc, char *argv[]) {
  MPI_Request sendreq;
  MPI_Request recvreq;
  MPI_Status status;
  int numprocs;
  int myid;
  int namelen;
  int left, right;
  int bufsize;
  char processor_name[MPI_MAX_PROCESSOR_NAME];
  char *sendbuf, *recvbuf;

  MPI_Init(&argc, &argv);
  MPI_Comm_size(MPI_COMM_WORLD, &numprocs);
  MPI_Comm_rank(MPI_COMM_WORLD, &myid);
  MPI_Get_processor_name(processor_name, &namelen);

  fprintf(stderr, "Process %d on %s\n", myid, processor_name);
  if(myid == 0) {
    fprintf(stderr, "There are %d total processors.\n", numprocs);
    ifstream f(argv[1]);
    if(!f.good()) {
      f.close();
      MPI_Finalize();
      fprintf(stderr, "File doesn't exist\n");
      exit(-1);
    } else {
      f.close();
    }
  }

  //MPI_Barrier(MPI_COMM_WORLD);

  // Decompose file
  uint32_t size = 0;
  char* data = file_block_decompose(argv[1], &size, MPI_COMM_WORLD, 1);
  if(data == NULL) {
    fprintf(stderr, "File allocation on processor %p failed.\n", myid);
    MPI_Finalize();
    exit(-1);
  }
  fprintf(stderr, "Processor %d read the file.\n", myid);
  fprintf(stdout, "%d: %.*s", myid, size, data);

  free(data);
  //MPI_Barrier(MPI_COMM_WORLD);

  // Begin construction


  MPI_Finalize();
  exit(0);
}
