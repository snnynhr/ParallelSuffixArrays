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
  }

  // Decompose file


  // Begin construction


  MPI_Finalize();
  exit(0);
}
