#include <algorithm>
#include <stdio.h>
#include <mpi.h>

#include "sort.h"

const int LOCALSZ = 1000;

inline int prng (int seed) {
  return (9253729 * seed + 2396403) % 32767;
}

int main() {
  int numprocs,
      myid;

  MPI_Init(NULL, NULL);
  MPI_Comm_size(MPI_COMM_WORLD, &numprocs);
  MPI_Comm_rank(MPI_COMM_WORLD, &myid);

  int arr[LOCALSZ];
  for (int i = 0; i < LOCALSZ; ++i)
    arr[i] = prng(myid*LOCALSZ+i);

  samplesort(arr, arr+LOCALSZ, std::less<int>(), MPI_INT, numprocs, myid);

  MPI_Finalize();
}
