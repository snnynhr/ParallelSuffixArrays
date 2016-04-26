#include <algorithm>
#include <assert.h>
#include <stdio.h>
#include <mpi.h>

#include "ssort.h"

inline int prng (int seed) {
  return (9253729 * seed + 2396403) % 32767;
}

static int *exclusive_sum(int *arr, size_t n) {
  int *sums = new int[n];
  sums[0] = 0;
  for (size_t i = 1; i < n; ++i)
    sums[i] = sums[i-1] + arr[i-1];
  return sums;
}

void sorttest(int *arr, int n, int numprocs, int myid) {
  int *all_sizes = NULL;
  int *correct_result = NULL;
  int *displacements = NULL;
  int *ssort_result = NULL;
  int total_size;

  if (myid == 0)
    all_sizes = new int[numprocs];

  MPI_Gather(&n, 1, MPI_INT, all_sizes, 1, MPI_INT, 0, MPI_COMM_WORLD);

  if (myid == 0) {
    displacements = exclusive_sum(all_sizes, numprocs);
    total_size = displacements[numprocs-1] + all_sizes[numprocs-1];
    correct_result = new int[total_size];
    ssort_result = new int[total_size];
  }

  MPI_Gatherv(arr, n, MPI_INT, correct_result, all_sizes, displacements,
              MPI_INT, 0, MPI_COMM_WORLD);

  ssort::samplesort(arr, arr+n, std::less<int>(), MPI_INT, numprocs, myid);

  // samplesort should preserve sizes of original input arrays,
  // so can still use all_sizes and displacements
  MPI_Gatherv(arr, n, MPI_INT, ssort_result, all_sizes, displacements,
              MPI_INT, 0, MPI_COMM_WORLD);

  if (myid == 0) {
    std::sort(correct_result, correct_result+total_size);

    for (int i = 0; i < total_size; i++)
      assert(correct_result[i] == ssort_result[i]);
  }

  delete[] all_sizes;
  delete[] correct_result;
  delete[] displacements;
}

int main() {
  int numprocs,
      myid;

  MPI_Init(NULL, NULL);
  MPI_Comm_size(MPI_COMM_WORLD, &numprocs);
  MPI_Comm_rank(MPI_COMM_WORLD, &myid);


  int n = 1000;
  int *arr = new int[n];
  for (int i = 0; i < n; ++i)
    arr[i] = prng(myid*n+i);
  sorttest(arr, n, numprocs, myid);
  delete[] arr;

  MPI_Barrier(MPI_COMM_WORLD);
  if (myid == 0)
    printf("Pass test 1\n");


  n = 1;
  arr = new int[n];
  for (int i = 0; i < n; ++i)
    arr[i] = prng(myid*n+i);
  sorttest(arr, n, numprocs, myid);
  delete[] arr;

  MPI_Barrier(MPI_COMM_WORLD);
  if (myid == 0)
    printf("Pass test 2\n");


  n = 1000 + prng(myid);
  arr = new int[n];
  for (int i = 0; i < n; ++i)
    arr[i] = prng(myid*n+i);
  sorttest(arr, n, numprocs, myid);
  delete[] arr;

  MPI_Barrier(MPI_COMM_WORLD);
  if (myid == 0)
    printf("Pass test 3\n");


  n = 1000;
  arr = new int[n];
  for (int i = 0; i < n; ++i)
    arr[i] = i % 2;
  sorttest(arr, n, numprocs, myid);
  delete[] arr;

  MPI_Barrier(MPI_COMM_WORLD);
  if (myid == 0)
    printf("Pass test 4\n");


  MPI_Finalize();
}
