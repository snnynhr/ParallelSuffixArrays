#include <algorithm>
#include <numeric>
#include <mpi.h>

#include "assert.h"

// TODO for implementation:
//
// handle case where some local arrays are of size 0.
//   need to change: local splitting
//                   gathering the local splitters
//                   iterating through sorted allsamples
//
// handle case where many of the global splitters are equal
//   (the code should still work, but it definitely won't distribute the data
//    evenly; some node will certainly not recieve any data)
//
// when sorting bucket_elems, it could be more efficient to do a p-way merge
// instead of using std::sort
//
// remove asserts + assert.h

// TODO for testing:
//   - small inputs
//   - inputs with many equal elements

// NOTES:
// Here I assume that the size of the input array (i.e. std::distance(begin, end))
// and the size of the buckets after splitting can be held in an int.
// The reason int is used instead of size_t is that sizes need to be passed
// using MPI, and there doesn't seem to be an MPI_Datatype for size_t.
// Moreover, MPI_Alltoall and MPI_Alltoallv require counts/displacements to be
// ints.
// Unfortunately, this means that this function must assume that it doesn't
// receieve some worst-case input where the buckets are distributed extremely
// unevenly.


static int *exclusive_sum(int *arr, size_t n) {
  int *sums = new int[n];
  sums[0] = 0;
  for (size_t i = 1; i < n; i++)
    sums[i] = sums[i-1] + arr[i-1];
  return sums;
}

// return array of p-1 splitter elements, where p is the number of processors.
// 
// assumes that p^2 is a reasonable number of elements to hold and sort on one
// processor
template <typename _RandomAccessIter, typename _Compare>
static void *get_splitters(_RandomAccessIter begin, _RandomAccessIter end,
                           _Compare comp, MPI_Datatype mpi_dtype,
                           int numprocs, int myid) {
  typedef typename std::iterator_traits<_RandomAccessIter>::value_type
           value_type;

  const unsigned size = std::distance(begin, end);
  const unsigned sample_size = numprocs - 1;

  // get p-1 local splitters, where p is the number of processors
  value_type *sample = new value_type[sample_size];
  _RandomAccessIter s_pos = begin;
  const unsigned jump = size / (sample_size+1);
  const unsigned leftover = size % (sample_size+1);
  for (unsigned i = 0; i < sample_size; ++i) {
    s_pos += jump + (i < leftover);
    assert(begin <= s_pos-1 && s_pos-1 < end);
    sample[i] = *(s_pos-1);
  }

  // send local splitters to processor 0
  value_type *all_samples = NULL;
  if (myid == 0)
    all_samples = new value_type[numprocs*sample_size];
  MPI_Gather(sample, sample_size, mpi_dtype,
             all_samples, sample_size, mpi_dtype, 0, MPI_COMM_WORLD);

  // get and broadcast p-1 global splitters, placing them in sample array
  if (myid == 0) {
    std::sort(all_samples, all_samples + numprocs*sample_size, comp);

    unsigned as_pos = 0;
    for (unsigned i = 0; i < sample_size; ++i) {
      as_pos += sample_size;
      assert(0 <= as_pos-1 && as_pos-1 < numprocs*sample_size);
      sample[i] = all_samples[as_pos-1];
    }

    printf("Splitters:");
    for (unsigned i = 0; i < sample_size; ++i)
      printf(" %d: %d,", i, sample[i]);
    printf("\n");
  }

  MPI_Bcast(sample, sample_size, mpi_dtype, 0, MPI_COMM_WORLD);

  delete[] all_samples;

  return (void *)sample;
}

// sort elements across all processors, placing the results back into the input
// array.
template <typename _RandomAccessIter, typename _Compare>
void samplesort(_RandomAccessIter begin, _RandomAccessIter end, _Compare comp,
                MPI_Datatype mpi_dtype, int numprocs, int myid) {
  // sort locally
  std::sort(begin, end, comp);

  if (numprocs <= 1)
    return;

  typedef typename std::iterator_traits<_RandomAccessIter>::value_type
           value_type;

  const int num_splitters = numprocs-1;
  value_type *splitters =
    (value_type *)get_splitters(begin, end, comp, mpi_dtype, numprocs, myid);

  // split local data into p buckets based on global splitters
  int *split_counts_send = new int[numprocs];
  _RandomAccessIter s_pos = begin;
  int i = 0;
  while (i < num_splitters) {
    _RandomAccessIter s_pos_next = std::lower_bound(s_pos, end, splitters[i], comp);
    split_counts_send[i] = std::distance(s_pos, s_pos_next);
    s_pos = s_pos_next;
    ++i;
  }
  split_counts_send[num_splitters] = std::distance(s_pos, end);

  // processor i will receive all elements in bucket i across all processors,
  // so send bucket sizes in order to know how much space to allocate
  int *split_counts_recv = new int[numprocs];
  MPI_Alltoall(split_counts_send, 1, MPI_INT,
               split_counts_recv, 1, MPI_INT, MPI_COMM_WORLD);

  int *displacements_send = exclusive_sum(split_counts_send, numprocs); 
  int *displacements_recv = exclusive_sum(split_counts_recv, numprocs); 

  int bucket_size =
    displacements_recv[numprocs-1] + split_counts_recv[numprocs-1];
  value_type *bucket_elems = new value_type[bucket_size];

  // send bucket elements
  MPI_Alltoallv(begin, split_counts_send, displacements_send, mpi_dtype,
                bucket_elems, split_counts_recv, displacements_recv, mpi_dtype,
                MPI_COMM_WORLD);

  // sort bucket elements
  std::sort(bucket_elems, bucket_elems + bucket_size, comp);

  printf("Proc %d: bucket holds %d to %d\n", myid, bucket_elems[0], bucket_elems[bucket_size-1]);

  delete[] splitters;
  delete[] split_counts_send;
  delete[] split_counts_recv;
  delete[] displacements_send;
  delete[] displacements_recv;
  delete[] bucket_elems;
}
