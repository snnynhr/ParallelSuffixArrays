#ifndef __SSORT__
#define __SSORT__

#include <algorithm>
#include <numeric>
#include <mpi.h>
#include <parallel/algorithm>

// TODO for implementation:
//
// handle case where some local arrays are of size 0.
//   need to change: local splitting
//                   gathering the local splitters
//                   iterating through sorted allsamples

// NOTES:
// Here I make the assumption that the sizes of the input arrays and the bucket
// arrays can be held in a signed int. The reason I do this is because MPI
// calls require counts and displacements to be ints, e.g. see MPI_Alltoallv.
// This unfortunately means that I make the assumption that we don't get some
// bad-case input where the bucket sizes are skewed and some processor gets a
// lot of data at once.

namespace ssort {

int *exclusive_sum(int *arr, size_t n) {
  int *sums = new int[n];
  sums[0] = 0;
  for (size_t i = 1; i < n; ++i) sums[i] = sums[i - 1] + arr[i - 1];
  return sums;
}

// Return the length of the intersection of [l1, r1) and [l2, r2)
size_t interval_overlap(size_t l1, size_t r1, size_t l2, size_t r2) {
  if (l2 < l1) {
    std::swap(l1, l2);
    std::swap(r1, r2);
  }

  if (r1 <= l2)
    return 0;
  else if (r1 >= r2)
    return r2 - l2;
  else
    return r1 - l2;
}

// Return array of p-1 splitter elements, where p is the number of processors.
//
// Assumes that p^2 is a reasonable number of elements to hold and sort on one
// processor
template <typename _Iter, typename _Compare>
void *get_splitters(_Iter begin, _Iter end, _Compare comp,
                    MPI_Datatype mpi_dtype, int numprocs, int myid,
                    MPI_Comm comm) {
  typedef typename std::iterator_traits<_Iter>::value_type value_type;

  const int size = std::distance(begin, end);
  const int sample_size = numprocs - 1;

  // get p-1 local splitters, where p is the number of processors
  value_type *sample = new value_type[sample_size];
  _Iter s_pos = begin;
  const int jump = size / (sample_size + 1);
  const int leftover = size % (sample_size + 1);
  for (int i = 0; i < sample_size; ++i) {
    s_pos += jump + (i < leftover);
    sample[i] = *(s_pos - 1);
  }

  // send local splitters to processor 0
  value_type *all_samples = NULL;
  if (myid == 0) all_samples = new value_type[numprocs * sample_size];

  double ag = MPI::Wtime();
  MPI_Gather(sample, sample_size, mpi_dtype, all_samples, sample_size,
             mpi_dtype, 0, comm);
  MPI_Barrier(comm);
  if (!myid) printf("SAMPLESORT: Gather time %f\n", MPI::Wtime() - ag);

  // get and broadcast p-1 global splitters, placing them in sample array
  if (myid == 0) {
    std::sort(all_samples, all_samples + numprocs * sample_size, comp);

    int as_pos = 0;
    for (int i = 0; i < sample_size; ++i) {
      as_pos += sample_size;
      sample[i] = all_samples[as_pos - 1];
    }
  }
  ag = MPI::Wtime();
  MPI_Bcast(sample, sample_size, mpi_dtype, 0, comm);
  MPI_Barrier(comm);
  if (!myid) printf("SAMPLESORT: Bcast time %f\n", MPI::Wtime() - ag);

  delete[] all_samples;

  return (void *)sample;
}

// Place input data into p buckets and give bucket i to processor i.
// Output is sorted on each processor
// Buckets are not guaranteed to be evenly sized.
template <typename _Iter, typename _Compare>
void *get_buckets(_Iter begin, _Iter end, _Compare comp, int *bucket_size_ptr,
                  MPI_Datatype mpi_dtype, int numprocs, int myid,
                  MPI_Comm comm) {
  typedef typename std::iterator_traits<_Iter>::value_type value_type;
  const int num_splitters = numprocs - 1;

  value_type *splitters = (value_type *)get_splitters(
      begin, end, comp, mpi_dtype, numprocs, myid, comm);

  // split local data into p buckets based on global splitters
  int *send_split_counts = new int[numprocs];
  _Iter s_pos = begin;
  int i = 0;
  while (i < num_splitters) {
    int same_idx = i;
    while (0 < same_idx && same_idx < num_splitters &&
           (!comp(splitters[same_idx - 1], splitters[same_idx]) &&
            !comp(splitters[same_idx], splitters[same_idx - 1])))
      ++same_idx;

    const int num_same = same_idx - i + 1;

    if (num_same == 1) {  // splitters[i] isn't repeated
      _Iter s_pos_next = std::lower_bound(s_pos, end, splitters[i], comp);
      send_split_counts[i] = std::distance(s_pos, s_pos_next);
      s_pos = s_pos_next;
      ++i;
    } else {
      // splitters[i] is repeated. Try to distribute elements evenly
      _Iter s_pos_next = std::upper_bound(s_pos, end, splitters[i], comp);
      const int dist = std::distance(s_pos, s_pos_next);
      const int jump = dist / (num_same - 1);
      const int leftover = dist % (num_same - 1);
      for (int j = 0; j < num_same; ++j)
        send_split_counts[i + j] = jump + (j < leftover);
      s_pos = s_pos_next;
      i = same_idx;
    }
  }
  send_split_counts[num_splitters] = std::distance(s_pos, end);

  // processor i will receive all elements in bucket i across all processors,
  // so send bucket sizes in order to know how much space to allocate
  int *recv_split_counts = new int[numprocs];
  double aa = MPI::Wtime();
  MPI_Alltoall(send_split_counts, 1, MPI_INT, recv_split_counts, 1, MPI_INT,
               comm);
  MPI_Barrier(comm);
  if (!myid) printf("SAMPLESORT: All to all time %f\n", MPI::Wtime() - aa);

  int *send_displacements = exclusive_sum(send_split_counts, numprocs);
  int *recv_displacements = exclusive_sum(recv_split_counts, numprocs);

  *bucket_size_ptr =
      recv_displacements[numprocs - 1] + recv_split_counts[numprocs - 1];
  value_type *bucket_elems = new value_type[*bucket_size_ptr];

  // send bucket elements
  double ag = MPI::Wtime();
  MPI_Alltoallv(begin, send_split_counts, send_displacements, mpi_dtype,
                bucket_elems, recv_split_counts, recv_displacements, mpi_dtype,
                comm);
  MPI_Barrier(comm);
  if (!myid) printf("SAMPLESORT: All to allv time %f\n", MPI::Wtime() - ag);

  // sort bucket elements
  ag = MPI::Wtime();
  std::sort(bucket_elems, bucket_elems + *bucket_size_ptr, comp);
  MPI_Barrier(comm);
  if (!myid) printf("SAMPLESORT: Bucket sort time %f\n", MPI::Wtime() - ag);

  delete[] splitters;
  delete[] send_split_counts;
  delete[] recv_split_counts;
  delete[] send_displacements;
  delete[] recv_displacements;

  return (void *)bucket_elems;
}

// Redistribute bucket elements to original input array (begin to end)
template <typename _Iter>
void redistribute(_Iter begin, _Iter end, void *bucket, int bucket_size,
                  MPI_Datatype mpi_dtype, int numprocs, int myid,
                  MPI_Comm comm) {
  typedef typename std::iterator_traits<_Iter>::value_type value_type;
  value_type *bucket_elems = (value_type *)bucket;

  int local_sizes[2];
  local_sizes[0] = std::distance(begin, end);  // size of original input array
  local_sizes[1] = bucket_size;

  int *all_sizes = new int[2 * numprocs];

  double ag = MPI::Wtime();
  MPI_Allgather(local_sizes, 2, MPI_INT, all_sizes, 2, MPI_INT, comm);
  MPI_Barrier(comm);
  if (!myid) printf("SAMPLESORT: Allgather time %f\n", MPI::Wtime() - ag);

  int *send_counts = new int[numprocs];
  int *recv_counts = new int[numprocs];

  size_t global_my_orig_begin = 0;
  size_t global_my_bucket_begin = 0;
  for (int i = 0; i < myid; ++i) {
    global_my_orig_begin += all_sizes[2 * i];
    global_my_bucket_begin += all_sizes[2 * i + 1];
  }
  size_t global_my_orig_end = global_my_orig_begin + all_sizes[2 * myid];
  size_t global_my_bucket_end =
      global_my_bucket_begin + all_sizes[2 * myid + 1];

  size_t curr_orig_begin = 0;
  size_t curr_bucket_begin = 0;
  for (int i = 0; i < numprocs; ++i) {
    size_t curr_orig_end = curr_orig_begin + all_sizes[2 * i];
    send_counts[i] =
        interval_overlap(curr_orig_begin, curr_orig_end, global_my_bucket_begin,
                         global_my_bucket_end);
    curr_orig_begin = curr_orig_end;

    size_t curr_bucket_end = curr_bucket_begin + all_sizes[2 * i + 1];
    recv_counts[i] = interval_overlap(curr_bucket_begin, curr_bucket_end,
                                      global_my_orig_begin, global_my_orig_end);
    curr_bucket_begin = curr_bucket_end;
  }

  int *send_displacements = exclusive_sum(send_counts, numprocs);
  int *recv_displacements = exclusive_sum(recv_counts, numprocs);

  double aa = MPI::Wtime();
  MPI_Alltoallv(bucket_elems, send_counts, send_displacements, mpi_dtype, begin,
                recv_counts, recv_displacements, mpi_dtype, comm);
  MPI_Barrier(comm);
  if (!myid) printf("SAMPLESORT: Alltoallv time %f\n", MPI::Wtime() - aa);

  delete[] all_sizes;
  delete[] send_counts;
  delete[] recv_counts;
  delete[] send_displacements;
  delete[] recv_displacements;
}

// sort elements across all processors, placing the results back into the input
// array.
template <typename _Iter, typename _Compare>
void samplesort(_Iter begin, _Iter end, _Compare comp, MPI_Datatype mpi_dtype,
                int numprocs, int myid, MPI_Comm comm) {
  // sort locally
  double ag = MPI::Wtime();
  std::sort(begin, end, comp);
  MPI_Barrier(comm);
  if (!myid) printf("SAMPLESORT: LOCAL SORT time %f\n", MPI::Wtime() - ag);

  if (numprocs <= 1) return;

  typedef typename std::iterator_traits<_Iter>::value_type value_type;
  ag = MPI::Wtime();
  int bucket_size;
  value_type *sorted_bucket = (value_type *)get_buckets(
      begin, end, comp, &bucket_size, mpi_dtype, numprocs, myid, comm);
  MPI_Barrier(comm);
  if (!myid) printf("SAMPLESORT: GET BUCKETS time %f\n", MPI::Wtime() - ag);

  // printf("proc %d bucket_size %d\n", myid, bucket_size);
  // printf("Proc %d: bucket holds %d to %d\n", myid, bucket_elems[0],
  // bucket_elems[bucket_size-1]);
  ag = MPI::Wtime();
  redistribute(begin, end, sorted_bucket, bucket_size, mpi_dtype, numprocs,
               myid, comm);
  MPI_Barrier(comm);
  if (!myid) printf("SAMPLESORT: REDISTRIBUTE time %f\n", MPI::Wtime() - ag);

  // printf("Proc %d: redistr holds %d to %d\n", myid, *begin, *(end-1));

  delete[] sorted_bucket;
}

}  // end namespace

#endif
