#include "suffix_array.h"
#include "../sort/ssort.h"
#include "../sais/sais.h"

/*
 * My SSM algorithm.
 */

struct compare_radix_css_elem : std::binary_function<css_elem, css_elem, bool> {
  compare_radix_css_elem(const char* data, const uint64_t size)
      : _data(data), _size(size) {}
  bool operator()(const css_elem& lhs, const css_elem& rhs) {
    uint64_t lindex = lhs.index;
    uint64_t rindex = rhs.index;

    uint64_t length = _size - std::max(lindex, rindex) - 8;

    for (uint64_t i = 0; i < length; i++) {
      if (_data[i + 8 + lindex] != _data[i + 8 + rindex]) {
        return _data[i + 8 + lindex] < _data[i + 8 + rindex];
      }
    }
    return lindex < rindex;
  }
  const char* _data;
  const uint64_t _size;
};

bool compare_css_elem(const css_elem& lhs, const css_elem& rhs) {
  return lhs.word < rhs.word;
}

SuffixArray::SuffixArray() {
  // Initialize datatype for css_elem
  int c = 2;
  int lengths[2] = {1, 1};
  MPI_Aint offsets[2] = {offsetof(struct css_elem, word),
                         offsetof(struct css_elem, index)};
  MPI_Datatype types[2] = {MPI_UNSIGNED_LONG_LONG, MPI_UNSIGNED_LONG_LONG};

  MPI_Type_struct(c, lengths, offsets, types, &mpi_css_elem);
  MPI_Type_commit(&mpi_css_elem);
};

int32_t SuffixArray::build(const char* data, uint32_t size,
                           uint64_t offset, int numprocs, int myid,
                           uint64_t* suffix_array, MPI_Comm comm) {
  MPI_Comm_size(comm, &numprocs);
  MPI_Comm_rank(comm, &myid);
  // If N is small, switch to single thread.

  // If N is med, switch to single core.

  // Set globals
  _size = size;
  _data = data;
  const char* node_data = data + offset;

  /*
   *  Component 1:
   *  S = <(T[i,i+7], i) : i \in [0,n)>
   */
  double elapsed = 0;
  if (!myid) {
    fprintf(stdout, "Building component 1\n");
    elapsed = MPI::Wtime();
  }

  css_elem* S = new css_elem[size]();
  if (S == NULL) {
    return -1;
  }

  // Construct 'S' array
  // S stores: [data[pos, pos+7], index].
  uint64_t word = 0;
  for (int i = 0; i < 7; i++) {
    uint64_t elem = static_cast<uint64_t>(node_data[i]);
    word = (word << 8) + elem;
  }
  for (uint64_t pos = 0; pos < size; pos++) {
    // Watch unsigned / signed
    // Read 8 chars

    uint64_t elem = static_cast<uint64_t>(node_data[pos + 7]);
    word = (word << 8) + elem;
    S[pos].word = word;
    S[pos].index = pos + offset;
  }

  /*
   *  Component 2:
   *  Sort S by first component.
   */
  MPI_Barrier(comm);
  if (!myid) {
    fprintf(stdout, "Runtime of component 1: %f\n\n", MPI::Wtime() - elapsed);
    elapsed = MPI::Wtime();
    fprintf(stdout, "Building component 2\n");
  }
  ssort::samplesort(S, S + size, compare_css_elem, mpi_css_elem, numprocs, myid,
                    comm);

  /*
   *  Component 3:
   *  P := name (S)
   */
  MPI_Barrier(comm);
  if (!myid) {
    fprintf(stdout, "Runtime of component 2: %f\n\n", MPI::Wtime() - elapsed);
    elapsed = MPI::Wtime();
    fprintf(stdout, "Building component 3\n");
  }

  // First we calculate a boolean if the curr string is not equal to next.
  // To check if not equal to next, last element of this process needs the
  // first element of next process. So we use SEND / RECV.
  if (myid != 0) {
    MPI_Send(S, 1, mpi_css_elem, myid - 1, 0, comm);
  }

  css_elem next;
  if (myid != numprocs - 1) {
    MPI_Recv(&next, 1, mpi_css_elem, myid + 1, 0, comm, MPI_STATUS_IGNORE);
  }

  bool* is_diff_from_adj = new bool[size]();
  if (is_diff_from_adj == NULL) {
    return -1;
  }

  // Calculate boolean for first element.
  if (myid == numprocs - 1) {
    // First processor has null predecessor, different by definition.
    is_diff_from_adj[size - 1] = 0;
  } else {
    // Check from process id myid - 1 (sent using MPI_Send).
    is_diff_from_adj[size - 1] = (S[size - 1].word == next.word);
  }

  const uint32_t NUM_THREADS = 12;
  const uint64_t thread_size = size / NUM_THREADS;
  //#pragma omp parallel for schedule(static) num_threads(NUM_THREADS)
  for (uint32_t index = 0; index < NUM_THREADS; index++) {
    // Each is different thread.
    uint64_t start = thread_size * index;
    uint64_t end = thread_size * (index + 1);
    if (index == NUM_THREADS - 1) {
      end = size - 1;
    }

    for (uint64_t j = start; j < end; j++) {
      is_diff_from_adj[j] = (S[j].word == S[j + 1].word);
    }

    uint64_t pos = start;
    if (index != 0) {
      while (pos < size && is_diff_from_adj[pos - 1] == 1) {
        pos++;
      }
    }

    if (index == NUM_THREADS - 1) {
      end++;
    }

    if (pos != size) {
      bool is_seq = false;
      uint64_t start_pos = -1;
      for (uint64_t j = pos; j < end; j++) {
        if (!is_seq) {
          // Not a current successive sequence.
          if (is_diff_from_adj[j] == 1) {
            start_pos = j;
            is_seq = true;
          }
          // else do nothing, keep moving
        } else {
          if (is_diff_from_adj[j] == 0) {
            // end of seq, local sort start_pos to i, inclusive.
            // naive comparator
            uint64_t local_size = j - start_pos + 1;
            std::sort(S + start_pos, S + start_pos + local_size,
                      compare_radix_css_elem(data, size));

            is_seq = false;
            start_pos = -1;
          }
          // else do nothing, keep moving
        }
      }
    }

    if (index != 0) {
      if (is_diff_from_adj[start - 1] == 1) {
        uint64_t pos_start = start - 1;
        uint64_t pos_end = start;
        while (pos_end < size && is_diff_from_adj[pos_end] == 1) {
          pos_end++;
        }
        while (pos_start >= 1 && is_diff_from_adj[pos_start - 1] == 1) {
          pos_start--;
        }
        std::sort(S + pos_start, S + pos_end + 1,
                  compare_radix_css_elem(data, size));
      }
    }
  }

  /*
   *  Component 7:
   *  Return last component of (s : s in S).
   */
  MPI_Barrier(comm);
  if (!myid) {
    fprintf(stdout, "Runtime of component 3: %f\n\n", MPI::Wtime() - elapsed);
    elapsed = MPI::Wtime();
    fprintf(stdout, "Building component 4\n");
  }

  for (uint64_t i = 0; i < size; i++) {
    suffix_array[i] = S[i].index;
  }

  MPI_Barrier(comm);
  if (!myid) {
    fprintf(stdout, "Runtime of component 4: %f\n\n", MPI::Wtime() - elapsed);
  }
  return 0;
}
