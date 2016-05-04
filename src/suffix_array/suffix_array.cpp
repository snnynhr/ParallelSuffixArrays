#include "suffix_array.h"
#include "../sort/ssort.h"

typedef struct dc3_elem {
  uint32_t word;
  uint64_t index;
} dc3_elem;

bool compare_dc3_elem(const dc3_elem lhs, const dc3_elem rhs) {
  return lhs.word < rhs.word;
}

bool compare_P_elem(const dc3_elem lhs, const dc3_elem rhs) {
  if ((lhs.index % 3) < (rhs.index % 3)) {
    return 1;
  } else {
    if (lhs.index / 3 < rhs.index / 3) {
      return 1;
    }
  }
  return 0;
}

SuffixArray::SuffixArray() {
  // Initialize datatype for dc3_elem
  int c = 2;
  int lengths[2] = {4, 8};
  MPI_Aint offsets[2] = {0, sizeof(uint32_t)};
  MPI_Datatype types[2] = {MPI_UNSIGNED, MPI_UNSIGNED_LONG_LONG};

  MPI_Type_struct(c, lengths, offsets, types, &mpi_dc3_elem);
  MPI_Type_commit(&mpi_dc3_elem);
};

void SuffixArray::build(const char* data, uint32_t size, int numprocs,
                        int myid) {
  // If N is small, switch to single thread.

  // If N is med, switch to single core.

  /*
   *  @TODO: Component 1:
   *  S = <(T[i,i+2], i) : i \in [0,n), i mod 3 \not= 0>
   */
  const uint64_t dc3_elem_array_size = ((size - 1) / 3) * 2 + ((size - 1) % 3);
  dc3_elem S[dc3_elem_array_size] = {0};

  // Construct 'S' array
  uint64_t pos = 0;
  uint64_t count = 0;
  while (count < dc3_elem_array_size) {
    if (pos % 3 != 0) {
      // Watch unsigned / signed
      // Read 3 chars
      uint32_t word = data[pos];
      word = (word << 8) + data[pos + 1];
      word = (word << 8) + data[pos + 2];
      S[count].word = word;
      count++;
    }
    pos++;
  }

  /*
   *  @TODO: Component 2:
   *  sort S by first component
   */
  ssort::samplesort(S, S + dc3_elem_array_size, compare_dc3_elem, mpi_dc3_elem,
                    numprocs, myid);

  /*
   *  @TODO: Component 3:
   *  P := name (S)
   */
  if (myid != numprocs - 1) {
    MPI_Send(S + (dc3_elem_array_size - 1), 1, mpi_dc3_elem, myid + 1, 0,
             MPI_COMM_WORLD);
  }

  dc3_elem start;
  if (myid != 0) {
    MPI_Recv(&start, 1, mpi_dc3_elem, myid - 1, 0, MPI_COMM_WORLD,
             MPI_STATUS_IGNORE);
  }

  uint64_t is_diff_from_adj[dc3_elem_array_size] = {0};

  // Check first element.
  if (myid == 0) {
    // First thread, different by definition
    is_diff_from_adj[0] = 1;
  } else {
    // Check from thread myid - 1 (p2p comm)
    is_diff_from_adj[0] = (S[0].word != start.word);
  }

  for (uint64_t i = 1; i < dc3_elem_array_size; i++) {
    is_diff_from_adj[i] = static_cast<uint64_t>((S[i].word != S[i - 1].word));
  }

  uint64_t names[size] = {0};
  // Barrier?
  MPI_Scan(is_diff_from_adj, names, size, MPI_UNSIGNED_LONG_LONG, MPI_SUM,
           MPI_COMM_WORLD);
  // Barrier?

  /*
   *  @TODO: Component 4:
   *  If names are not unique:
   *  Permute (r,i) in P such that they are sorted by (i mod 3, i div 3)
   *  SA^{12} = pDC3(<c : (c,i) in P>)
   *  P := <(j+1, mapBack(SA^{12}[j], n/3)) : j < 2n/3>
   */
  uint64_t total = 0;
  if (myid == numprocs - 1) {
    total = names[size - 1];
    if (total == dc3_elem_array_size) {
      MPI_Bcast(&total, 1, MPI_UNSIGNED_LONG_LONG, myid, MPI_COMM_WORLD);
    } else {
      MPI_Bcast(&total, 0, MPI_UNSIGNED_LONG_LONG, myid, MPI_COMM_WORLD);
    }
  } else {
    MPI_Bcast(&total, 0, MPI_UNSIGNED_LONG_LONG, myid, MPI_COMM_WORLD);
  }

  // @TODO: Fix since word can be 64 bit.
  // Generate P array.
  dc3_elem P[dc3_elem_array_size];
  for (uint64_t i = 0; i < dc3_elem_array_size; i++) {
    P[i].word = names[i];
    P[i].index = S[i].index;
  }

  // Not unique
  if (total == 0) {
    // Permute.
    ssort::samplesort(P, P + dc3_elem_array_size, compare_P_elem, mpi_dc3_elem, numprocs, id);
    for(uint64_t i = 0; i < dc3_elem_array_size; i++) {
      // reuse stack memory;
      names[i] = P[i].word;
    }

    // @TODO: Local compute: need to expand to distributed
    //sais_int()
  }

  /*
   *  @TODO: Component 5:
   *  S_0 := <(T[i], T[i+1], c', c'', i) : i mod 3 = 0), (c',i+1), (c'', i+2) in
   * P>
   */

  /*
   *  @TODO: Component 6:
   *  S_1 := <(c, T[i], c', i) : i mod 3 = 1), (c,i), (c', i+1) in P>
   */

  /*
   *  @TODO: Component 7:
   *  S_2 := <(c, T[i], T[i+1], c'', i) : i mod 3 = 2), (c,i), (c'', i+2) in P>
   */

  /*
   *  @TODO: Component 8:
   *  Sort S_0 union S_1 union S_2 using compare operator in paper.
   */

  /*
   *  @TODO: Component 9:
   *  Return last component of (s : s in S).
   */
}