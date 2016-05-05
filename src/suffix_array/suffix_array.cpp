#include "suffix_array.h"
#include "../sort/ssort.h"

typedef struct dc3_elem {
  uint32_t word;
  uint64_t index;
} dc3_elem;

typedef struct dc3_tuple_elem {
  uint32_t word;
  uint64_t name1;
  uint64_t name2;
  uint64_t index;
} dc3_tuple_elem;

bool compare_dc3_elem(const dc3_elem& lhs, const dc3_elem& rhs) {
  return lhs.word < rhs.word;
}

bool compare_sortedP_elem(const dc3_elem& lhs, const dc3_elem& rhs) {
  return lhs.index < rhs.index;
}

bool compare_P_elem(const dc3_elem& lhs, const dc3_elem& rhs) {
  if ((lhs.index % 3) < (rhs.index % 3)) {
    return 1;
  } else if (lhs.index % 3 == rhs.index % 3) {
    if (lhs.index / 3 < rhs.index / 3) {
      return 1;
    }
  }
  return 0;
}

bool compare_tuple_elem(const dc3_tuple_elem& lhs, const dc3_tuple_elem& rhs) {
  uint32_t l_id = lhs.word >> 16;
  uint32_t r_id = rhs.word >> 16;
  if (l_id != 0 && r_id != 0) {
    return (lhs.name1 < rhs.name1);
  } else if (l_id == 1 || r_id == 1) {
    std::tuple<uint32_t, uint64_t> ll(0xFF & (lhs.word >> 8), lhs.name1);
    std::tuple<uint32_t, uint64_t> rr(0xFF & (rhs.word >> 8), rhs.name1);
    return ll < rr;
  } else if (l_id == 2 || r_id == 2) {
    std::tuple<uint32_t, uint32_t, uint64_t> ll(0xFF & (lhs.word >> 8),
                                                0xFF & (lhs.word), lhs.name2);
    std::tuple<uint32_t, uint32_t, uint64_t> rr(0xFF & (rhs.word >> 8),
                                                0xFF & (rhs.word), rhs.name2);
    return ll < rr;
  } else {
    std::tuple<uint32_t, uint64_t> ll(0xFF & (lhs.word >> 8), lhs.name1);
    std::tuple<uint32_t, uint64_t> rr(0xFF & (rhs.word >> 8), rhs.name1);
    return ll < rr;
  }
}

SuffixArray::SuffixArray() {
  // Initialize datatype for dc3_elem
  int c = 2;
  int lengths[2] = {1, 1};
  MPI_Aint offsets[2] = {offsetof(struct dc3_elem, word),
                         offsetof(struct dc3_elem, index)};
  MPI_Datatype types[2] = {MPI_UNSIGNED, MPI_UNSIGNED_LONG_LONG};

  MPI_Type_struct(c, lengths, offsets, types, &mpi_dc3_elem);
  MPI_Type_commit(&mpi_dc3_elem);

  // Initialize datatype for dc3_tuple_elem
  int _c = 4;
  int _lengths[4] = {1, 1, 1, 1};
  MPI_Aint _offsets[4] = {
      offsetof(dc3_tuple_elem, word), offsetof(dc3_tuple_elem, name1),
      offsetof(dc3_tuple_elem, name2), offsetof(dc3_tuple_elem, index)};
  MPI_Datatype _types[4] = {MPI_UNSIGNED, MPI_UNSIGNED_LONG_LONG,
                            MPI_UNSIGNED_LONG_LONG, MPI_UNSIGNED_LONG_LONG};

  MPI_Type_struct(_c, _lengths, _offsets, _types, &mpi_dc3_tuple_elem);
  MPI_Type_commit(&mpi_dc3_tuple_elem);
};

int32_t SuffixArray::build(const char* data, uint32_t size, uint64_t file_size,
                           uint64_t offset, int numprocs, int myid,
                           uint64_t* suffix_array) {
  // If N is small, switch to single thread.

  // If N is med, switch to single core.

  /*
   *  @TODO: Component 1:
   *  S = <(T[i,i+2], i) : i \in [0,n), i mod 3 \not= 0>
   */
  uint32_t sm = (3 - (offset % 3)) % 3;
  const uint64_t dc3_elem_array_size =
      sm + ((size - sm - 1) / 3) * 2 + ((size - sm - 1) % 3);
  dc3_elem* S = new dc3_elem[dc3_elem_array_size]();
  if (S == NULL) {
    return -1;
  }

  // Construct 'S' array
  uint64_t pos = offset;
  uint64_t count = 0;
  while (count < dc3_elem_array_size) {
    if (pos % 3 != 0) {
      // Watch unsigned / signed
      // Read 3 chars
      uint32_t word = data[pos - offset];
      word = (word << 8) + data[pos - offset + 1];
      word = (word << 8) + data[pos - offset + 2];
      S[count].word = word;
      S[count].index = pos;
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

  // To check if not equal to previous, first element needs last element of
  // previous process.
  if (myid != numprocs - 1) {
    MPI_Send(S + (dc3_elem_array_size - 1), 1, mpi_dc3_elem, myid + 1, 0,
             MPI_COMM_WORLD);
  }

  dc3_elem start;
  if (myid != 0) {
    MPI_Recv(&start, 1, mpi_dc3_elem, myid - 1, 0, MPI_COMM_WORLD,
             MPI_STATUS_IGNORE);
  }

  uint64_t* is_diff_from_adj = new uint64_t[dc3_elem_array_size]();
  if (is_diff_from_adj == NULL) {
    return -1;
  }
  // Check first element.
  if (myid == 0) {
    // First thread, different by definition
    is_diff_from_adj[0] = 1;
  } else {
    // Check from thread myid - 1 (p2p comm)
    is_diff_from_adj[0] = (S[0].word != start.word);
  }

  // Local scanned is_diff to calculates names.
  for (uint64_t i = 1; i < dc3_elem_array_size; i++) {
    is_diff_from_adj[i] = static_cast<uint64_t>((S[i].word != S[i - 1].word)) +
                          is_diff_from_adj[i - 1];
  }

  MPI_Barrier(MPI_COMM_WORLD);  // test only

  uint64_t prefix_sum = 0;
  MPI_Exscan(&is_diff_from_adj[dc3_elem_array_size - 1], &prefix_sum, 1,
             MPI_UNSIGNED_LONG_LONG, MPI_SUM, MPI_COMM_WORLD);

  // Update names with prefix sum
  uint64_t* names = is_diff_from_adj;
  for (uint64_t i = 1; i < dc3_elem_array_size; i++) {
    names[i] += prefix_sum;
  }

  /*
   *  @TODO: Component 4:
   *  If names are not unique:
   *  Permute (r,i) in P such that they are sorted by (i mod 3, i div 3)
   *  SA^{12} = pDC3(<c : (c,i) in P>)
   *  P := <(j+1, mapBack(SA^{12}[j], n/3)) : j < 2n/3>
   */
  uint64_t total = 0;
  if (myid == numprocs - 1) {
    total = names[dc3_elem_array_size - 1];
    if (total == ((file_size - 1) / 3) * 2 + ((file_size - 1) % 3)) {
      total = 1;
      MPI_Bcast(&total, 1, MPI_UNSIGNED_LONG_LONG, numprocs - 1,
                MPI_COMM_WORLD);
    } else {
      total = 0;
      MPI_Bcast(&total, 1, MPI_UNSIGNED_LONG_LONG, numprocs - 1,
                MPI_COMM_WORLD);
    }
  } else {
    MPI_Bcast(&total, 1, MPI_UNSIGNED_LONG_LONG, numprocs - 1,
    MPI_COMM_WORLD);
  }

  // // @TODO: Fix since word can be 64 bit.
  // // Generate P array.
  // dc3_elem P[dc3_elem_array_size];
  // for (uint64_t i = 0; i < dc3_elem_array_size; i++) {
  //   P[i].word = names[i];
  //   P[i].index = S[i].index;
  // }

  // // Not unique
  // if (total == 0) {
  //   // Permute.
  //   ssort::samplesort(P, P + dc3_elem_array_size, compare_P_elem,
  //   mpi_dc3_elem,
  //                     numprocs, myid);
  //   for (uint64_t i = 0; i < dc3_elem_array_size; i++) {
  //     // reuse stack memory;
  //     names[i] = P[i].word;
  //   }

  //   // @TODO: Local compute: need to expand to distributed
  //   // sais_int()
  // }

  // // Sort by second element.
  // ssort::samplesort(P, P + dc3_elem_array_size, compare_sortedP_elem,
  //                   mpi_dc3_elem, numprocs, myid);

  // if (myid != 0) {
  //   MPI_Send(P, 1, mpi_dc3_elem, myid - 1, 1, MPI_COMM_WORLD);
  // }

  // dc3_elem next;
  // if (myid != numprocs - 1) {
  //   MPI_Recv(&next, 1, mpi_dc3_elem, myid - 1, 1, MPI_COMM_WORLD,
  //            MPI_STATUS_IGNORE);
  // }

  // /*
  //  *  @TODO: Component 5:
  //  *  S_0 := <(T[i], T[i+1], c', c'', i) : i mod 3 = 0), (c',i+1), (c'', i+2)
  //  in
  //  * P>
  //  */
  // /*
  //  *  @TODO: Component 6:
  //  *  S_1 := <(c, T[i], c', i) : i mod 3 = 1), (c,i), (c', i+1) in P>
  //  */
  // /*
  //  *  @TODO: Component 7:
  //  *  S_2 := <(c, T[i], T[i+1], c'', i) : i mod 3 = 2), (c,i), (c'', i+2) in
  //  P>
  //  */

  // dc3_tuple_elem SS[size];
  // for (uint64_t i = 0; i < size; i++) {
  //   uint32_t d = i % 3;
  //   uint32_t word = d;
  //   word <<= 10;
  //   word = word + (static_cast<uint32_t>(data[i]) & 0xFF);
  //   word <<= 8;
  //   word = word + (static_cast<uint32_t>(data[i + 1]) & 0xFF);
  //   SS[i].word = word;
  //   SS[i].index = i;
  // }

  // for (uint64_t i = 0; i < dc3_elem_array_size; i++) {
  //   uint64_t curr = P[i].index;
  //   if (curr % 3 == 1) {
  //     SS[curr - 1].name1 = P[i].word;
  //     SS[curr].name1 = P[i].word;
  //     if (curr - 2 > 0) {
  //       SS[curr - 2].name2 = P[i].word;
  //     }
  //   } else {
  //     SS[curr - 2].name2 = P[i].word;
  //     SS[curr - 1].name2 = P[i].word;
  //     SS[curr].name1 = P[i].word;
  //   }
  // }

  // uint64_t w = next.index;
  // if (w % 3 == 1) {
  //   SS[w - 1].name1 = next.word;
  //   SS[w - 2].name2 = next.word;
  // } else {
  //   SS[w - 2].name2 = next.word;
  //   SS[w - 1].name2 = next.word;
  // }

  // /*
  //  *  @TODO: Component 8:
  //  *  Sort S_0 union S_1 union S_2 using compare operator in paper.
  //  */

  // ssort::samplesort(SS, SS + size, compare_tuple_elem, mpi_dc3_tuple_elem,
  //                   numprocs, myid);

  // /*
  //  *  @TODO: Component 9:
  //  *  Return last component of (s : s in S).
  //  */
  // for (uint64_t i = 0; i < size; i++) {
  //   suffix_array[i] = SS[i].index;
  // }
  return 0;
}