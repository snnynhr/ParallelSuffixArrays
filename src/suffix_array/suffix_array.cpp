#include "suffix_array.h"
#include "../sort/ssort.h"
#include "../sais/sais.h"

typedef struct dc3_elem {
  uint32_t word;
  uint32_t index;
} dc3_elem;

typedef struct dc3_tuple_elem {
  uint32_t word;
  uint32_t name1;
  uint32_t name2;
  uint32_t index;
} dc3_tuple_elem;

bool compare_dc3_elem(const dc3_elem& lhs, const dc3_elem& rhs) {
  return lhs.word < rhs.word;
}

bool compare_sortedP_elem(const dc3_elem& lhs, const dc3_elem& rhs) {
  return lhs.index < rhs.index;
}

bool compare_P_elem(const dc3_elem& lhs, const dc3_elem& rhs) {
  return (lhs.index % 3 < rhs.index % 3) 
      || ((lhs.index % 3 == rhs.index % 3) && (lhs.index / 3 < rhs.index / 3));
}

// returns (a1, b1) < (a2, b2)
inline bool tuple2_comp(uint32_t a1, uint32_t b1, uint32_t a2, uint32_t b2) {
  return (a1 < a2) 
      || ((a1 == a2) && (b1 < b2));
}

inline bool tuple3_comp(uint32_t a1, uint32_t b1, uint32_t c1,
                       uint32_t a2, uint32_t b2, uint32_t c2) {
  return (a1 < a2) 
      || ((a1 == a2) && (b1 < b2))
      || ((a1 == a2) && (b1 == b2) && (c1 < c2));
}

bool compare_tuple_elem(const dc3_tuple_elem& lhs, const dc3_tuple_elem& rhs) {
  // Get which array (S_0,S_1,S_2) which corresponds to element.
  uint32_t l_id = lhs.word >> 16;
  uint32_t r_id = rhs.word >> 16;

  // If both are not 0:
  if (l_id != 0 && r_id != 0) {
    return (lhs.name1 < rhs.name1);
  } else if (l_id == 1) { // at least 1 is 1:
      return tuple2_comp(0xFF & (lhs.word >> 8), lhs.name2, 0xFF & (rhs.word >> 8), rhs.name1);
  } else if (r_id == 1) { // at least 1 is 1:
      return tuple2_comp(0xFF & (lhs.word >> 8), lhs.name1, 0xFF & (rhs.word >> 8), rhs.name2);
  } else if (l_id == 2 || r_id == 2) {
    // At least 1 is 2:
    return tuple3_comp (0xFF & (lhs.word >> 8), 0xFF & (lhs.word), lhs.name2,
                        0xFF & (rhs.word >> 8), 0xFF & (rhs.word), rhs.name2);
  } else {
    // Both are 0:
    return tuple2_comp(0xFF & (lhs.word >> 8), lhs.name1,
                       0xFF & (rhs.word >> 8), rhs.name1);
  }
}

inline uint64_t map_back(uint64_t i, uint64_t s) {
  return i < s ? 3 * i + 1 : 3 * (i - s) + 2;
}

SuffixArray::SuffixArray() {
  // Initialize datatype for dc3_elem
  int c = 2;
  int lengths[2] = {1, 1};
  MPI_Aint offsets[2] = {offsetof(struct dc3_elem, word),
                         offsetof(struct dc3_elem, index)};
  MPI_Datatype types[2] = {MPI_UNSIGNED, MPI_UNSIGNED};

  MPI_Type_struct(c, lengths, offsets, types, &mpi_dc3_elem);
  MPI_Type_commit(&mpi_dc3_elem);

  // Initialize datatype for dc3_tuple_elem
  int _c = 4;
  int _lengths[4] = {1, 1, 1, 1};
  MPI_Aint _offsets[4] = {
      offsetof(dc3_tuple_elem, word), offsetof(dc3_tuple_elem, name1),
      offsetof(dc3_tuple_elem, name2), offsetof(dc3_tuple_elem, index)};
  MPI_Datatype _types[4] = {MPI_UNSIGNED, MPI_UNSIGNED,
                            MPI_UNSIGNED, MPI_UNSIGNED};

  MPI_Type_struct(_c, _lengths, _offsets, _types, &mpi_dc3_tuple_elem);
  MPI_Type_commit(&mpi_dc3_tuple_elem);
};

int32_t SuffixArray::build(const char* data, uint32_t size, uint32_t file_size,
                           uint32_t offset, int numprocs, int myid,
                           uint32_t* suffix_array) {
  // If N is small, switch to single thread.

  // If N is med, switch to single core.

  /*
   *  Component 1:
   *  S = <(T[i,i+2], i) : i \in [0,n), i mod 3 \not= 0>
   */
  double elapsed = 0;
  if (!myid) {
    fprintf(stdout, "Building component 1\n");
    elapsed = MPI::Wtime();
  }
  // We need to calculate the number of positions which are not 0 mod 3.
  const uint32_t sm = (3 - (offset % 3)) % 3;
  const uint32_t dc3_elem_array_size =
      sm + ((size - sm - 1) / 3) * 2 + ((size - sm - 1) % 3);
  dc3_elem* S = new dc3_elem[dc3_elem_array_size]();
  if (S == NULL) {
    return -1;
  }

  // Construct 'S' array
  // S stores: [data[pos, pos+2], index].
  uint32_t pos = offset;
  uint32_t count = 0;
  while (count < dc3_elem_array_size) {
    if (pos % 3 != 0) {
      // Watch unsigned / signed
      // Read 3 chars
      uint32_t word = static_cast<uint32_t>(data[pos - offset]);
      word = (word << 8) + static_cast<uint32_t>(data[pos - offset + 1]);
      word = (word << 8) + static_cast<uint32_t>(data[pos - offset + 2]);
      S[count].word = word;
      S[count].index = pos;
      count++;
    }
    pos++;
  }

  /*
   *  Component 2:
   *  Sort S by first component.
   */
  MPI_Barrier(MPI_COMM_WORLD);
  if (!myid) {
    fprintf(stdout, "Runtime of component 1: %f\n\n", MPI::Wtime() - elapsed);
    elapsed = MPI::Wtime();
    fprintf(stdout, "Building component 2\n");
  }
  ssort::samplesort(S, S + dc3_elem_array_size, compare_dc3_elem, mpi_dc3_elem,
                    numprocs, myid);

  /*
   *  Component 3:
   *  P := name (S)
   */
  MPI_Barrier(MPI_COMM_WORLD);
  if (!myid) {
    fprintf(stdout, "Runtime of component 2: %f\n\n", MPI::Wtime() - elapsed);
    elapsed = MPI::Wtime();
    fprintf(stdout, "Building component 3\n");
  }

  // First we calculate a boolean if the curr string is not equal to previous.
  // To check if not equal to previous, first element of this process needs the
  // last element of previous process. So we use SEND / RECV.
  if (myid != numprocs - 1) {
    MPI_Send(S + (dc3_elem_array_size - 1), 1, mpi_dc3_elem, myid + 1, 0,
             MPI_COMM_WORLD);
  }

  dc3_elem start;
  if (myid != 0) {
    MPI_Recv(&start, 1, mpi_dc3_elem, myid - 1, 0, MPI_COMM_WORLD,
             MPI_STATUS_IGNORE);
  }

  uint32_t* is_diff_from_adj = new uint32_t[dc3_elem_array_size]();
  if (is_diff_from_adj == NULL) {
    return -1;
  }

  // Calculate boolean for first element.
  if (myid == 0) {
    // First processor has null predecessor, different by definition.
    is_diff_from_adj[0] = 1;
  } else {
    // Check from process id myid - 1 (sent using MPI_Send).
    is_diff_from_adj[0] = (S[0].word != start.word);
  }

  // Calculate booleans for rest and simultaenously update prefix sum.
  for (uint32_t i = 1; i < dc3_elem_array_size; i++) {
    is_diff_from_adj[i] = static_cast<uint32_t>((S[i].word != S[i - 1].word)) +
                          is_diff_from_adj[i - 1];
  }

  MPI_Barrier(MPI_COMM_WORLD);  // test only

  // We want to produce a scan over the is_diff array across all processors.
  // The sum of each individual array is is_diff[_size - 1]. We do an
  // exclusive scan on this to propagate the partial sums.
  uint32_t prefix_sum = 0;
  MPI_Exscan(&is_diff_from_adj[dc3_elem_array_size - 1], &prefix_sum, 1,
             MPI_UNSIGNED, MPI_SUM, MPI_COMM_WORLD);

  // Update names with prefix sum.
  uint32_t* names = is_diff_from_adj;
  for (uint32_t i = 0; i < dc3_elem_array_size; i++) {
    names[i] += prefix_sum;
  }

  /*
   *  Component 4:
   *  If names are not unique:
   *  Permute (r,i) in P such that they are sorted by (i mod 3, i div 3)
   *  SA^{12} = pDC3(<c : (c,i) in P>)
   *  P := <(j+1, mapBack(SA^{12}[j], n/3)) : j < 2n/3>
   */
  MPI_Barrier(MPI_COMM_WORLD);
  if (!myid) {
    fprintf(stdout, "Runtime of component 3: %f\n\n", MPI::Wtime() - elapsed);
    elapsed = MPI::Wtime();
    fprintf(stdout, "Building component 4\n");
  }

  uint32_t total = 0;
  uint32_t is_unique = 0;
  if (myid == numprocs - 1) {
    // To check if all lexicographical prefixes are unique, we check that the
    // last entry of the prefix sum in the last process (sum) is equal to the
    // number of total elements not 0 mod 3. We calculate this using the
    // filesize.
    total = names[dc3_elem_array_size - 1];
    is_unique = (total == ((file_size - 1) / 3) * 2 + ((file_size - 1) % 3));
  }
  MPI_Bcast(&is_unique, 1, MPI_UNSIGNED, numprocs - 1, MPI_COMM_WORLD);

  // @TODO: We can probably reuse S.
  // Generate P array. This stores [name, index].
  dc3_elem* P = new dc3_elem[dc3_elem_array_size];
  for (uint32_t i = 0; i < dc3_elem_array_size; i++) {
    P[i].word = names[i];
    P[i].index = S[i].index;
  }

  // Not unique
  if (!is_unique) {
    // Permute.
    ssort::samplesort(P, P + dc3_elem_array_size, compare_P_elem, mpi_dc3_elem,
                      numprocs, myid);

    MPI_Barrier(MPI_COMM_WORLD);
    double recursivet = 0;
    if (!myid)
      recursivet = MPI::Wtime();


    // Use SAIS for recursive case. TODO expand to distributed version
    //

    for (uint32_t i = 0; i < dc3_elem_array_size; i++)
      names[i] = P[i].word;

    int* sizes = NULL;
    int* displ = NULL;
    int* all_names = NULL;
    int* all_SA = NULL;
    uint32_t total_size;
    if (myid == numprocs - 1) {
      sizes = new int[numprocs];
      displ = new int[numprocs];
    }

    int dc3_elem_array_size_int = dc3_elem_array_size;
    double gt = MPI :: Wtime();
    // send sizes of local arrays in preparation for sending the local arrays
    MPI_Gather(&dc3_elem_array_size_int, 1, MPI_INT, &sizes[0], 1, MPI_INT,
               numprocs - 1, MPI_COMM_WORLD);
    MPI_Barrier(MPI_COMM_WORLD);
    if (!myid) printf("Gather time %f\n", MPI::Wtime() - gt);

    if (myid == numprocs - 1) {
      displ[0] = 0;
      for (int i = 1; i < numprocs; i++)  // exclusive scan
        displ[i] = displ[i - 1] + sizes[i - 1];
      total_size = displ[numprocs - 1] + sizes[numprocs - 1];

      all_names = new int[total_size];
    }

    // send local arrays to root
    double ag = MPI :: Wtime();

    MPI_Gatherv(names, dc3_elem_array_size_int, MPI_UNSIGNED, all_names, sizes,
                displ, MPI_INT, numprocs - 1, MPI_COMM_WORLD);
    MPI_Barrier(MPI_COMM_WORLD);
    if (!myid) printf("Allgather time %f\n", MPI::Wtime() - ag);

    if (myid == numprocs - 1) {
      all_SA = new int[total_size];

      // recurse
      sais_int(all_names, all_SA, total_size, total + 1);

      delete[] all_names;
    }

    int* local_SA = new int[dc3_elem_array_size];

    // send result of recursive call back to nodes
    MPI_Barrier(MPI_COMM_WORLD);
    ag = MPI :: Wtime();

    MPI_Scatterv(all_SA, sizes, displ, MPI_INT, local_SA,
                 dc3_elem_array_size_int, MPI_INT, numprocs - 1,
                 MPI_COMM_WORLD);
    MPI_Barrier(MPI_COMM_WORLD);
    if (!myid) printf("Scatterv time %f\n", MPI::Wtime() - ag);

    int global_idx;

    // tell each processor what their index their array starts on globally
    ag = MPI :: Wtime();

    MPI_Scatter(displ, 1, MPI_INT, &global_idx, 1, MPI_INT, numprocs - 1,
                MPI_COMM_WORLD);
    MPI_Barrier(MPI_COMM_WORLD);
    if (!myid) printf("Scatter time %f\n", MPI::Wtime() - ag);
    if (!myid) fprintf(stdout, "Runtime of sais call: %f\n\n", MPI::Wtime() - recursivet);

    for (int i = 0; i < dc3_elem_array_size_int; i++) {
      P[i].word = global_idx + i + 1;
      P[i].index = map_back(local_SA[i], (file_size + 1) / 3);
    }

    delete[] sizes;
    delete[] displ;
    delete[] all_SA;
    delete[] local_SA;

  }


  // Sort P by second element. This aids in next component's construction.
  ssort::samplesort(P, P + dc3_elem_array_size, compare_sortedP_elem,
                    mpi_dc3_elem, numprocs, myid);

  /*
   *  @TODO: Component 5:
   *  S_0 := <(T[i], T[i+1], c', c'', i) : i mod 3 = 0), (c',i+1), (c'', i+2)
   in P>
   *  S_1 := <(c, T[i], c', i) : i mod 3 = 1), (c,i), (c', i+1) in P>
   *  S_2 := <(c, T[i], T[i+1], c'', i) : i mod 3 = 2), (c,i), (c'', i+2) in
   P>
   */
  MPI_Barrier(MPI_COMM_WORLD);
  if (!myid) {
    fprintf(stdout, "Runtime of component 4: %f\n\n", MPI::Wtime() - elapsed);
    elapsed = MPI::Wtime();
    fprintf(stdout, "Building component 5\n");
  }

  // We need the first two elements of the next process.
  if (myid != 0) {
    MPI_Send(&P[0], 2, mpi_dc3_elem, myid - 1, 0, MPI_COMM_WORLD);
  }

  dc3_elem* next2 = new dc3_elem[2]();
  if (next2 == NULL) {
    return -1;
  }
  if (myid != numprocs - 1) {
    MPI_Recv(next2, 2, mpi_dc3_elem, myid + 1, 0, MPI_COMM_WORLD,
             MPI_STATUS_IGNORE);
  }

  // Create the tuple array.
  dc3_tuple_elem* SS = new dc3_tuple_elem[size]();
  if (SS == NULL) {
    return -1;
  }

  for (uint32_t i = 0; i < size; i++) {
    // Calculate which of (S_0, S_1, S_2) this index is.
    uint32_t array_num = (i + offset) % 3;

    // Word stores [arraynum][char i][char i + 1]
    uint32_t word = array_num;
    word <<= 8;
    word = word + static_cast<uint32_t>(data[i]);
    word <<= 8;
    word = word + static_cast<uint32_t>(data[i + 1]);
    SS[i].word = word;

    // Use offset to calculate global index.
    SS[i].index = i + offset;
  }

  for (uint32_t i = 0; i < dc3_elem_array_size; i++) {
    // Get current index.
    uint32_t curr_index = P[i].index;

    // Get local offset of SS array.
    uint32_t local_offset = curr_index - offset;

    // x = 0 mod 3 depends on x + 1, x + 2
    // x = 1 mod 3 depends on x, x + 1
    // x = 2 mod 3 depends on x, x + 2

    // Index is 1 mod 3.
    if (curr_index % 3 == 1) {
      // If prev element (0 mod 3) resides in this process, update.
      // prev process | [ 0 mod 3] [ 1 mod 3 ]
      if (local_offset >= 1) {
        SS[local_offset - 1].name1 = P[i].word;
      }

      // Update current element.
      SS[local_offset].name1 = P[i].word;

      // If the prev-prev (2 mod 3) element resides in this process, update it.
      if (local_offset >= 2) {
        SS[local_offset - 2].name2 = P[i].word;
      }
    } else {
      // Index is 2 mod 3.
      // Check if in bounds
      if (local_offset >= 2) {
        SS[local_offset - 2].name2 = P[i].word;
      }
      if (local_offset >= 1) {
        SS[local_offset - 1].name2 = P[i].word;
      }
      SS[local_offset].name1 = P[i].word;
    }
  }

  // Update the SS array using elements received from next process.
  if (myid != numprocs - 1) {
    uint32_t next_index = next2[0].index;
    uint32_t local_offset = next_index - offset;

    // Index is x = 1 mod 3.
    if (next_index % 3 == 1) {
      // If the (x - 1) = 0 mod 3 element is in this process (as opposed to the
      // next process), update it. This also implies that x - 2 is in this
      // process, so update that as well.
      if (local_offset - 1 < size) {
        SS[local_offset - 1].name1 = next2[0].word;
        SS[local_offset - 1].name2 = next2[1].word;
      }

      // x - 2 's name will always be updated.
      SS[local_offset - 2].name2 = next2[0].word;
    } else {
      // Index is x = 2 mod 3. Since the first element of the next process is 2
      // mod 3, this implies x - 1 must be in this process. Hence update both
      // x - 1 and x - 2.
      SS[local_offset - 2].name2 = next2[0].word;
      SS[local_offset - 1].name2 = next2[0].word;
    }
  }

  /*
   *  Component 6:
   *  Sort S_0 union S_1 union S_2 using compare operator in paper.
   */
  MPI_Barrier(MPI_COMM_WORLD);
  if (!myid) {
    fprintf(stdout, "Runtime of component 5: %f\n\n", MPI::Wtime() - elapsed);
    elapsed = MPI::Wtime();
    fprintf(stdout, "Building component 6\n");
  }

  ssort::samplesort(SS, SS + size, compare_tuple_elem, mpi_dc3_tuple_elem,
                    numprocs, myid);

  /*
   *  Component 7:
   *  Return last component of (s : s in S).
   */
  MPI_Barrier(MPI_COMM_WORLD);
  if (!myid) {
    fprintf(stdout, "Runtime of component 6: %f\n\n", MPI::Wtime() - elapsed);
    elapsed = MPI::Wtime();
    fprintf(stdout, "Building component 7\n");
  }

  for (uint32_t i = 0; i < size; i++) {
    suffix_array[i] = SS[i].index;
  }

  MPI_Barrier(MPI_COMM_WORLD);
  if (!myid) {
    fprintf(stdout, "Runtime of component 7: %f\n\n", MPI::Wtime() - elapsed);
  }

  delete[] P;
  delete[] S;
  delete[] SS;
  delete[] is_diff_from_adj;
  delete[] next2;

  return 0;
}
