#include "suffix_array.h"
#include "../sort/ssort.h"

typedef struct dc3_elem {
  uint32_t word;
  uint64_t index;
} dc3_elem;

bool compare_dc3_elem(const dc3_elem lhs, const dc3_elem rhs) {
  return lhs.word < rhs.word;
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
  /*
*  @TODO: Component 1:
*  S = <(T[i,i+2], i) : i \in [0,n), i mod 3 \not= 0>
*/

  const uint64_t dc3_elem_array_size = ((size - 1) / 3) * 2 + ((size - 1) % 3);
  dc3_elem dc3_elem_array[dc3_elem_array_size] = {0};

  // Construct 'S' array
  uint64_t pos = 0;
  uint64_t count = 0;
  while (count < dc3_elem_array_size) {
    if (pos % 3 != 0) {
      // Watch unsigned / signed
      uint32_t word = data[pos];
      word = (word << 8) + data[pos + 1];
      word = (word << 8) + data[pos + 2];
      dc3_elem_array[count].word = word;
      count++;
    }
    pos++;
  }

  /*
   *  @TODO: Component 2:
   *  sort S by first component
   */
  ssort::samplesort(dc3_elem_array, dc3_elem_array + dc3_elem_array_size,
                    compare_dc3_elem, mpi_dc3_elem, numprocs, myid);

  /*
   *  @TODO: Component 3:
   *  P := name (S)
   */

  /*
   *  @TODO: Component 4:
   *  If names are not unique:
   *  Permute (r,i) in P such that they are sorted by (i mod 3, i div 3)
   *  SA^{12} = pDC3(<c : (c,i) in P>)
   *  P := <(j+1, mapBack(SA^{12}[j], n/3)) : j < 2n/3>
   */

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