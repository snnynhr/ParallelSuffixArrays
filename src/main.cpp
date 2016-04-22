#include <stdio.h>
#include <stdlib.h>
#include "mpi.h"
#include "fileio.h"

using namespace std;

int main(int argc, char* argv[]) {
  int numprocs;
  int myid;
  int namelen;
  char processor_name[MPI_MAX_PROCESSOR_NAME];

  MPI_Init(&argc, &argv);
  MPI_Comm_size(MPI_COMM_WORLD, &numprocs);
  MPI_Comm_rank(MPI_COMM_WORLD, &myid);
  MPI_Get_processor_name(processor_name, &namelen);

  fprintf(stderr, "Process %d on %s\n", myid, processor_name);
  if (myid == 0) {
    fprintf(stderr, "There are %d total processors.\n", numprocs);
    ifstream f(argv[1]);
    if (!f.good()) {
      f.close();
      MPI_Finalize();
      fprintf(stderr, "File doesn't exist\n");
      exit(-1);
    } else {
      f.close();
    }
  }

  // MPI_Barrier(MPI_COMM_WORLD);

  // Decompose file
  uint64_t size = 0;
  char* data = file_block_decompose(argv[1], size, MPI_COMM_WORLD, 1);
  if (data == NULL) {
    fprintf(stderr, "File allocation on processor %d failed.\n", myid);
    MPI_Finalize();
    exit(-1);
  }
  fprintf(stdout, "%d: %.*s", myid, static_cast<uint32_t>(size), data);

  /*
   *  Begin construction!
   */

  /*
   *  @TODO: Component 1:
   *  S = <(T[i,i+2], i) : i \in [0,n), i mod 3 \not= 0>
   */

  /*
   *  @TODO: Component 2:
   *  sort S by first component
   */

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

  // MPI_Barrier(MPI_COMM_WORLD);

  // Done
  free(data);
  MPI_Finalize();
  exit(0);
}
