#include <mpi.h>

// C++ includes
#include <string.h>
#include <fstream>
#include <streambuf>
#include <sstream>
#include <iostream>
#include <iomanip>
#include <stdint.h>
#include <stdlib.h>

size_t get_filesize(const char* filename);
char * file_block_decompose(const char* filename, uint32_t* size, MPI_Comm comm = MPI_COMM_WORLD, uint32_t alignment = 32);
