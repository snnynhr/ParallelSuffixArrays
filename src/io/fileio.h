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
char* file_block_decompose(const char* filename, uint64_t& size,
                           uint64_t& file_size, uint64_t& offset,
                           MPI_Comm comm = MPI_COMM_WORLD,
                           uint64_t alignment = 32, uint32_t extra = 2);
