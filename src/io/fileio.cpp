#include "fileio.h"

size_t get_filesize(const char* filename) {
  std::ifstream in(filename, std::ifstream::ate | std::ifstream::binary);
  return in.tellg();
}

char* file_block_decompose(const char* filename, uint64_t& size,
                           uint64_t& file_size, uint64_t& offset, MPI_Comm comm,
                           uint64_t alignment) {
  // get size of input file
  file_size = get_filesize(filename);

  // get communication parameters
  int32_t p, rank;
  MPI_Comm_size(comm, &p);
  MPI_Comm_rank(comm, &rank);

  const uint64_t num_aligned_blocks = (file_size + alignment - 1) / alignment;
  const uint32_t mod = num_aligned_blocks % p;

  const uint64_t proc_num_aligned_blocks =
      num_aligned_blocks / static_cast<uint64_t>(p) +
      (static_cast<uint64_t>(rank) < mod);

  const uint64_t proc_size = proc_num_aligned_blocks * alignment;

  if (static_cast<uint32_t>(rank) < mod) {
    offset = rank * proc_num_aligned_blocks * alignment;
  } else {
    offset = mod * (proc_num_aligned_blocks + 1) * alignment +
             (rank - mod) * proc_num_aligned_blocks * alignment;
  }

  if (rank < p - 1) {
    size = proc_size;
  } else {
    size = proc_size - (alignment - (file_size % alignment));
  }

  if (rank == 0) {
    fprintf(stdout, "Filesize %zu and block size %zu\n", file_size, size);
  }

  // open file
  std::ifstream t(filename);

  t.seekg(offset);
  char* data;
  try {
    data = new char[size + 2];
  } catch (std::bad_alloc& ba) {
    return NULL;
  }

  if (rank < p - 1) {
    t.readsome(data, size + 2);
  } else {
    t.readsome(data, size);
    data[size + 1] = 0;
    data[size + 2] = 0;
  }
  return data;
}

// void write_files(const std::string& filename, _Iterator begin, _Iterator end,
// MPI_Comm comm = MPI_COMM_WORLD)
// {
//     // get MPI Communicator properties
//     int rank, p;
//     MPI_Comm_size(comm, &p);
//     MPI_Comm_rank(comm, &rank);

//     // get max rank string length:
//     std::stringstream sslen;
//     sslen << p;
//     int rank_slen = sslen.str().size();

//     // concat rank at end of filename
//     std::stringstream ss;
//     ss << filename << "." << std::setfill('0') << std::setw(rank_slen) << p
//     << "." << std::setfill('0') << std::setw(rank_slen) << rank;

//     // open file with stream
//     //std::cerr << "writing to file " << ss.str() << std::endl;
//     std::ofstream outfs(ss.str());

//     // write the content into the file, sep by newline
//     while (begin != end)
//     {
//         outfs << *(begin++) << std::endl;
//     }
//     outfs.close();
// }
