#include "fileio.h"

size_t get_filesize(const char* filename)
{
    std::ifstream in(filename, std::ifstream::ate | std::ifstream::binary);
    return in.tellg();
}

const char * file_block_decompose(const char* filename, MPI_Comm comm, uint32_t alignment)
{
    // get size of input file
    const std::size_t file_size = get_filesize(filename);

    // get communication parameters
    int p, rank;
    MPI_Comm_size(comm, &p);
    MPI_Comm_rank(comm, &rank);


    uint32_t size = 0;
    const uint32_t numblocks = (file_size + alignment - 1) / alignment;

    const uint32_t blocks = numblocks / p + (rank < numblocks % p);

    if(rank < p - 1) {
        size = blocks * alignment;
    } else {
        size = blocks * alignment - file_size % alignment;
    }

    const uint32_t offset = p * blocks * alignment;

    // open file
    std::ifstream t(filename);

    std::streambuf* sbuf= t.rdbuf();
    sbuf->pubseekpos(offset, std::ios_base::in);
    char* data = static_cast<char*>(malloc(sizeof(size)));

    if(data == NULL) {
        return NULL;
    }

    sbuf->sgetn(data, size);
    return data;
}

// void write_files(const std::string& filename, _Iterator begin, _Iterator end, MPI_Comm comm = MPI_COMM_WORLD)
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
//     ss << filename << "." << std::setfill('0') << std::setw(rank_slen) << p << "." << std::setfill('0') << std::setw(rank_slen) << rank;

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
