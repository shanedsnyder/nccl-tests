/*************************************************************************
 * Copyright (c) 2016-2022, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#include "cuda_runtime.h"
#include "common.h"
#include <vector>
#include <fstream>
#include <sstream>
#include <algorithm>
#include <string>

#define DEBUG_PRINT(fmt, ...) \
  do { \
    if (enable_debug && mpi_rank == 0) { \
      printf("[DEBUG] " fmt "\n", ##__VA_ARGS__); \
    } \
  } while(0)

#define DEBUG_PRINT_ALL(fmt, ...) \
  do { \
    if (enable_debug) { \
      printf("[DEBUG rank %d] " fmt "\n", mpi_rank, ##__VA_ARGS__); \
    } \
  } while(0)

#define FATAL_ERROR(fmt, ...) \
  do { \
    if (mpi_rank == 0) { \
      printf("Error: " fmt "\n", ##__VA_ARGS__); \
    } \
    MPI_Abort(MPI_COMM_WORLD, 1); \
  } while(0)

// Global variables for AlltoAllv traffic matrix and options
static int mpi_rank;
static bool enable_debug = false;
static char* matrix_file = nullptr; // XXX default in-tree matrix file?
static std::vector<std::vector<size_t>> traffic_matrix;
static size_t global_max_send_bytes = 0;
static size_t global_max_recv_bytes = 0;

void AlltoAllvParseEnv() {
  const char* matrix_file_env = getenv("ALLTOALLV_MATRIX_FILE");
  if (matrix_file_env) matrix_file = const_cast<char*>(matrix_file_env);
  if (getenv("ALLTOALLV_DEBUG")) enable_debug = true;
}

void AlltoAllvReadTrafficMatrix(int nranks) {
  if (!matrix_file) {
    FATAL_ERROR("No matrix file specified. Use ALLTOALLV_MATRIX_FILE env var.");
  }
  std::ifstream file(matrix_file);
  if (!file.is_open()) {
    FATAL_ERROR("Unable to open matrix file %s.", matrix_file);
  }

  // Read lines until we have nranks or hit EOF
  std::vector<std::string> lines;
  std::string line;
  int matrix_len = 0;
  while (matrix_len < nranks && std::getline(file, line)) {
    if (!line.empty()) {
      lines.push_back(line);
      matrix_len++;
    }
  }

  // Validate that we have enough ranks in the matrix
  if (matrix_len < nranks) {
    FATAL_ERROR("Requested nranks (%d) is larger than matrix size (%d) in file %s.",
                nranks, matrix_len, matrix_file);
  }

  traffic_matrix.clear();
  traffic_matrix.resize(nranks, std::vector<size_t>(nranks));

  // Single pass: validate columns and assign values
  for (int i = 0; i < nranks; i++) {
    std::stringstream ss(lines[i]);
    std::string cell;
    int j = 0;
    while (std::getline(ss, cell, ' ')) {
      if (!cell.empty()) {
        if (j >= nranks) {
          // More columns than needed, ignore the rest
          break;
        }
        traffic_matrix[i][j] = (size_t)std::stod(cell);
        j++;
      }
    }

    if (j < nranks) {
      FATAL_ERROR("Row %d has only %d columns in file %s, but nranks is %d.",
                  i, j, matrix_file, nranks);
    }
        
    if (enable_debug && mpi_rank == 0 && i < 5) {
      std::stringstream debug_ss;
      debug_ss << "Matrix row " << i << ": ";
      for (int k = 0; k < std::min(nranks, 8); k++) {
        debug_ss << traffic_matrix[i][k] << " ";
      }
      if (nranks > 8) debug_ss << "...";
      DEBUG_PRINT("%s", debug_ss.str().c_str());
    }
  }

  file.close();
  DEBUG_PRINT("Successfully loaded traffic matrix (%dx%d) from %s", nranks, nranks, matrix_file);
}

void AlltoAllvGetCollByteCount(size_t *sendcount, size_t *recvcount, size_t *paramcount, size_t *sendInplaceOffset, size_t *recvInplaceOffset, size_t count, size_t eltSize, int nranks) {
  // derive a representative per-peer element count from the matrix maximums
  size_t max_count = std::max(global_max_send_bytes, global_max_recv_bytes) / eltSize;
  *paramcount = (max_count / nranks) & -(16/eltSize);
  // use precomputed byte maximums divided by element size
  *sendcount = global_max_send_bytes / eltSize;
  *recvcount = global_max_recv_bytes / eltSize;
  // alltoAllv doesn't support in-place operations
  *sendInplaceOffset = 0;
  *recvInplaceOffset = 0;
}

testResult_t AlltoAllvInitData(struct threadArgs* args, ncclDataType_t type, ncclRedOp_t op, int root, int rep, int in_place) {
  size_t elt_size = wordSize(type);
  int nranks = args->nProcs*args->nThreads*args->nGpus;

  for (int i=0; i<args->nGpus; i++) {
    CUDACHECK(cudaSetDevice(args->gpus[i]));
    int rank = ((args->proc*args->nThreads + args->thread)*args->nGpus + i);

    // zero out full recv and expected buffers, as expectedBytes is based on max across
    // ranks (not this rank's exact value) and unused tails must ultimately compare equal
    CUDACHECK(cudaMemset(args->recvbuffs[i], 0, args->expectedBytes));
    CUDACHECK(cudaMemset(args->expected[i], 0, args->expectedBytes));

    // prepare this rank's send buffer
    size_t my_send_bytes = 0;
    for (int dst = 0; dst < nranks; dst++) my_send_bytes += traffic_matrix[rank][dst];
    size_t my_send_count = my_send_bytes / elt_size;
    if (my_send_count > 0) {
      void* data = in_place ? args->recvbuffs[i] : args->sendbuffs[i];
      TESTCHECK(InitData(data, my_send_count, 0, type, ncclSum, 33*rep + rank, 1, 0));
    }

    // prepare this rank's expected buffer, which is a concatenation of variable-length
    // segments received from each source in rank order
    size_t recv_offset = 0;
    for (int src = 0; src < nranks; src++) {
      size_t src_recv_bytes = traffic_matrix[src][rank];
      size_t src_recv_count = src_recv_bytes / elt_size;
      if (src_recv_count == 0) continue;
      // offset within the source's send stream where my segment begins (in elements)
      size_t src_offset = 0;
      for (int k = 0; k < rank; k++) src_offset += traffic_matrix[src][k];
      void* exp_ptr = (char*)args->expected[i] + recv_offset;
      TESTCHECK(InitData(exp_ptr, src_recv_count, src_offset / elt_size, type, ncclSum, 33*rep + src, 1, 0));
      recv_offset += src_recv_bytes;
    }
    CUDACHECK(cudaDeviceSynchronize());
  }
  // We don't support in-place alltoallv
  args->reportErrors = in_place ? 0 : 1;
  return testSuccess;
}

void AlltoAllvGetBw(size_t count, int typesize, double sec, double* algBw, double* busBw, int nranks) {
  // use max send or recv bytes across all ranks
  size_t max_sendrecv_bytes = std::max(global_max_send_bytes, global_max_recv_bytes);
  double baseBw = (double)max_sendrecv_bytes / 1.0E9 / sec;

  *algBw = baseBw;
  double factor = ((double)(nranks-1))/((double)(nranks));
  *busBw = baseBw * factor;
}

testResult_t AlltoAllvRunColl(void* sendbuff, size_t sendoffset, void* recvbuff, size_t recvoffset, size_t count, ncclDataType_t type, ncclRedOp_t op, int root, ncclComm_t comm, cudaStream_t stream, int deviceImpl) {
  if (deviceImpl == 0) {
    char* sptr = (char*)sendbuff + sendoffset;
    char* rptr = (char*)recvbuff + recvoffset;
#if NCCL_VERSION_CODE >= NCCL_VERSION(2,7,0)
    int nranks, nccl_rank;
    NCCLCHECK(ncclCommCount(comm, &nranks));
    NCCLCHECK(ncclCommUserRank(comm, &nccl_rank));
    NCCLCHECK(ncclGroupStart());
    for (int r=0; r<nranks; r++) {
      // use traffic matrix for variable send/recv counts
      size_t send_bytes = traffic_matrix[nccl_rank][r];
      size_t send_count = send_bytes / wordSize(type);
      size_t recv_bytes = traffic_matrix[r][nccl_rank];
      size_t recv_count = recv_bytes / wordSize(type);
      if (send_count > 0) {
        NCCLCHECK(ncclSend(sptr, send_count, type, r, comm, stream));
        sptr += send_bytes;
      }
      if (recv_count > 0) {
        NCCLCHECK(ncclRecv(rptr, recv_count, type, r, comm, stream));
        rptr += recv_bytes;
      }
    }
    NCCLCHECK(ncclGroupEnd());
    return testSuccess;
#else
  printf("NCCL 2.7 or later is needed for alltoallv. This test was compiled with %d.%d.\n", NCCL_MAJOR, NCCL_MINOR);
  return testNcclError;
#endif
  } else {
    return testNotImplemented;
  }
}

struct testColl alltoAllvTest = {
  "AlltoAllv",
  AlltoAllvGetCollByteCount,
  AlltoAllvInitData,
  AlltoAllvGetBw,
  AlltoAllvRunColl
};

void AlltoAllvGetBuffSize(size_t *sendcount, size_t *recvcount, size_t count, int nranks) {
  MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);  // for debug printing
  AlltoAllvParseEnv();
  AlltoAllvReadTrafficMatrix(nranks);

  // calculate max send and recv bytes across all ranks
  size_t max_send_bytes = 0;
  size_t max_recv_bytes = 0;
  for (int i = 0; i < nranks; i++) {
    size_t rank_sendbytes = 0;
    size_t rank_recvbytes = 0;
    for (int j = 0; j < nranks; j++) {
      rank_sendbytes += traffic_matrix[i][j];
      rank_recvbytes += traffic_matrix[j][i];
    }
    DEBUG_PRINT_ALL("Rank %d: send %zu bytes, recv %zu bytes", i, rank_sendbytes, rank_recvbytes);
    if (rank_sendbytes > max_send_bytes) max_send_bytes = rank_sendbytes;
    if (rank_recvbytes > max_recv_bytes) max_recv_bytes = rank_recvbytes;
  }
  global_max_send_bytes = max_send_bytes;
  global_max_recv_bytes = max_recv_bytes;
  DEBUG_PRINT_ALL("Max buffer sizes - Send: %zu, Recv: %zu", global_max_send_bytes, global_max_recv_bytes);

  // eltSize=1 is expected here, so output counts are bytes
  *sendcount = global_max_send_bytes;
  *recvcount = global_max_recv_bytes;
}

testResult_t AlltoAllvRunTest(struct threadArgs* args, int root, ncclDataType_t type, const char* typeName, ncclRedOp_t op, const char* opName) {
  args->collTest = &alltoAllvTest;
  ncclDataType_t *run_types;
  const char **run_typenames;
  int type_count;

  if (args->minbytes != args->maxbytes) {
    FATAL_ERROR("AlltoAllv requires -b and -e options to be set to the same value (e.g., -b 32M -e 32M).");
  }
  size_t total_bytes_req = std::max(global_max_send_bytes, global_max_recv_bytes);
  if (args->maxbytes < total_bytes_req) {
    FATAL_ERROR("AlltoAllv requires at least %zu bytes (from input traffic matrix), but maxBytes (-e) is %zu. Increase -e.", total_bytes_req, args->maxbytes);
  }

  if ((int)type != -1) {
    type_count = 1;
    run_types = &type;
    run_typenames = &typeName;
  } else {
    type_count = test_typenum;
    run_types = test_types;
    run_typenames = test_typenames;
  }

  for (int i=0; i<type_count; i++) {
    TESTCHECK(TimeTest(args, run_types[i], run_typenames[i], (ncclRedOp_t)0, "none", -1));
  }
  return testSuccess;
}

struct testEngine alltoAllvEngine = {
  .getBuffSize = AlltoAllvGetBuffSize,
  .runTest = AlltoAllvRunTest
};

#pragma weak ncclTestEngine=alltoAllvEngine
