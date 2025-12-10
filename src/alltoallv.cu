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

// scale input traffic matrix values by a KiB
#define TRAFFIC_SCALE 1024

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
static bool matrix_loaded = false;
static size_t global_max_send = 0;
static size_t global_max_recv = 0;
static size_t rank_send_bytes = 0;

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
        traffic_matrix[i][j] = (size_t)std::stod(cell) * TRAFFIC_SCALE;
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

  // Try to load traffic matrix into memory on first call
  if (!matrix_loaded) {
    MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);  // Only needed for debug printing
    AlltoAllvParseEnv();
    AlltoAllvReadTrafficMatrix(nranks);
    matrix_loaded = true;
  }

  // Calculate maximum send/receive counts across ALL ranks
  *sendcount = 0;
  *recvcount = 0;

  for (int i = 0; i < nranks; i++) {
    size_t rank_sendbytes = 0;
    size_t rank_recvbytes = 0;

    for (int j = 0; j < nranks; j++) {
      rank_sendbytes += traffic_matrix[i][j];  // What rank i sends total
      rank_recvbytes += traffic_matrix[j][i];  // What rank i receives total
    }

    DEBUG_PRINT_ALL("Rank %d: send %zu bytes, recv %zu bytes", i, rank_sendbytes, rank_recvbytes);

    *sendcount = std::max(*sendcount, rank_sendbytes/eltSize);
    *recvcount = std::max(*recvcount, rank_recvbytes/eltSize);
  }

  DEBUG_PRINT_ALL("Max buffer sizes - Send: %zu, Recv: %zu", *sendcount * eltSize, *recvcount * eltSize);

  // Avoid recalculating this later
  global_max_send = *sendcount;
  global_max_recv = *recvcount;

  // For AlltoAllv, paramcount isn't used in the traditional sense since we don't call
  // a single NCCL collective. However, we need to set it for interface compatibility.
  // We'll use the overall input count divided by the number of ranks as a representative value.
  *paramcount = (count/nranks) & -(16/eltSize);

  // AlltoAllv doesn't support in-place operations
  *sendInplaceOffset = 0;
  *recvInplaceOffset = 0;
}

// XXX This needs to be fixed to support data validation
testResult_t AlltoAllvInitData(struct threadArgs* args, ncclDataType_t type, ncclRedOp_t op, int root, int rep, int in_place) {
  size_t sendcount = args->sendBytes / wordSize(type);
  size_t recvcount = args->expectedBytes / wordSize(type);
  int nranks = args->nProcs*args->nThreads*args->nGpus;

  for (int i=0; i<args->nGpus; i++) {
    CUDACHECK(cudaSetDevice(args->gpus[i]));
    int rank = ((args->proc*args->nThreads + args->thread)*args->nGpus + i);
    CUDACHECK(cudaMemset(args->recvbuffs[i], 0, args->expectedBytes));
    void* data = in_place ? args->recvbuffs[i] : args->sendbuffs[i];
    TESTCHECK(InitData(data, sendcount, 0, type, ncclSum, 33*rep + rank, 1, 0));
    for (int j=0; j<nranks; j++) {
      size_t partcount = sendcount/nranks;
      TESTCHECK(InitData((char*)args->expected[i] + j*partcount*wordSize(type), partcount, rank*partcount, type, ncclSum, 33*rep + j, 1, 0));
    }
    CUDACHECK(cudaDeviceSynchronize());
  }
  // We don't support in-place alltoallv
  args->reportErrors = in_place ? 0 : 1;
  return testSuccess;
}

void AlltoAllvGetBw(size_t count, int typesize, double sec, double* algBw, double* busBw, int nranks) {
  // Calculate total data transferred by this rank (focusing on sent data)
  double baseBw = (double)(rank_send_bytes) / 1.0E9 / sec;

  *algBw = baseBw;
  double factor = ((double)(nranks-1))/((double)(nranks));
  *busBw = baseBw * factor;
}

testResult_t AlltoAllvRunColl(void* sendbuff, size_t sendoffset, void* recvbuff, size_t recvoffset, size_t count, ncclDataType_t type, ncclRedOp_t op, int root, ncclComm_t comm, cudaStream_t stream, int deviceImpl) {
  if (deviceImpl == 0) {
    char* sptr = (char*)sendbuff + sendoffset;
    char* rptr = (char*)recvbuff + recvoffset;
#if NCCL_VERSION_CODE >= NCCL_VERSION(2,7,0)
    int nRanks, ncclRank;
    NCCLCHECK(ncclCommCount(comm, &nRanks));
    NCCLCHECK(ncclCommUserRank(comm, &ncclRank));
    NCCLCHECK(ncclGroupStart());
    for (int r=0; r<nRanks; r++) {
      // use traffic matrix for variable send/recv counts
      size_t sendBytes = traffic_matrix[ncclRank][r];
      size_t sendCount = sendBytes / wordSize(type);
      size_t recvBytes = traffic_matrix[r][ncclRank];
      size_t recvCount = recvBytes / wordSize(type);
      if (sendCount > 0) {
        NCCLCHECK(ncclSend(sptr, sendCount, type, r, comm, stream));
        sptr += sendBytes;
      }
      if (recvCount > 0) {
        NCCLCHECK(ncclRecv(rptr, recvCount, type, r, comm, stream));
        rptr += recvBytes;
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
  size_t paramcount, sendInplaceOffset, recvInplaceOffset;
  AlltoAllvGetCollByteCount(sendcount, recvcount, &paramcount, &sendInplaceOffset, &recvInplaceOffset, count, /*eltSize=*/1, nranks);
}

testResult_t AlltoAllvRunTest(struct threadArgs* args, int root, ncclDataType_t type, const char* typeName, ncclRedOp_t op, const char* opName) {
  args->collTest = &alltoAllvTest;

  if (args->reportErrors) {
    FATAL_ERROR("AlltoAllv does not support data validation. Run with -c 0 to disable data checking.");
  }

  if (args->minbytes != args->maxbytes) {
    FATAL_ERROR("AlltoAllv requires single size testing. Set -b and -e to the same value (e.g., -b 32M -e 32M).");
  }

  ncclDataType_t *run_types;
  const char **run_typenames;
  int type_count;

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
    // Set maxbytes for this specific data type to ensure steps=1 (shift=0)
    size_t max_elements = std::max(global_max_send, global_max_recv);
    //args->maxbytes = max_elements * wordSize(run_types[i]);

    TESTCHECK(TimeTest(args, run_types[i], run_typenames[i], (ncclRedOp_t)0, "none", -1));
  }
  return testSuccess;
}

struct testEngine alltoAllvEngine = {
  .getBuffSize = AlltoAllvGetBuffSize,
  .runTest = AlltoAllvRunTest
};

#pragma weak ncclTestEngine=alltoAllvEngine
