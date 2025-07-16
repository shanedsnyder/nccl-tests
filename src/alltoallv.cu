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

#define DEBUG_PRINT(rank, fmt, ...) \
  do { \
    if (enable_debug && rank == 0) { \
      printf("[DEBUG] " fmt "\n", ##__VA_ARGS__); \
    } \
  } while(0)

#define DEBUG_PRINT_ALL(rank, fmt, ...) \
  do { \
    if (enable_debug) { \
      printf("[DEBUG rank %d] " fmt "\n", rank, ##__VA_ARGS__); \
    } \
  } while(0)

// Global variables for AlltoAllv traffic matrix and options
static int rank;
static bool enable_debug = false;
static char* matrix_file = nullptr; // XXX default in-tree matrix file?
static std::vector<std::vector<size_t>> traffic_matrix;
static bool matrix_loaded = false;

void AlltoAllvParseEnv() {
  const char* matrix_file_env = getenv("ALLTOALLV_MATRIX_FILE");
  if (matrix_file_env) matrix_file = const_cast<char*>(matrix_file_env);
  if (getenv("ALLTOALLV_DEBUG")) enable_debug = true;
}

void AlltoAllvReadTrafficMatrix(int nranks) {
  if (!matrix_file) {
    if (rank == 0) {
      printf("Error: No matrix file specified. Use ALLTOALLV_MATRIX_FILE env var.\n");
    }
    MPI_Abort(MPI_COMM_WORLD, 1);
  }
  std::ifstream file(matrix_file);
  if (!file.is_open()) {
    if (rank == 0) {
      printf("Error: Unable to open matrix file %s.\n", matrix_file);
    }
    MPI_Abort(MPI_COMM_WORLD, 1);
  }

  traffic_matrix.clear();
  traffic_matrix.resize(nranks, std::vector<size_t>(nranks));

  std::string line;
  for (int i = 0; i < nranks && std::getline(file, line); i++) {
    std::stringstream ss(line);
    std::string cell;

    // XXX general error handling for parsing matrix file
    for (int j = 0; j < nranks && std::getline(ss, cell, ' '); j++) {
      traffic_matrix[i][j] = (size_t)std::stod(cell);
    }
        
    if (enable_debug && rank == 0 && i < 5) {
      printf("[DEBUG] Matrix row %d: ", i);
      for (int j = 0; j < std::min(nranks, 8); j++) {
        printf("%zu ", traffic_matrix[i][j]);
      }
      if (nranks > 8) printf("...");
      printf("\n");
    }
  }

  // XXX handle mismatches between matrix and nranks
  file.close();
}
// Get traffic count for src->dst pair
size_t AlltoAllvGetTrafficCount(int src, int dst, int nranks) {
  // XXX is this needed? or should the program have exited by now?
  if (!matrix_loaded) {
    AlltoAllvReadTrafficMatrix(nranks);
  }

  // XXX handle this in matrix loop bounds or something. if removed, we can likely drop nranks from the function signature or just remove the function entirely
  if (src >= nranks || dst >= nranks) {
    return 0; // No traffic for ranks beyond matrix size
  }

  return traffic_matrix[src][dst];
}

void AlltoAllvGetCollByteCount(size_t *sendcount, size_t *recvcount, size_t *paramcount, size_t *sendInplaceOffset, size_t *recvInplaceOffset, size_t count, size_t eltSize, int nranks) {

  // Try to load traffic matrix into memory on first call
  if (!matrix_loaded) {
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    AlltoAllvParseEnv();
    AlltoAllvReadTrafficMatrix(nranks);
    matrix_loaded = true;
  }

  *sendcount = 0;
  *recvcount = 0;

  // Calculate send count (sum of what this rank sends to all others)
  for (int r = 0; r < nranks; r++) {
    size_t traffic = AlltoAllvGetTrafficCount(rank, r, nranks);
    *sendcount += traffic;
    DEBUG_PRINT_ALL(rank, "Send to rank %d: %zu elements", r, traffic);
  }

  // Calculate recv count (sum of what this rank receives from all others)
  for (int r = 0; r < nranks; r++) {
    size_t traffic = AlltoAllvGetTrafficCount(r, rank, nranks);
    *recvcount += traffic;
    DEBUG_PRINT_ALL(rank, "Recv from rank %d: %zu elements", r, traffic);
  }

  DEBUG_PRINT_ALL(rank, "Buffer sizes - Send: %zu, Recv: %zu", *sendcount, *recvcount);

// XXX double check paramcount is right for alltoallv
  *paramcount = (count/nranks) & -(16/eltSize);
  *sendInplaceOffset = 0;
  *recvInplaceOffset = 0;
  fprintf(stderr, "SDS     AlltoAllvGetCollByteCount: count=%zu, eltSize=%zu, nranks=%d, paramcount=%zu, sendcount=%zu, recvcount=%zu\n", count, eltSize, nranks, *paramcount, *sendcount, *recvcount);
  exit(1);
}

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
  double baseBw = (double)(count * nranks * typesize) / 1.0E9 / sec;

  *algBw = baseBw;
  double factor = ((double)(nranks-1))/((double)(nranks));
  *busBw = baseBw * factor;
}

testResult_t AlltoAllvRunColl(void* sendbuff, void* recvbuff, size_t count, ncclDataType_t type, ncclRedOp_t op, int root, ncclComm_t comm, cudaStream_t stream) {
  int nRanks;
  NCCLCHECK(ncclCommCount(comm, &nRanks));
  size_t rankOffset = count * wordSize(type);

#if NCCL_MAJOR < 2 || NCCL_MINOR < 7
  printf("NCCL 2.7 or later is needed for alltoall. This test was compiled with %d.%d.\n", NCCL_MAJOR, NCCL_MINOR);
  return testNcclError;
#else
  NCCLCHECK(ncclGroupStart());
  for (int r=0; r<nRanks; r++) {
    NCCLCHECK(ncclSend(((char*)sendbuff)+r*rankOffset, count, type, r, comm, stream));
    NCCLCHECK(ncclRecv(((char*)recvbuff)+r*rankOffset, count, type, r, comm, stream));
  }
  NCCLCHECK(ncclGroupEnd());
  return testSuccess;
#endif
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
    TESTCHECK(TimeTest(args, run_types[i], run_typenames[i], (ncclRedOp_t)0, "none", -1));
  }
  return testSuccess;
}

struct testEngine alltoAllvEngine = {
  AlltoAllvGetBuffSize,
  AlltoAllvRunTest
};

#pragma weak ncclTestEngine=alltoAllvEngine
