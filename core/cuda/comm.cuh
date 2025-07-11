#pragma once
/*
Copyright 2025
Linkping Alibaba Cloude Air (alpha) 2025 Leyi Ye
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/
#ifndef COMM_CUH
#define COMM_CUH

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <nccl.h>
#include <stdio.h>
#include <pthread.h>
#include <functional>

#define CUDACHECK(cmd)                                                                              \
    do {                                                                                            \
        cudaError_t err = cmd;                                                                      \
        if (err != cudaSuccess) {                                                                   \
            printf("Failed: Cuda error %s:%d '%s'\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
            exit(EXIT_FAILURE);                                                                     \
        }                                                                                           \
    } while (0)

#define NCCLCHECK(cmd)                                                                              \
    do {                                                                                            \
        ncclResult_t res = cmd;                                                                     \
        if (res != ncclSuccess) {                                                                   \
            printf("Failed, NCCL error %s:%d '%s'\n", __FILE__, __LINE__, ncclGetErrorString(res)); \
            exit(EXIT_FAILURE);                                                                     \
        }                                                                                           \
    } while (0)

typedef enum {
    testSuccess = 0,
    testError = 1,
} testResult_t;

#define TESTCHECK(cmd)                                                                              \
    do {                                                                                            \
        testResult_t res = cmd;                                                                     \
        if (res != testSuccess) {                                                                   \
            printf("Failed, Test error %s:%d '%s'\n", __FILE__, __LINE__, ncclGetErrorString(res)); \
            exit(EXIT_FAILURE);                                                                     \
        }                                                                                           \
    } while (0)

#define DEFAULT_DEVICES_NUM 8

class LinkPingTimer {
public:
    static void TimerProfile(const char* op_name, std::function<void()> func, cudaStream_t stream);
    static void Warmup(const char* op_name, std::function<void()> func, cudaStream_t stream, int warmup_iters);
};

//LINKPING_TIMER("ncclAllReduce", 
//               NCCLCHECK(ncclAllReduce(send_ptr, recv_ptr, 10000, ncclFloat, ncclSum, comm, s)); 
//               CUDACHECK(cudaStreamSynchronize(s)), s);
#define LINKPING_TIMER(name, code_block, stream)                                                    \
    LinkPingTimer::TimerProfile(name, [&]() { code_block; }, stream)

// warmup 5 次
// LINKPING_WARMUP("ncclAllReduce", 
//     NCCLCHECK(ncclAllReduce(send_ptr, recv_ptr, usr_par.size, ncclFloat, ncclSum, comm, s)), s, 5);
#define LINKPING_WARMUP(name, code_block, stream, warmup_iters)                                     \
    LinkPingTimer::Warmup(name, [&]() { code_block; }, stream, warmup_iters)

template<typename T>
__global__ void InitDataKernel(T*data, size_t size);

extern void InitData(void* data_ptr, size_t size, ncclDataType_t type, cudaStream_t stream);

extern void AllReduceGetBw(size_t count, int typesize, double sec, double* algBw, double* busBw, int nranks);

#endif // COMM_CUH