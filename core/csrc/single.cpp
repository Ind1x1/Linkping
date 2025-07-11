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
#include "single.h"
#include "utils.h"

#include "../cuda/comm.cuh"

#include <iostream>
#include <getopt.h>
#include <cstring>
#include <sys/time.h>
#include <signal.h>
#include <memory>
#include <unistd.h>
#include <cuda_runtime.h>
#include <iomanip>
#include <pthread.h>

#define MAX_GPU 16
static double bandwidth_matrix[MAX_GPU][MAX_GPU] = {0};

static int          device_count = DEFAULT_DEVICES_NUM;

void Singlep2p::Barrier(ThreadArgs* args) {
    thread_local int epoch = 0;
    static pthread_mutex_t lock[2] = {PTHREAD_MUTEX_INITIALIZER, PTHREAD_MUTEX_INITIALIZER};
    static pthread_cond_t cond[2] = {PTHREAD_COND_INITIALIZER, PTHREAD_COND_INITIALIZER};
    static int counter[2] = {0, 0};
  
    pthread_mutex_lock(&lock[epoch]);
    if(++counter[epoch] == args->nThreads)
      pthread_cond_broadcast(&cond[epoch]);
  
    if(args->thread+1 == args->nThreads) {
      while(counter[epoch] != args->nThreads)
        pthread_cond_wait(&cond[epoch], &lock[epoch]);
      counter[epoch] = 0;
      pthread_cond_broadcast(&cond[epoch]);
    }
    else {
      while(counter[epoch] != 0)
        pthread_cond_wait(&cond[epoch], &lock[epoch]);
    }
    pthread_mutex_unlock(&lock[epoch]);
    epoch ^= 1;
}

void Singlep2p::usage(const char *argv0){
    std::cout << "Usage: \n"
              << " start linkping single p2p test\n"
              << "Options: \n"
              << " -s, --size=SIZE    Set the size of the message to send (default: 100000)\n"
              << " -n, --iters=ITERS  Set the number of iterations to run (default: 100)\n"
              << " -t, --type=TYPE    Set the type of the message to send (default: float)\n"
              << " -k, --keep-running Keep running the test\n"
              << std::endl;
}

int Singlep2p::parse_command_line(int argc, char *argv[], user_params &usr_par)
{
    while (1) {
        int c;
        static struct option long_options[] = {
            { "size",          1, nullptr, 's' },
            { "iters",         1, nullptr, 'n' },
            { "type",          1, nullptr, 't' },
            { "keep-running",  0, nullptr, 'k' },
            { 0 }
        };
        c = getopt_long(argc, argv, "s:n:t:k", long_options, nullptr);
        if (c == -1)
            break;
        switch (c) {
        case 's':
            usr_par.size = strtol(optarg, nullptr, 0);
            break;
        case 'n':
            usr_par.iters = strtol(optarg, nullptr, 0);
            break;
        case 't':
            usr_par.type = std::string(optarg);
            break;
        case 'k':
            usr_par.keep_running = true;
            break;
        }
    }
    return 0;
}

void* Singlep2p::thread_main(void* arg) {
    ThreadArgs* args = static_cast<ThreadArgs*>(arg);
    int rank = args->rank;
    int device_count = args->device_count;
    int size = args->usr_par.size;
    bool keep_running = args->usr_par.keep_running;
    float** send_ptrs = args->send_ptrs;
    float** recv_ptrs = args->recv_ptrs;
    int iters = args->usr_par.iters;
    cudaStream_t s;
    CUDACHECK(cudaSetDevice(rank));
    CUDACHECK(cudaStreamCreate(&s));
    cudaEvent_t start, end;
    CUDACHECK(cudaEventCreate(&start));
    CUDACHECK(cudaEventCreate(&end));
    printf("rank %d, device %d, stream %p, start %p, end %p\n", rank, rank, s, start, end);
    CUDACHECK(cudaStreamSynchronize(s));
    Barrier(args);
    
    do {
        int shift = 1;
        for (int round = 0; round < device_count - 1; ++round) {
            int dst = (rank + shift) % device_count;
            if (rank == dst) {
                continue;
            }
            Barrier(args);
            float* send_ptr = send_ptrs[rank];
            float* recv_ptr = recv_ptrs[dst];
            CUDACHECK(cudaStreamSynchronize(s));
            float total_time_ms = 0.0f;
            Barrier(args);
           // warmup
            for (int k = 0; k < WARMUP_ITERS; ++k) {
                CUDACHECK(cudaMemcpyPeerAsync(recv_ptr, dst, send_ptr, rank, size * sizeof(float), s));
            }
            CUDACHECK(cudaStreamSynchronize(s));
            for (int k = 0; k < iters; ++k) {
                CUDACHECK(cudaEventRecord(start, s));
                CUDACHECK(cudaMemcpyPeerAsync(recv_ptr, dst, send_ptr, rank, size * sizeof(float), s));
                CUDACHECK(cudaEventRecord(end, s));
                CUDACHECK(cudaEventSynchronize(end));
                float elapsed_ms = 0.0f;
                CUDACHECK(cudaEventElapsedTime(&elapsed_ms, start, end));
                total_time_ms += elapsed_ms;
                Barrier(args);
            }
            float avg_time_ms = total_time_ms / iters;
            double seconds = avg_time_ms / 1000.0;
            double bandwidth = (size * sizeof(float)) / seconds / 1e9;
            bandwidth_matrix[rank][dst] = bandwidth;
            Barrier(args);
            shift = (shift + 1);
        }
        Barrier(args);
        if (rank == 0) {
            printf("\nP2P Bandwidth Topo Matrix (GB/s):\n");
            printf("%6s", " ");
            for (int dst = 0; dst < device_count; ++dst) {
                printf("%10d", dst);
            }
            printf("\n");
            for (int src = 0; src < device_count; ++src) {
                printf("%4d |", src);
                for (int dst = 0; dst < device_count; ++dst) {
                    if (src == dst)
                        printf("%10s", "--");
                    else
                        printf("%10.2f", bandwidth_matrix[src][dst]);
                }
                printf("\n");
            }
            printf("\nTopo test finished.\n\n");
        }
        Barrier(args);
        if (!keep_running) break;
    } while (1);
    return nullptr;
}

int Singlep2p::main(int argc, char *argv[]) {
    int                      ret_val = 0;
    Singlep2p::user_params   usr_par;

    ret_val = parse_command_line(argc, argv, usr_par);
    if (ret_val){
        return ret_val;
    }

    CUDACHECK(cudaGetDeviceCount(&device_count));
    for (int i = 0; i < device_count; ++i) {
        for (int j = 0; j < device_count; ++j) {
            if (i != j) {
                int canAccessPeer = 0;
                cudaDeviceCanAccessPeer(&canAccessPeer, i, j);
                if (canAccessPeer) {
                    cudaSetDevice(i);
                    cudaDeviceEnablePeerAccess(j, 0);
                }
            }
        }
    }

    // 统一分配所有GPU的send/recv指针
    float* send_ptrs[MAX_GPU];
    float* recv_ptrs[MAX_GPU];
    for (int i = 0; i < device_count; ++i) {
        cudaSetDevice(i);
        cudaMalloc(&send_ptrs[i], usr_par.size * sizeof(float));
        cudaMalloc(&recv_ptrs[i], usr_par.size * sizeof(float));
        InitData(send_ptrs[i], usr_par.size * sizeof(float), ncclFloat, 0);
    }

    pthread_t threads[MAX_GPU];
    ThreadArgs thread_args[MAX_GPU];
    for (int i = 0; i < device_count; i++) {
        thread_args[i].usr_par = usr_par;
        thread_args[i].thread = i;
        thread_args[i].nThreads = device_count;
        thread_args[i].rank = i;
        thread_args[i].device_count = device_count;
        thread_args[i].send_ptrs = send_ptrs;
        thread_args[i].recv_ptrs = recv_ptrs;
        pthread_create(&threads[i], NULL, thread_main, &thread_args[i]);
    }
    for (int i = 0; i < device_count; i++) {
        pthread_join(threads[i], NULL);
    }
    for (int i = 0; i < device_count; ++i) {
        cudaSetDevice(i);
        cudaFree(send_ptrs[i]);
        cudaFree(recv_ptrs[i]);
    }
    return 0;
}