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

static int          device_count = DEFAULT_DEVICES_NUM;

void Client::Barrier(ThreadArgs* args) {
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

void*  Singlep2p::thread_main(void* arg) {
    ThreadArgs* args = static_cast<ThreadArgs*>(arg);
    //TODO:
    Singlep2p::user_params usr_par = args->usr_par;
    int rank = args->rank; 
    bool keep_running = usr_par.keep_running;
    int dst_rank = (rank + 1) % device_count;

    float *recv_ptr;
    CUDACHECK(cudaSetDevice(dst_rank));
    CUDACHECK(cudaMalloc(&recv_ptr, usr_par.size * sizeof(float)));

    CUDACHECK(cudaSetDevice(rank));
    cudaStream_t s;
    float *send_ptr;
    CUDACHECK(cudaStreamCreate(&s));
    CUDACHECK(cudaMalloc(&send_ptr, usr_par.size * sizeof(float)));

    InitData(send_ptr, usr_par.size * sizeof(float), ncclFloat, s);

    CUDACHECK(cudaStreamSynchronize(s));
    Barrier(args);
    
    CUDACHECK(cudaStreamSynchronize(s));
    Barrier(args);

    if (keep_running){
        LINKPING_P2P("P2P", Launch_linkpingp2p(send_ptr, recv_ptr, usr_par.size, s), s, usr_par.size, sizeof(float));
        cudaStreamSynchronize(s);
        Barrier(args);
        if (rank == 0){
            printf("--------------------------------\n");
        }
        Barrier(args);
    }

    CUDACHECK(cudaStreamSynchronize(s));
    CUDACHECK(cudaFree(send_ptr));
    CUDACHECK(cudaFree(recv_ptr));
    CUDACHECK(cudaStreamDestroy(s));
    return nullptr;
}

int Singlep2p::main(int argc, char *argv[]) {

    int                      ret_val = 0;
    Singlep2p::user_params       usr_par;

    ret_val = parse_command_line(argc, argv, usr_par);
    if (ret_val){
        return ret_val;
    }

    CUDACHECK(cudaGetDeviceCount(&device_count));
    pthread_t threads[device_count];

    ThreadArgs thread_args[device_count];
    for (int i = 0; i < device_count; i++) {
        thread_args[i].usr_par = usr_par;   
        thread_args[i].thread = i;
        thread_args[i].nThreads = device_count;
        thread_args[i].rank = i;
        thread_args[i].device_count = device_count;

        ret_val = pthread_create(&threads[i], NULL, thread_main, &thread_args[i]);
        if (ret_val) {
            std::cerr << "Failed to create thread " << i << std::endl;
            return 1;
        }
    }

    for (int i = 0; i < device_count; i++) {
        pthread_join(threads[i], NULL);
    }

    std::cout << "All threads joined. " << std::endl;
    //FIXME:

    return 0;
}