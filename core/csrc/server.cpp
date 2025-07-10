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

#include "server.h"
#include "socket.h"
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

int debug;
int debug_fast_path;

static volatile int keep_running = 1;

static ncclUniqueId ncclId;
static int          device_count = DEFAULT_DEVICES_NUM;

void sigint_handler(int dummy)
{
    keep_running = 0;
}

void Server::usage(const char *argv0){
    std::cout << "Usage: \n"
              << " start linkping test\n"
              << "Options: \n"
              << " -p, --port=PORT    Set the port to listen on (default: 18515)\n"
              << " -s, --size=SIZE    Set the size of the message to send (default: 4096)\n"
              << " -n, --iters=ITERS  Set the number of iterations to run (default: 1000)\n"
              << " -a, --addr=ADDR    Set the address to connect to (default: 127.0.0.1)\n"
              << std::endl;
}

int Server::parse_command_line(int argc, char *argv[], user_params &usr_par)
{
    while (1) {
        int c;
        static struct option long_options[] = {
            { "persistent",    0, nullptr, 'P' },
            { "addr",          1, nullptr, 'a' },
            { "port",          1, nullptr, 'p' },
            { "size",          1, nullptr, 's' },
            { "iters",         1, nullptr, 'n' },
            { "sg_list-len",   1, nullptr, 'l' },
            { "debug-mask",    1, nullptr, 'D' },
            { "rd",            0, nullptr, 'R' },
            { "nv",            0, nullptr, 'N' },
            { 0 }
        };
        c = getopt_long(argc, argv, "Pra:p:s:n:l:D:RN", long_options, nullptr);
        if (c == -1)
            break;
        switch (c) {
        case 'P':
            usr_par.persistent = 1;
            break;
        case 'a':
            utils::get_addr(std::string(optarg), (struct sockaddr *) &usr_par.hostaddr);
            break;
        case 'p':
            usr_par.port = strtol(optarg, nullptr, 0);
            if (usr_par.port < 0 || usr_par.port > 65535) {
                usage(argv[0]);
                return 1;
            }
            break;
        case 's':
            usr_par.size = strtol(optarg, nullptr, 0);
            break;
        case 'n':
            usr_par.iters = strtol(optarg, nullptr, 0);
            break;
        case 'l':
            usr_par.num_sges = strtol(optarg, nullptr, 0);
            break;
        case 'D':
            debug           = (strtol(optarg, nullptr, 0) >> 0) & 1; /*bit 0*/
            debug_fast_path = (strtol(optarg, nullptr, 0) >> 1) & 1; /*bit 1*/
            break;
        case 'R':
            usr_par.rdma = 1;
            break;
        case 'N':
            usr_par.nvlink = 1;
            break;
        default:
            usage(argv[0]);
            return 1;
        }
    }
    if (optind < argc) {
        usage(argv[0]);
        return 1;
    }
    return 0;
}

void* Server::thread_main(void* arg) {
    ThreadArgs* args = static_cast<ThreadArgs*>(arg);
    int rank = args->rank;
    int device_count = args->device_count;
    int persistent = args->persistent;
    int port = args->port;
    unsigned long size = args->size;
    int iters = args->iters;
    int num_sges = args->num_sges;
    int rdma = args->rdma;
    int nvlink = args->nvlink;
    ncclUniqueId nccl_id = args->ncclId;

    CUDACHECK(cudaSetDevice(rank));
    ncclComm_t comm;
    NCCLCHECK(ncclCommInitRank(&comm, device_count * 2, nccl_id, rank));
    
    float *send_ptr;
    float *recv_ptr;
    cudaStream_t s;

    CUDACHECK(cudaMalloc(&send_ptr, 10000 * sizeof(float)));
    CUDACHECK(cudaMalloc(&recv_ptr, 10000 * sizeof(float)));
    CUDACHECK(cudaStreamCreate(&s));

    InitData(send_ptr, 10000 * sizeof(float), ncclFloat, s);
    NCCLCHECK(ncclAllReduce(send_ptr, recv_ptr, 10000, ncclFloat, ncclSum, comm, s));

    std::cout << "AllReduce done." << std::endl;

    CUDACHECK(cudaStreamSynchronize(s));
    CUDACHECK(cudaFree(send_ptr));
    CUDACHECK(cudaFree(recv_ptr));
    CUDACHECK(cudaStreamDestroy(s));
    NCCLCHECK(ncclCommDestroy(comm));
    return nullptr;
}

int Server::main(int argc, char *argv[]) {

    struct timeval           start;
    int                      ret_val = 0;
    Server::user_params       usr_par;

    ret_val = parse_command_line(argc, argv, usr_par);
    if (ret_val){
        return ret_val;
    }
    
    struct sigaction act;
    act.sa_handler = sigint_handler;
    sigaction(SIGINT, &act, NULL);
    
    auto sockfd_deleter = [](int* fd) { if (fd && *fd >= 0) close(*fd); };
    std::unique_ptr<int, decltype(sockfd_deleter)> sockfd_ptr(new int(-1), sockfd_deleter);
    
    std::cout << "Listening to remote client on port " << usr_par.port << "..." << std::endl;
    *sockfd_ptr = linkping::Socket::openServerSocket(usr_par.port);
    if (*sockfd_ptr < 0) {
        std::cerr << "Failed to open server socket.\n";
        return 1;
    }
    std::cout << "Connection accepted." << std::endl;

    if (gettimeofday(&start, nullptr)) {
        std::cerr << "gettimeofday failed: " << strerror(errno) << "!" << std::endl;
        return 1; 
    }
    
    NCCLCHECK(ncclGetUniqueId(&ncclId));
    ssize_t send_size = send(*sockfd_ptr, &ncclId.internal, sizeof(ncclId.internal), 0);
    if (send_size < 0) {
        std::cerr << "Failed to send nccl uniqueID to client: "  << std::endl;
        return 1;
    }
    std::cout << "NCCL ID sent to client successfully." << std::endl;
    
    NCCLCHECK(ncclGetDeviceCount(&device_count));
    pthread_t threads[device_count];

    ThreadArgs thread_args[device_count];
    for (int i = 0; i < device_count; i++) {
        thread_args[i].persistent = usr_par.persistent;
        thread_args[i].port = usr_par.port;
        thread_args[i].size = usr_par.size;
        thread_args[i].iters = usr_par.iters;
        thread_args[i].num_sges = usr_par.num_sges;
        thread_args[i].rdma = usr_par.rdma;
        thread_args[i].nvlink = usr_par.nvlink;
        thread_args[i].rank = i;
        thread_args[i].device_count = device_count;
        thread_args[i].ncclId = ncclId;
        
        ret_val = pthread_create(&threads[i], NULL, thread_main, &thread_args[i]);
        if (ret_val) {
            std::cerr << "Failed to create thread " << i << std::endl;
            return 1;
        }
    }

    for (int i = 0; i < device_count; i++) {
        pthread_join(threads[i], NULL);
    }

    std::cout << "All threads joined." << std::endl;
    //FIXME:

    return 0;
}