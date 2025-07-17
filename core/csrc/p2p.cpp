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

#include "socket.h"
#include "utils.h"
#include "p2p.h"

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

void P2P::usage(const char *argv0){
    std::cout << "Usage: \n"
              << " start linkping test\n"
              << "Options: \n"
              << " -s, --size=SIZE    Set the size of the message to send (default: 10000000)\n"
              << " -n, --iters=ITERS  Set the number of iterations to run (default: 10)\n"
              << " -t, --type=TYPE    Set the type of the message to send (default: float)\n"
              << " -r, --srcRank=SRC_RANK    Set the source rank (default: 0)\n"
              << " -d, --dstRank=DST_RANK    Set the destination rank (default: 1)\n"
              << std::endl;
}

int P2P::parse_command_line(int argc, char *argv[], user_params &usr_par)
{
    while (1) {
        int c;
        static struct option long_options[] = {
            { "size",          1, nullptr, 's' },
            { "iters",         1, nullptr, 'n' },
            { "type",          1, nullptr, 't' },
            { "keep_running",  0, nullptr, 'k' },
            { "srcRank",       1, nullptr, 'r' },
            { "dstRank",       1, nullptr, 'd' },
            { 0 }
        };
        c = getopt_long(argc, argv, "s:n:t:r:d:", long_options, nullptr);
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
        case 'r':
            usr_par.srcRank = strtol(optarg, nullptr, 0);
            break;
        case 'd':
            usr_par.dstRank = strtol(optarg, nullptr, 0);
            break;
        case 'k':
            usr_par.keep_running = true;
            break;
        }
    }
    return 0;
}  



int P2P::main(int argc, char *argv[]) {
    int                ret_val = 0;
    P2P::user_params   usr_par;

    ret_val = parse_command_line(argc, argv, usr_par);
    if (ret_val){
        return ret_val;
    }

    int deviceCount;
    CUDACHECK(cudaGetDeviceCount(&deviceCount));

    if (usr_par.srcRank >= deviceCount || usr_par.dstRank >= deviceCount) {
        std::cerr << "Error: Device index out of range. Available devices: 0-" << (deviceCount-1) << std::endl;
        return -1;
    }

    CUDACHECK(cudaSetDevice(usr_par.srcRank));
    int canAccessPeer = 0;
    cudaDeviceCanAccessPeer(&canAccessPeer, usr_par.srcRank, usr_par.dstRank);
    if (canAccessPeer) {
        CUDACHECK(cudaSetDevice(usr_par.srcRank));
        CUDACHECK(cudaDeviceEnablePeerAccess(usr_par.dstRank, 0));
    }
    std::cout << std::string(100, '-') << std::endl;
    std::cout << "srcRank: " << usr_par.srcRank << ", dstRank: " << usr_par.dstRank << std::endl;
    std::cout << "Size: " << usr_par.size << std::endl;
    std::cout << "iters: " << (usr_par.keep_running ? "Loop" : std::to_string(usr_par.iters))
          << std::endl;
    std::cout << "type: " << usr_par.type << std::endl;
    std::cout << "P2P can Access: "
          << (canAccessPeer ? "Nvlink" : "PCIe")
          << std::endl;
    std::cout << std::string(100, '-') << std::endl;

    cudaStream_t s;
    CUDACHECK(cudaSetDevice(usr_par.srcRank));
    float* send_ptr;
    // 确保size是4的倍数
    if (usr_par.size % 4 != 0) {
        usr_par.size = (usr_par.size / 4) * 4;
        std::cout << "Warning: Size adjusted to " << usr_par.size << " to ensure it's a multiple of 4" << std::endl;
    }
    CUDACHECK(cudaMalloc(&send_ptr, usr_par.size * sizeof(float)));
    float* recv_ptr;
    CUDACHECK(cudaSetDevice(usr_par.dstRank));
    CUDACHECK(cudaMalloc(&recv_ptr, usr_par.size * sizeof(float)));
    CUDACHECK(cudaSetDevice(usr_par.srcRank));
    CUDACHECK(cudaStreamCreate(&s));
    InitData(send_ptr, usr_par.size, ncclFloat, s);
    std::cout << std::setw(20) << std::left << "#bytes" 
              << std::setw(20) << std::left << "#iterations" 
              << std::setw(20) << std::left << "#overhead(ms)" 
              << std::setw(20) << std::left << "#bandwidth(GB/s)" 
              << std::setw(20) << std::left << "#Total(GB)" << std::endl;
    cudaEvent_t start, stop;
    CUDACHECK(cudaEventCreate(&start));
    CUDACHECK(cudaEventCreate(&stop));
    
    size_t total_uint4 = usr_par.size / 4;
    size_t chunkSizeInElement = total_uint4 / (96*512);

    int total_GB = 0;
    if (usr_par.keep_running) {
        int i = 1;
        while(1) {
        float elapsed_time = 0.0f;
        //elapsed_time = Launch_linkpingp2p_simple(recv_ptr, send_ptr, usr_par.size, s);
        elapsed_time = Launch_stridingMemcpy(reinterpret_cast<uint4*>(recv_ptr), reinterpret_cast<uint4*>(send_ptr), chunkSizeInElement, s);
        double total_bytes = static_cast<double>(usr_par.size) * sizeof(float);
        double bandwidth = total_bytes / (elapsed_time / 1000.0) / 1e9; // GB/s
        total_GB += total_bytes / 1e9;
        std::cout << std::setw(20) << std::left << std::scientific << (usr_par.size * sizeof(float)) << std::fixed
                  << std::setw(20) << std::left << (i+1)
                  << std::setw(20) << std::left << std::fixed << std::setprecision(2) << elapsed_time
                  << std::setw(20) << std::left << std::fixed << std::setprecision(4) << bandwidth 
                  << std::setw(20) << std::left << (total_GB) << std::endl;
        i++;
        }
    }

    for (int i = 0; i < usr_par.iters; i++) {
        float elapsed_time = 0.0f;
        //elapsed_time = Launch_linkpingp2p_simple(recv_ptr, send_ptr, usr_par.size, s);
        elapsed_time = Launch_stridingMemcpy(reinterpret_cast<uint4*>(recv_ptr), reinterpret_cast<uint4*>(send_ptr), chunkSizeInElement, s);
        double total_bytes = static_cast<double>(usr_par.size) * sizeof(float);
        double bandwidth = total_bytes / (elapsed_time / 1000.0) / 1e9; // GB/s
        total_GB += total_bytes / 1e9;
        std::cout << std::setw(20) << std::left << std::scientific << (usr_par.size * sizeof(float)) << std::fixed
                  << std::setw(20) << std::left << (i+1)
                  << std::setw(20) << std::left << std::fixed << std::setprecision(2) << elapsed_time
                  << std::setw(20) << std::left << std::fixed << std::setprecision(4) << bandwidth 
                  << std::setw(20) << std::left << (total_GB) << std::endl;
    }
    
    CUDACHECK(cudaEventDestroy(start));
    CUDACHECK(cudaEventDestroy(stop));
    CUDACHECK(cudaFree(send_ptr));
    CUDACHECK(cudaFree(recv_ptr));
    CUDACHECK(cudaStreamDestroy(s));
    std::cout << "P2P test completed" << std::endl;
    return 0;
}