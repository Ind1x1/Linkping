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

#include "core/csrc/server.h"
#include "core/csrc/socket.h"
#include "core/csrc/utils.h"


#include <iostream>
#include <getopt.h>
#include <cstring>
#include <sys/time.h>
#include <signal.h>
#include <memory>
#include <unistd.h>

int debug;
int debug_fast_path;

static volatile int keep_running = 1;

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
    
    //FIXME:
    return 0;
}