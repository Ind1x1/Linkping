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

#include "core/csrc/client.h"
#include "core/csrc/socket.h"
#include "core/csrc/utils.h"


#include <iostream>
#include <getopt.h>
#include <cstring>
#include <sys/time.h>
#include <signal.h>
#include <memory>
#include <unistd.h>


void Client::usage(const char *argv0){
    std::cout << "Usage: \n"
              << " start linkping test\n"
              << "Options: \n"
              << " -p, --port=PORT    Set the port to listen on (default: 18515)\n"
              << " -s, --size=SIZE    Set the size of the message to send (default: 4096)\n"
              << " -n, --iters=ITERS  Set the number of iterations to run (default: 1000)\n"
              << " -a, --addr=ADDR    Set the address to connect to (default: 127.0.0.1)\n"
              << std::endl;
}

int Client::parse_command_line(int argc, char *argv[], user_params &usr_par)
{
    while (1) {
        int c;
        static struct option long_options[] = {
            { "port",          1, nullptr, 'p' },
            { "size",          1, nullptr, 's' },
            { "iters",         1, nullptr, 'n' },
            { "addr",          1, nullptr, 'a' },
            { 0 }
        };
        c = getopt_long(argc, argv, "p:s:n:a:", long_options, nullptr);
        if (c == -1)
            break;
        switch (c) {
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
        case 'a':
            utils::get_addr(std::string(optarg), (struct sockaddr *) &usr_par.hostaddr);
            break;
        }
    }
    if (optind == argc) {
        std::cerr << "Error: missing server address\n";
        usage(argv[0]);
        return 1;
    } else if (optind == argc - 1) {
        usr_par.servername = argv[optind];
    }
    else if (optind < argc) {
        usage(argv[0]);
        return 1;
    }

    return 0;
}


int Client::main(int argc, char *argv[]) {

    int                      ret_val = 0;
    Client::user_params       usr_par;

    ret_val = parse_command_line(argc, argv, usr_par);
    if (ret_val){
        return ret_val;
    }

    if (!usr_par.hostaddr.sa_family) {
        std::cout << "Error: host ip is missing in the command line. " << std::endl;
        usage(argv[0]);
        ret_val = 1;
        return ret_val;
    }

    std::cout << "Connecting to server " << usr_par.servername << " on port " << usr_par.port << std::endl;
    auto sockfd_deleter = [](int* fd) { if (fd && *fd >= 0) close(*fd); };
    std::unique_ptr<int, decltype(sockfd_deleter)> sockfd_ptr(new int(-1), sockfd_deleter);

    *sockfd_ptr = linkping::Socket::openClientSocket(usr_par.servername, usr_par.port);
    if (*sockfd_ptr < 0) {
        std::cerr << "Failed to connect to server.\n";
        return 1;
    }
    //FIXME: 

    return 0;
}