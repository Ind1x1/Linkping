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
#include "server.h"
#include "client.h"

#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <system_error>
#include <unistd.h>
#include <string.h>
#include <signal.h>
#include <sys/types.h>
#include <sys/socket.h>
#include <sys/time.h>
#include <netdb.h>
#include <malloc.h>
#include <getopt.h>
#include <arpa/inet.h>
#include <time.h>
#include <vector>

#define LINKPING_VERSION "0.1.0"

int main(int argc, char *argv[])
{
    if (argc < 2) {
        Server::usage(argv[0]);
        return 1;
    }

    std::string mode = argv[1];
    if (mode != "server" && mode != "client"){
        std::cerr << "Error: Ineffective working mode\n";
        Server::usage(argv[0]);
        return 1;
    }

    std::cout << "Linkping version " << LINKPING_VERSION << std::endl;
    if (mode == "server"){
        std::cout << "Linkping started in server mode." << std::endl;
        std::cout << "\n" << std::endl;
        return Server::main(argc - 1, argv + 1);
    } else if (mode == "client"){
        std::cout << "Linkping started in client mode." << std::endl;
        std::cout << "\n" << std::endl;
        return Client::main(argc - 1, argv + 1);
    } else if (mode == "single"){
        std::cout << "Linkping started in single mode." << std::endl;
        std::cout << "\n" << std::endl;
        return Single::main(argc - 1, argv + 1);
    }
}