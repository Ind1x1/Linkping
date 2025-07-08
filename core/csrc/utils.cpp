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

#include "core/csrc/utils.h"

#include <iostream>
#include <cstring>
#include <netdb.h>
#include <sys/types.h>
#include <sys/socket.h>
#include <sys/time.h>
#include <unistd.h>

namespace utils {
     
bool get_addr(const std::string& dst, struct sockaddr* addr) {
    struct addrinfo hints = {};
    struct addrinfo* res = nullptr;
    int ret;

    hints.ai_family = AF_UNSPEC;  
    hints.ai_socktype = SOCK_STREAM; 
    hints.ai_flags = AI_PASSIVE;     

    ret = getaddrinfo(dst.c_str(), nullptr, &hints, &res);
    if (ret != 0) {
        std::cerr << "getaddrinfo failed (" << gai_strerror(ret) 
                  << ") - invalid hostname or IP address: " << dst << std::endl;
        return false;
    }

    if (res->ai_family == AF_INET) {
        std::memcpy(addr, res->ai_addr, sizeof(struct sockaddr_in));
    } else if (res->ai_family == AF_INET6) {
        std::memcpy(addr, res->ai_addr, sizeof(struct sockaddr_in6));
    } else {
        std::cerr << "Unsupported address family: " << res->ai_family << std::endl;
        freeaddrinfo(res);
        return false;
    }

    freeaddrinfo(res);
    return true;
}

bool print_run_time(const struct timeval& start, unsigned long size, int iters)
{
    struct timeval end;
    double usec;
    long long bytes;

    if (gettimeofday(&end, nullptr) != 0) {
        std::perror("gettimeofday");
        return false;
    }

    usec = (end.tv_sec - start.tv_sec) * 1000000.0 + (end.tv_usec - start.tv_usec);
    
    bytes = static_cast<long long>(size) * iters;

    std::cout << bytes << " bytes in " << (usec / 1000000.0) << " seconds = "
              << (bytes * 8.0 / usec) << " Mbit/sec" << std::endl;
    
    std::cout << iters << " iters in " << (usec / 1000000.0) << " seconds = "
              << (usec / iters) << " usec/iter" << std::endl;

    return true;
}

}