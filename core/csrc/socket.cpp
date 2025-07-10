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

#include <iostream>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <unistd.h>
#include <sys/types.h>
#include <sys/socket.h>
#include <netdb.h>
#include <arpa/inet.h>
#include <stdexcept>
#include <memory>

namespace linkping {

int Socket::openServerSocket(int port)
{
    struct addrinfo *res, *t;
    struct addrinfo hints = {
        .ai_flags    = AI_PASSIVE,
        .ai_family   = AF_UNSPEC,
        .ai_socktype = SOCK_STREAM
    };
    
    char* service = nullptr;
    int ret_val;
    int sockfd;
    int tmp_sockfd = -1;

    ret_val = asprintf(&service, "%d", port);
    if (ret_val < 0) {
        std::cerr << "Failed to allocate memory for service string" << std::endl;
        return -1;
    }
    std::unique_ptr<char, decltype(&free)> service_guard(service, free);

    ret_val = getaddrinfo(nullptr, service, &hints, &res);
    if (ret_val < 0) {
        std::cerr << gai_strerror(ret_val) << " for port " << port << std::endl;
        return -1;
    }

    std::unique_ptr<addrinfo, decltype(&freeaddrinfo)> res_guard(res, freeaddrinfo);

    for (t = res; t; t = t->ai_next) {
        tmp_sockfd = socket(t->ai_family, t->ai_socktype, t->ai_protocol);
        if (tmp_sockfd >= 0) {
            int optval = 1;

            setsockopt(tmp_sockfd, SOL_SOCKET, SO_REUSEADDR, &optval, sizeof(optval));

            if (!bind(tmp_sockfd, t->ai_addr, t->ai_addrlen))
                break;
            close(tmp_sockfd);
            tmp_sockfd = -1;
        }
    }

    if (tmp_sockfd < 0) {
        std::cerr << "Couldn't listen to port " << port << std::endl;
        return -1;
    }

    if (listen(tmp_sockfd, 1) < 0) {
        std::cerr << "listen() failed" << std::endl;
        close(tmp_sockfd);
        return -1;
    }

    sockfd = accept(tmp_sockfd, nullptr, nullptr);
    close(tmp_sockfd);
    if (sockfd < 0) {
        std::cerr << "accept() failed" << std::endl;
        return -1;
    } 
    
    return sockfd;
}

int Socket::openClientSocket(const std::string& host, int port)
{
    struct addrinfo *res, *t;
    struct addrinfo hints = {
        .ai_flags    = 0,
        .ai_family   = AF_UNSPEC,
        .ai_socktype = SOCK_STREAM
    };
    
    char* service = nullptr;
    int ret_val;
    int sockfd = -1;

    ret_val = asprintf(&service, "%d", port);
    if (ret_val < 0) {
        std::cerr << "Failed to allocate memory for service string" << std::endl;
        return -1;
    }
    std::unique_ptr<char, decltype(&free)> service_guard(service, free);

    ret_val = getaddrinfo(host.c_str(), service, &hints, &res);
    if (ret_val < 0) {
        std::cerr << gai_strerror(ret_val) << " for host " << host << ":" << port << std::endl;
        return -1;
    }

    std::unique_ptr<addrinfo, decltype(&freeaddrinfo)> res_guard(res, freeaddrinfo);

    for (t = res; t; t = t->ai_next) {
        sockfd = socket(t->ai_family, t->ai_socktype, t->ai_protocol);
        if (sockfd >= 0) {
            if (!connect(sockfd, t->ai_addr, t->ai_addrlen))
                break;
            close(sockfd);
            sockfd = -1;
        }
    }

    if (sockfd < 0) {
        std::cerr << "Couldn't connect to " << host << ":" << port << std::endl;
        return -1;
    }

    return sockfd;
}

void Socket::closeSocket(int sockfd)
{
    if (sockfd >= 0) {
        close(sockfd);
    }
}

} // namespace linkping
