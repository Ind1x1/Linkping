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
#ifndef SOCKET_H
#define SOCKET_H

#include <string>

namespace linkping {

/**
 * @brief Socket 
 * 
 */
class Socket {
public:
    /**
     * 打开服务器socket并等待连接
     * @param port 要监听的端口号
     * @return 成功返回已连接的socket文件描述符，失败返回-1
     */
    static int openServerSocket(int port);
    
    /**
     * 打开客户端socket并连接到服务器
     * @param host 服务器主机名或IP地址
     * @param port 服务器端口号
     * @return 成功返回已连接的socket文件描述符，失败返回-1
     */
    static int openClientSocket(const std::string& host, int port);
    
    /**
     * 关闭socket连接
     * @param sockfd socket文件描述符
     */
    static void closeSocket(int sockfd);
};

} // namespace linkping

#endif // SOCKET_H
