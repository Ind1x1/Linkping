#pragma once
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
#include <netdb.h>
#include <string>

class P2P {
    public:
        struct user_params {
            unsigned long         size = 10000000;
            int                   iters = 10;
            std::string           type = "float";
            bool                  keep_running = false;
            int                   srcRank = 0;
            int                   dstRank = 1;
        };
        /**
         * @brief main
         * 
         * @param argc 
         * @param argv 
         * @return int 
         */
        static int main(int argc, char *argv[]);
        /**
         * @brief usage
         * 
         * @param argv0 
         */
        static void usage(const char *argv0);
        /**
         * @brief parse_command_line
         */
        static int parse_command_line(int argc, char *argv[], user_params &usr_par);
};