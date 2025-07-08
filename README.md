# Linkping
Nvlink/RDMA bandwidth test


g++ -std=c++17 -Icore/csrc -o linkping core/launcher.cpp core/csrc/server.cpp core/csrc/client.cpp core/csrc/utils.cpp core/csrc/socket.cpp