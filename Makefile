CXX = g++
NVCC = nvcc
CXXFLAGS = -std=c++17 -Icore/csrc -Icore -I/usr/local/cuda/include -Wall -O2
NVCCFLAGS = -std=c++17 -Icore/csrc -Icore -I/usr/local/cuda/include -O2
LDFLAGS = -L/usr/local/cuda/lib64 -lnccl -lcudart -lpthread

SRCS = core/launcher.cpp core/csrc/server.cpp core/csrc/client.cpp core/csrc/utils.cpp core/csrc/socket.cpp core/cuda/comm.cu core/csrc/single.cpp core/csrc/p2p.cpp 
OBJS = core/launcher.o core/csrc/server.o core/csrc/client.o core/csrc/utils.o core/csrc/socket.o core/cuda/comm.o core/csrc/single.o core/csrc/p2p.o 

linkping: $(OBJS)
	$(CXX) $(CXXFLAGS) -o $@ $(OBJS) $(LDFLAGS)

%.o: %.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

%.o: %.cu
	$(NVCC) $(NVCCFLAGS) -c $< -o $@

clean:
	rm -rf core/launcher.o core/csrc/server.o core/csrc/client.o core/csrc/utils.o core/csrc/socket.o core/cuda/comm.o core/csrc/single.o core/csrc/p2p.o core/csrc/p2p_ll_test.o linkping

.PHONY: clean