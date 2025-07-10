CXX = g++
NVCC = nvcc
CXXFLAGS = -std=c++17 -Icore/csrc -Icore -I/usr/local/cuda/include -Wall -02
NVCCFLAGS = -std=c++17 -Icore/csrc -Icore -I/usr/local/cuda/include -02
LDFLAGS = -L/usr/local/cuda/lib64 -lnccl -lcudart -lpthread

SRCS = core/launcher.cpp core/csrc/server.cpp core/csrc/client.cpp core/csrc/utils.cpp core/csrc/socket.cpp core/cuda/comm.cu
OBJS = core/launcher.o core/csrc/server.o core/csrc/client.o core/csrc/utils.o core/csrc/socket.o core/cuda/comm.o

linkping: $(OBJS)
	$(CXX) $(CXXFLAGS) -o $@ $(OBJS) $(LDFLAGS)

%.o: %.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

%.o: %.cu
	nvcc $(CXXFLAGS) -c $< -o $@

clean:
	rm -rf core/launcher.o core/csrc/server.o core/csrc/client.o core/csrc/utils.o core/csrc/socket.o core/cuda/comm.o linkping

.PHONY: clean