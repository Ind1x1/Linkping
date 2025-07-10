CXX = g++
CXXFLAGS = -std=c++17 -Icsrc -Wall -O2
LDFLAGS = -lnccl -lcudart

SRCS = launcher.cpp csrc/server.cpp csrc/client.cpp csrc/utils.cpp csrc/socket.cpp cuda/comm.cu
OBJS = $(SRCS:.cpp=.o) $(SRCS:.cu=.o)

linkping: $(OBJS)
	$(CXX) $(CXXFLAGS) -o $@ $(OBJS) $(LDFLAGS)

%.o: %.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

%.o: %.cu
	nvcc $(CXXFLAGS) -c $< -o $@

clean:
	rm -f launcher.o csrc/server.o csrc/client.o csrc/utils.o csrc/socket.o cuda/comm.o linkping