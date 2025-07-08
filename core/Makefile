CXX = g++
CXXFLAGS = -std=c++17 -Icore/csrc -Wall -O2

SRCS = core/launcher.cpp core/csrc/server.cpp core/csrc/client.cpp core/csrc/utils.cpp core/csrc/socket.cpp
OBJS = $(SRCS:.cpp=.o)

linkping: $(OBJS)
	$(CXX) $(CXXFLAGS) -o $@ $(OBJS)

%.o: %.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

clean:
	rm -f core/launcher.o core/csrc/server.o core/csrc/client.o core/csrc/utils.o core/csrc/socket.o linkping