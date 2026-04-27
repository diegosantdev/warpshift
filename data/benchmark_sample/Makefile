CXX ?= hipcc
CXXFLAGS = -O3 -std=c++14

all: saxpy

saxpy: main.cu
	$(CXX) $(CXXFLAGS) -o $@ $<

clean:
	rm -f saxpy
