NVCC ?= /usr/local/cuda/bin/nvcc
HOST_CXX ?= /usr/bin/g++-15
ARCH ?= sm_120
NVCCFLAGS ?= -O3 -std=c++17 -arch=$(ARCH) -ccbin $(HOST_CXX) -lineinfo
LDLIBS ?=
PROGRAM ?= hello_world
PROGRAMS := $(patsubst %.cu,bin/%,$(wildcard *.cu))

.PHONY: all run clean
all: $(PROGRAMS)

bin:
	mkdir -p bin

bin/%: %.cu | bin
	$(NVCC) $(NVCCFLAGS) $< -o $@ $(LDLIBS)

run: bin/$(PROGRAM)
	./bin/$(PROGRAM)

clean:
	rm -rf bin
