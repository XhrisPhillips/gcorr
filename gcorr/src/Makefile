# Makefile for gpu benchmark programs

CUDA    = /Developer/NVIDIA/CUDA-8.0

# CC must be able to generate dependencies with -MM option
CC      = gcc -O3 -Wall -g
NVCC	= nvcc -O3
NVDEBUG = -lineinfo -g -G

EXECS = testgpukernel benchmark_gxkernel

SOURCES	= gxkernel.cu testgpukernel.cu benchmark_gxkernel.cu

OBJECTS = $(SOURCES:.cu=.o)

CUFLAGS	=

CLINK	= -lcudart -lcufft -lcurand

%.o : %.cu
	$(NVCC) $(CUFLAGS) -c $<

%.o : %.c
#	$(CC) -c -O3 -ffast-math $<
	$(CC) -c  $<

all: $(EXECS)

testgpukernel : gxkernel.cu testgpukernel.cu
	$(NVCC) $(CUFLAGS) $(CLINK) -o $@ $^

benchmark_gxkernel : gxkernel.cu benchmark_gxkernel.cu
	$(NVCC) $(CUFLAGS) $(CLINK) -o $@ $^

clean:
	rm -f $(DEP)
	rm -f *.linkinfo
	rm -f $(OBJECTS)
	rm -f $(EXECS)

