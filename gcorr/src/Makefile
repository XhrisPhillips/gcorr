# Makefile for gpu benchmark programs

#CUDA    = /Developer/NVIDIA/CUDA-8.0

# CC must be able to generate dependencies with -MM option
CC      = gcc -O3 -Wall -g
NVCC	= nvcc -O3 -arch sm_61
NVDEBUG = nvcc -lineinfo -g -G

EXECS = testgpukernel benchmark_gxkernel validate_xcorr

SOURCES	= gxkernel.cu testgpukernel.cu benchmark_gxkernel.cu validate_xcorr.cu

OBJECTS = $(SOURCES:.cu=.o)

CUFLAGS	= -I../../fxkernel/src/

CFLAGS  = 

CLINK	= -lcudart -lcufft -lcurand

%.o : %.cu
	$(NVCC) $(CUFLAGS) -c $<

%.o : %.cpp
	$(CC) $(CFLAGS) -c $<

%.o : %.c
#	$(CC) -c -O3 -ffast-math $<
	$(CC) -c  $<

all: $(EXECS)

debug: NVCC = $(NVDEBUG)
debug: all
testgpukernel : gxkernel.cu testgpukernel.cu ../../fxkernel/src/common.cpp
	$(NVCC) $(CUFLAGS) $(CLINK) -o $@ $^

benchmark_gxkernel : gxkernel.cu benchmark_gxkernel.cu
	$(NVCC) $(CUFLAGS) $(CLINK) -o $@ $^

validate_xcorr : gxkernel.cu validate_xcorr.cu ../../fxkernel/src/common.cpp
	$(NVCC) $(CUFLAGS) $(CLINK) -o $@ $^

clean:
	rm -f $(DEP)
	rm -f *.linkinfo
	rm -f $(OBJECTS)
	rm -f $(EXECS)

