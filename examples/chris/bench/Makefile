# Makefile for gpu benchmark programs

CUDA    = /Developer/NVIDIA/CUDA-8.0

# CC must be able to generate dependencies with -MM option
CC      = gcc -O3 -Wall -g
#NVCC	= nvcc -ccbin clang-3.8 -lineinfo --compiler-options -fpie 
NVCC	= nvcc -lineinfo 

EXECS = gpufft gpubench statistics cpubench gpuauto gpuspeed freeMem gpucorr

SOURCES	= gpufft.cu gpubench.cu statistics.cu

OBJECTS = $(SOURCES:.cu=.o)

CUFLAGS	=

CLINK	= -lcudart -lcufft -lnpps

IPPINC = -I/opt/intel/ipp/include
IPPLINK = -L/opt/intel/lib -L/opt/intel/ipp/lib -lipps -lippvm -lippcore -liomp5

%.o : %.cu
	$(NVCC) $(CUFLAGS) -c $<

%.o : %.c
#	$(CC) -c -O3 -ffast-math $<
	$(CC) -c  $<

all: $(EXECS)

gauto : gauto.o
	$(NVCC) $(CUFLAGS) $(CLINK) -o $@ $^

freeMem : freeMem.o
	$(NVCC) $(CUFLAGS) $(CLINK) -o $@ $^

iauto : iauto.c
	$(CC) -o $@ $(IPPINC) $(IPPLINK) $^

gpufft : gpufft.o
	$(NVCC) $(CUFLAGS) $(CLINK) -o $@ $^

gpubench : gpubench.o
	$(NVCC) -DALL $(CUFLAGS) $(CLINK) -o $@ $^ -lm

gpucorr : gpucorr.o
	$(NVCC) $(CUFLAGS) $(CLINK) -o $@ $^

gpuspeed : gpuspeed.o
	$(NVCC) $(CUFLAGS) $(CLINK) -o $@ $^

filterspeed : filterspeed.o
	$(NVCC) $(CUFLAGS) -lcurand $(CLINK) -o $@ $^

gputurn : gputurn.o
	$(NVCC) -DALL $(CUFLAGS) $(CLINK) -o $@ $^

gpuauto : gpuauto.o
	$(NVCC) -DALL $(CUFLAGS) $(CLINK) -o $@ $^

statistics : statistics.o
	$(NVCC) -DALL $(CUFLAGS) $(CLINK) -o $@ $^

cpubench : cpubench.o
	$(CC)  -o $@ $^ -lm -O3 -lm -lfftw3f

ippbench.o: ippbench.c
	$(CC) -c -o $@ $^ -O3 -ffast-math $(IPPINC)

ippbench : ippbench.o
	$(CC)  -o $@ $^ -lm -O3 -ffast-math $(IPPLINK)

clean :
	rm -f $(DEP)
	rm -f *.linkinfo
	rm -f $(OBJECTS)
	rm -f $(EXECS)

