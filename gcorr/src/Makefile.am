# This Makefile.am specifies the programs and libraries to build. Several
# different targets can be specified in this single file.
# Compiler flags



# bin means install in bindir, PROGRAMS means building a program
# You can have multiple programs. See slide 307 for different types of target.
# Header files are not 
# compiled, but should be listed as dependencies so that
# they get distributed.
CC      = gcc -O3 -Wall -g
NVCC    = nvcc -O3 -arch=sm_86 -lineinfo -maxrregcount 64
NVDEBUG = nvcc -lineinfo -g -G
AM_CFLAGS  =
#CLINK   = -lcudart -lcufft -lcurand -lstdc++ -lm -L${IPPROOT}/lib/intel64 -L${IPPROOT}/ipp/lib/intel64 -L${IPPROOT}/compiler/lib/intel64 -lipps -lippvm -lippcore
CLINK   = -lcudart -lcufft -lcurand -lstdc++ -lm 
#CUFLAGS = -I../../fxkernel/src/ -I${IPPROOT}ipp/include
CUFLAGS = -I../../fxkernel/src/
EXEEXT =

bin_PROGRAMS = testgpukernel benchmark_gxkernel validate_xcorr testgpukernel_half 
testgpukernel_SOURCES = testgpukernel.cu gxkernel.cu ../../fxkernel/src/common.cpp
testgpukernel_half_SOURCES = gxkernel.cu testgpukernel.cu ../../fxkernel/src/common.cpp
benchmark_gxkernel_SOURCES = gxkernel.cu benchmark_gxkernel.cu
validate_xcorr_SOURCES = gxkernel.cu validate_xcorr.cu ../../fxkernel/src/common.cpp 

testgpukernel$(EXEEXT) : testgpukernel.cu gxkernel.cu ../../fxkernel/src/common.cpp
	$(NVCC) -DUSING_CUDA $(CUFLAGS)  $(CLINK) -o $@ $^

testgpukernel_half$(EXEEXT) : gxkernel.cu testgpukernel.cu ../../fxkernel/src/common.cpp
	$(NVCC) -DUSING_CUDA  -DUSEHALF $(CUFLAGS) -arch=sm_86 $(CLINK) -o $@ $^

benchmark_gxkernel$(EXEEXT) : gxkernel.cu benchmark_gxkernel.cu
	$(NVCC) $(CUFLAGS) $(CLINK) -o $@ $^

validate_xcorr$(EXEEXT) : gxkernel.cu validate_xcorr.cu ../../fxkernel/src/common.cpp
	$(NVCC) -DUSING_CUDA $(CUFLAGS) $(CLINK) -o $@ $^


