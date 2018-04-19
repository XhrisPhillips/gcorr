.PHONY: all clean realclean

makefiledir:=$(dir $(realpath $(lastword $(MAKEFILE_LIST))))

testsrcdir:=$(makefiledir)
gbenchdir:=$(makefiledir)/google-benchmark
kernels:=$(makefiledir)/../src

VPATH = $(kernels):$(testsrcdir)

all:: unit ubench

CXXFLAGS=-O3 -g

NVCC?=nvcc
NVCCFLAGS+=-O3 -g

INCLUDES=-I$(testsrcdir) -I$(kernels) -I/opt/cuda/include

define cxx-compile
$(CXX) $(OPTFLAGS) $(CXXFLAGS) -std=c++11 $(CPPFLAGS) $(INCLUDES) -o $@ -c $<
endef

define cuda-compile
$(NVCC) $(OPTFLAGS) $(NVCCFLAGS) -std=c++11 $(CPPFLAGS) $(INCLUDES) -o $@ -c $<
endef

define cxx-link
$(NVCC) -o $@ $^ $(LDFLAGS) $(LDLIBS)
endef

#

libbench.a: CXXFLAGS=-std=c++11 -O3 -DNDEBUG -I$(gbenchdir)/include
libbench.a: $(patsubst %.cc,%.o,$(wildcard $(gbenchdir)/src/*.cc))
	$(AR) rcs $@ $^

gtest-all.o: CXXFLAGS=-std=c++11 -O3 -DNDEBUG
gtest-all.o: gtest-all.cc gtest.h
	$(cxx-compile)

#

wrappers.o: wrappers.cu wrappers.h gxkernel.h gpu_array.h
	$(cuda-compile)

mkdata.o: mkdata.cu mkdata.h
	$(cuda-compile)

unit.o: unit.cc wrappers.h mkdata.h gtest.h
	$(cxx-compile)

ubench.o: INCLUDES+=-I$(gbenchdir)/include
ubench.o: ubench.cc wrappers.h mkdata.h gpu_array.h
	$(cxx-compile)

gxkernel.o: gxkernel.cu gxkernel.h
	$(cuda-compile)

#

unit: gtest-all.o unit.o wrappers.o gxkernel.o mkdata.o
	$(cxx-link)

ubench: ubench.o wrappers.o gxkernel.o mkdata.o libbench.a
	$(cxx-link)

#

clean:
	rm -f *.o unit ubench

realclean: clean
	rm -f gtest-all.o libbench.a

