.PHONY: clean all realclean test

makefiledir:=$(dir $(realpath $(lastword $(MAKEFILE_LIST))))
srcdir:=$(makefiledir)/src
gbenchdir:=$(makefiledir)/google-benchmark

VPATH = $(srcdir)

all:: ubench

sources:=$(wildcard $(srcdir)/*.cc)
gbench-sources:=$(wildcard $(gbenchdir)/src/*.cc)

-include $(patsubst %.cc,obj/gbench/%.d,$(notdir $(gbench-sources)))

CXXFLAGS=-O3 -g

NVCC?=nvcc
NVCCFLAGS+=-O3 -g

extra-includes=-I$(gbenchdir)/include -I/opt/cuda/include

define cxx-compile
$(CXX) $(CXXFLAGS) -std=c++11 -MMD -MP $(CPPFLAGS) $(extra-includes) -o $@ -c $<
endef

define cuda-compile
$(NVCC) $(NVCCFLAGS) -std=c++11 $(CPPFLAGS) $(extra-includes) -o $@ -c $<
endef

define cxx-link
$(NVCC) -o $@ $^ $(LDFLAGS) $(LDLIBS)
endef


libbench.a: $(patsubst %.cc,obj/gbench/%.o,$(notdir $(gbench-sources)))
	$(AR) rcv $@ $^

obj/gbench/%.o: $(gbenchdir)/src/%.cc
	$(cxx-compile)

ubench: obj/ubench.o obj/kernels.o libbench.a
	$(cxx-link)

obj/%.o: %.cc
	$(cxx-compile)

obj/%.o: %.cu
	$(cuda-compile)

clean:
	rm -f obj/*.o obj/gbench/*.o

realclean: clean
	rm -f ubench libbench.a obj/*.d obj/gbemch/*.d

