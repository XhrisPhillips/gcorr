# This Makefile.am specifies the programs and libraries to build. Several
# different targets can be specified in this single file.

# bin means install in bindir, PROGRAMS means building a program
# You can have multiple programs. See slide 307 for different types of target.
# Header files are not compiled, but should be listed as dependencies so that
# they get distributed.
bin_PROGRAMS = testfxkernel bench_fxkernel

testfxkernel_SOURCES = testfxkernel.cpp fxkernel.cpp common.cpp
bench_fxkernel_SOURCES = bench_fxkernel.cpp fxkernel.cpp common.cpp

bench_fxkernel_CXXFLAGS = $(PTHREAD_CFLAGS)
bench_fxkernel_LDADD = $(PTHREAD_LIBS)
