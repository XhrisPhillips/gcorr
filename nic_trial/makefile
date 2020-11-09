.PHONY: help

help::
	$(ECHO) "Makefile Usage:"
	$(ECHO) "  make all"
	$(ECHO) "      Command to generate executions."
	$(ECHO) ""
	$(ECHO) "  make clean "
	$(ECHO) "      Command to remove all execution related files."
	$(ECHO) ""

DEBUG ?= 1
ifeq ($(DEBUG), 1)
    CXXFLAGS := -DDEBUG
else
    CXXFLAGS := -DNDEBUG
endif

CXX  := gcc
ECHO := @echo
RM   := rm -f
TEMP_FILES := *~
COUNTER := counter
EXES := $(COUNTER)

LDFLAGS := -lpsrdada -lcudart -lcuda -lm -lrt -lcufft -lpthread -lvdifio -L/usr/local/cuda-10.2/lib64 -L/usr/local/lib -L$(HOME)/.local/lib -L$(DIFXROOT)/lib -fopenmp
CXXFLAGS += -g -I/usr/local/cuda-10.2/include -I/usr/local/include -I$(HOME)/.local/include -I$(DIFXROOT)/include

HEADERS := 
SHARED_SRCS :=
SHARED_OBJS := $(patsubst %.c, %.o, $(EXE_SHARED_SRCS))
COUNTER_OBJS := $(COUNTER).o $(SHARED_OBJS)
COUNTER_SRCS := $(COUNTER).c $(SHARED_SRCS)

%.o: %.c $(HEADERS)
	$(CXX) -c -o $@ $< $(CXXFLAGS)

$(COUNTER):$(COUNTER_OBJS)
	$(CXX) -o $@ $^ $(CXXFLAGS) $(LDFLAGS)

clean:
	$(RM) $(COUNTER_OBJS) $(TEMP_FILES) $(EXES)

.PHONY: all clean
all: $(EXES)
.NOTPARALLEL: clean

.PHONY: incremental
incremental: all

nothing:
