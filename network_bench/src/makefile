DEBUG ?= 1
ifeq ($(DEBUG), 1)
    C_FLAGS = -DDEBUG
else
    C_FLAGS = -DNDEBUG
endif

C_FLAGS      += -g
#CU_FLAGS     = -rdc=true -Wno-deprecated-gpu-targets -gencode arch=compute_20,code=sm_20 -gencode arch=compute_20,code=sm_21 -gencode arch=compute_30,code=sm_30 -gencode arch=compute_35,code=sm_35 -gencode arch=compute_50,code=sm_50 -gencode arch=compute_52,code=sm_52
#CU_FLAGS     = -rdc=true -Wno-deprecated-gpu-targets --default-stream per-thread
CU_FLAGS     = -Wno-deprecated-gpu-targets --default-stream per-thread #-arch=sm_30 \
 -gencode=arch=compute_20,code=sm_20 \
 -gencode=arch=compute_30,code=sm_30 \
 -gencode=arch=compute_50,code=sm_50 \
 -gencode=arch=compute_52,code=sm_52 \
 -gencode=arch=compute_60,code=sm_60 \
 -gencode=arch=compute_61,code=sm_61 \
 -gencode=arch=compute_61,code=compute_61 \
--ptxas-options=-v 	

NVCC         = nvcc
GCC          = gcc
SRC_DIR      = .
OBJ_DIR      = .

#LIBS         = -lpsrdada -lcudart -lcuda -lm -lrt -lcufft -lpthread
#LIB_DIRS     = -L/usr/local/cuda/lib64 -L/usr/local/lib 
#INCLUDE_DIRS = -I/usr/local/include -L/usr/local/cuda/include

LIBS         = -lrt -lpthread -lm
LIB_DIRS     = #-L/usr/local/lib
INCLUDE_DIRS = #-I/usr/local/include

all: sendto hires_timer clocks setitimer_test timer_settime timer_settime_1 nanosleep_test clean 

sendto:sendto.o
	$(GCC) -o sendto sendto.o $(LIB_DIRS) $(LIBS)

clocks:clocks.o
	$(GCC) -o clocks clocks.o $(LIB_DIRS) $(LIBS)

hires_timer:hires_timer.o
	$(GCC) -o hires_timer hires_timer.o $(LIB_DIRS) $(LIBS)

setitimer_test:setitimer_test.o
	$(GCC) -o setitimer_test setitimer_test.o $(LIB_DIRS) $(LIBS)

timer_settime:timer_settime.o
	$(GCC) -o timer_settime timer_settime.o $(LIB_DIRS) $(LIBS)

timer_settime_1:timer_settime_1.o
	$(GCC) -o timer_settime_1 timer_settime_1.o $(LIB_DIRS) $(LIBS)

nanosleep_test:nanosleep_test.o
	$(GCC) -o nanosleep_test nanosleep_test.o $(LIB_DIRS) $(LIBS)

%.o: %.c
	$(GCC) $(INCLUDE_DIRS) $(C_FLAGS) -c $<

clean:
	rm -f *.o *~
