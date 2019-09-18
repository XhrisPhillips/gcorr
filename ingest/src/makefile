NVCC = nvcc

NVCCFLAGS += -I/home/den15c/.local/include 
#LDFLAGS += -L/home/den15c/.local/lib -Wl,-Bdynamic -lpsrdada -Wl,-Bstatic -lvdifio -lm -lpthread
LDFLAGS += -L/home/den15c/.local/lib -lpsrdada -lvdifio -lm -lpthread  

OBJECTS = ingest.o log.o
SRC_DIR := .

INGEST_FILE_SRCS += $(SRC_DIR)/ingest_file.c $(SRC_DIR)/ingest.c  $(SRC_DIR)/log.c 
INGEST_FILE_HDRS += $(SRC_DIR)/ingest_file.h $(SRC_DIR)/ingest.h  $(SRC_DIR)/log.h 
INGEST_FILE = ingest_file
EXECUTABLE += $(INGEST_FILE)

#INGEST_NETWORK_SRCS += $(SRC_DIR)/ingest_network.c $(SRC_DIR)/ingest.c 
#INGEST_NETWORK_HDRS += $(SRC_DIR)/ingest_network.h $(SRC_DIR)/ingest.h 
#INGEST_NETWORK = ingest_network
#EXECUTABLE += $(INGEST_NETWORK)

.PHONY: exe
exe: $(EXECUTABLE)

.PHONY: all clean
all: exe

$(INGEST_FILE): $(INGEST_FILE).o $(OBJECTS)
	$(NVCC) $(NVCCFLAGS) $(LDFLAGS) $(INGEST_FILE).o $(OBJECTS) -o $(INGEST_FILE) 

$(INGEST_NETWORK): $(INGEST_NETWORK).o $(OBJECTS)
	$(NVCC) $(NVCCFLAGS) $(LDFLAGS) $(INGEST_NETWORK).o $(OBJECTS) -o $(INGEST_NETWORK)

%.o: %.c
	$(NVCC) $(NVCCFLAGS) -c $<

RMDIR = rm -rf

clean:
	-$(RMDIR) *.o *~
