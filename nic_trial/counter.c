#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif

#include <byteswap.h>
#include <inttypes.h>
#include <time.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdbool.h>
#include <pthread.h>
#include <arpa/inet.h>
#include <sys/socket.h>
#include <byteswap.h>
#include <linux/un.h>
#include <unistd.h>
#include <getopt.h>

// PSRDADA related headers
#include "dada_hdu.h"
#include "dada_def.h"
#include "ipcio.h"
#include "ascii_header.h"
#include "daemon.h"
#include "futils.h"
#include "ipcbuf.h"

// PSRDADA with CUDA support 
#include "dada_cuda.h"

// VDIF headers
#include "vdifio.h"

#define MAX_STRLEN   1024
#define VDIF_HDRSIZE 32
#define UPDATE_INTERVAL 1

double time_diff(struct timespec start,
		 struct timespec current);

void usage(){
  fprintf(stdout,
	  "counter - count number of packets received from NiC\n"
	  "\n"
	  "Usage: counter [options]\n"
	  "  -H/-host <HOSTNAME>        Remote host to connect to\n"
	  "  -p/-port <PORT>            Port number for receive\n"
	  "  -d/-duration <DUR>         Time in seconds to run, 0 to run forever\n"
	  "  -M/-bandwidth <BANWIDTH>   Channel bandwidth in MHz\n"
	  "  -f/-framesize <FRAMESIZE>  Data frame size for VDIF data (bytes)\n"
	  "  -n/-nchan <N>              Number of channels to assume in stream\n"
	  "  -b/-bits <N>               Number of bits/channel\n"
	  "  -w/-window <SIZE>          UDP window size (MB)\n"
	  "  -T/-nthread <NUM>          Number of threads (VDIF only)\n"
	  "  -t/-timeout <SEC>          Time out of UDP socket, 0 (default) to disable it \n"
	  "  -r/-reuse                  SO_REUSEADDR socket option, 0 (default) to disable it \n"
	  "  -k/-key <KEY>              PSRDADA HEX key\n"
	  "  -N/-nframe <NUM>           PSRDADA buffer block size in the number of data frames\n"
	  "  -s/-sod                    Enable sod, default to disable\n"
	  "  -D/-template               PSRDADA header template"
	  "  -h/-help                   This list\n"
	  );
}

struct option options[] = {
  {"duration",  required_argument, 0, 'd'},
  {"bandwidth", required_argument, 0, 'M'},
  {"port",      required_argument, 0, 'p'},
  {"host",      required_argument, 0, 'H'},
  {"window",    required_argument, 0, 'w'},
  {"framesize", required_argument, 0, 'F'},
  {"nchan",     required_argument, 0, 'n'},
  {"bits",      required_argument, 0, 'b'},
  {"nthread",   required_argument, 0, 'T'},
  {"timeout",   required_argument, 0, 't'},
  {"key",       required_argument, 0, 'k'},
  {"nframe",    required_argument, 0, 'N'},
  {"template",  required_argument, 0, 'D'},
  {"reuse",     no_argument,       0, 'r'},
  {"sod",       no_argument,       0, 's'},
  {"help",      no_argument,       0, 'h'},
  
  {0, 0, 0, 0}
};

// taskset -c 0 ./counter -d 10 -M 128 -H 10.17.4.1 -p 10000 -w 10 -F 4096 -n 1 -b 16 -T 1 -t 10 -r -k dada -N 1024
// taskset -c 0 ./counter -d 10 -M 128 -H 10.17.4.1 -p 10000 -w 10 -F 4096 -n 1 -b 16 -T 1 -t 1 -r -N 1024 -k dada -s -D psrdada_bigcat.txt

int main(int argc, char *argv[]){
  int opt;
  int status;
  float ftmp;

  char template[MAX_STRLEN]; ///< String to hold PSRDADA header template
  int nframe; ///< PSRDADA buffer block size in the number of data frames
  key_t key; ///< Hex key of PSRDADA ring buffer
  
  char hostname[MAX_STRLEN]; ///< UDP IP
  int reuse = 0; ///< Default not to enable SO_REUSEADDR socket option 
  int timeout = 0; ///< Socket time out in seconds
  int duration = 0; ///< Receive data for given seconds
  int window_size = 0; ///< MBytes
  int sod = 0; ///< Default to disable sod
  
  int bits; ///< bits per channel
  int framesize; ///< Data frame size in bytes
  int port; ///< UDP port
  int bandwidth; ///< Bandwidth in MHz
  int nchan; ///< Number of channels
  int nthread; ///< Number of threads
  
  while ((opt = getopt_long_only(argc, argv,"hH:n:t:T:M:w:b:f:F:N:D:", 
				 options, NULL )) != -1) {
    
    switch (opt) {
      
    case 'w':
      status = sscanf(optarg, "%f", &ftmp);
      if (status!=1){
	fprintf(stderr, "ERROR: Bad window option %s\n", optarg);
	return EXIT_FAILURE;
      }
      window_size = ftmp * 1024 * 1024;
      fprintf(stdout, "INFO: Window size is %d bytes\n", window_size);
      break;
     
    case 'N':
      status = sscanf(optarg, "%d", &nframe);
      if (status!=1){
	fprintf(stderr, "ERROR: Bad nframe %s\n", optarg);
	return EXIT_FAILURE;
      }
      fprintf(stdout, "INFO: Each PSRDADA ring buffer blcok has %d data fremas\n", nframe);
      break;
     
    case 'k':
      status = sscanf(optarg, "%x", &key);
      if (status!=1){
	fprintf(stderr, "ERROR: Bad key %s\n", optarg);
	return EXIT_FAILURE;
      }
      fprintf(stdout, "INFO: PSRDADA ring buffer key is %x\n", key);
      break;
     
    case 'M':
      status = sscanf(optarg, "%d", &bandwidth);
      if (status!=1){
     	fprintf(stderr, "ERROR: Bad bandwidth option %s\n", optarg);
	return EXIT_FAILURE;
      }
      fprintf(stdout, "INFO: Bandwidth is %d MHz\n", bandwidth);
      break;

    case 'p':
      status = sscanf(optarg, "%d", &port);
      if (status!=1){
	fprintf(stderr, "ERROR: Bad port option %s\n", optarg);
	return EXIT_FAILURE;
      }
      fprintf(stdout, "INFO: Port number is %d\n", port);
      break;
      
    case 'd':
      status = sscanf(optarg, "%d", &duration);
      if (status!=1){
	fprintf(stderr, "ERROR: Bad duration option %s\n", optarg);
	return EXIT_FAILURE;
      }
      fprintf(stdout, "INFO: Duration is %d seconds\n", duration);
      break;

    case 'F':
      status = sscanf(optarg, "%d", &framesize);
      if (status!=1){
        fprintf(stderr, "ERROR: Bad framesize option %s\n", optarg);
	return EXIT_FAILURE;
      }
      fprintf(stdout, "INFO: Data frame size is %d bytes \n", framesize);
      break;

    case 'n':
      status = sscanf(optarg, "%d", &nchan);
      if (status!=1){
	fprintf(stderr, "ERROR: Bad nchan option %s\n", optarg);
	return EXIT_FAILURE;
      }
      fprintf(stdout, "INFO: Nchan is %d \n", nchan);
      break;

    case 'b':
      status = sscanf(optarg, "%d", &bits);
      if (status!=1){
	fprintf(stderr, "ERROR: Bad bits option %s\n", optarg);
	return EXIT_FAILURE;
      }
      fprintf(stdout, "INFO: Bits per channel is %d \n", bits);
      break;

    case 'T':
      status = sscanf(optarg, "%d", &nthread);
      if (status!=1){
        fprintf(stderr, "ERROR: Bad nthread option %s\n", optarg);
	return EXIT_FAILURE;
      }
      fprintf(stdout, "INFO: Nthread is %d \n", nthread);
      break;

    case 't':
      status = sscanf(optarg, "%d", &timeout);
      if (status!=1){
	fprintf(stderr, "ERROR: Bad time time out option %s\n", optarg);
	return EXIT_FAILURE;
      }
      if(timeout){
	fprintf(stdout, "INFO: Time out is %d seconds\n", timeout);
      }
      else{
	fprintf(stdout, "INFO: Time out is disabled\n");
      }
      break;
      
    case 'r':
      reuse = 1;
      fprintf(stdout, "INFO: SO_REUSEADDR will be enabled\n");
      break;

    case 's':
      sod = 1;
      fprintf(stdout, "INFO: SOD will be enabled\n");
      break;

    case 'H':
      if (strlen(optarg)>MAX_STRLEN) {
	fprintf(stderr, "ERROR: Hostname is too long\n");
	return EXIT_FAILURE;
      }
      strcpy(hostname, optarg);
      fprintf(stdout, "INFO: Hostname is %s\n", hostname);
      break;

    case 'D':
      if (strlen(optarg)>MAX_STRLEN) {
	fprintf(stderr, "ERROR: PSRDADA header template file name is too long\n");
	return EXIT_FAILURE;
      }
      strcpy(template, optarg);
      fprintf(stdout, "INFO: PSRDADA header template file is %s\n", template);
      break;

    case 'h':
      usage();
      return EXIT_FAILURE;
      break;
    
    case '?':
    default:
      break;
    }
  }

  fprintf(stdout, "INFO: Command line parse done\n\n");

  /* Setup PSRDADA ring buffer */
  dada_hdu_t *hdu = dada_hdu_create(NULL);
  if(!hdu){
    fprintf(stderr, "ERROR: Can not create hdu, "
	    "which happens at \"%s\", line [%d], "
	    "has to abort.\n", __FILE__, __LINE__);
    
    exit(EXIT_FAILURE);  
  }
  // Required if the key is different from default value dada
  dada_hdu_set_key(hdu, key); 
  if(dada_hdu_connect(hdu)){ 
    fprintf(stderr, "ERROR: Can not connect to hdu, "
	    "which happens at \"%s\", line [%d], "
	    "has to abort.\n", __FILE__, __LINE__);
    
    exit(EXIT_FAILURE);    
  }

  // make ourselves the write client
  if(dada_hdu_lock_write(hdu)) {
    fprintf(stderr, "ERROR: Error locking write to HDU, "
	    "which happens at \"%s\", line [%d], "
	    "has to abort.\n", __FILE__, __LINE__);
    
    exit(EXIT_FAILURE);
  }

  ipcbuf_t *data_block   = (ipcbuf_t *)(hdu->data_block);
  ipcbuf_t *header_block = (ipcbuf_t *)(hdu->header_block);

  // Check the ring buffer blovk size
  uint64_t blksz = nthread*nframe*framesize; // Expected buffer block size
  if(blksz != ipcbuf_get_bufsz(data_block))  {
    fprintf(stderr, "CAPTURE_ERROR: Buffer size mismatch, "
	    "%"PRIu64" vs %"PRIu64", "
	    "which happens at \"%s\", line [%d], "
	    "has to abort.\n",
	    blksz,
	    ipcbuf_get_bufsz(data_block),
	    __FILE__, __LINE__);
    
    exit(EXIT_FAILURE);    
  }

  // Disable sod when required
  if(!sod){
    if(ipcbuf_disable_sod(data_block)){
      fprintf(stderr, "CAPTURE_ERROR: Can not write data before start, "
	      "which happens at \"%s\", line [%d], has to abort.\n",
	      __FILE__, __LINE__);
      
      exit(EXIT_FAILURE);
    }
  }
  
  /* Create UDP socket */
  int sock;
  sock = socket(AF_INET, SOCK_DGRAM, IPPROTO_UDP);

  /* Enable time out when required */
  if(timeout){
    struct timeval tout = {timeout, 0};
    setsockopt(sock, SOL_SOCKET, SO_RCVTIMEO,
	       (const char*)&tout, sizeof(tout));
  }

  /* Enable REUSEADDR when required */
  if (reuse){
    setsockopt(sock, SOL_SOCKET, SO_REUSEADDR,
	       &reuse, sizeof(reuse));
  }
  
  /* Setup UDP socket receive buffer size */
  if(window_size){
    if (setsockopt(sock, SOL_SOCKET, SO_RCVBUF,
		   &window_size, sizeof(window_size))) {
      fprintf(stderr, "ERROR: Could not set socket RCVBUF\n");
    }
  }
  
  /* Check what the socket receive size actually was set to */
  socklen_t winlen = sizeof(window_size);
  if (getsockopt(sock, SOL_SOCKET, SO_RCVBUF,
		 &window_size, &winlen)) {
    fprintf(stderr, "ERROR: Could not get socket RCVBUF size\n");
  }
  fprintf(stdout, "INFO: Socket buffersize is %d Kbytes\n", window_size/1024);
  
  /* Bind socket */
  struct sockaddr_in sa = {0};
  sa.sin_family      = AF_INET;
  sa.sin_port        = htons(port);
  sa.sin_addr.s_addr = inet_addr(hostname);
  if(bind(sock,
	  (struct sockaddr *)&sa,
	  sizeof(sa))) {
    fprintf(stderr, "ERROR: Can not bind to %s_%d"
	    ", which happens at \"%s\", "
	    "line [%d], has to abort.\n",
	    inet_ntoa(sa.sin_addr),		\
	    ntohs(sa.sin_port),			\
	    __FILE__, __LINE__);
    
    close(sock);

    exit(EXIT_FAILURE);
  }

  // Fill in the first header buffer block
  // For now, no useful information inside
  char *hdrbuf = ipcbuf_get_next_write(header_block);
  if(!hdrbuf){
    fprintf(stderr, "CAPTURE_ERROR: Error getting header_buf, "
	    "which happens at \"%s\", line [%d], has to abort.\n",
	    __FILE__, __LINE__);
    exit(EXIT_FAILURE);
  }
  if(fileread(template, hdrbuf, DADA_DEFAULT_HEADER_SIZE) < 0){
    fprintf(stderr, "CAPTURE_ERROR: Error reading header file, "
	    "which happens at \"%s\", line [%d], has to abort.\n",
	    __FILE__, __LINE__);
    
    exit(EXIT_FAILURE);
  }
  // We should get reference time from the first packet and 
  // fill it into the header buffer
  // We can also setup reference time from command line and
  // wait until we see it from network packet
  
  /* donot set header parameters anymore */
  if(ipcbuf_mark_filled(header_block, DADA_DEFAULT_HEADER_SIZE)){
    fprintf(stderr, "CAPTURE_ERROR: Error ipcbuf_mark_filled, "
	    "which happens at \"%s\", line [%d], has to abort.\n",
	    __FILE__, __LINE__);
    
    exit(EXIT_FAILURE);
  }


  /* Do capture */
  struct timespec start;
  clock_gettime(CLOCK_REALTIME, &start);

  struct sockaddr_in fromsa = {0};
  socklen_t fromlen = sizeof(fromsa);

  int pktsz = framesize + VDIF_HDRSIZE;
  char *buf = (char *)malloc(pktsz);
  fprintf(stdout, "INFO: Data frame plus header size is %d bytes \n", pktsz);
  
  uint64_t counter     = 0;
  uint64_t lastcounter = 0;
  double elapsed_time  = 0;
  int update_threshold = UPDATE_INTERVAL;
  uint64_t lastseconds = 0;
  uint64_t lastframe   = 0;  
  do{
    if(recvfrom(sock,
		(void *)buf,
		pktsz,
		0,
		(struct sockaddr *)&fromsa,
		&fromlen) == -1){      
      fprintf(stderr, "ERROR: Can not receive data from %s_%d"
	      ", which happens at \"%s\", "
	      "line [%d], has to abort.\n",
	      inet_ntoa(sa.sin_addr),		\
	      ntohs(sa.sin_port),		\
	      __FILE__, __LINE__);
      
      close(sock);
      
      exit(EXIT_FAILURE);
    }
    
    vdif_header *vheader = (vdif_header*)buf;
    uint64_t thisframe   = getVDIFFrameNumber(vheader);
    uint64_t thisseconds = getVDIFFrameSecond(vheader);
    int threadID = getVDIFThreadID(vheader);
    fprintf(stdout,
    	    "frame: %"PRIu64"\t"
    	    "second: %"PRIu64"\t"
    	    "thread: %d\n",
    	    thisframe,
    	    thisseconds,
    	    threadID);
    
    struct timespec current;
    clock_gettime(CLOCK_REALTIME, &current);
    elapsed_time = time_diff(start, current);

    //if(elapsed_time > update_threshold){
    if(thisseconds > lastseconds){
      fprintf(stdout, "INFO: We got %"PRIu64" packets in %d seconds.\n",
	      counter-lastcounter, UPDATE_INTERVAL);
      
      fprintf(stdout, "INFO: This frame is %"PRIu64" and last frame is %"PRIu64".\n\n",
	      thisframe, lastframe);
      lastcounter = counter;
      update_threshold = update_threshold + UPDATE_INTERVAL;
    }    
    lastseconds = thisseconds;
    lastframe   = thisframe;
    
    counter++;
  }while(elapsed_time<duration);
    
  /* Free buffer */
  free(buf);
  
  fprintf(stdout, "INFO: We got %"PRIu64" packets in %d seconds.\n", counter, duration);
  return EXIT_SUCCESS;
}

double time_diff(struct timespec start,
		 struct timespec current){
  return (current.tv_sec-start.tv_sec +
	  (current.tv_nsec-start.tv_nsec)/1E9);
}
