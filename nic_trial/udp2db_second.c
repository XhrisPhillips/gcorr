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
#include <signal.h>

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
#define SECDAY       86400.0
#define MJD1970      40587.0
#define MAXPKTSIZE   65536

static volatile int finished = false;
void sigaction_handler() {
  finished = true;
}

// taskset -c 10 vlbi_fake -vdif  -H 10.17.4.1 -p 10000 -udp 10000 -d 10000 -novtp -bandwidth 128 -sleep 3.5 -b 16 -nchan 1 -complex -nthread 1
// ./pipeline_second.py -c config_second.yaml -e -v

void usage(){
  fprintf(stdout,
	  "udp2db - count number of packets received from NiC \n"
	  "\n"
	  "Usage: udp2db [options] \n"
	  "  -H/-host <HOSTNAME>        Remote host to connect to \n"
	  "  -p/-port <PORT>            Port number for receive \n"
	  "  -d/-duration <DUR>         Time in seconds to run \n"
	  "  -M/-bandwidth <BANWIDTH>   Channel bandwidth in MHz \n"
	  "  -f/-framesize <FRAMESIZE>  Frame size with VDIF header (bytes) \n"
	  "  -n/-nchan <N>              Number of channels to assume in stream \n"
	  "  -b/-bits <N>               Number of bits/channel \n"
	  "  -w/-window <SIZE>          UDP window size (MB) \n"
	  "  -T/-nthread <NUM>          Number of threads \n"
	  "  -t/-timeout <SEC>          Time out of UDP socket, 0 (default) to disable it \n"
	  "  -r/-reuse                  SO_REUSEADDR socket option, 0 (default) to disable it \n"
	  "  -k/-key <KEY>              PSRDADA HEX key \n"
	  "  -s/-sod                    Enable sod, default to disable \n"
	  "  -D/-template               PSRDADA header template \n"
	  "  -c/-copy                   Copy data to ring buffer \n"
	  "  -h/-help                   This list \n"
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
  {"template",  required_argument, 0, 'D'},
  {"reuse",     no_argument,       0, 'r'},
  {"copy",      no_argument,       0, 'c'},
  {"sod",       no_argument,       0, 's'},
  {"help",      no_argument,       0, 'h'},
  
  {0, 0, 0, 0}
};

// taskset -c 0 ./udp2db -d 10 -M 128 -H 10.17.4.1 -p 10000 -w 10 -F 10000 -n 1 -b 16 -T 1 -t 1 -r -N 1024 -k dada -s -D psrdada_bigcat.txt -c 

int main(int argc, char *argv[]){
  
  // Setup single handler
  struct sigaction act;
  act.sa_handler = sigaction_handler;
  //act.sa_handler = signal_handler;
  sigaction(SIGINT, &act, NULL);
  
  int opt;
  int status;
  float ftmp;

  char template[MAX_STRLEN]; ///< String to hold PSRDADA header template
  key_t key; ///< Hex key of PSRDADA ring buffer
  
  char hostname[MAX_STRLEN]; ///< UDP IP
  int reuse = 0; ///< Default not to enable SO_REUSEADDR socket option 
  int timeout = 0; ///< Socket time out in seconds
  int duration = 0; ///< Receive data for given seconds
  int window_size = 0; ///< MBytes
  int sod = 0; ///< Default to disable sod
  int copy = 0; ///< Default not to copy data to ring buffer
  
  int bits; ///< bits per channel
  int framesize; ///< Frame size in bytes (with vdif header)
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
      fprintf(stdout, "INFO: Proposed frame size (with vdif header) is %d bytes \n", framesize);
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

    case 'c':
      copy = 1;
      fprintf(stdout, "INFO: Data will be copied to ring buffer\n");
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

  // make ourselves as the write client
  if(dada_hdu_lock_write(hdu)) {
    fprintf(stderr, "ERROR: Error locking write to HDU, "
	    "which happens at \"%s\", line [%d], "
	    "has to abort.\n", __FILE__, __LINE__);
    
    exit(EXIT_FAILURE);
  }

  // Get data and header block for later use
  ipcbuf_t *data_block   = (ipcbuf_t *)(hdu->data_block);
  ipcbuf_t *header_block = (ipcbuf_t *)(hdu->header_block);

  // Adjust framesize first to make sure that we have integer frames per second
  uint64_t bytes_per_sec = 2E6*bits*nchan*bandwidth/8;
  int datasize = framesize - VDIF_HDRSIZE; // Only need data frame size here
  while(datasize > 0){
    if(bytes_per_sec%datasize == 0){
      break;
    }
    datasize -= 8;
  }
  fprintf(stdout, "INFO: %"PRIu64" bytes per second\n", bytes_per_sec);
  fprintf(stdout, "INFO: Final frame size (without vdif header) "
	  "in bytes is %d.\n", datasize);
  
  // Get number of frames per second
  uint64_t nbytes_per_blk = nthread*bytes_per_sec;   // nbytes per block, block has 1 second
  uint64_t nframe_per_blk = nbytes_per_blk/datasize; // nframe per block, block has 1 second
  fprintf(stdout, "INFO: %"PRIu64" bytes per block\n", nbytes_per_blk);
  fprintf(stdout, "INFO: %"PRIu64" data frames per block\n", nframe_per_blk);
  
  // Check the ring buffer block size
  if(nbytes_per_blk != ipcbuf_get_bufsz(data_block))  {
    fprintf(stderr, "ERROR: Buffer size mismatch, "
	    "%"PRIu64" vs %"PRIu64", "
	    "which happens at \"%s\", line [%d], "
	    "has to abort.\n",
	    nbytes_per_blk,
	    ipcbuf_get_bufsz(data_block),
	    __FILE__, __LINE__);
    
    exit(EXIT_FAILURE);    
  }
  
  // Disable sod when required
  if(!sod){
    if(ipcbuf_disable_sod(data_block)){
      fprintf(stderr, "ERROR: Can not write data before start, "
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
    if(setsockopt(sock, SOL_SOCKET, SO_RCVTIMEO,
		  (const char*)&tout, sizeof(tout))){
      
      fprintf(stderr, "ERROR: Could not set socket RECVTIMEO\n");

      close(sock);
      exit(EXIT_FAILURE);
    }
  }

  /* Enable REUSEADDR when required */
  if (reuse){
    if(setsockopt(sock, SOL_SOCKET, SO_REUSEADDR,
		  &reuse, sizeof(reuse))){
      fprintf(stderr, "ERROR: Could not enable socket REUSEADDR\n");

      close(sock);
      exit(EXIT_FAILURE);
    }
  }
  
  /* Setup UDP socket receive buffer size */
  if(window_size){
    if (setsockopt(sock, SOL_SOCKET, SO_RCVBUF,
		   &window_size, sizeof(window_size))) {
      fprintf(stderr, "ERROR: Could not set socket RCVBUF\n");

      close(sock);
      exit(EXIT_FAILURE);
    }
  }
  
  /* Check what the socket receive size actually was set to */
  socklen_t winlen = sizeof(window_size);
  if (getsockopt(sock, SOL_SOCKET, SO_RCVBUF,
		 &window_size, &winlen)) {
    fprintf(stderr, "ERROR: Could not get socket RCVBUF size\n");
    
    close(sock);
    exit(EXIT_FAILURE);
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

  // Setup socket receive buffer
  framesize = datasize + VDIF_HDRSIZE; // Add header back here
  fprintf(stdout, "INFO: Final frame size (with vdif header) "
	  "in bytes is %d.\n", framesize);
  fprintf(stdout, "INFO: Data frame plus header size is %d bytes \n", framesize);

  // Get first frame and decode reference timestamp 
  vdif_header *vheader = NULL;
  char buf[MAXPKTSIZE];
  if(recv(sock,
	  (void *)buf,
	  MAXPKTSIZE,
	  MSG_WAITALL) == -1){      
    fprintf(stderr, "ERROR: Can not receive data from %s_%d"
	    ", which happens at \"%s\", "
	    "line [%d], has to abort.\n",
	    inet_ntoa(sa.sin_addr),		\
	    ntohs(sa.sin_port),			\
	    __FILE__, __LINE__);
    
    close(sock);    
    exit(EXIT_FAILURE);
  }
  vheader = (vdif_header*)buf;
  uint64_t lastseconds = getVDIFFrameSecond(vheader);
  
  while(true){ // Wait until we get next int seconds
    if(recv(sock,
	    (void *)buf,
	    MAXPKTSIZE,
	    MSG_WAITALL) == -1){      
      fprintf(stderr, "ERROR: Can not receive data from %s_%d"
	      ", which happens at \"%s\", "
	      "line [%d], has to abort.\n",
	      inet_ntoa(sa.sin_addr),		\
	      ntohs(sa.sin_port),		\
	      __FILE__, __LINE__);
      
      close(sock);    
      exit(EXIT_FAILURE);
    }
    vheader = (vdif_header*)buf;
    
    if(getVDIFFrameSecond(vheader) > lastseconds){
      break;
    }
  }
  lastseconds = getVDIFFrameSecond(vheader);
  uint64_t thisseconds = lastseconds;
  uint64_t seconds0    = lastseconds;
  
  int epoch_mjd    = getVDIFEpochMJD(vheader);
  double mjd_start = epoch_mjd + lastseconds/SECDAY;
  time_t seconds_from_1970 = (epoch_mjd - MJD1970)*SECDAY + lastseconds;
    
  // Start time without fraction second
  char utc_start[MAX_STRLEN];
  strftime (utc_start, MAX_STRLEN, DADA_TIMESTR, gmtime(&seconds_from_1970)); 
  
  // Fill in the first header buffer block
  char *hdrbuf = ipcbuf_get_next_write(header_block);
  if(!hdrbuf){
    fprintf(stderr, "ERROR: Error getting header_buf, "
	    "which happens at \"%s\", line [%d], has to abort.\n",
	    __FILE__, __LINE__);

    close(sock);
    exit(EXIT_FAILURE);
  }
  if(fileread(template, hdrbuf, DADA_DEFAULT_HEADER_SIZE) < 0){
    fprintf(stderr, "ERROR: Error reading header file, "
	    "which happens at \"%s\", line [%d], has to abort.\n",
	    __FILE__, __LINE__);

    close(sock);
    exit(EXIT_FAILURE);
  }  
  if(ascii_header_set(hdrbuf, "BYTES_PER_SECOND", "%"PRIu64"", bytes_per_sec)){
    fprintf(stderr, "ERROR: Error setting BYTES_PER_SECOND, "
	    "which happens at \"%s\", line [%d], has to abort.\n", __FILE__, __LINE__);

    close(sock);
    exit(EXIT_FAILURE);
  }  
  if(ascii_header_set(hdrbuf, "MJD_START", "%.15f", mjd_start)){
    fprintf(stderr, "ERROR: Error setting MJD_START, "
	    "which happens at \"%s\", line [%d], has to abort.\n", __FILE__, __LINE__);

    close(sock);
    exit(EXIT_FAILURE);
  }  
  if(ascii_header_set(hdrbuf, "UTC_START", "%s", utc_start)) {
    fprintf(stderr, "ERROR: Error setting UTC_START, "
	    "which happens at \"%s\", line [%d], has to abort.\n", __FILE__, __LINE__);

    close(sock);
    exit(EXIT_FAILURE);
  }
  uint64_t picoseconds = 0;
  if(ascii_header_set(hdrbuf, "PICOSECONDS", "%"PRIu64"", picoseconds)){
    fprintf(stderr, "ERROR: Error setting PICOSECONDS, "
	    "which happens at \"%s\", line [%d], has to abort.\n", __FILE__, __LINE__);

    close(sock);
    exit(EXIT_FAILURE);
  }
  if(ascii_header_set(hdrbuf, "BW", "%d", bandwidth)){
    fprintf(stderr, "ERROR: Error getting BW, "
	    "which happens at \"%s\", line [%d], has to abort.\n", __FILE__, __LINE__);

    close(sock);
    exit(EXIT_FAILURE);
  }
  if(ascii_header_set(hdrbuf, "NCHAN", "%d", nchan)){
    fprintf(stderr, "ERROR: Error getting NCHAN, "
	    "which happens at \"%s\", line [%d], has to abort.\n", __FILE__, __LINE__);
    
    close(sock);
    exit(EXIT_FAILURE);
  }
  if(ascii_header_set(hdrbuf, "NBIT", "%d", bits)){
    fprintf(stderr, "ERROR: Error getting NBIT, "
	    "which happens at \"%s\", line [%d], has to abort.\n", __FILE__, __LINE__);
    
    close(sock);
    exit(EXIT_FAILURE);
  }
  double tsamp = 0.5/bandwidth; // in microsecond
  if(ascii_header_set(hdrbuf, "TSAMP", "%.8f", tsamp)){
    fprintf(stderr, "ERROR: Error setting TSAMP, "
	    "which happens at \"%s\", line [%d], has to abort.\n", __FILE__, __LINE__);
    
    close(sock);
    exit(EXIT_FAILURE);
  }  
  
  /* donot set header parameters anymore */
  if(ipcbuf_mark_filled(header_block, DADA_DEFAULT_HEADER_SIZE)){
    fprintf(stderr, "ERROR: Error ipcbuf_mark_filled, "
	    "which happens at \"%s\", line [%d], has to abort.\n",
	    __FILE__, __LINE__);

    close(sock);
    exit(EXIT_FAILURE);
  }

  /* Do capture */
  // Get the first data block
  char *cbuf = ipcbuf_get_next_write(data_block);
  uint64_t counter = 0;
  while((!finished) && ((thisseconds-seconds0)<duration)){
    if(recv(sock,
	    (void *)buf,
	    MAXPKTSIZE,
	    MSG_WAITALL) == -1){      
      fprintf(stderr, "ERROR: Can not receive data from %s_%d"
	      ", which happens at \"%s\", "
	      "line [%d], has to abort.\n",
	      inet_ntoa(sa.sin_addr),		\
	      ntohs(sa.sin_port),		\
	      __FILE__, __LINE__);
      
      close(sock);
      exit(EXIT_FAILURE);
    }
    counter ++; // increase counter by one
    
    vheader = (vdif_header*)buf;
    thisseconds = getVDIFFrameSecond(vheader);
    //fprintf(stdout, "this seconds is %"PRIu64"\n", thisseconds);
    //fprintf(stdout, "last seconds is %"PRIu64"\n", lastseconds);
    
    int threadID       = getVDIFThreadID(vheader);
    uint64_t thisframe = getVDIFFrameNumber(vheader);
    
    if(thisseconds > lastseconds){ // Assume that we do not lost data for entire second
      lastseconds = thisseconds;
      
      // Mark current buffer as filled
      if(ipcbuf_mark_filled(data_block, nbytes_per_blk)){
	fprintf(stderr, "ERROR: ipcio_close_block failed, "
		"which happens at \"%s\", line [%d], has to abort.\n", __FILE__, __LINE__);
	
	close(sock);
	exit(EXIT_FAILURE);	
      }
      
      // Get a new buffer block
      cbuf = ipcbuf_get_next_write(data_block); 
      if(cbuf == NULL){
	fprintf(stderr, "ERROR: open_buffer failed, "
		"which happens at \"%s\", line [%d], has to abort.\n", __FILE__, __LINE__);
	
	close(sock);
	exit(EXIT_FAILURE);
      }
      
      // Report traffice status of previous buffer block
      fprintf(stdout, "INFO: thisframe is %"PRIu64"\n", thisframe);
      fprintf(stdout, "INFO: Expected %"PRIu64", "
	      "lost %"PRIu64" frames and %f%% lost in 1 seconds.\n",
	      nframe_per_blk,
	      nframe_per_blk - counter,
	      100.0*(nframe_per_blk-counter)/(float)nframe_per_blk);
      fflush(stdout);
      
      counter = 0; // Reset counter at this point
    }
    
    memcpy(cbuf+datasize*(nthread*thisframe+threadID), buf+VDIF_HDRSIZE, datasize);
  }
  
  // Enable eod at the end
  if(ipcbuf_is_writing(data_block)){
    if(ipcbuf_enable_eod(data_block)){
      fprintf(stderr, "ERROR: Can not enable eod, "
	      "which happens at \"%s\", line [%d].\n", __FILE__, __LINE__);
      
      close(sock);
      exit(EXIT_FAILURE);
    }
  }
  
  // Close socket
  close(sock);
  
  return EXIT_SUCCESS;
}
