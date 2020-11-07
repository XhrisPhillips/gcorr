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

#include "dada_cuda.h"
#include "ipcbuf.h"

#include "vdifio.h"

#define MAX_STRLEN 1024
#define UPDATE_INTERVAL 1

double time_diff(struct timespec start,
		 struct timespec current);

void usage(){
  fprintf(stdout,
	  "counter - count number of packets received from NiC\n"
	  "\n"
	  "Usage: counter [options]\n"
	  " -i IP address to receive data from \n"
	  " -p port number to receive data from \n"
	  " -l number of seconds to receive data \n"
	  " -s UDP packet size \n"
	  " -o time out in seconds, 0 to disable time out \n"
	  " -b socket buffer size in MBytes, 0 to use default buffer size \n"
	  " -r enable SO_REUSEADDR socket option \n"
	  " -h show help\n"
	  );
}

// taskset -c 0 ./counter -i 10.17.4.1 -p 10000 -l 10 -s 4096 -o 10 -b 10 -r 0
int main(int argc, char *argv[]){
  int arg;
  int status;
  int reuse = 0; ///< Default not to enable SO_REUSEADDR socket option 
  double length; ///< Receive data for given seconds
  char ip[MAX_STRLEN]; ///< UDP IP
  int port; ///< UDP port
  int pktsz; ///< UDP packet size in bytes
  int bufsz; ///< Socket buffer size in MBytes
  int time_out; ///< Socket time out in seconds
  
  /* read in argument from command line */
  while((arg=getopt(argc,argv,"p:s:l:hi:o:b:r:")) != -1)
    {
      switch(arg)
	{
	case 'h':
	  usage();
	  exit(EXIT_FAILURE);
	  
	case 'l':
	  status = sscanf(optarg, "%lf", &length);
	  if(status == 1){
	    fprintf(stdout, "INFO: Count packets for %f seconds.\n", length);
	  }
	  else{
	    fprintf(stderr, "ERROR: Wrong length option %s\n", optarg);
	  }
	  break;

	case 'i':
	  status = sscanf(optarg, "%s", ip);
	  if (status == 1){
	    fprintf(stdout, "INFO: Count packages from IP %s.\n", ip);
	  }
	  else{
	    fprintf(stderr, "ERROR: Wrong IP option %s\n", optarg);
	  }
	  break;
	  
	case 'p':
	  status = sscanf(optarg, "%d", &port);
	  if (status == 1){
	    fprintf(stdout, "INFO: Count packages from port %d.\n", port);
	  }
	  else{
	    fprintf(stderr, "ERROR: Wrong port option %s\n", optarg);
	  }
	  break;
	  
	case 's':
	  status = sscanf(optarg, "%d", &pktsz);
	  if (status == 1){
	    fprintf(stdout, "INFO: Packet size is %d bytes.\n", pktsz);
	  }
	  else{
	    fprintf(stderr, "ERROR: Wrong packet size option %s\n", optarg);
	  }
	  break;
	  
	case 'o':
	  status = sscanf(optarg, "%d", &time_out);
	  if(status == 1){
	    if(time_out){
	      fprintf(stdout, "INFO: Socket time out in %d seconds.\n", time_out);
	    }
	    else{
	      fprintf(stdout, "INFO: Time out disabled.\n");
	    }
	  }
	  else{
	    fprintf(stderr, "ERROR: Wrong time out option %s\n", optarg);
	  }
	  break;
	  
	case 'b':
	  status = sscanf(optarg, "%d", &bufsz);
	  if(status == 1){
	    if(bufsz){
	      fprintf(stdout, "INFO: Socket buffer size is %d MBytes.\n", bufsz);
	    }
	    else{
	      fprintf(stdout, "INFO: Use default socket buffer size.\n");
	    }
	  }
	  else{
	    fprintf(stderr, "ERROR: Wrong socket buffer size option %s\n", optarg);
	  }
	  break;

	case 'r':
	  reuse = 1;
	  break;
	  
	default:
	  usage();
	  exit(EXIT_FAILURE);
	  
	}
    }
  
  /* Create UDP socket */
  int sock;
  sock = socket(AF_INET, SOCK_DGRAM, IPPROTO_UDP);

  /* Enable time out when required */
  if(time_out){
    struct timeval tout = {time_out, 0};
    setsockopt(sock, SOL_SOCKET, SO_RCVTIMEO,
	       (const char*)&time_out, sizeof(time_out));
  }

  /* Enable REUSEADDR when required */
  if (reuse){
    setsockopt(sock, SOL_SOCKET, SO_REUSEADDR,
	       &reuse, sizeof(reuse));
  }
  
  /* Setup UDP socket receive buffer size */
  int udpbufbytes;
  if(bufsz){
    udpbufbytes = bufsz*1024*1024;
    if (setsockopt(sock, SOL_SOCKET, SO_RCVBUF,
		   &udpbufbytes, sizeof(udpbufbytes))) {
      fprintf(stderr, "ERROR: Could not set socket RCVBUF\n");
    }
  }
  
  /* Check what the socket receive size actually was set to */
  socklen_t winlen = sizeof(udpbufbytes);
  if (getsockopt(sock, SOL_SOCKET, SO_RCVBUF,
		 &udpbufbytes, &winlen)) {
    fprintf(stderr, "ERROR: Could not get socket RCVBUF size\n");
  }
  fprintf(stdout, "Socket buffersize is %d Kbytes\n", udpbufbytes/1024);
  
  /* Bind socket */
  struct sockaddr_in sa = {0};
  sa.sin_family      = AF_INET;
  sa.sin_port        = htons(port);
  sa.sin_addr.s_addr = inet_addr(ip);
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

  /* Do capture */
  struct timespec start;
  clock_gettime(CLOCK_REALTIME, &start);

  struct sockaddr_in fromsa = {0};
  socklen_t fromlen = sizeof(fromsa);
  
  uint64_t counter     = 0;
  uint64_t lastcounter = 0;
  double elapsed_time  = 0;
  int update_threshold = UPDATE_INTERVAL;
  char *buf            = (char *)malloc(pktsz);
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
    //fprintf(stdout,
    //	    "frame: %"PRIu64"\t"
    //	    "second: %"PRIu64"\t"
    //	    "thread: %d\n",
    //	    thisframe,
    //	    thisseconds,
    //	    threadID);
    
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
  }while(elapsed_time<length);

  /* Free buffer */
  free(buf);
  
  fprintf(stdout, "INFO: We got %"PRIu64" packets in %f seconds.\n", counter, length);
  return EXIT_SUCCESS;
}

double time_diff(struct timespec start,
		 struct timespec current){
  return (current.tv_sec-start.tv_sec +
	  (current.tv_nsec-start.tv_nsec)/1E9);
}
