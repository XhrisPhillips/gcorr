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
	  " -h show help\n"
	  );
}

// ./counter -i 10.17.4.2 -p 14700 -l 100 -s 8196
int main(int argc, char *argv[]){

  double length;
  int port;
  char ip[MAX_STRLEN];
  int arg;
  int pktsz;
  
  /* read in argument from command line */
  while((arg=getopt(argc,argv,"p:s:l:hi:")) != -1)
    {
      switch(arg)
	{
	case 'h':
	  usage();
	  exit(EXIT_FAILURE);
	  
	case 'l':
	  sscanf(optarg, "%lf", &length);
	  fprintf(stdout, "INFO: Count packets for %f seconds.\n", length);
	  break;

	case 'i':
	  sscanf(optarg, "%s", ip);
	  fprintf(stdout, "INFO: Count packages from IP %s.\n", ip);
	  break;
	  
	case 'p':
	  sscanf(optarg, "%d", &port);
	  fprintf(stdout, "INFO: Count packages from port %d.\n", port);
	  break;
	  
	case 's':
	  sscanf(optarg, "%d", &pktsz);
	  fprintf(stdout, "INFO: Packet size is %d bytes.\n", pktsz);
	  break;
	  
	default:
	  usage();
	  exit(EXIT_FAILURE);
	  
	}
    }

  /* Setup UDP socket */
  int sock;
  sock = socket(AF_INET, SOCK_DGRAM, IPPROTO_UDP);

  struct timeval tout = {1, 0};
  setsockopt(sock,
	     SOL_SOCKET,
	     SO_RCVTIMEO,
	     (const char*)&tout,
	     sizeof(tout));

  int enable = 1;
  setsockopt(sock,
	     SOL_SOCKET,
	     SO_REUSEADDR,
	     &enable,
	     sizeof(enable));

  struct sockaddr_in sa = {0};
  sa.sin_family      = AF_INET;
  sa.sin_port        = htons(port);
  sa.sin_addr.s_addr = inet_addr(ip);

  /* Bind socket */
  if(bind(sock,
	  (struct sockaddr *)&sa,
	  sizeof(sa)) == -1) {
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
  struct sockaddr_in fromsa = {0};
  socklen_t fromlen = sizeof(fromsa);
  
  struct timespec start;
  clock_gettime(CLOCK_REALTIME, &start);

  uint64_t counter          = 0;
  uint64_t previous_counter = 0;
  double elapsed_time  = 0;
  int update_threshold = UPDATE_INTERVAL;
  char *buf            = (char *)malloc(pktsz);
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

    uint32_t *ptr = (uint32_t*)buf;

    uint64_t writebuf = ptr[0];    
    uint32_t seconds_from_epoch = writebuf&0x3FFFFFFF;
    fprintf(stdout, "seconds from epoch is %zu .\n", seconds_from_epoch);

    writebuf = ptr[1];    
    uint32_t data_frame = writebuf&0x00FFFFFF;
    uint32_t epoch      = writebuf&0x3F000000;
    
    fprintf(stdout, "data from within second is %zu .\n", data_frame);
    fprintf(stdout, "epoch is %zu .\n", epoch);
    
    struct timespec current;
    clock_gettime(CLOCK_REALTIME, &current);
    elapsed_time = time_diff(start, current);

    if(elapsed_time > update_threshold){
      fprintf(stdout, "INFO: We got %"PRIu64" packets in %d seconds.\n",
	      counter-previous_counter,
	      UPDATE_INTERVAL);
      previous_counter = counter;
      update_threshold = update_threshold + UPDATE_INTERVAL;
    }
    
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
