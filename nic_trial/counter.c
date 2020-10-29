#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif

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

//#include "dada_cuda.h"
//#include "ipcbuf.h"

#define MAX_STRLEN 1024
#define PKTSZ      8196

double time_diff(struct timespec start,
		 struct timespec end);

void usage(){
  fprintf(stdout,
	  "counter - count number of packets received from NiC\n"
	  "\n"
	  "Usage: counter [options]\n"
	  " -i IP address to receive data from \n"
	  " -p port number to receive data from \n"
	  " -l number of seconds to receive data \n"
	  " -h show help\n"
	  );
}

// ./counter -i 10.17.4.2 -p 14700 -l 100
int main(int argc, char *argv[]){

  uint64_t length;
  int port;
  char ip[MAX_STRLEN];
  int arg;
  
  /* read in argument from command line */
  while((arg=getopt(argc,argv,"p:l:hi:")) != -1)
    {
      switch(arg)
	{
	case 'h':
	  usage();
	  exit(EXIT_FAILURE);
	  
	case 'l':
	  sscanf(optarg, "%"SCNd64"", &length);
	  fprintf(stdout, "INFO: We will counter packets for %"PRIu64" seconds.\n", length);
	  break;

	case 'i':
	  sscanf(optarg, "%s", ip);
	  fprintf(stdout, "INFO: We will counter packages from IP %s.\n", ip);
	  break;
	  
	case 'p':
	  sscanf(optarg, "%d", &port);
	  fprintf(stdout, "INFO: We will counter packages from port %d.\n", port);
	  break;
	  
	default:
	  usage();
	  exit(EXIT_FAILURE);
	  
	}
    }

  /* Setup UDP socket */
  int sock;
  sock = socket(AF_INET, SOCK_DGRAM, IPPROTO_UDP);

  int enable = 1;
  struct timeval tout = {1, 0};
  setsockopt(sock, SOL_SOCKET, SO_RCVTIMEO, (const char*)&tout, sizeof(tout));
  setsockopt(sock, SOL_SOCKET, SO_REUSEADDR, &enable, sizeof(enable));

  struct sockaddr_in sa = {0};
  memset(&sa, 0x00, sizeof(sa));
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
  char buf[PKTSZ];
  struct timespec start, current;
  clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &start);
  clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &current);

  uint64_t counter = 0;
  while(time_diff(start, current)<length){
    if(recvfrom(sock, (void *)buf, PKTSZ, 0, (struct sockaddr *)&fromsa, &fromlen) == -1){      
      fprintf(stderr, "ERROR: Can not receive data from %s_%d"
	      ", which happens at \"%s\", "
	      "line [%d], has to abort.\n",
	      inet_ntoa(sa.sin_addr),		\
	      ntohs(sa.sin_port),		\
	      __FILE__, __LINE__);
      
      close(sock);
      
      exit(EXIT_FAILURE);
    }
    counter++;
    
    clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &current);
  }

  fprintf(stdout, "INFO: We got %"PRIu64" packets in %f seconds.", counter, length);
  return EXIT_SUCCESS;
}

double time_diff(struct timespec start,
		 struct timespec end){
  return (end.tv_sec-start.tv_sec +
	  (end.tv_nsec-start.tv_nsec)/1E9);
}
