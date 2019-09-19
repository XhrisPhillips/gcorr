#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif

#include <time.h>
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <inttypes.h>
#include <math.h>
#include <arpa/inet.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <unistd.h>

#define MSTR_LEN 1024

typedef struct conf_t
{
  int ptype_monitor, ptype_spectral;
  char ip[MSTR_LEN];
  int port;
  int pktsz;
  double rate;
  useconds_t usleep;
  double length;
  uint64_t npacket;
}conf_t; 

void usage ()
{
  fprintf (stdout,
	   "sendto - Send data with given data rate to a given network interface which is defined by IP and port number\n"
	   "\n"
	   "Usage: sendto [options]\n"
	   " -a  The required data rate in Gbps \n"
	   " -b  The UDP packet size in bytes \n"
	   " -c  The IP send data to \n"
	   " -d  The port send data to \n"
	   " -e  The duration of data sending in seconds \n"
	   );
}

// gcc -o sendto sendto.c
// ./sendto -a 1.0 -b 8096 -c 10.17.2.1 -d 17100 -e 10
int main(int argc, char *argv[])
{
  int arg, ret;
  conf_t conf;
  struct timespec start, stop;
  double elapsed_time;
  uint64_t i;

  /* Light it up */
  while((arg=getopt(argc,argv,"a:b:hc:d:e:")) != -1)
    {
      switch(arg)
	{
	case 'h':
	  usage();
	  exit(EXIT_FAILURE);
	  
	case 'a':	  
	  sscanf (optarg, "%lf", &conf.rate);
	  break;
	  
	case 'b':
	  sscanf (optarg, "%d", &conf.pktsz);
	  break;
	  
	case 'c':
	  sscanf(optarg, "%s", conf.ip);
	  break;
	  
	case 'd':
	  sscanf(optarg, "%d", &conf.port);
	  break;

	case 'e':	  
	  sscanf (optarg, "%lf", &conf.length);
	  break;
	}
    }
  conf.usleep = (useconds_t) (1E6 * (8.0 * conf.pktsz) / (conf.rate * 1024. * 1024 * 1024)); // Sleep time in micro seconds
  conf.npacket = conf.length * conf.rate * 1024 * 1024 * 1024 / (8.0 * conf.pktsz);

  struct timespec nsleep, nsleep_rem;
  nsleep.tv_sec = 0;
  nsleep.tv_nsec = conf.usleep*1000;
  nsleep_rem.tv_sec = 0;
  nsleep_rem.tv_nsec = 0;
  fprintf(stdout, "\n");
  fprintf(stdout, "The data rate is %.1f Gbps\n", conf.rate);
  fprintf(stdout, "The packet size is %d bytes \n", conf.pktsz);
  fprintf(stdout, "The network interface is %s:%d \n", conf.ip, conf.port);
  fprintf(stdout, "The sleep time of packet sending is %"PRIu64" micro seconds\n", (uint64_t)conf.usleep);
  fprintf(stdout, "%"PRIu64" packets should be sent in %.1f seconds\n", conf.npacket, conf.length);  
  fprintf(stdout, "%ld\t%ld\n", (long)nsleep.tv_sec, (long)nsleep.tv_nsec);
  
  /* Do the job */
  clock_gettime(CLOCK_REALTIME, &start); // Timer at the beginning
  //for (i = 0; i < conf.npacket; i++)
  //for (i = 0; i < 100; i++)
  for (i = 0; i < 1; i++)
    {
      //fprintf(stdout, "%"PRIu64"\t%"PRIu64"\n", i, conf.npacket);
      //ret = usleep(conf.usleep * 1);
      //fprintf(stdout, "%d\t", ret);
      clock_gettime(CLOCK_MONOTONIC, &start); // Timer at the beginning
      
      //ret = nanosleep(&nsleep, NULL);
      //ret = nanosleep(&nsleep, &nsleep_rem);
      ret = clock_nanosleep(CLOCK_MONOTONIC, 0, &nsleep, &nsleep_rem);
      
      clock_gettime(CLOCK_MONOTONIC, &stop);  // Timer at the end
      elapsed_time = (stop.tv_sec - start.tv_sec) + (stop.tv_nsec - start.tv_nsec)/1.0E9L;
      fprintf(stdout, "%.1E seconds are used to send %.1f seconds data with the data rate of %.1f Gbps\n", elapsed_time, conf.length, conf.rate);
      fprintf(stdout, "%ld\t%ld\n", (long)nsleep_rem.tv_sec, (long)nsleep_rem.tv_nsec);
      fprintf(stdout, "%ld\t%ld\n", (long)nsleep.tv_sec, (long)nsleep.tv_nsec);
      fprintf(stdout, "\n");
      //fprintf(stdout, "%d\n", ret);
    }
  clock_gettime(CLOCK_REALTIME, &stop);  // Timer at the end
  
  elapsed_time = (stop.tv_sec - start.tv_sec) + (stop.tv_nsec - start.tv_nsec)/1.0E9L;
  fprintf(stdout, "%.1E seconds are used to send %.1f seconds data with the data rate of %.1f Gbps\n", elapsed_time, conf.length, conf.rate);
  fprintf(stdout, "\n");
  
  return EXIT_SUCCESS;
}
