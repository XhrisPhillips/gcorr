#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif

#include <sys/time.h>
#include <signal.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <unistd.h>
#include <arpa/inet.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <string.h>
#include <inttypes.h>

#define MSTR_LEN 1024

// gcc setitimer_test.c -o setitimer_test -lrt
void missed_alarm(int signum) {
  /* we missed a timer signal, so won't be sending packets fast enough!!! */
  write(2, "Missed Alarm!\n", 14); /* can't use printf in a signal handler */
}

int main(int argc, char *argv[])
{
  useconds_t sleep_time = 16;
  struct itimerval timer;
  timer.it_interval.tv_sec = timer.it_value.tv_sec = 0;
  timer.it_interval.tv_usec = timer.it_value.tv_usec = sleep_time;   /* 16 microseconds */
  
  sigset_t alarm_sig;
  int signum;
  struct timespec start, stop;
  double elapsed_time;
  struct sockaddr_in sa;
  int enable=1;
  int sock;
  int port = 17100;
  char ip[MSTR_LEN] = {'0'};
  socklen_t tolen = sizeof(sa);
  char buf[8192];
  int errno = 0;
  uint64_t pkt_number = 0;
  double length = 10;
  
  memset((char *) &sa, 0, sizeof(sa));
  sprintf(ip, "10.17.2.2");
  fprintf(stdout, "%s\n", ip);
  sock = socket(AF_INET, SOCK_DGRAM, IPPROTO_UDP);
  sa.sin_family      = AF_INET;
  sa.sin_port        = htons(port);
  sa.sin_addr.s_addr = inet_addr(ip);
  setsockopt(sock, SOL_SOCKET, SO_REUSEADDR, &enable, sizeof(enable));
  
  if (setitimer(ITIMER_REAL, &timer, 0) < 0) {
    perror("setitimer");
    return EXIT_FAILURE;
  }
  sigemptyset(&alarm_sig);
  sigaddset(&alarm_sig, SIGALRM);
  elapsed_time = 0;
  clock_gettime(CLOCK_BOOTTIME, &start); // Timer at the beginning
  while (elapsed_time<length) {
    signal(SIGALRM, missed_alarm);
    sigwait(&alarm_sig, &signum); /* wait until the next signal */
    //if(sendto(sock, (void *)buf, 8192, 0,(struct sockaddr *)&sa, tolen) == -1){
    //  fprintf(stdout, "Fail to send with errno %d\n", errno);
    //  break;
    //}
    clock_gettime(CLOCK_BOOTTIME, &stop);  // Timer at the end
    elapsed_time = (stop.tv_sec - start.tv_sec) + (stop.tv_nsec - start.tv_nsec)/1.0E9L;
    pkt_number++;
  }
  fprintf(stdout, "%"PRIu64"\t%f\t%f in %f seconds\n", pkt_number, length/(double)sleep_time*1E6, pkt_number/(length/(double)sleep_time*1E6), length);

  return EXIT_SUCCESS;
}
