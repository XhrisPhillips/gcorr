#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif

//#include <sys/time.h>
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
#include <pthread.h>
#include <sched.h>
#include <assert.h>

static int get_thread_policy(pthread_attr_t *attr)
{
  int policy;
  int rs = pthread_attr_getschedpolicy(attr,&policy);
  assert(rs==0);
  switch(policy) {
  case SCHED_FIFO:
    printf("policy= SCHED_FIFO\n");
    break;
  case SCHED_RR:
    printf("policy= SCHED_RR\n");
    break;
  case SCHED_OTHER:
    printf("policy=SCHED_OTHER\n");
    break;
  default:
    printf("policy=UNKNOWN\n");
    break;
  }
  return policy;
}

static void set_thread_policy(pthread_attr_t *attr,int policy)
{
  get_thread_policy(attr);
  int rs = pthread_attr_setschedpolicy(attr,policy);
  assert(rs==0);
  get_thread_policy(attr);
}

int main(int argc, char *argv[])
{
  struct timespec nsleep;
  struct timespec start;
  struct timespec stop;
  double length;
  double elapsed_time;
  uint64_t n = 10;
  uint64_t i;
  //pthread_attr_t attr;
  //int rs;

  //rs = pthread_attr_init(&attr);
  //assert(rs==0);
  //
  //set_thread_policy(&attr,SCHED_RR);
  ////sched_get_priority_max(SCHED_RR);
  //sched_setscheduler(SCHED_RR);

  int policy;
  policy = sched_getscheduler(0);
  switch(policy) {
  case SCHED_OTHER:
    printf("SCHED_OTHER\n");
    break;
  case SCHED_RR:
    printf("SCHED_RR\n");
    break;
  case SCHED_FIFO:
    printf("SCHED_FIFO\n");
    break;
  default:
    printf("Unknown...\n");
  }
  struct sched_param sp = { .sched_priority = 1 };
  int ret = sched_setscheduler(0, SCHED_FIFO, &sp);
  //fprintf(stdout, "%d\n", sched_get_priority_max(SCHED_FIFO));
  //fprintf(stdout, "%d\n", sched_get_priority_min(SCHED_FIFO));
  if (ret == -1) {
    perror("sched_setscheduler");
    return 1;
  }
  
  policy = sched_getscheduler(0);
  switch(policy) {
  case SCHED_OTHER:
    printf("SCHED_OTHER\n");
    break;
  case SCHED_RR:
    printf("SCHED_RR\n");
    break;
  case SCHED_FIFO:
    printf("SCHED_FIFO\n");
    break;
  default:
    printf("Unknown...\n");
  }
  
  nsleep.tv_sec = 0;
  nsleep.tv_nsec = 10000;
  length = n*nsleep.tv_nsec/1.0E9;

  clock_gettime(CLOCK_REALTIME, &start); // Timer at the beginning
  for(i=0; i<n; i++)
    nanosleep(&nsleep, &nsleep);
  clock_gettime(CLOCK_REALTIME, &stop); // Timer at the beginning
  elapsed_time = (stop.tv_sec - start.tv_sec) + (stop.tv_nsec - start.tv_nsec)/1.0E9L;

  fprintf(stdout, "%f\t%f\t%f\n", 1.0E9*elapsed_time/n, (double)nsleep.tv_nsec, (1.0E9*elapsed_time/n - (double)nsleep.tv_nsec)/(double)nsleep.tv_nsec);

  //get_thread_policy(&attr);
  //
  //rs = pthread_attr_destroy(&attr);
  //assert(rs==0);
  
  return EXIT_SUCCESS;
}
