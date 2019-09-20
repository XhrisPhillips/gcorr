#include <sys/time.h>
#include <signal.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <unistd.h>

// gcc setitimer_test.c -o setitimer_test -lrt
void missed_alarm(int signum) {
  /* we missed a timer signal, so won't be sending packets fast enough!!! */
  write(2, "Missed Alarm!\n", 14); /* can't use printf in a signal handler */
}

int main(int argc, char *argv[])
{
  struct itimerval timer;
  timer.it_interval.tv_sec = timer.it_value.tv_sec = 0;
  timer.it_interval.tv_usec = timer.it_value.tv_usec = 16;   /* 16 microseconds */
  if (setitimer(ITIMER_REAL, &timer, 0) < 0) {
    perror("setitimer");
    return EXIT_FAILURE;
  }
  
  sigset_t alarm_sig;
  int signum;
  struct timespec start, stop;
  double elapsed_time;

  sigemptyset(&alarm_sig);
  sigaddset(&alarm_sig, SIGALRM);
  while (1) {
    clock_gettime(CLOCK_BOOTTIME, &start); // Timer at the beginning
    
    signal(SIGALRM, missed_alarm);
    sigwait(&alarm_sig, &signum); /* wait until the next signal */

    clock_gettime(CLOCK_BOOTTIME, &stop);  // Timer at the end
    elapsed_time = (stop.tv_sec - start.tv_sec) + (stop.tv_nsec - start.tv_nsec)/1.0E9L;
    fprintf(stdout, "%E seconds\n", elapsed_time);
  }

  return EXIT_SUCCESS;
}
