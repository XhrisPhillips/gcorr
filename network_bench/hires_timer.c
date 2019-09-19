/*   hires_timer.c: 100 microsecond timer example program   */
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <unistd.h>
#include <signal.h>
#include <time.h>

void print_time (int);

timer_t timer1;

int main (int argc, char **argv)
{
  timer_t timer2;
  struct itimerspec new_value, old_value;
  struct sigaction action;
  struct sigevent sevent;
  sigset_t set;
  int signum;

  /* SIGALRM for printing time */
  memset (&action, 0, sizeof (struct sigaction));
  action.sa_handler = print_time;
  if (sigaction (SIGALRM, &action, NULL) == -1)
    perror ("sigaction");

  /* for program completion */
  memset (&sevent, 0, sizeof (struct sigevent));
  sevent.sigev_notify = SIGEV_SIGNAL;
  sevent.sigev_signo = SIGRTMIN;

  if (timer_create (CLOCK_MONOTONIC, NULL, &timer1) == -1)
    perror ("timer_create");


  new_value.it_interval.tv_sec = 0;
  new_value.it_interval.tv_nsec = 100000;  /* 100 us */
  new_value.it_value.tv_sec = 0;
  new_value.it_value.tv_nsec = 100000;     /* 100 us */
  if (timer_settime (timer1, 0, &new_value, &old_value) == -1)
    perror ("timer_settime");

  if (sigemptyset (&set) == -1)
    perror ("sigemptyset");

  if (sigaddset (&set, SIGRTMIN) == -1)
    perror ("sigaddset");

  if (sigprocmask (SIG_BLOCK, &set, NULL) == -1)
    perror ("sigprocmask");

  if (timer_create (CLOCK_MONOTONIC, &sevent, &timer2) == -1)
    perror ("timer_create");

  new_value.it_interval.tv_sec = 0;
  new_value.it_interval.tv_nsec = 0;  /* one time timer, no reset */
  new_value.it_value.tv_sec = 0;
  new_value.it_value.tv_nsec = 1000000; /* 1 ms */
  if (timer_settime (timer2, 0, &new_value, &old_value) == -1)
    perror ("timer_settime");

  /* wait for completion signal (1 ms)  */
  if (sigwait (&set, &signum) == -1)
    perror ("sigwait");

  exit (EXIT_SUCCESS);
}

void print_time (int signum)
{   
  struct timespec tp;
  char buffer [80];

  if (clock_gettime (CLOCK_MONOTONIC, &tp) == -1)
    perror ("clock_gettime");

  sprintf (buffer, "%ld s %ld ns overrun = %d\n", tp.tv_sec,
	   tp.tv_nsec, timer_getoverrun (timer1));
  write (STDOUT_FILENO, buffer, strlen (buffer));
}
