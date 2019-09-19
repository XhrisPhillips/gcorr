/* clocks.c: find clock resolution */
// gcc clocks.c -o clocks -lrt

#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <unistd.h>
#include <time.h>

int main (int argc, char *argv[])
{
  struct timespec res;  
  
  if (clock_getres (CLOCK_REALTIME, &res) == -1)
    perror ("clock_getres: CLOCK_REALTIME"); 
  else  
    printf ("CLOCK_REALTIME: %ld s, %ld ns\n", res.tv_sec, 
            res.tv_nsec);  
  
  if (clock_getres (CLOCK_MONOTONIC, &res) == -1) 
    perror ("clock_getres: CLOCK_MONOTONIC"); 
  else  
    printf ("CLOCK_MONOTONIC: %ld s, %ld ns\n", res.tv_sec,
            res.tv_nsec);
  
  if (clock_getres (CLOCK_MONOTONIC_RAW, &res) == -1) 
    perror ("clock_getres: CLOCK_MONOTONIC_RAW");  
  else 
    printf ("CLOCK_MONOTONIC_RAW: %ld s, %ld ns\n", res.tv_sec,
            res.tv_nsec);
  
  if (clock_getres (CLOCK_PROCESS_CPUTIME_ID, &res) == -1) 
    perror ("clock_getres: CLOCK_PROCESS_CPUTIME_ID"); 
  else 
    printf ("CLOCK_PROCESS_CPUTIME_ID: %ld s, %ld ns\n", res.tv_sec,
            res.tv_nsec);
  
  if (clock_getres (CLOCK_THREAD_CPUTIME_ID, &res) == -1)
    perror ("clock_getres: CLOCK_THREAD_CPUTIME_ID");
  else
    printf ("CLOCK_THREAD_CPUTIME_ID: %ld s, %ld ns\n", res.tv_sec,
            res.tv_nsec);
  
  return EXIT_SUCCESS;
}
