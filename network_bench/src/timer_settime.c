#include <stdio.h>
#include <time.h>
#include <signal.h>

timer_t gTimerid;

void start_timer(void)
{

  struct itimerspec value;

  value.it_value.tv_sec = 0;//waits for 5 seconds before sending timer signal
  value.it_value.tv_nsec = 8000;

  value.it_interval.tv_sec = 0;//sends timer signal every 5 seconds
  value.it_interval.tv_nsec = 8000;

  timer_create (CLOCK_REALTIME, NULL, &gTimerid);

  timer_settime (gTimerid, 0, &value, NULL);

}

void stop_timer(void)
{

  struct itimerspec value;

  value.it_value.tv_sec = 0;
  value.it_value.tv_nsec = 0;

  value.it_interval.tv_sec = 0;
  value.it_interval.tv_nsec = 0;

  timer_settime (gTimerid, 0, &value, NULL);
}

void timer_callback(int sig)
{
  printf("Catched timer signal: %d...!!\n", sig);
}

int main(int ac, char **av)
{
  int signum;
  sigset_t alarm_sig;

  start_timer();
  
  sigemptyset(&alarm_sig);
  sigaddset(&alarm_sig, SIGALRM);
  (void) signal(SIGALRM, timer_callback);
  
  while(1){
    
    sigwait(&alarm_sig, &signum); /* wait until the next signal */
    //fprintf(stdout, "HERE\n");
  }
}
