#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif

#include <pthread.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <string.h>
#include "log.h"

pthread_mutex_t log_mutex = PTHREAD_MUTEX_INITIALIZER;

FILE *log_open(char *fname, const char *mode)
{  
  FILE *fp = fopen(fname, mode);
  if(fp == NULL)
    {
      fprintf(stderr, "Can not open log file %s\n", fname);
      exit(EXIT_FAILURE);
    }
  return fp;
}

int log_add(FILE *fp, const char *type, int flush, const char *format, ...)
{
  struct tm *local = NULL;
  time_t rawtime;
  char buffer[MSTR_LEN] = {'\0'};
  va_list args;

  pthread_mutex_lock(&log_mutex);
  
  /* Get current time */
  time(&rawtime);
  local = localtime(&rawtime);

  /* Get real message */
  va_start(args, format);
  vsprintf(buffer, format, args);
  va_end (args);
  
  /* Write to log file */
  fprintf(fp, "[%s] %s\t%s\n", strtok(asctime(local), "\n"), type, buffer);
  
  /* Flush it if required */
  if(flush)
    fflush(fp);

  pthread_mutex_unlock(&log_mutex);
  
  return EXIT_SUCCESS;
}

int log_close(FILE *fp)
{
  if(fp!=NULL)
    fclose(fp);
  
  return EXIT_SUCCESS;
}
