#include <stdio.h>
#include <stdlib.h>
#include <strings.h>
#include <argp.h>

#include <cuComplex.h>
#include <npp.h>
#include "gxkernel.h"

/*
 * Code to test the kernels in the gxkernel.cu.
 */

void preLaunchCheck() {
  cudaError_t error;

  error = cudaGetLastError();
  
  if (error != cudaSuccess) {
    fprintf(stderr, "Error: Previous CUDA failure: \"%s\". Exiting\n",
	    cudaGetErrorString(error));
    exit(EXIT_FAILURE);
  }
}

void postLaunchCheck() {
  cudaError_t error;

  error = cudaGetLastError();
  
  if (error != cudaSuccess) {
    fprintf(stderr, "Error: Failure Launching kernel: \"%s\". Exiting\n",
	    cudaGetErrorString(error));
    exit(EXIT_FAILURE);
  }
}

const char *argp_program_version = "benchmark_gxkernel 1.0";
static char doc[] = "benchmark_gxkernel -- testing performance of various kernels";
static char args_doc[] = "";

/* Our command line options */
static struct argp_option options[] = {
  { "loops", 'n', "NLOOPS", 0, "run each performance test NLOOPS times" },
  { "threads", 't', "NTHREADS", 0, "run with NTHREADS threads on each test" },
  { 0 }
};

struct arguments {
  int nloops;
  int nthreads;
};

/* The option parser */
static error_t parse_opt(int key, char *arg, struct argp_state *state) {
  struct arguments *arguments = (struct arguments *)state->input;

  switch (key) {
  case 'n':
    arguments->nloops = atoi(arg);
    break;
  case 't':
    arguments->nthreads = atoi(arg);
    break;
  }
  return 0;
}

/* The argp parser */
static struct argp argp = { options, parse_opt, args_doc, doc };

void time_stats(float *timearray, int ntime, float *average, float *min, float *max) {
  int i = 0;
  *average = 0.0;
  for (i = 0; i < ntime; i++) {
    *average += timearray[i];
    if (i == 0) {
      *min = timearray[i];
      *max = timearray[i];
    } else {
      *min = (timearray[i] < *min) ? timearray[i] : *min;
      *max = (timearray[i] > *max) ? timearray[i] : *max;
    }
  }

  if (ntime > 0) {
    *average /= (float)ntime;
  }
  return;
}

int main(int argc, char *argv[]) {
  cudaError_t status;
  
  /* Default argument values first. */
  struct arguments arguments;
  arguments.nloops = 100;
  arguments.nthreads = 512;
  argp_parse(&argp, argc, argv, 0, 0, &arguments);

  printf("BENCHMARK PROGRAM STARTS\n\n");

  
#ifndef NOADDCOMPLEX
  /*
   * This benchmarks the performance of the complex number adder.
   */
  cuFloatComplex a, b;
  int i;
  float *dtime_addcomplex=NULL, averagetime_addcomplex = 0.0;
  float mintime_addcomplex = 0.0, maxtime_addcomplex = 0.0;
  cudaEvent_t start_test_addcomplex, end_test_addcomplex;
  dtime_addcomplex = (float *)malloc(arguments.nloops * sizeof(float));
  for (i = 0; i < arguments.nloops; i++) {
    // Set the complex number values.
    a = 1.0 + 2.0 * I;
    b = 2.0 + 3.0 * I;
    
    preLaunchCheck();
    cudaEventRecord(start_test_addcomplex, 0);
    cuCaddIf<<<1, arguments.nthreads>>(&a, b);
    cudaEventRecord(stop_test_addcomplex, 0);
    cudaEventSynchronize(stop_test_addcomplex);
    cudaEventElapsedTime(&(dtime_addcomplex[i]), start_test_addcomplex, stop_test_addcomplex);
    postLaunchCheck();
  }
  // Do some statistics.
  (void)time_stats(dtime_addcomplex, arguments.nloops, &averagetime_addcomplex,
		   &mintime_addcomplex, &maxtime_addcomplex);
  printf("\n==== ROUTINE: cuCaddIf ====\n");
  printf("Iterations | Average time | Min time | Max time |\n");
  printf("%5d     | %8.3f ms | %8.3f ms | %8.3f ms |\n", arguments.nloops,
	 averagetime_addcomplex, mintime.addcomplex, maxtime_addcomplex);
  cudaEventDestroy(start_test_addcomplex);
  cudaEventDestroy(end_test_addcomplex);
#endif
  
}


