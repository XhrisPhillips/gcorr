#include <stdio.h>
#include <stdlib.h>
#include <strings.h>
#include <argp.h>
#include <complex.h>
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
  { "antennas", 'a', "NANTENNAS", 0, "assume NANTENNAS antennas when required" },
  { "channels", 'c', "NCHANNELS", 0, "assume NCHANNELS frequency channels when required" },
  { 0 }
};

struct arguments {
  int nloops;
  int nthreads;
  int nantennas;
  int nchannels;
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
  case 'a':
    arguments->nantennas = atoi(arg);
    break;
  case 'c':
    arguments->nchannels = atoi(arg);
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

#define NOFRINGEROTATE

int main(int argc, char *argv[]) {
  cudaError_t status;
  
  /* Default argument values first. */
  struct arguments arguments;
  arguments.nloops = 100;
  arguments.nthreads = 512;
  arguments.nantennas = 6;
  arguments.nchannels = 2048;
  int npolarisations = 2;
  
  argp_parse(&argp, argc, argv, 0, 0, &arguments);

  printf("BENCHMARK PROGRAM STARTS\n\n");

#ifndef NOUNPACK
  /*
   * This benchmarks unpacker kernels.
   */
  cuComplex **unpackedData = new cuComplex*[arguments.nantennas * npolarisations];
  int8_t **packedData, pb;
  float *dtime_unpack=NULL, averagetime_unpack = 0.0;
  float mintime_unpack = 0.0, maxtime_unpack = 0.0;
  cudaEvent_t start_test_unpack, end_test_unpack;
  dtime_unpack = (float *)malloc(arguments.nloops * sizeof(float));
  int i, j, k, l, unpackBlocks;

  // Allocate the memory.
  int packedBytes = arguments.nchannels * 2 * npolarisations / 8;
  packedData = new int8_t*[arguments.nantennas];
  for (i = 0; i < arguments.nantennas; i++) {
    gpuErrchk(cudaMalloc(&packedData[i], packedBytes));
  }

  for (i = 0; i < arguments.nantennas * npolarisations; i++) {
    gpuErrchk(cudaMalloc(&unpackedData[i], arguments.nchannels * sizeof(cuComplex)));
  }

  unpackBlocks = arguments.nchannels / npolarisations / arguments.nthreads;
  for (i = 0; i < arguments.nloops; i++) {
    // Generate some random 2 bit data each loop.
    for (j = 0; j < arguments.nantennas; j++) {
      for (k = 0; k < packedBytes; k++) {
	pb = 0;
	for (l = 0; l < 4; l++) {
	  pb = pb | (rand() % 4) << (l * 2);
	}
	gpuErrchk(cudaMemcpy(&packedData[j][k], &pb, (size_t)sizeof(int8_t), cudaMemcpyHostToDevice));
      }
    }

    // Now do the unpacking.
    preLaunchCheck();
    cudaEventRecord(start_test_unpack, 0);
    for (j = 0; j < arguments.nantennas; j++) {
      unpack2bit_2chan<<<unpackBlocks, arguments.nthreads>>>(unpackedData, packedData[j], j);
    }
    cudaEventRecord(end_test_unpack, 0);
    cudaEventSynchronize(end_test_unpack);
    cudaEventElapsedTime(&(dtime_unpack[i]), start_test_unpack, end_test_unpack);
    postLaunchCheck();
  }
  (void)time_stats(dtime_unpack, arguments.nloops, &averagetime_unpack,
		   &mintime_unpack, &maxtime_unpack);
  printf("\n==== ROUTINE: unpack2bit_2chan ====\n");
  printf("Iterations | Average time | Min time | Max time |\n");
  printf("%5d     | %8.3f ms | %8.3f ms | %8.3f ms |\n", arguments.nloops,
	 averagetime_unpack, mintime_unpack, maxtime_unpack);
  
  
  // Clean up.
  cudaEventDestroy(start_test_unpack);
  cudaEventDestroy(end_test_unpack);

#endif
  
  
#ifndef NOFRINGEROTATE
  /*
   * This benchmarks the performance of the fringe rotator kernel.
   */
  cuComplex **unpacked = new cuComplex*[arguments.nantennas * npolarisations];
  int i, j, k;
  unsigned long long GPUalloc = 0;
  float *dtime_addcomplex=NULL, averagetime_addcomplex = 0.0;
  float mintime_addcomplex = 0.0, maxtime_addcomplex = 0.0;
  cudaEvent_t start_test_addcomplex, end_test_addcomplex;
  dtime_addcomplex = (float *)malloc(arguments.nloops * sizeof(float));
  // Prepare the large arrays.
  for (i = 0; i < arguments.nantennas * npolarisations; i++) {
    gpuErrchk(cudaMalloc(&unpacked[i], arguments.nchannels * sizeof(cuComplex)));
    GPUalloc += arguments.nchannels * sizeof(cuComplex);
  }
  for (i = 0; i < arguments.nloops; i++) {
    // Set the complex number values.
    for (j = 0; j < arguments.nantennas * npolarisations; j++) {
      for (k = 0; k < arguments.nchannels; k++) {
	unpacked[j][k] = make_cuComplex(4.0 * ((float)rand() / (float)RAND_MAX),
					4.0 * ((float)rand() / (float)RAND_MAX));
      }
    }
    
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


