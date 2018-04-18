#include <stdio.h>
#include <stdlib.h>
#include <strings.h>
#include <argp.h>
#include <complex.h>
#include <cuComplex.h>
#include <npp.h>
#include <cuda.h>
#include <curand.h>
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
  { "samples", 's', "NSAMPLES", 0, "assume NSAMPLES when unpacking" },
  { "bandwidth", 'b', "BANDWIDTH", 0, "the bandwidth in Hz" },
  { "verbose", 'v', 0, 0, "output more" },
  { "bits", 'B', "NBITS", 0, "number of bits assumed in the data" },
  { "complex", 'I', 0, 0, "the data input is complex sampled" },
  { 0 }
};

struct arguments {
  int nloops;
  int nthreads;
  int nantennas;
  int nchannels;
  int nsamples;
  int bandwidth;
  int verbose;
  int nbits;
  int complexdata;
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
    break;
  case 's':
    arguments->nsamples = atoi(arg);
    break;
  case 'b':
    arguments->bandwidth = atoi(arg);
    break;
  case 'v':
    arguments->verbose = 1;
    break;
  case 'B':
    arguments->nbits = atoi(arg);
    break;
  case 'C':
    arguments->complexdata = 1;
    break;
  }
  return 0;
}

/* The argp parser */
static struct argp argp = { options, parse_opt, args_doc, doc };

void time_stats(float *timearray, int ntime, float *average, float *min, float *max) {
  int i = 0;
  *average = 0.0;
  for (i = 1; i < ntime; i++) {
    *average += timearray[i];
    if (i == 1) {
      *min = timearray[i];
      *max = timearray[i];
    } else {
      *min = (timearray[i] < *min) ? timearray[i] : *min;
      *max = (timearray[i] > *max) ? timearray[i] : *max;
    }
  }

  if ((ntime - 1) > 0) {
    *average /= (float)(ntime - 1);
  }
  return;
}

#define NOFRINGEROTATE

int main(int argc, char *argv[]) {
  
  /* Default argument values first. */
  struct arguments arguments;
  arguments.nloops = 100;
  arguments.nthreads = 512;
  arguments.nantennas = 6;
  arguments.nchannels = 2048;
  arguments.nsamples = 1<<23;
  arguments.bandwidth = 64e6;
  arguments.verbose = 0;
  arguments.nbits = 2;
  arguments.complexdata = 0;
  int npolarisations = 2;
  
  argp_parse(&argp, argc, argv, 0, 0, &arguments);

  // Always discard the first trial.
  arguments.nloops += 1;
  
  printf("BENCHMARK PROGRAM STARTS\n\n");

#ifndef NOUNPACK
  /*
   * This benchmarks unpacker kernels.
   */
  cuComplex **unpacked = new cuComplex*[arguments.nantennas * npolarisations];
  cuComplex **unpacked2 = new cuComplex*[arguments.nantennas];
  cuComplex **unpackedData;
  int8_t **packedData;
  float *dtime_unpack=NULL, *dtime_unpack2=NULL, *dtime_unpack3=NULL; 
  float averagetime_unpack = 0.0, mintime_unpack = 0.0, maxtime_unpack = 0.0;
  float averagetime_unpack2 = 0.0, mintime_unpack2 = 0.0, maxtime_unpack2 = 0.0;
  float averagetime_unpack3 = 0.0, mintime_unpack3 = 0.0, maxtime_unpack3 = 0.0;
  float implied_time;
  cudaEvent_t start_test_unpack, end_test_unpack;
  cudaEvent_t start_test_unpack2, end_test_unpack2;
  cudaEvent_t start_test_unpack3, end_test_unpack3;
  curandGenerator_t gen;
  dtime_unpack = (float *)malloc(arguments.nloops * sizeof(float));
  dtime_unpack2 = (float *)malloc(arguments.nloops * sizeof(float));
  dtime_unpack3 = (float *)malloc(arguments.nloops * sizeof(float));
  int i, j, unpackBlocks;

  // Allocate the memory.
  int packedBytes = arguments.nsamples * 2 * npolarisations / 8;
  packedData = new int8_t*[arguments.nantennas];
  for (i = 0; i < arguments.nantennas; i++) {
    gpuErrchk(cudaMalloc(&packedData[i], packedBytes));
  }

  for (i = 0; i < arguments.nantennas * npolarisations; i++) {
    gpuErrchk(cudaMalloc(&unpacked[i], arguments.nsamples * sizeof(cuComplex)));
  }
  gpuErrchk(cudaMalloc(&unpackedData, arguments.nantennas * npolarisations * sizeof(cuComplex*)));
  gpuErrchk(cudaMemcpy(unpackedData, unpacked, arguments.nantennas * npolarisations * sizeof(cuComplex*), cudaMemcpyHostToDevice));

  for (i = 0; i < arguments.nantennas; i++) {
    gpuErrchk(cudaMalloc(&unpacked2[i], arguments.nsamples * npolarisations * sizeof(cuComplex)));
  }
  
  unpackBlocks = arguments.nsamples / npolarisations / arguments.nthreads;
  printf("Each test will run with %d threads, %d blocks\n", arguments.nthreads, unpackBlocks);
  printf("  nsamples = %d\n", arguments.nsamples);
  printf("  nantennas = %d\n", arguments.nantennas);
  
  cudaEventCreate(&start_test_unpack);
  cudaEventCreate(&end_test_unpack);
  cudaEventCreate(&start_test_unpack2);
  cudaEventCreate(&end_test_unpack2);
  cudaEventCreate(&start_test_unpack3);
  cudaEventCreate(&end_test_unpack3);
  // Generate some random data.
  curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT);
  curandSetPseudoRandomGeneratorSeed(gen, time(NULL));
  for (i = 0; i < arguments.nantennas; i++) {
    curandGenerateUniform(gen, (float*)packedData[i], packedBytes * (sizeof(int8_t) / sizeof(float)));
  }
  curandDestroyGenerator(gen);
  for (i = 0; i < arguments.nloops; i++) {
    if (arguments.verbose) {
      printf("\nLOOP %d\n", i);
    }

    // Now do the unpacking.
    preLaunchCheck();
    if (arguments.verbose) {
      printf("  RUNNING KERNEL... ");
    }
    cudaEventRecord(start_test_unpack, 0);
    for (j = 0; j < arguments.nantennas; j++) {
      old_unpack2bit_2chan<<<unpackBlocks, arguments.nthreads>>>(unpackedData, packedData[j], j);
    }
    cudaEventRecord(end_test_unpack, 0);
    cudaEventSynchronize(end_test_unpack);
    cudaEventElapsedTime(&(dtime_unpack[i]), start_test_unpack, end_test_unpack);
    if (arguments.verbose) {
      printf("  done in %8.3f ms.\n", dtime_unpack[i]);
    }
    postLaunchCheck();

    preLaunchCheck();
    if (arguments.verbose) {
      printf("  RUNNING KERNEL 2... ");
    }
    cudaEventRecord(start_test_unpack2, 0);
    for (j = 0; j < arguments.nantennas; j++) {
      unpack2bit_2chan<<<unpackBlocks, arguments.nthreads>>>(unpacked2[j], packedData[j]);
    }
    cudaEventRecord(end_test_unpack2, 0);
    cudaEventSynchronize(end_test_unpack2);
    cudaEventElapsedTime(&(dtime_unpack2[i]), start_test_unpack2, end_test_unpack2);
    if (arguments.verbose) {
      printf("  done in %8.3f ms.\n", dtime_unpack2[i]);
    }
    postLaunchCheck();

    preLaunchCheck();
    if (arguments.verbose) {
      printf("  RUNNING KERNEL 3... ");
    }
    cudaEventRecord(start_test_unpack3, 0);
    for (j = 0; j < arguments.nantennas; j++) {
      unpack2bit_2chan_fast<<<unpackBlocks, arguments.nthreads>>>(unpacked2[j], packedData[j]);
    }
    cudaEventRecord(end_test_unpack3, 0);
    cudaEventSynchronize(end_test_unpack3);
    cudaEventElapsedTime(&(dtime_unpack3[i]), start_test_unpack3, end_test_unpack3);
    if (arguments.verbose) {
      printf("  done in %8.3f ms.\n", dtime_unpack3[i]);
    }
    postLaunchCheck();
  }
  (void)time_stats(dtime_unpack, arguments.nloops, &averagetime_unpack,
		   &mintime_unpack, &maxtime_unpack);
  (void)time_stats(dtime_unpack2, arguments.nloops, &averagetime_unpack2,
		   &mintime_unpack2, &maxtime_unpack2);
  (void)time_stats(dtime_unpack3, arguments.nloops, &averagetime_unpack3,
       &mintime_unpack3, &maxtime_unpack3);
  implied_time = (float)arguments.nsamples;
  if (arguments.complexdata) {
    // Bandwidth is the same as the sampling rate.
    implied_time /= (float)arguments.bandwidth;
    // But the data is twice as big.
    implied_time /= 2;
  } else {
    implied_time /= 2 * (float)arguments.bandwidth;
  }
  printf("\n==== ROUTINE: old_unpack2bit_2chan ====\n");
  printf("Iterations | Average time |  Min time   |  Max time   | Data time  | Speed up  |\n");
  printf("%5d      | %8.3f ms  | %8.3f ms | %8.3f ms | %8.3f s | %8.3f  |\n", (arguments.nloops - 1),
	 averagetime_unpack, mintime_unpack, maxtime_unpack, implied_time,
	 ((implied_time * 1e3) / averagetime_unpack));
  printf("\n==== ROUTINE: unpack2bit_2chan ====\n");
  printf("Iterations | Average time |  Min time   |  Max time   | Data time  | Speed up  |\n");
  printf("%5d      | %8.3f ms  | %8.3f ms | %8.3f ms | %8.3f s | %8.3f  |\n", (arguments.nloops - 1),
	 averagetime_unpack2, mintime_unpack2, maxtime_unpack2, implied_time,
	 ((implied_time * 1e3) / averagetime_unpack2));
  printf("\n==== ROUTINE: unpack2bit_2chan_fast ====\n");
  printf("Iterations | Average time |  Min time   |  Max time   | Data time  | Speed up  |\n");
  printf("%5d      | %8.3f ms  | %8.3f ms | %8.3f ms | %8.3f s | %8.3f  |\n", (arguments.nloops - 1),
   averagetime_unpack3, mintime_unpack3, maxtime_unpack3, implied_time,
   ((implied_time * 1e3) / averagetime_unpack3));
  
  
  // Clean up.
  cudaEventDestroy(start_test_unpack);
  cudaEventDestroy(end_test_unpack);
  cudaEventDestroy(start_test_unpack2);
  cudaEventDestroy(end_test_unpack2);

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


