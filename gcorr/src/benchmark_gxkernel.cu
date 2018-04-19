#include <stdio.h>
#include <stdlib.h>
#include <strings.h>
#include <argp.h>
#include <complex.h>
#include <cuComplex.h>
#include <npp.h>
#include <cuda.h>
#include <curand.h>
#include <cufft.h>
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
  curandGenerator_t gen;
  
  argp_parse(&argp, argc, argv, 0, 0, &arguments);

  // Always discard the first trial.
  arguments.nloops += 1;

  // Calculate the number of FFTs  
  int numffts = arguments.nsamples / arguments.nchannels;
  if (arguments.complexdata == 0)
  {
    numffts /= 2;
  }
  if (numffts % 8) {
    printf("Unable to proceed, numffts must be divisible by 8!\n");
    exit(0);
  }

  printf("BENCHMARK PROGRAM STARTS\n\n");

  /*
   * This benchmarks unpacker kernels.
   */
  cuComplex **unpacked = new cuComplex*[arguments.nantennas * npolarisations];
  cuComplex **unpackedData, *unpackedData2;
  int8_t **packedData, **packedData8;
  int32_t *sampleShift;
  float *dtime_unpack=NULL, *dtime_unpack2=NULL, *dtime_unpack3=NULL, *dtime_unpack4=NULL;
  float *dtime_delaycalc=NULL;
  float averagetime_unpack = 0.0, mintime_unpack = 0.0, maxtime_unpack = 0.0;
  float averagetime_unpack2 = 0.0, mintime_unpack2 = 0.0, maxtime_unpack2 = 0.0;
  float averagetime_unpack3 = 0.0, mintime_unpack3 = 0.0, maxtime_unpack3 = 0.0;
  float averagetime_unpack4 = 0.0, mintime_unpack4 = 0.0, maxtime_unpack4 = 0.0;
  float implied_time;
  cudaEvent_t start_test_unpack, end_test_unpack;
  cudaEvent_t start_test_unpack2, end_test_unpack2;
  cudaEvent_t start_test_unpack3, end_test_unpack3;
  cudaEvent_t start_test_unpack4, end_test_unpack4;
  cudaEvent_t start_test_delaycalc, end_test_delaycalc;
  dim3 FringeSetblocks;
  // TODO
  double *gpuDelays;
  double lo, sampletime;
  float *rotationPhaseInfo, *fractionalSampleDelays;

  dtime_unpack = (float *)malloc(arguments.nloops * sizeof(float));
  dtime_unpack2 = (float *)malloc(arguments.nloops * sizeof(float));
  dtime_unpack3 = (float *)malloc(arguments.nloops * sizeof(float));
  dtime_unpack4 = (float *)malloc(arguments.nloops * sizeof(float));
  dtime_delaycalc = (float *)malloc(arguments.nloops * sizeof(float));
  int i, j, unpackBlocks;

  FringeSetblocks = dim3(8, arguments.nantennas);
  
  // Allocate the memory.
  int packedBytes = arguments.nsamples * 2 * npolarisations / 8;
  int packedBytes8 = packedBytes * 4;
  packedData = new int8_t*[arguments.nantennas];
  packedData8 = new int8_t*[arguments.nantennas];
  for (i = 0; i < arguments.nantennas; i++) {
    gpuErrchk(cudaMalloc(&packedData[i], packedBytes));
    gpuErrchk(cudaMalloc(&packedData8[i], packedBytes8));
  }
  for (i = 0; i < arguments.nantennas * npolarisations; i++) {
    gpuErrchk(cudaMalloc(&unpacked[i], arguments.nsamples * sizeof(cuComplex)));
  }
  gpuErrchk(cudaMalloc(&unpackedData, arguments.nantennas * npolarisations * sizeof(cuComplex*)));
  gpuErrchk(cudaMemcpy(unpackedData, unpacked, arguments.nantennas * npolarisations * sizeof(cuComplex*), cudaMemcpyHostToDevice));
  gpuErrchk(cudaMalloc(&unpackedData2, arguments.nantennas * npolarisations * arguments.nsamples * sizeof(cuComplex)));

  /* Allocate memory for the sample shifts vector */
  gpuErrchk(cudaMalloc(&sampleShift, arguments.nantennas * numffts * sizeof(int)));
  gpuErrchk(cudaMemset(sampleShift, 0, arguments.nantennas * numffts * sizeof(int)));
  
  unpackBlocks = arguments.nsamples / npolarisations / arguments.nthreads;
  printf("Each unpacking test will run with %d threads, %d blocks\n", arguments.nthreads, unpackBlocks);
  printf("  nsamples = %d\n", arguments.nsamples);
  printf("  nantennas = %d\n", arguments.nantennas);
  
  cudaEventCreate(&start_test_unpack);
  cudaEventCreate(&end_test_unpack);
  cudaEventCreate(&start_test_unpack2);
  cudaEventCreate(&end_test_unpack2);
  cudaEventCreate(&start_test_unpack3);
  cudaEventCreate(&end_test_unpack3);
  cudaEventCreate(&start_test_unpack4);
  cudaEventCreate(&end_test_unpack4);
  cudaEventCreate(&start_test_delaycalc);
  cudaEventCreate(&end_test_delaycalc);
  // Generate some random data.
  curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT);
  curandSetPseudoRandomGeneratorSeed(gen, time(NULL));
  for (i = 0; i < arguments.nantennas; i++) {
    curandGenerateUniform(gen, (float*)packedData[i], packedBytes * (sizeof(int8_t) / sizeof(float)));
    curandGenerateUniform(gen, (float*)packedData8[i], packedBytes8 * (sizeof(int8_t) / sizeof(float)));
  }
  curandDestroyGenerator(gen);
  for (i = 0; i < arguments.nloops; i++) {
    if (arguments.verbose) {
      printf("\nLOOP %d\n", i);
    }

    // Run the delay calculator.
    preLaunchCheck();
    if (arguments.verbose) {
      printf("  RUNNING DELAY KERNEL...");
    }
    cudaEventRecord(start_test_delaycalc, 0);
    calculateDelaysAndPhases<<<FringeSetblocks, numffts/8>>>(gpuDelays, lo, sampletime,
							     arguments.nchannels,
							     arguments.nchannels,
							     rotationPhaseInfo,
							     sampleShift,
							     fractionalSampleDelays);
    cudaEventRecord(end_test_delaycalc, 0);
    cudaEventSynchronize(end_test_delaycalc);
    cudaEventElapsedTime(&(dtime_delaycalc[i]), start_test_delaycalc, end_test_delaycalc);
    if (arguments.verbose) {
      printf("  done in %8.3f ms.\n", dtime_delaycalc[i]);
    }
    postLaunchCheck();
    
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
      unpack2bit_2chan<<<unpackBlocks, arguments.nthreads>>>(&unpackedData2[2*j*arguments.nsamples], packedData[j]);
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
      init_2bitLevels();
      unpack2bit_2chan_fast<<<unpackBlocks, arguments.nthreads>>>(&unpackedData2[2*j*arguments.nsamples], packedData[j], sampleShift);
    }
    cudaEventRecord(end_test_unpack3, 0);
    cudaEventSynchronize(end_test_unpack3);
    cudaEventElapsedTime(&(dtime_unpack3[i]), start_test_unpack3, end_test_unpack3);
    if (arguments.verbose) {
      printf("  done in %8.3f ms.\n", dtime_unpack3[i]);
    }
    postLaunchCheck();

    preLaunchCheck();
    if (arguments.verbose) {
      printf("  RUNNING KERNEL 4... ");
    }
    cudaEventRecord(start_test_unpack4, 0);
    for (j = 0; j < arguments.nantennas; j++) {
      init_2bitLevels();
      unpack8bitcomplex_2chan<<<unpackBlocks, arguments.nthreads>>>(&unpackedData2[2*j*arguments.nsamples], packedData8[j]);
    }
    cudaEventRecord(end_test_unpack4, 0);
    cudaEventSynchronize(end_test_unpack4);
    cudaEventElapsedTime(&(dtime_unpack4[i]), start_test_unpack4, end_test_unpack4);
    if (arguments.verbose) {
      printf("  done in %8.3f ms.\n", dtime_unpack4[i]);
    }
    postLaunchCheck();
  }
  (void)time_stats(dtime_unpack, arguments.nloops, &averagetime_unpack,
		   &mintime_unpack, &maxtime_unpack);
  (void)time_stats(dtime_unpack2, arguments.nloops, &averagetime_unpack2,
		   &mintime_unpack2, &maxtime_unpack2);
  (void)time_stats(dtime_unpack3, arguments.nloops, &averagetime_unpack3,
       &mintime_unpack3, &maxtime_unpack3);
  (void)time_stats(dtime_unpack4, arguments.nloops, &averagetime_unpack4,
       &mintime_unpack4, &maxtime_unpack4);
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
  printf("\n==== ROUTINE: unpack28itcomplex_2chan ====\n");
  printf("Iterations | Average time |  Min time   |  Max time   | Data time  | Speed up  |\n");
  printf("%5d      | %8.3f ms  | %8.3f ms | %8.3f ms | %8.3f s | %8.3f  |\n", (arguments.nloops - 1),
   averagetime_unpack4, mintime_unpack4, maxtime_unpack4, implied_time,
   ((implied_time * 1e3) / averagetime_unpack4));
  
  
  // Clean up.
  cudaEventDestroy(start_test_unpack);
  cudaEventDestroy(end_test_unpack);
  cudaEventDestroy(start_test_unpack2);
  cudaEventDestroy(end_test_unpack2);
  cudaEventDestroy(start_test_unpack3);
  cudaEventDestroy(end_test_unpack3);
  cudaEventDestroy(start_test_unpack4);
  cudaEventDestroy(end_test_unpack4);
  cudaEventDestroy(start_test_delaycalc);
  cudaEventDestroy(end_test_delaycalc);


  /*
   * This benchmarks the performance of the fringe rotator kernel.
   */
  cuComplex *unpackedFR;
  /* A suitable array has already been defined and populated. */
  unpackedFR = unpackedData2;
  float *dtime_fringerotate=NULL, averagetime_fringerotate = 0.0;
  float mintime_fringerotate = 0.0, maxtime_fringerotate = 0.0;
  float *dtime_fringerotate2=NULL, averagetime_fringerotate2 = 0.0;
  float mintime_fringerotate2 = 0.0, maxtime_fringerotate2 = 0.0;
  float *rotVec;
  cudaEvent_t start_test_fringerotate, end_test_fringerotate;
  cudaEvent_t start_test_fringerotate2, end_test_fringerotate2;
  dim3 fringeBlocks;
  dtime_fringerotate = (float *)malloc(arguments.nloops * sizeof(float));
  dtime_fringerotate2 = (float *)malloc(arguments.nloops * sizeof(float));
  
  // Work out the block and thread numbers.
  fringeBlocks = dim3((arguments.nchannels / arguments.nthreads), numffts, arguments.nantennas);
  printf("\n\nEach fringe rotation test will run:\n");
  printf("  nsamples = %d\n", arguments.nsamples);
  printf("  nchannels = %d\n", arguments.nchannels);
  printf("  nffts = %d\n", numffts);
  
  cudaEventCreate(&start_test_fringerotate);
  cudaEventCreate(&end_test_fringerotate);
  cudaEventCreate(&start_test_fringerotate2);
  cudaEventCreate(&end_test_fringerotate2);

  /* Allocate memory for the rotation vector. */
  gpuErrchk(cudaMalloc(&rotVec, arguments.nantennas * numffts * 2 * sizeof(float)));
  /* Fill it with random data. */
  curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT);
  curandSetPseudoRandomGeneratorSeed(gen, time(NULL));
  curandGenerateUniform(gen, rotVec, arguments.nantennas * numffts * 2);
  curandDestroyGenerator(gen);

  for (i = 0; i < arguments.nloops; i++) {
    
    preLaunchCheck();
    cudaEventRecord(start_test_fringerotate2, 0);

    //setFringeRotation<<<FringeSetblocks, numffts/8>>>(rotVec);
    FringeRotate2<<<fringeBlocks, arguments.nthreads>>>(unpackedFR, rotVec);
    
    cudaEventRecord(end_test_fringerotate2, 0);
    cudaEventSynchronize(end_test_fringerotate2);
    cudaEventElapsedTime(&(dtime_fringerotate2[i]), start_test_fringerotate2,
			 end_test_fringerotate2);
    postLaunchCheck();

    preLaunchCheck();
    cudaEventRecord(start_test_fringerotate, 0);

    //setFringeRotation<<<FringeSetblocks, numffts/8>>>(rotVec);
    FringeRotate<<<fringeBlocks, arguments.nthreads>>>(unpackedFR, rotVec);
    
    cudaEventRecord(end_test_fringerotate, 0);
    cudaEventSynchronize(end_test_fringerotate);
    cudaEventElapsedTime(&(dtime_fringerotate[i]), start_test_fringerotate,
			 end_test_fringerotate);
    postLaunchCheck();

  }
  // Do some statistics.
  (void)time_stats(dtime_fringerotate, arguments.nloops, &averagetime_fringerotate,
		   &mintime_fringerotate, &maxtime_fringerotate);
  (void)time_stats(dtime_fringerotate2, arguments.nloops, &averagetime_fringerotate2,
		   &mintime_fringerotate2, &maxtime_fringerotate2);
  printf("\n==== ROUTINES: FringeRotate ====\n");
  printf("Iterations | Average time |  Min time   |  Max time   | Data time  | Speed up  |\n");
  printf("%5d      | %8.3f ms  | %8.3f ms | %8.3f ms | %8.3f s | %8.3f  |\n",
	 (arguments.nloops - 1),
	 averagetime_fringerotate, mintime_fringerotate, maxtime_fringerotate, implied_time,
	 ((implied_time * 1e3) / averagetime_fringerotate));
  printf("\n==== ROUTINES: FringeRotate2 ====\n");
  printf("Iterations | Average time |  Min time   |  Max time   | Data time  | Speed up  |\n");
  printf("%5d      | %8.3f ms  | %8.3f ms | %8.3f ms | %8.3f s | %8.3f  |\n",
	 (arguments.nloops - 1),
	 averagetime_fringerotate2, mintime_fringerotate2, maxtime_fringerotate2, implied_time,
	 ((implied_time * 1e3) / averagetime_fringerotate2));
  cudaEventDestroy(start_test_fringerotate);
  cudaEventDestroy(end_test_fringerotate);
  cudaEventDestroy(start_test_fringerotate2);
  cudaEventDestroy(end_test_fringerotate2);


  /*
   * This benchmarks the performance of the FFT.
   */
  cufftHandle plan;
  cudaEvent_t start_test_fft, end_test_fft;
  float *dtime_fft=NULL, averagetime_fft = 0.0;
  float mintime_fft = 0.0, maxtime_fft = 0.0;
  cuComplex *channelisedData, *baselineData;
  int nbaseline = arguments.nantennas * (arguments.nantennas - 1) / 2;
  int parallelAccum = (int)ceil(arguments.nthreads / arguments.nchannels + 1);
  int rc;
  while (parallelAccum && numffts % parallelAccum) parallelAccum--;
  if (parallelAccum == 0) {
    printf("Error: can not determine block size for the cross correlator!\n");
    exit(0);
  }
  dtime_fft = (float *)malloc(arguments.nloops * sizeof(float));

  cudaEventCreate(&start_test_fft);
  cudaEventCreate(&end_test_fft);

  printf("\n\nEach fringe rotation test will run:\n");
  printf("  parallelAccum = %d\n", parallelAccum);
  printf("  nbaselines = %d\n", nbaseline);
  
  /* Allocate the necessary arrays. */
  gpuErrchk(cudaMalloc(&baselineData, nbaseline * 4 * arguments.nchannels *
		       parallelAccum * sizeof(cuComplex)));
  gpuErrchk(cudaMalloc(&channelisedData, arguments.nantennas * npolarisations *
		       arguments.nsamples * sizeof(cuComplex)));
  if (rc = cufftPlan1d(&plan, arguments.nchannels, CUFFT_C2C,
		       2 * arguments.nantennas * numffts) != CUFFT_SUCCESS) {
    printf("FFT planning failed! %d\n", rc);
    exit(0);
  }
  for (i = 0; i < arguments.nloops; i++) {

    preLaunchCheck();
    cudaEventRecord(start_test_fft, 0);
    if (cufftExecC2C(plan, unpackedFR, channelisedData, CUFFT_FORWARD) != CUFFT_SUCCESS) {
      printf("FFT execution failed!\n");
      exit(0);
    }
    
    cudaEventRecord(end_test_fft, 0);
    cudaEventSynchronize(end_test_fft);
    cudaEventElapsedTime(&(dtime_fft[i]), start_test_fft,
			 end_test_fft);
    postLaunchCheck();

  }
  cufftDestroy(plan);
    
  // Do some statistics.
  (void)time_stats(dtime_fft, arguments.nloops, &averagetime_fft,
		   &mintime_fft, &maxtime_fft);
  printf("\n==== ROUTINES: cufftExecC2C ====\n");
  printf("Iterations | Average time |  Min time   |  Max time   | Data time  | Speed up  |\n");
  printf("%5d      | %8.3f ms  | %8.3f ms | %8.3f ms | %8.3f s | %8.3f  |\n",
	 (arguments.nloops - 1),
	 averagetime_fft, mintime_fft, maxtime_fft, implied_time,
	 ((implied_time * 1e3) / averagetime_fft));

  cudaEventDestroy(start_test_fft);
  cudaEventDestroy(end_test_fft);
  
#ifndef NOCROSSCORRACCUM
  /*
   * This benchmarks the performance of the cross-correlator and accumulator
   * combination.
   */
  
  
  
#endif
  
}


