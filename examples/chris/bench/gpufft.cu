
#include <stdio.h>
#include <stdlib.h>
#include <strings.h>
#include <cufft.h>
#include <getopt.h>


int main(int argc, char *argv[]) {
  int nfft, batch, opt, tmp, ss, repeat;
  size_t nsample;
  float dtime;
  cudaError_t status;
  cufftReal *idata;
  cufftComplex *odata, *icdata;
  cufftResult result;
  cufftHandle plan;
  cudaEvent_t start_exec, stop_exec, start_fft, stop_fft;

  int ffts[]  = {4, 16,32,64,128,256,1024,4096,8192,16384};
  
  repeat = 10;
  nsample = 1024*1024*8;

  struct option options[] = {
    {"repeat", 1, 0, 'r'}, 
    {0, 0, 0, 0}
  };

  while (1) {
    opt=getopt_long_only(argc, argv, "n:r:", 
			 options, NULL);

    if (opt==EOF) break;

    switch (opt) {
      
    case 'r':
      ss = sscanf(optarg, "%d", &tmp);
      if (ss!=1)
        fprintf(stderr, "Bad -repeat option %s\n", optarg);
      else {
	repeat = tmp;
      }
      break; 

    case '?':
    default:
      break;
    }
  }

  cudaEventCreate(&start_exec);
  cudaEventCreate(&stop_exec);
  cudaEventCreate(&start_fft);
  cudaEventCreate(&stop_fft);

  // Start total time event
  cudaEventRecord(start_exec, 0);


  // Allocate memory on the host
  status = cudaMalloc(&idata, nsample*sizeof(cufftReal));
  if (status != cudaSuccess) {
    fprintf(stderr, "Error: cudaMalloc failed (1)\n");
    return EXIT_FAILURE;
  }
  status = cudaMalloc(&icdata, nsample*sizeof(cufftComplex));
  if (status != cudaSuccess) {
    fprintf(stderr, "Error: cudaMalloc failed (2)\n");
    return EXIT_FAILURE;
  }
  status = cudaMalloc(&odata, nsample*sizeof(cufftComplex));
  if (status != cudaSuccess) {
    fprintf(stderr, "Error: cudaMalloc failed (3)\n");
    return EXIT_FAILURE;
  }

  printf("======= Real to Complex ========\n");
  printf("   n   |    time     |    1 GHz    |  Bandwidth |\n");

  int N = sizeof(ffts)/sizeof(int);
  for (int j=0; j<N; j++) {
    nfft = ffts[j]*2;

    // Setup the FFT
    batch = nsample/nfft;
    //result = cufftPlan1d(&plan, nfft, CUFFT_R2C, batch);
    result = cufftPlanMany(&plan, 1, &nfft, NULL, 1, 0, NULL, 1, 0, CUFFT_R2C, batch);
    if (result != CUFFT_SUCCESS) {
      fprintf(stderr, "cufftPlan1d failed with status %d\n", result);
      return EXIT_FAILURE;
    }
//    cufftSetCompatibilityMode(plan, CUFFT_COMPATIBILITY_NATIVE);

    cudaEventRecord(start_fft, 0);
    for (int i=0; i<repeat; i++) {
      result = cufftExecR2C(plan, idata, odata);
    }
    cudaEventRecord(stop_fft, 0);
    cudaEventSynchronize(stop_fft);
    cudaEventElapsedTime(&dtime, start_fft, stop_fft);

    printf("%6d | %8.3f ms | %8.3f ms | %6.1f MHz |\n", nfft/2, dtime, dtime*4e9/((float)nfft*batch*repeat),
	   (float)nfft*batch*repeat/4/1e6/dtime*1000);

    // Destroy plan
    cufftDestroy(plan);
  }

  printf("\n\n======= Complex to Complex ========\n");
  printf("   n   |    time     |    1 GHz    |  Bandwidth  |\n");

  N = sizeof(ffts)/sizeof(int);
  for (int j=0; j<N; j++) {
    nfft = ffts[j];

    // Setup the FFT
    batch = nsample/nfft;

    result = cufftPlanMany(&plan, 1, &nfft, NULL, 1, 0, NULL, 1, 0, CUFFT_C2C, batch);
    if (result != CUFFT_SUCCESS) {
      fprintf(stderr, "cufftPlanMany failed with status %d\n", result);
      return EXIT_FAILURE;
    }
//    cufftSetCompatibilityMode(plan, CUFFT_COMPATIBILITY_NATIVE);

    cudaEventRecord(start_fft, 0);
    for (int i=0; i<repeat; i++) {
      result = cufftExecC2C(plan, icdata, odata, CUFFT_FORWARD);
    }
    cudaEventRecord(stop_fft, 0);
    cudaEventSynchronize(stop_fft);
    cudaEventElapsedTime(&dtime, start_fft, stop_fft);

    printf("%6d | %8.3f ms | %8.3f ms | %6.1f MHz |\n", nfft, dtime, dtime*2e9/((float)nfft*batch*repeat),
	   (float)nfft*batch*repeat/2/1e6/(dtime/1000));

    // Destroy plan
    cufftDestroy(plan);

  }

  cudaEventRecord(stop_exec, 0);
  cudaEventSynchronize(stop_exec);
  cudaEventElapsedTime(&dtime, start_exec, stop_exec);
  printf("\nTotal executution time =  %.3f ms\n", dtime);


  // Free allocated memory
  cudaFree(idata);
  cudaFree(icdata);
  cudaFree(odata);

  cudaEventDestroy(start_fft);
  cudaEventDestroy(stop_fft);
  cudaEventDestroy(start_exec);
  cudaEventDestroy(stop_exec);
  

}
