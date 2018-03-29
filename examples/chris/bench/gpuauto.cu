#include <stdio.h>
#include <stdlib.h>
#include <strings.h>
#include <stdint.h>
#include <cufft.h>
#include <getopt.h>

#define RESTFREQ  1420.405752  /* MHz */
#define C  299792458 /* m/s */

void fillGPUarray_8bit(int8_t *data, uint64_t n);

__global__ void unpack8bits_kernel(float *rcp, float *lcp, const int8_t *src) {

  const size_t i = blockDim.x * blockIdx.x + threadIdx.x;
  const size_t j = i*2;

  rcp[i] = static_cast<float>(src[j]);
  lcp[i] = static_cast<float>(src[j+1]);
}


int main(int argc, char *argv[]) {
  int nfft, batch, opt, tmp, ss, repeat, nthread, nblock;
  uint64_t sampPersec;
  float dtime, bandwidth;
  cudaError_t status;
  int8_t *idata;
  cufftReal *fdata;
  cufftComplex *odata;
  cufftResult result;
  cufftHandle plan;
  cudaEvent_t start_exec, stop_exec, start_fft, stop_fft;

  int ffts[]  = {1024,8192,16384,32768,65536,131072};
  
  size_t memsize = 1024*1024*32*4;
  uint32_t nsample = memsize/sizeof(float);

  bandwidth = 128;
  repeat = 10;
  nthread = 512;

  struct option options[] = {
    {"nthread", 1, 0, 'n'}, 
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

    case 'n':
      ss = sscanf(optarg, "%d", &tmp);
      if (ss!=1)
        fprintf(stderr, "Bad -nthread option %s\n", optarg);
      else {
	nthread = tmp;
      }
      break; 

    case '?':
    default:
      break;
    }
  }

  nblock = nsample/nthread;
  nsample = nblock*nthread;

  sampPersec = bandwidth*1e6*2*2; // # Real samples per second

  float totalvelocity = bandwidth/RESTFREQ*C/1000;

  cudaEventCreate(&start_exec);
  cudaEventCreate(&stop_exec);
  cudaEventCreate(&start_fft);
  cudaEventCreate(&stop_fft);

  // Start total time event
  cudaEventRecord(start_exec, 0);

  // Allocate memory on the host
  status = cudaMalloc(&idata, nsample*sizeof(int8_t));
  if (status != cudaSuccess) {
    fprintf(stderr, "Error: cudaMalloc failed (%s at line %d)\n", __FILE__, __LINE__);
    return EXIT_FAILURE;
  }

  // Allocate memory on the host
  status = cudaMalloc(&fdata, nsample*sizeof(cufftReal));
  if (status != cudaSuccess) {
    fprintf(stderr, "Error: cudaMalloc failed (%s at line %d)\n", __FILE__, __LINE__);
    return EXIT_FAILURE;
  }

  status = cudaMalloc(&odata, nsample*sizeof(cufftComplex));
  if (status != cudaSuccess) {
    fprintf(stderr, "Error: cudaMalloc failed (%s at line %d)\n", __FILE__, __LINE__);
    return EXIT_FAILURE;
  }

  fillGPUarray_8bit(idata, nsample);

  int N = sizeof(ffts)/sizeof(int);
  for (int j=0; j<N; j++) {
    nfft = ffts[j]*2;

    // Setup the FFT
    batch = nsample/nfft;
    result = cufftPlan1d(&plan, nfft, CUFFT_R2C, batch);
    if (result != CUFFT_SUCCESS) {
      fprintf(stderr, "cufftPlan1d failed with status %d\n", result);
      return EXIT_FAILURE;
    }
    //    cufftSetCompatibilityMode(plan, CUFFT_COMPATIBILITY_NATIVE);

    cudaEventRecord(start_fft, 0);
    for (int i=0; i<repeat; i++) {
      
      unpack8bits_kernel<<<nblock/2,nthread>>>(fdata, &fdata[nsample/2], idata);

      result = cufftExecR2C(plan, fdata, odata);
    }
    cudaEventRecord(stop_fft, 0);
    cudaEventSynchronize(stop_fft);
    cudaEventElapsedTime(&dtime, start_fft, stop_fft);

    printf("%6d | %8.3f ms | %8.3f ms | %6.1f km/s |\n", nfft/2, dtime, dtime*sampPersec/((float)nfft*batch*repeat), totalvelocity/(nfft/2));

    // Destroy plan
    cufftDestroy(plan);

  }
  cudaEventRecord(stop_exec, 0);
  cudaEventSynchronize(stop_exec);
  cudaEventElapsedTime(&dtime, start_exec, stop_exec);
  printf("\nTotal executution time =  %.3f ms\n", dtime);


  // Free allocated memory
  cudaFree(idata);
  cudaFree(odata);

  cudaEventDestroy(start_fft);
  cudaEventDestroy(stop_fft);
  cudaEventDestroy(start_exec);
  cudaEventDestroy(stop_exec);
  

}

void fillGPUarray_8bit(int8_t *data, uint64_t n) {
  int8_t *r;
  uint64_t N, i;

  if (n<1024*1024) 
    N = n;
  else
    N = 1024*1024;

  r = (int8_t*)malloc(N*sizeof(int8_t));

  for (i=0; i<N; i++) {
    r[i] = random() & 0xFF;
  }

  i = 0;
  while (i < n) {
    N = n-i;
    if (N > 1024*1024) N = 1024*1024;

    cudaError_t status = cudaMemcpy (&data[i], r, N, cudaMemcpyHostToDevice);
    if (status != cudaSuccess) {
      fprintf(stderr, "Error: cudaMemcpy failed\n");
      exit(EXIT_FAILURE);
    }
    
    i += N;
  }

  free(r);

}
