#include <stdio.h>
#include <stdlib.h>
#include <strings.h>
#include <cuda.h>
#include <curand.h>
#include <getopt.h>

#define TIMING

#define gpuErrchk(ans) { __gpuErrchk((ans), __FILE__, __LINE__); }
inline void __gpuErrchk(cudaError_t code, const char *file, int line, bool abort=true) {
  if (code != cudaSuccess) {
    fprintf(stderr,"gpuErrchk: %s %s %d\n", cudaGetErrorString(code), file, line);
    if (abort) exit(code);
  }
}

#define CudaCheckError()  __cudaCheckError( __FILE__, __LINE__ )
inline void __cudaCheckError( const char *file, const int line ) {
  cudaError err = cudaGetLastError();
  if ( cudaSuccess != err ) {
    fprintf( stderr, "cudaCheckError() failed at %s:%i : %s\n",
	     file, line-1, cudaGetErrorString( err ) );
    exit(err);
  }
}

#define CURAND_CALL(x) {__curand_call((x), __FILE__, __LINE__); }
inline void __curand_call(curandStatus_t code, const char *file, int line) {
  if (code != CURAND_STATUS_SUCCESS) {	
    fprintf(stderr, "Curand error (%d) at %s:%d\n", code, file, line);	
    exit(EXIT_FAILURE); 
  }
}

#ifdef TIMING
#define STARTTIME(x)  cudaEventRecord(start_ ## x, 0);
#define RESETTIME(x) x ## time = 0.0;

#define STOPTIME(x) {\
  cudaEventRecord(stop_ ## x, 0); \
  cudaEventSynchronize(stop_ ## x); \
  cudaEventElapsedTime(&dtime, start_ ## x, stop_ ## x); \
  x ## time += dtime; }


#define TIMINGEVENT(x) \
  cudaEvent_t start_ ## x, stop_ ## x; \
  double x ## time = 0.0; \
  gpuErrchk(cudaEventCreate(&start_ ## x)); \
  gpuErrchk(cudaEventCreate(&stop_ ## x));

#else
#define STARTTIME(x)
#define STOPTIME(x)
#define RESETTIME(x)
#endif

__global__ void warmup(float *input, float *output) {

  const int i = threadIdx.x + blockIdx.x * blockDim.x;
  output[i] = input[i] * input[i];
}


__global__ void pfbFilter(float *filtered, float *unfiltered, float *taps, const int ntaps) {

  const int nfft = blockDim.x;
  const int i = threadIdx.x + threadIdx.y*blockDim.x + blockIdx.x * blockDim.x * blockDim.y;
  
  filtered[i] = unfiltered[i] * taps[threadIdx.x];
  for (int j=1; j<ntaps; j++) {
    filtered[i] += unfiltered[i + j*nfft] * taps[threadIdx.x + j*nfft];
  }
}


__global__ void pfbFilter4(float *filtered, float *unfiltered, float *taps, const int ntaps) {

  const int nfft = blockDim.x;
  const int i = threadIdx.x + threadIdx.y*blockDim.x*4 + blockIdx.x*blockDim.x*blockDim.y*4;
  
  filtered[i] = unfiltered[i] * taps[threadIdx.x];
  filtered[i+nfft] = unfiltered[i+nfft] * taps[threadIdx.x];
  filtered[i+nfft*2] = unfiltered[i+nfft*2] * taps[threadIdx.x];
  filtered[i+nfft*3] = unfiltered[i+nfft*3] * taps[threadIdx.x];
  for (int j=1; j<ntaps; j++) {
    filtered[i] += unfiltered[i + j*nfft] * taps[threadIdx.x + j*nfft];
    filtered[i+nfft] += unfiltered[i + (j+1)*nfft] * taps[threadIdx.x + j*nfft];
    filtered[i+nfft] += unfiltered[i + (j+2)*nfft] * taps[threadIdx.x + j*nfft];
    filtered[i+nfft] += unfiltered[i + (j+3)*nfft] * taps[threadIdx.x + j*nfft];
  }
}


__global__ void initTaps(float *taps) {
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  float x;

  int nfft = gridDim.x;
  int ntap = blockDim.x;
  int M = nfft * ntap;
  
  x = (float)i/float(nfft) - (float)ntap/2.0;
  if (x==0.0)
    x = 1;
  else
    x = sinpif(x)/(x*M_PI); // Sinc
  taps[i] = x*(0.5 - 0.5*cospif(2*(float)i/(float)(M-1)));
}

#define MAXTAPS 256
__device__ __constant__ float ctaps[MAXTAPS];

void initConstTaps(int nfft, int ntaps) {
  int i;
  int M = nfft * ntaps;
  float x, mytaps[MAXTAPS];


  if (M>MAXTAPS) {
    fprintf(stderr, "Error: MAXTAPS < nfft * ntaps (%d: %dx%d)\n", MAXTAPS, nfft, ntaps);
    exit(1);
  }
  
  for (i=0; i<M; i++) {
  
    x = (float)i/float(nfft) - (float)ntaps/2.0;
    if (x==0.0)
      x = 1;
    else
      x = sin(x*M_PI)/(x*M_PI); // Sinc
    mytaps[i] = x*(0.5 - 0.5*cos(2*(float)i/(float)(M-1))*M_PI);
    gpuErrchk(cudaMemcpyToSymbol(ctaps, mytaps, sizeof(ctaps)));
  }
}

__global__ void pfbFilterConstant(float *filtered, float *unfiltered, const int ntaps) {

  const int nfft = blockDim.x;
  const int i = threadIdx.x + threadIdx.y*blockDim.x + blockIdx.x * blockDim.x * blockDim.y;

  filtered[i] = unfiltered[i] * ctaps[threadIdx.x];
  for (int j=1; j<ntaps; j++) {
    filtered[i] += unfiltered[i + j*nfft] * ctaps[threadIdx.x + j*nfft];
  }
}

__global__ void pfbFilterShared(float *filtered, float *unfiltered, float *taps, const int ntaps) {
  extern __shared__ float shared_taps[];
  
  const int nfft = blockDim.x;
  const int i = threadIdx.x + threadIdx.y*blockDim.x + blockIdx.x * blockDim.x * blockDim.y;
  if (i<ntaps*nfft) {
    shared_taps[i] = taps[i];
  }
  __syncthreads();

  
  filtered[i] = unfiltered[i] * shared_taps[threadIdx.x];
  for (int j=1; j<ntaps; j++) {
    filtered[i] += unfiltered[i + j*nfft] * shared_taps[threadIdx.x + j*nfft];
  }
}



int main(int argc, char *argv[]) {
  int opt, tmp, i, status, blocks;
  float *taps, *input, *output;
  curandGenerator_t gen;
  dim3 threads;
  
#ifdef TIMING
  float dtime;
  TIMINGEVENT(filter);
  TIMINGEVENT(exec);
#endif

  int repeat = 10;
  int ntap = 16;
  int nfft = 16;
  int nsample = 1024*1024*4;
  int nthread = 512;

  struct option options[] = {
    {"repeat", 1, 0, 'r'}, 
    {"ntap", 1, 0, 't'}, 
    {"nfft", 1, 0, 'n'}, 
    {"nthread", 1, 0, 'T'}, 
    {0, 0, 0, 0}
  };

#define CASEINT(ch,var)                                     \
  case ch:                                                  \
    status = sscanf(optarg, "%d", &tmp);                    \
    if (status!=1)                                          \
      fprintf(stderr, "Bad %s option %s\n", #var, optarg);  \
    else                                                    \
      var = tmp;                                            \
    break

  
  while (1) {
    opt=getopt_long_only(argc, argv, "n:r:t:T:", 
			 options, NULL);

    if (opt==EOF) break;

    switch (opt) {

      CASEINT('r', repeat);
      CASEINT('t', ntap);
      CASEINT('n', nfft);
      CASEINT('T', nthread);
      
    case '?':
    default:
      break;
    }
  }

  // Check numerology works

  if (nthread % nfft) {
    fprintf(stderr, "Inconsistent number of threads and fft channels (%d,%d)\n", nthread, nfft);
    exit(1);
  }

  // Nsample needs to be divisible by nfft and other stuff

  while (nsample) {
    if ((nsample%nfft==0) && (nsample%nthread==0) && (nsample/4%nthread==0)) break;
    nsample--;
  }
  if (nsample==0) {
    fprintf(stderr, "Cannot fit samples into %d fft points\n", nfft);
    exit(1);
  }

  threads = dim3(nfft, nthread/nfft);
  blocks = nsample/nthread;

  printf("nsample = %d\n", nsample);
  printf("Blocks = %d\n", blocks);
  printf("Threads = (%d,%d)\n", threads.x, threads.y);
  
  // Initialise Taps
  gpuErrchk(cudaMalloc(&taps, (nfft*ntap)*sizeof(float)));
  initTaps<<<nfft,ntap>>>(taps);
  CudaCheckError();

  initConstTaps(nfft,ntap);
  
  // Allocate memory on the host
  gpuErrchk(cudaMalloc(&input, (nsample+(ntap-1)*nfft)*sizeof(float)));
  gpuErrchk(cudaMalloc(&output, nsample*sizeof(float)));

  // Fill input array with Guassian noise

  CURAND_CALL(curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT));
  CURAND_CALL(curandSetPseudoRandomGeneratorSeed(gen, time(NULL)));
  CURAND_CALL(curandGenerateNormal(gen, input, (nsample+(ntap-1)*nfft), 0.0, 10.0)); 
  CURAND_CALL(curandDestroyGenerator(gen));

  //warmup<<<nsample/512,512>>>(input, output);
  // Run each kernel once as a warmup
  pfbFilter<<<blocks,threads>>>(output, input, taps, ntap);
  CudaCheckError();
  pfbFilter4<<<blocks/4,threads>>>(output, input, taps, ntap);
  CudaCheckError();
  pfbFilterConstant<<<blocks,threads>>>(output, input, ntap);
  CudaCheckError();
  pfbFilterShared<<<blocks,threads, ntap*nfft*sizeof(float)>>>(output, input, taps, ntap);    
  CudaCheckError();
  
  // Start total time event
  STARTTIME(exec);

  for (i=0; i<repeat; i++) {
    STARTTIME(filter);
    pfbFilter<<<blocks,threads>>>(output, input, taps, ntap);
    CudaCheckError();
    STOPTIME(filter);
  }
#ifdef TIMING
  printf("Filter       %3.0f msec\n", filtertime);
#endif
 

  RESETTIME(filter);
  for (i=0; i<repeat; i++) {
    STARTTIME(filter);
    pfbFilter4<<<blocks/4,threads>>>(output, input, taps, ntap);
    CudaCheckError();
    STOPTIME(filter);
  }
#ifdef TIMING
  printf("Filter4      %3.0f msec\n", filtertime);
#endif

  RESETTIME(filter);
  for (i=0; i<repeat; i++) {
    STARTTIME(filter);
    pfbFilterConstant<<<blocks,threads>>>(output, input, ntap);
    CudaCheckError();
    STOPTIME(filter);
  }
#ifdef TIMING
  printf("FilterConst  %3.0f msec\n", filtertime);
#endif

  RESETTIME(filter);
  for (i=0; i<repeat; i++) {
    STARTTIME(filter);
    pfbFilterShared<<<blocks,threads, ntap*nfft*sizeof(float)>>>(output, input, taps, ntap);
    CudaCheckError();
    STOPTIME(filter);
  }
#ifdef TIMING
  printf("FilterShared %3.0f msec\n", filtertime);
#endif


  
  STOPTIME(exec);
#ifdef TIMING
  printf("\n");
  printf("Total time %.1f sec\n\n", exectime/1000);
#endif

  // Free allocated memory
  cudaFree(taps);
  cudaFree(input);
  cudaFree(output);
}
