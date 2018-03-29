#include <stdio.h>
#include <stdlib.h>
#include <strings.h>
#include <sys/time.h>
#include <cufft.h>
#include <getopt.h>

#define DEFAULTMEM 500 // MBytes

#define MAXBLOCKS 65535  // Maximum number of blocks in the grid

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess) 
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
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


__global__ void unpack8bitcomplex(const int8_t *src, cuFloatComplex *dest) {

  const size_t i = blockDim.x * blockIdx.x + threadIdx.x;

  dest[i] = make_cuFloatComplex(src[i*2], src[i*2+1]);
}

__global__ void unpack8bitcomplex2(const int8_t *src, cuFloatComplex *dest1, cuFloatComplex *dest2) {

  const size_t i = blockDim.x * blockIdx.x + threadIdx.x;
  dest1[i] = make_cuFloatComplex(src[i*4], src[i*4+1]);
  dest2[i] = make_cuFloatComplex(src[i*4+2], src[i*4+3]);
}

__global__ void unpack8bitcomplex3(const int8_t *src, cuFloatComplex *dest1, cuFloatComplex *dest2, int loop) {

  const size_t i = blockDim.x * blockIdx.x + threadIdx.x;
  const size_t k = blockDim.x * gridDim.x;

  for (int n=0; n<loop; n++) {
    dest1[i+n*k] = make_cuFloatComplex(src[(i+n*k)*4], src[(i+n*k)*4+1]);
    dest2[i+n*k] = make_cuFloatComplex(src[(i+n*k)*4+2], src[(i+n*k)*4+3]);
  }
}


__global__ void fillArray(int8_t *dest, int loop) {
  const size_t i = blockDim.x * blockIdx.x + threadIdx.x;
  const size_t k = blockDim.x * gridDim.x;
  for (int n=0; n<loop; n++) {
    dest[i+n*k] = sin((i+n*k)/(float)100.0)*30;
  }
}

__global__ void PowerInterleaved(float4 *src, float4 *dest) {
 
  const size_t i = blockDim.x * blockIdx.x + threadIdx.x;

  // Cross pols
  dest[i].x  += src[i].x * src[i].x + src[i].y * src[i].y;
  dest[i].y  += src[i].z * src[i].z + src[i].w * src[i].w;
  // Parallel pols
  dest[i].z += src[i].x * src[i].z + src[i].y * src[i].w;
  dest[i].w += src[i].y * src[i].z - src[i].x * src[i].w;
}

__global__ void Power(cufftComplex *src1, cufftComplex *src2, float4 *dest) {
 
  const size_t i = blockDim.x * blockIdx.x + threadIdx.x;

  // Cross pols
  dest[i].x  += src1[i].x * src1[i].x + src1[i].y * src1[i].y;
  dest[i].y  += src2[i].x * src2[i].x + src2[i].y * src2[i].y;
  // Parallel pols
  dest[i].z += src1[i].x * src2[i].x + src1[i].y * src2[i].y;
  dest[i].w += src1[i].y * src2[i].x - src1[i].x * src2[i].y;
}

__global__ void Accumulate(float4 *src, float4 *dest, int loop) {
  const size_t i = blockDim.x * blockIdx.x + threadIdx.x;
  const size_t k = blockDim.x * gridDim.x;

  dest[i] = src[i];

  for (int n=1; n<loop; n++) {
    dest[i].x  += src[i+n*k].x;
    dest[i].y  += src[i+n*k].y;
    dest[i].z  += src[i+n*k].z;
    dest[i].w  += src[i+n*k].w;
  }
}

__global__ void PowerAccumulate(cufftComplex *src1, cufftComplex *src2, float4 *dest, int loop) { 
  const size_t i = blockDim.x * gridDim.y * blockIdx.x + threadIdx.x + blockIdx.y * blockDim.x;
  const size_t k = blockDim.x * gridDim.x * gridDim.y;

  // Cross pols
  dest[i].x  += src1[i].x * src1[i].x + src1[i].y * src1[i].y;
  dest[i].y  += src2[i].x * src2[i].x + src2[i].y * src2[i].y;
  // Parallel pols
  dest[i].z += src1[i].x * src2[i].x + src1[i].y * src2[i].y;
  dest[i].w += src1[i].y * src2[i].x - src1[i].x * src2[i].y;

  for (int n=1; n<loop; n++) {
    dest[i].x  += src1[i+n*k].x * src1[i+n*k].x + src1[i+n*k].y * src1[i+n*k].y;
    dest[i].y  += src2[i+n*k].x * src2[i+n*k].x + src2[i+n*k].y * src2[i+n*k].y;;
    dest[i].z  += src1[i+n*k].x * src2[i+n*k].x + src1[i+n*k].y * src2[i+n*k].y;
    dest[i].w  += src1[i+n*k].y * src2[i+n*k].x - src1[i+n*k].x * src2[i+n*k].y;
  }
}

#define TIMING

#ifdef TIMING
#define STARTTIME(x)  cudaEventRecord(start_ ## x, 0);

#define STOPTIME(x) {\
    cudaEventRecord(stop_ ## x, 0); \
    cudaEventSynchronize(stop_ ## x); \
    cudaEventElapsedTime(&dtime, start_ ## x, stop_ ## x); \
    x ## time += dtime; }

#else
#define STARTTIME(x) 
#define STOPTIME(x)
#endif

int main(int argc, char *argv[]) {
  int batch, opt, tmp, ss, blocks, nloop;
  size_t nsample, free, total;
  float dtime;
  cudaError_t status;
  int8_t *idata;
  cufftComplex *cdata, *odata;
  cufftResult result;
  cufftHandle plan;
  cudaEvent_t start_exec, stop_exec;
#ifdef TIMING
  cudaEvent_t start_float, stop_float, start_fft, stop_fft, start_power, stop_power, start_accumulate, stop_accumulate;
  double floattime, ffttime, accumulatetime;
#endif
  int nthread = 256;
  int nfft = 1024;
  int averagethreads = 5e4; 
  int repeat = 10;
  int mem = DEFAULTMEM; 

  struct option options[] = {
    {"repeat", 1, 0, 'r'}, 
    {"nfft", 1, 0, 'n'}, 
    {"memory", 1, 0, 'm'}, 
    {"nthread", 1, 0, 'N'}, 
    {0, 0, 0, 0}
  };

  while (1) {
    opt=getopt_long_only(argc, argv, "n:r:m:N:", 
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
        fprintf(stderr, "Bad -nfft option %s\n", optarg);
      else {
	nfft = tmp;
      }
      break; 

    case 'm':
      ss = sscanf(optarg, "%d", &tmp);
      if (ss!=1)
        fprintf(stderr, "Bad -mem option %s\n", optarg);
      else {
	mem = tmp;
      }
      break; 

    case 'N':
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

  gpuErrchk(cudaEventCreate(&start_exec));
  gpuErrchk(cudaEventCreate(&stop_exec));
  
#ifdef TIMING
  gpuErrchk(cudaEventCreate(&start_float));
  gpuErrchk(cudaEventCreate(&stop_float));
  gpuErrchk(cudaEventCreate(&start_fft));
  gpuErrchk(cudaEventCreate(&stop_fft));
  gpuErrchk(cudaEventCreate(&start_power));
  gpuErrchk(cudaEventCreate(&stop_power));
  gpuErrchk(cudaEventCreate(&start_accumulate));
  gpuErrchk(cudaEventCreate(&stop_accumulate));
  floattime = 0;
  //powertime = 0;
  accumulatetime = 0;
  ffttime = 0;
#endif

  //  2 pol
  nsample = mem*1024*1024/((1+4+4)*2*2);  // Byte->float->float * complex * 2pol
  nsample = (nsample/(nfft*8))*(nfft*8); // Round to multiple of fft size
  batch = nsample/nfft;

  // How many blocks to run for the accumulation - make sure it fits with an even number
  
  int nchunk = averagethreads/nfft+2;
  while (nsample % (nfft*nchunk)) {
    nchunk--;
  }
  int avloop = nsample/(nchunk*nfft);

  int accumThreads = nfft;
  dim3 accumBlocks(nchunk);
  if (nfft>1024) {
    accumThreads = 1024;
    accumBlocks.y = nfft/1024;
  }

  printf("accumBlocks: %d,%d,%d\n",accumBlocks.x, accumBlocks.y, accumBlocks.z);
  printf("accumThreads=%d\n",accumThreads);
  printf("totalThreads=%d\n",accumThreads*accumBlocks.x*accumBlocks.y);
  printf("Avloop=%d\n", avloop);
  printf("Nsample=%d\n", nsample);
  printf("batch=%d\n", batch);
  printf("nchunk=%d\n",nchunk);
  printf("nfft=%d\n",nfft);
  printf("Total Accum=%d\n", accumBlocks.x*accumBlocks.y*avloop*accumThreads);
  
  // Allocate memory on the host
  status = cudaMalloc(&idata, nsample*sizeof(int8_t)*4);  // 2 pol & complex
  if (status != cudaSuccess) {
    fprintf(stderr, "Error: cudaMalloc failed (1)\n");
    return EXIT_FAILURE;
  }
  status = cudaMalloc(&cdata, nsample*sizeof(cufftComplex)*2); // 2pol
  if (status != cudaSuccess) {
    fprintf(stderr, "Error: cudaMalloc failed (2)\n");
    return EXIT_FAILURE;
  }
  status = cudaMalloc(&odata, nsample*sizeof(cufftComplex)*2); // 2pol
  if (status != cudaSuccess) {
    fprintf(stderr, "Error: cudaMalloc failed (3)\n");
    return EXIT_FAILURE;
  }

  status = cudaMemGetInfo(&free, &total);
  if (status != cudaSuccess) {
    fprintf(stderr, "Error: cudaMemGetInfo failed with %d\n", status); 
  } else {
    printf("GPU available memory %.1f/%.1f MBbytes\n", free/1024.0/1024, total/1024.0/1024);
  }

  
  nloop = 1;
  blocks = nsample*4/nthread;
  while (blocks>MAXBLOCKS) {
    nloop *=2;
    blocks = nsample*4/nthread/nloop;
  }
  fillArray<<<blocks,nthread>>>(idata,nloop);
  CudaCheckError();

#ifdef TIMING
  floattime = 0;
  //powertime = 0;
  ffttime = 0;
  accumulatetime = 0;
#endif

  // Assume interleaved samples
  result = cufftPlanMany(&plan, 1, &nfft, NULL, 1, 0, NULL, 1, 0, CUFFT_C2C, batch*2);
  if (result != CUFFT_SUCCESS) {
    fprintf(stderr, "cufftPlanMany failed with status %d\n", result);
    return EXIT_FAILURE;
  }
//  cufftSetCompatibilityMode(plan, CUFFT_COMPATIBILITY_NATIVE);
  
  // Start total time event
  gpuErrchk(cudaEventRecord(start_exec, 0));


  for (int i=0; i<repeat; i++) {  

    // Convert to float
    STARTTIME(float);
    unpack8bitcomplex2<<<nsample/nthread,nthread>>>(idata, cdata, &cdata[nsample]);
    CudaCheckError();
    STOPTIME(float);

    STARTTIME(fft);
    result = cufftExecC2C(plan, cdata, odata, CUFFT_FORWARD);
    CudaCheckError();
    STOPTIME(fft);

    if (nfft>1024) {
      STARTTIME(accumulate);
      PowerAccumulate<<<accumBlocks,accumThreads>>>(odata,&odata[nsample], (float4*)cdata, avloop);
      CudaCheckError();
      STOPTIME(accumulate);
    } else {
      STARTTIME(accumulate);
      PowerAccumulate<<<accumBlocks,accumThreads>>>(odata,&odata[nsample], (float4*)cdata, avloop);
      CudaCheckError();
      STOPTIME(accumulate);
    }
  }
   
  gpuErrchk(cudaEventRecord(stop_exec, 0));
  gpuErrchk(cudaEventSynchronize(stop_exec));
  gpuErrchk(cudaEventElapsedTime(&dtime, start_exec, stop_exec));
  
  printf("   n   |    time     |    1 GHz    |  Bandwidth |  **DUAL POL**\n");
  printf("%6d | %8.3f ms | %8.3f ms | %6.1f MHz |\n", nfft, dtime, dtime*1e9/((float)nfft*batch*repeat),
	 (float)nfft*batch*repeat/1e6/(dtime/1000));

#ifdef TIMING
  printf("\n");
  printf("Integer->float  %.0f msec\n", floattime);
  printf("FFT             %.0f msec\n", ffttime);
  //printf("Power           %.0f msec\n", powertime);
  printf("Accumulate      %.0f msec\n", accumulatetime);
#endif

  // Destroy plan
  cufftDestroy(plan);

  // Free allocated memory
  cudaFree(idata);
  cudaFree(cdata);
  cudaFree(odata);

  //cudaEventDestroy(start_fft);
  //cudaEventDestroy(stop_fft);
  cudaEventDestroy(start_exec);
  cudaEventDestroy(stop_exec);

}

