#include <stdio.h>
#include <stdlib.h>
#include <strings.h>
#include <sys/time.h>
#include <cuComplex.h>
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

void freeMem() {
  cudaError_t status;
  size_t free, total;
  status = cudaMemGetInfo(&free, &total);
  if (status != cudaSuccess) {
    fprintf(stderr, "Error: cudaMemGetInfo failed with %d\n", status); 
    exit(EXIT_FAILURE);							
  }
  printf("GPU memory available %.1f/%.1f MBbytes\n", free/1024.0/1024, total/1024.0/1024);
}


__global__ void fillArray(cuComplex *dest, int loop) {
  int nchan = blockDim.x * gridDim.x;
  const size_t i = threadIdx.x + blockIdx.x * blockDim.x + blockIdx.y * 2 *  nchan * loop;

  for (int n=0; n<loop*2; n++) {
    dest[i+nchan*n].x = sin((i+n*nchan)/(float)100.0)*3.5;
    dest[i+nchan*n].x = cos((i+n*nchan)/(float)100.0)*3.5;
  }
}


// Add complex number another number 
__host__ __device__ static __inline__ void cuCaddIf(cuFloatComplex *a, cuFloatComplex b)
{
  (*a).x += b.x;
  (*a).y += b.y;
}

// Multiply a complex number by the conjugate of the other
__host__ __device__ static __inline__ cuComplex cuCmulConjf (cuComplex x, cuComplex y)
{
    cuFloatComplex prod;
    prod = make_cuFloatComplex  ((cuCrealf(x) * cuCrealf(y)) + (cuCimagf(x) * cuCimagf(y)),
                                 (cuCimagf(x) * cuCrealf(y)) - cuCrealf(x) * cuCimagf(y));
    return prod;
}

/* Cross correlate and accumulate nant antenna data

   ants is an array of array pointers for each telescope. There are nant*2 arrays (dual pol)
   Each antenna array has nchan frequency points, repeated XX times.
   accum contains the cross correlatipn values - there is nant*(nant-1)*2*4 values repeated XX times

*/


__global__ void CrossCorr(cuComplex **ants, cuComplex **accum, int nant, int nchunk) { 

  int nchan = blockDim.x * gridDim.x;
  size_t ichan = threadIdx.x + blockIdx.x * blockDim.x + blockIdx.y * nchan * nchunk;
  const size_t ochan = threadIdx.x + blockIdx.x * blockDim.x + blockIdx.y * nchan;

  //printf("%ld %ld\n", ichan, ochan);

  
  int i,j, l, b;
  for (l=0; l<nchunk; l++) {
    b=0;
    for (i=0; i<nant-1; i++) {
      for (j=i+1; j<nant; j++) {
	cuCaddIf(&accum[b++][ochan], cuCmulConjf(ants[i*2][ichan], ants[j*2][ichan]));
	cuCaddIf(&accum[b++][ochan], cuCmulConjf(ants[i*2][ichan], ants[j*2+1][ichan]));
	cuCaddIf(&accum[b++][ochan], cuCmulConjf(ants[i*2+1][ichan], ants[j*2][ichan]));
	cuCaddIf(&accum[b++][ochan], cuCmulConjf(ants[i*2+1][ichan], ants[j*2+1][ichan]));
      }
    }
    ichan += nchan*2;
  }
}

__global__ void CrossCorrShared(cuComplex **ants, cuComplex **accum, int nant, int nchunk) { 

  extern __shared__ cuComplex antShar[];
  
  int nchan = blockDim.x * gridDim.x;
  size_t ichan = threadIdx.x + blockIdx.x * blockDim.x + blockIdx.y * nchan * nchunk;
  const size_t ochan = threadIdx.x + blockIdx.x * blockDim.x + blockIdx.y * nchan;

  //printf("%ld %ld\n", ichan, ochan);

  int i,j, l, b;
  for (l=0; l<nchunk; l++) {
    if (threadIdx.x<nant*2) antShar[threadIdx.x] = ants[threadIdx.x][ichan];
    __syncthreads();
    
    b=0;
    for (i=0; i<nant-1; i++) {
      for (j=i+1; j<nant; j++) {
	cuCaddIf(&accum[b++][ochan], cuCmulConjf(antShar[i*2], antShar[j*2]));
	cuCaddIf(&accum[b++][ochan], cuCmulConjf(antShar[i*2], antShar[j*2+1]));
	cuCaddIf(&accum[b++][ochan], cuCmulConjf(antShar[i*2+1], antShar[j*2]));
	cuCaddIf(&accum[b++][ochan], cuCmulConjf(antShar[i*2+1], antShar[j*2+1]));
      }
    }
    ichan += nchan*2;
  }
}

__global__ void CrossCorrShared2(cuComplex **ants, cuComplex **accum, int nant, int nchunk) { 

  extern __shared__ cuComplex antShar[];
  
  int nchan = blockDim.x * gridDim.x;
  size_t ichan = threadIdx.x + blockIdx.x * blockDim.x + blockIdx.y * nchan * nchunk;
  const size_t ochan = threadIdx.x + blockIdx.x * blockDim.x + blockIdx.y * nchan;

  int i,j, l, b;
  if (threadIdx.x<nant*2 && blockIdx.y<nchunk)
    antShar[threadIdx.x + blockIdx.y*nant*2] = ants[threadIdx.x][ichan+blockIdx.y*nchan*2];
  __syncthreads();
  
  for (l=0; l<nchunk; l++) {
    
    b=0;
    for (i=0; i<nant-1; i++) {
      for (j=i+1; j<nant; j++) {
	cuCaddIf(&accum[b++][ochan], cuCmulConjf(antShar[i*2+l*nant*2], antShar[j*2l*nant*2]));
	cuCaddIf(&accum[b++][ochan], cuCmulConjf(antShar[i*2l*nant*2], antShar[j*2+1+l*nant*2]));
	cuCaddIf(&accum[b++][ochan], cuCmulConjf(antShar[i*2+1+l*nant*2], antShar[j*2l*nant*2]));
	cuCaddIf(&accum[b++][ochan], cuCmulConjf(antShar[i*2+1l*nant*2], antShar[j*2+1+l*nant*2]));
      }
    }
  }
}


__global__ void CrossCorrShared3(cuComplex **ants, cuComplex **accum, int nant, int nchunk) { 

  extern __shared__ cuComplex sharData[];

  cuComplex *antShar = sharData;
  cuComplex *accumShar = &sharData[nant*2];
  
  int nchan = blockDim.x * gridDim.x;
  size_t ichan = threadIdx.x + blockIdx.x * blockDim.x + blockIdx.y * nchan * nchunk;
  const size_t ochan = threadIdx.x + blockIdx.x * blockDim.x + blockIdx.y * nchan;

  //printf("%ld %ld\n", ichan, ochan);

  int i,j, l, b;
  for (l=0; l<nchunk; l++) {
    if (threadIdx.x<nant*2) antShar[threadIdx.x] = ants[threadIdx.x][ichan];
    __syncthreads();
    
    b=0;
    for (i=0; i<nant-1; i++) {
      for (j=i+1; j<nant; j++) {
	cuCaddIf(&accum[b++][ochan], cuCmulConjf(antShar[i*2], antShar[j*2]));
	cuCaddIf(&accum[b++][ochan], cuCmulConjf(antShar[i*2], antShar[j*2+1]));
	cuCaddIf(&accum[b++][ochan], cuCmulConjf(antShar[i*2+1], antShar[j*2]));
	cuCaddIf(&accum[b++][ochan], cuCmulConjf(antShar[i*2+1], antShar[j*2+1]));
      }
    }
    ichan += nchan*2;
  }
}



int main(int argc, char *argv[]) {
  int opt, tmp, ss;
  float dtime;
  cudaError_t status;

  cuComplex **antData_h, **baseline_h, **antData, **baseline;
  cudaEvent_t start_exec, stop_exec;
  int threads, parallelAccum;
  dim3 blocks;
  
  int nchan = 128;
  int nant = 6;
  int targetThreads = 50e4; 
  int repeat = 24;
  int mem = DEFAULTMEM; 

  struct option options[] = {
    {"repeat", 1, 0, 'r'}, 
    {"nchan", 1, 0, 'n'}, 
    {"nant", 1, 0, 'N'}, 
    {"memory", 1, 0, 'm'}, 
    {"targetthreads", 1, 0, 't'}, 
    {0, 0, 0, 0}
  };

  while (1) {
    opt=getopt_long_only(argc, argv, "n:r:m:N:t:", 
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

    case 't':
      ss = sscanf(optarg, "%d", &tmp);
      if (ss!=1)
        fprintf(stderr, "Bad -targetthreads option %s\n", optarg);
      else {
	targetThreads = tmp;
      }
      break; 

    case 'n':
      ss = sscanf(optarg, "%d", &tmp);
      if (ss!=1)
        fprintf(stderr, "Bad -nchan option %s\n", optarg);
      else {
	nchan = tmp;
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
        fprintf(stderr, "Bad -nant option %s\n", optarg);
      else {
	nant = tmp;
      }
      break; 

    case '?':
    default:
      break;
    }
  }

  freeMem();
  
  gpuErrchk(cudaEventCreate(&start_exec));
  gpuErrchk(cudaEventCreate(&stop_exec));
  
  mem *= 1024*1024;

  int blockchan;
  if (nchan<=512) {
    threads = nchan;
    blockchan = 1;
  } else {
    threads = 512;
    blockchan = nchan/512;
  }
  parallelAccum = targetThreads/nchan;
  blocks = dim3(blockchan, parallelAccum);
  printf("Threads:  %d, %d, %d\n", threads, blockchan, parallelAccum);
  

  int memPerChunk = nant*2*nchan*sizeof(cuComplex)*parallelAccum;
  int nchunk = mem / memPerChunk;
  if (nchunk==0) nchunk=1;
  int nbaseline = nant*(nant-1)/2;
  
  printf("DEBUG: nchunk=%d\n", nchunk);
  
  // Allocate memory on the GPU
  antData_h = (cuComplex**)malloc(nant*2*sizeof(cuComplex*));
  if (antData_h==NULL) {
      fprintf(stderr, "Error: malloc failed\n");
      return EXIT_FAILURE;
  }
  
  printf("Alloc %lld elements\n", (long long)nchan*parallelAccum*nchunk);
  for (int i=0; i<nant*2; i++) {
    status = cudaMalloc(&antData_h[i], nchan*parallelAccum*nchunk*sizeof(cuComplex));
    if (status != cudaSuccess) {
      fprintf(stderr, "Error: cudaMalloc failed\n");
      return EXIT_FAILURE;
    }
  }
  status = cudaMalloc(&antData, nant*2*sizeof(cuComplex*));
  if (status != cudaSuccess) {
    fprintf(stderr, "Error: cudaMalloc failed\n");
    return EXIT_FAILURE;
  }
  gpuErrchk(cudaMemcpy(antData, antData_h, nant*2*sizeof(cuComplex*), cudaMemcpyHostToDevice));

  baseline_h = (cuComplex**)malloc(nbaseline*4*sizeof(cuComplex*));
  if (baseline_h==NULL) {
      fprintf(stderr, "Error: malloc failed\n");
      return EXIT_FAILURE;
  }
  for (int i=0; i<nbaseline*4; i++) {
    status = cudaMalloc(&baseline_h[i], nchan*parallelAccum*sizeof(cuComplex));
    if (status != cudaSuccess) {
      fprintf(stderr, "Error: cudaMalloc failed (1)\n");
      return EXIT_FAILURE;
    }
    gpuErrchk(cudaMemset(baseline_h[i], 0, nchan*parallelAccum*sizeof(cuComplex)));
    
  }
  status = cudaMalloc(&baseline, nbaseline*4*sizeof(cuComplex*));
  if (status != cudaSuccess) {
    fprintf(stderr, "Error: cudaMalloc failed\n");
    return EXIT_FAILURE;
  }
  gpuErrchk(cudaMemcpy(baseline, baseline_h, nbaseline*4*sizeof(cuComplex*), cudaMemcpyHostToDevice));
  
  freeMem();

  for (int i=0;i<nant;i++) {
    fillArray<<<blocks,threads>>>(antData_h[i],nchunk);
    CudaCheckError();
  }
  
  // Start total time event
  gpuErrchk(cudaEventRecord(start_exec, 0));
  for (int i=0; i<repeat; i++) {  
    // Cross Correlate
    CrossCorr<<<blocks,threads>>>(antData, baseline, nant, nchunk);
    CudaCheckError();
  }
  gpuErrchk(cudaEventRecord(stop_exec, 0));
  gpuErrchk(cudaEventSynchronize(stop_exec));
  gpuErrchk(cudaEventElapsedTime(&dtime, start_exec, stop_exec));
  printf("That took %.3f ms\n", dtime);

  // Start total time event
  gpuErrchk(cudaEventRecord(start_exec, 0));
  for (int i=0; i<repeat; i++) {  
    // Cross Correlate
    CrossCorrShared<<<blocks,threads,nant*2*sizeof(cuComplex)>>>(antData, baseline, nant, nchunk);
    CudaCheckError();
  }
   
  gpuErrchk(cudaEventRecord(stop_exec, 0));
  gpuErrchk(cudaEventSynchronize(stop_exec));
  gpuErrchk(cudaEventElapsedTime(&dtime, start_exec, stop_exec));

  printf("Shared took %.3f ms\n", dtime);

  // Start total time event
  gpuErrchk(cudaEventRecord(start_exec, 0));
  for (int i=0; i<repeat; i++) {  
    // Cross Correlate
    CrossCorrShared2<<<blocks,threads,nant*2*nchunk*sizeof(cuComplex)>>>(antData, baseline, nant, nchunk);
    CudaCheckError();
  }
  gpuErrchk(cudaEventRecord(stop_exec, 0));
  gpuErrchk(cudaEventSynchronize(stop_exec));
  gpuErrchk(cudaEventElapsedTime(&dtime, start_exec, stop_exec));
  printf("Shared2 took %.3f ms\n", dtime);


  // Start total time event
  gpuErrchk(cudaEventRecord(start_exec, 0));
  for (int i=0; i<repeat; i++) {  
    // Cross Correlate
    CrossCorrShared3<<<blocks,threads,nant*2*sizeof(cuComplex)>>>(antData, baseline, nant, nchunk);
    CudaCheckError();
  }
  gpuErrchk(cudaEventRecord(stop_exec, 0));
  gpuErrchk(cudaEventSynchronize(stop_exec));
  gpuErrchk(cudaEventElapsedTime(&dtime, start_exec, stop_exec));
  printf("Shared3 took %.3f ms\n", dtime);
  
  // Free allocated memory
  for (int i=0; i<nant*2; i++) {
    cudaFree(antData_h[i]);
  }
  for (int i=0; i<nbaseline*4; i++) {
    cudaFree(baseline_h[i]);
  }
  cudaFree(antData);
  cudaFree(baseline);
  free(antData_h);
  free(baseline_h);
  
  cudaEventDestroy(start_exec);
  cudaEventDestroy(stop_exec);

}

