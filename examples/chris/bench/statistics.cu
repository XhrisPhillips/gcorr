#include <stdio.h>
#include <stdlib.h>
#include <strings.h>
#include <getopt.h>

void postLaunchCheck();
void preLaunchCheck();
void cpuStats(float data[], int n);

#define FLOAT float

__global__ void sumSqr_kernel(float *data, FLOAT *results, unsigned int n) {

  extern __shared__ FLOAT sdata[];

  // perform first level of reduction,
  // reading from global memory, writing to shared memory
  
  unsigned int blockSize = blockDim.x;
  unsigned int tid = threadIdx.x;
  unsigned int i = blockIdx.x*blockSize*2 + threadIdx.x;
  unsigned int gridSize = blockSize*2*gridDim.x;

  FLOAT mySum = 0;
  FLOAT mySum2 = 0;

  // we reduce multiple elements per thread.  The number is determined by the 
  // number of active thread blocks (via gridDim).  More blocks will result
  // in a larger gridSize and therefore fewer elements per thread
  while (i < n) {         
    mySum += data[i];
    mySum2 += data[i]*data[i];
    mySum += data[i+blockSize];  
    mySum2 += data[i+blockSize]*data[i+blockSize];  
    i += gridSize;
  } 

  // each thread puts its local sum into shared memory 
  sdata[tid] = mySum;
  sdata[tid+blockSize] = mySum2;
  __syncthreads();

  // do reduction in shared mem
  if (blockSize >= 1024) { if (tid < 512) { 
      sdata[tid] = mySum = mySum + sdata[tid + 512]; 
      sdata[tid+blockSize] = mySum2 = mySum2 + sdata[tid + 512 + blockSize]; 
    } __syncthreads(); }
  if (blockSize >= 512) { if (tid < 256) { 
      sdata[tid] = mySum = mySum + sdata[tid + 256]; 
      sdata[tid+blockSize] = mySum2 = mySum2 + sdata[tid + 256 + blockSize]; 
    } __syncthreads(); }

  if (blockSize >= 256) { if (tid < 128) { 
      sdata[tid] = mySum = mySum + sdata[tid + 128]; 
      sdata[tid+blockSize] = mySum2 = mySum2 + sdata[tid + 128 + blockSize]; 
    } __syncthreads(); }


  if (blockSize >= 128) { if (tid <  64) { 
      sdata[tid] = mySum = mySum + sdata[tid +  64]; 
      sdata[tid+blockSize] = mySum2 = mySum2 + sdata[tid + 64 + blockSize]; 
    } __syncthreads(); }
    
  if (tid < 32) {
    // now that we are using warp-synchronous programming (below)
    // we need to declare our shared memory volatile so that the compiler
    // doesn't reorder stores to it and induce incorrect behavior.
    volatile FLOAT* smem = sdata;

    if (blockSize >=  64) { 
      smem[tid] = mySum = mySum + smem[tid + 32]; 
      smem[tid+blockSize] = mySum2 = mySum2 + smem[tid + 32 + blockSize]; 
    }
    if (blockSize >=  32) { 
      smem[tid] = mySum = mySum + smem[tid + 16];  
      smem[tid+blockSize] = mySum2 = mySum2 + smem[tid + 16 + blockSize]; 
   }
    if (blockSize >=  16) { 
      smem[tid] = mySum = mySum + smem[tid +  8]; 
      smem[tid+blockSize] = mySum2 = mySum2 + smem[tid + 8 + blockSize]; 
    }
    if (blockSize >=   8) { 
      smem[tid] = mySum = mySum + smem[tid +  4]; 
      smem[tid+blockSize] = mySum2 = mySum2 + smem[tid + 4 + blockSize]; 
    }
    if (blockSize >=   4) { 
      smem[tid] = mySum = mySum + smem[tid +  2]; 
      smem[tid+blockSize] = mySum2 = mySum2 + smem[tid + 2 + blockSize]; 
    }
    if (blockSize >=   2) { 
      smem[tid] = mySum = mySum + smem[tid +  1]; 
      smem[tid+blockSize] = mySum2 = mySum2 + smem[tid + 1 + blockSize]; 
    }
  }
    
  // write result for this block to global mem 
  if (tid == 0) {
    results[blockIdx.x] = sdata[0];
    results[blockIdx.x+gridDim.x] = sdata[blockSize];
  }
}



int main(int argc, char *argv[]) {
  int opt, tmp, ss, i;
  float *d_idata, *h_idata, dtime;
  FLOAT *d_results, *h_results;
  double sum, sum2;
  cudaError_t status;
  cudaEvent_t start_exec, stop_exec, start_test, stop_test;

  int nsample = 16*1024*1024; // Aprox 64 MB floating point data
  int nthread = 256;
  int repeat = 1;
  int nblock = 64;

  struct option options[] = {
    {"nthread", 1, 0, 'n'}, 
    {"nblock", 1, 0, 'b'}, 
    {"repeat", 1, 0, 'r'}, 
    {0, 0, 0, 0}
  };

  while (1) {
    opt=getopt_long_only(argc, argv, "n:r:b:", 
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

    case 'b':
      ss = sscanf(optarg, "%d", &tmp);
      if (ss!=1)
        fprintf(stderr, "Bad -nblock option %s\n", optarg);
      else {
	nblock = tmp;
      }
      break; 

    case '?':
    default:
      break;
    }
  }

  cudaEventCreate(&start_exec);
  cudaEventCreate(&stop_exec);
  cudaEventCreate(&start_test);
  cudaEventCreate(&stop_test);

  nsample = (nsample/(nblock*nthread))*nblock*nthread;   // Round

  printf("Number of samples = %d\n", nsample);
  printf("Number of repeats = %d\n", repeat);
  printf("Number of threads = %d\n", nthread);
  printf("Number of blocks = %d\n", nblock);
  printf("Processing approx %lu MB per iteration\n", nsample*sizeof(float)/1024/1024);

  int bytes = nsample*sizeof(float);

  // Allocate memory on the Device
  status = cudaMalloc(&d_idata, bytes);
  if (status != cudaSuccess) {
    fprintf(stderr, "Error: cudaMalloc failed\n");
    return EXIT_FAILURE;
  }

  status = cudaMalloc(&d_results, nblock*sizeof(FLOAT)*2);
  if (status != cudaSuccess) {
    fprintf(stderr, "Error: cudaMalloc failed\n");
    return EXIT_FAILURE;
  }

  // Allocate pinned Host memory 
  status = cudaMallocHost(&h_idata, bytes);
  if (status != cudaSuccess) {
    fprintf(stderr, "Error: cudaMallocHost failed\n");
    return EXIT_FAILURE;
  }

  status = cudaMallocHost(&h_results, nblock*sizeof(FLOAT)*2);
  if (status != cudaSuccess) {
    fprintf(stderr, "Error: cudaMallocHost failed\n");
    return EXIT_FAILURE;
  }

  // Fill Host memory with random data
  for (i=0; i<nsample; i++) {
    h_idata[i] = drand48()*4-1;
  }
  
  // Copy to device
  cudaMemcpy(d_idata, h_idata, bytes, cudaMemcpyHostToDevice);

  // Start total time event
  cudaEventRecord(start_exec, 0);

  sum = 0;
  sum2 = 0;

  preLaunchCheck();
  cudaEventRecord(start_test, 0);
  for (int j=0; j<repeat; j++) {
    sumSqr_kernel<<<nblock,nthread,nthread*sizeof(FLOAT)*2>>>(d_idata, d_results, nsample);

    // Copy partial result back to host and finalize sum on CPU
    cudaMemcpy(h_results, d_results, sizeof(FLOAT)*2*nblock, cudaMemcpyDeviceToHost);

    for (i=0; i<nblock; i++) {
      sum += h_results[i];
      sum2 +=  h_results[i+nblock];
    }
  }
  cudaEventRecord(stop_test, 0);
  cudaEventSynchronize(stop_test);
  cudaEventElapsedTime(&dtime, start_test, stop_test);
  postLaunchCheck();

  double n = (long)nsample*repeat;
  printf("\nGPU Average = %.8f\n", sum/n);
  printf("GPU SD      = %.8f\n\n", sqrt(sum2/n - sum*sum/(n*n)));

  cpuStats(h_idata, nsample);

  printf("That took %8.3f ms\n\n", dtime);

  printf("      |    Time     |    1 GHz     |   Realtime   |\n");
  printf("Stats | %8.3f ms |  %8.3f ms | %8.1f MHz | \n", dtime, dtime*4e9/((float)nsample*repeat), 
	 (float)nsample*repeat/4e6/dtime*1000);

  cudaEventRecord(stop_exec, 0);
  cudaEventSynchronize(stop_exec);
  cudaEventElapsedTime(&dtime, start_exec, stop_exec);
  printf("\nTotal executution time =  %.3f ms\n", dtime);


  // Free allocated memory
  cudaFree(d_idata);
  cudaFree(d_results);
  cudaFree(h_idata);

  cudaEventDestroy(start_test);
  cudaEventDestroy(stop_test);
  cudaEventDestroy(start_exec);
  cudaEventDestroy(stop_exec);

}

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

void cpuStats(float data[], int n) {
  int i;
  double sum=0, sum2=0;

  for (i=0; i<n; i++) {
    sum += data[i];
    sum2 += data[i]*data[i];
  }

  printf("\nCPU Average = %.8f\n", sum/n);
  printf("CPU SD      = %.8f\n\n", sqrt(sum2/n - sum*sum/((long)n*n)));;

} 