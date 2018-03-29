#include <stdio.h>
#include <stdlib.h>
#include <strings.h>
#include <getopt.h>

#include <cuComplex.h>
#include <npp.h>

void postLaunchCheck();
void preLaunchCheck();

#define UNPACK
#define POLSWAP
#define PACK

#ifdef UNPACK
__global__ void unpack8bit_kernel(float *dest, const int8_t *src) {

  const size_t i = blockDim.x * blockIdx.x + threadIdx.x;

  dest[i] = src[i];
}

__global__ void unpack8bit_2chan_kernel(float *dest, const int8_t *src, int N) {

  const size_t i = blockDim.x * blockIdx.x + threadIdx.x;
  const size_t j = i*2;

  dest[i] = static_cast<float>(src[j]);
  dest[i+N] = static_cast<float>(src[j+1]);
}

__global__ void unpack8bit_4chan_kernel(float *dest, const int8_t *src, int N) {

  const size_t i = blockDim.x * blockIdx.x + threadIdx.x;
  const size_t j = i*4;

  dest[i] = static_cast<float>(src[j]);
  dest[i+N] = static_cast<float>(src[j+1]);
  dest[i+N*2] = static_cast<float>(src[j+2]);
  dest[i+N*3] = static_cast<float>(src[j+3]);
}

__global__ void unpack8bit_8chan_kernel(float *dest, const int8_t *src, int N) {

  const size_t i = blockDim.x * blockIdx.x + threadIdx.x;
  const size_t j = i*8;

  dest[i] = static_cast<float>(src[j]);
  dest[i+N] = static_cast<float>(src[j+1]);
  dest[i+N*2] = static_cast<float>(src[j+2]);
  dest[i+N*3] = static_cast<float>(src[j+3]);
  dest[i+N*4] = static_cast<float>(src[j+4]);
  dest[i+N*5] = static_cast<float>(src[j+5]);
  dest[i+N*6] = static_cast<float>(src[j+6]);
  dest[i+N*7] = static_cast<float>(src[j+7]);
}

#endif


#ifdef PACK
#define VERYHIGH 0x0
#define HIGH     0x1
#define LOW      0x2
#define VERYLOW  0x3

__global__ void pack2bit_kernel(int8_t *dest, const float *src, const float thresh[3]) {

  const size_t i = (blockDim.x * blockIdx.x + threadIdx.x);
  int j;

  dest[i] = 0;
  for (j=0; j<4; j++) {  // 4 samples/byte
    if (src[i*4+j] > thresh[1]) {
      if (src[i*4+j] > thresh[0]) 
	dest[i] += VERYHIGH<<(j*2);
      else 
	dest[i] += HIGH<<(j*2);
    } else {
      if (src[i*4+j] < thresh[2]) 
	dest[i] += VERYLOW<<(j*2);
      else 
	dest[i] += LOW<<(j*2);
    }
  }
}

#define PACKIT(X,Y) {\
  if (src[X] > thresh[1]) { \
    if (src[X] > thresh[2]) \
      dest[i] += VERYHIGH<<(Y); \
    else \
      dest[i] += HIGH<<(Y); \
  } else { \
    if (src[X] < thresh[0]) \
      dest[i] += VERYLOW<<(Y); \
    else \
      dest[i] += LOW<<(Y); \
    } \
    X++; \
  }

__global__ void pack2bit_dualpol_kernel(int8_t *dest, const float *src, const float thresh[3]) {
  const size_t i = (blockDim.x * blockIdx.x + threadIdx.x);
  int j, k;

  j = i*2;
  k = j+blockDim.x*gridDim.x*2;

  dest[i] = 0;

  PACKIT(j,0);
  PACKIT(k,2);
  PACKIT(j,4);
  PACKIT(k,6);
}

__global__ void pack2bit_dualpol_32chan_kernel(int8_t *dest, const float *src, const float thresh[3]) {
  const size_t l = (blockDim.x * blockIdx.x + threadIdx.x)*16;
  int i, j, k;

  j = l*2;
  k = j+blockDim.x*gridDim.x*16;
  for (i=l; i<l+16; i++) {
    dest[i] = 0;

    PACKIT(j,0);
    PACKIT(k,2);
    PACKIT(j,4);
    PACKIT(k,6);
  }
}

#endif

#ifdef POLSWAP

#define NSPEC 32

__constant__ cuFloatComplex cgain[NSPEC*2];

// Assume data has been run through filterbank already so is complex and layed out as 
// A1,B1,C1,....,A2,B2,C3,... for pol 1 (nchan*N floats) and the same for pol 2
__global__ void linear2circular_kernel(cuFloatComplex *data, int nchan, int N, cuFloatComplex *gain) {

  int i = blockDim.x * blockIdx.x + threadIdx.x;
  int c = i % nchan;
  cuFloatComplex temp;

  data[i] = cuCmulf(data[i], gain[c]);
  data[i+N] = cuCmulf(data[i+N], gain[c+nchan]);

  temp = cuCsubf(data[i], data[i+N]);
  data[i+N] = cuCaddf(data[i], data[i+N]);
  data[i] = temp;
}

__global__ void linear2circular_kernel2(cuFloatComplex *data, int nchan, int N, cuFloatComplex *gain) {
  extern __shared__ cuFloatComplex sgain[];

  int tid = threadIdx.x;
  int i = blockDim.x * blockIdx.x + tid;
  int c = i % nchan;
  cuFloatComplex temp;

  if (tid<nchan) {
    sgain[tid] = gain[tid];
    sgain[tid+nchan] = gain[tid+nchan];
  }
  __syncthreads();

  data[i] = cuCmulf(data[i], sgain[c]);
  data[i+N] = cuCmulf(data[i+N], sgain[c+nchan]);

  temp = cuCsubf(data[i], data[i+N]);
  data[i+N] = cuCaddf(data[i], data[i+N]);
  data[i] = temp;
}

__global__ void linear2circular_kernel3(cuFloatComplex *data, int nchan, int N) {

  int tid = threadIdx.x;
  int i = blockDim.x * blockIdx.x + tid;
  int c = i % nchan;
  cuFloatComplex temp;

  data[i] = cuCmulf(data[i], cgain[c]);
  data[i+N] = cuCmulf(data[i+N], cgain[c+nchan]);

  temp = cuCsubf(data[i], data[i+N]);
  data[i+N] = cuCaddf(data[i], data[i+N]);
  data[i] = temp;
}

__global__ void linear2circular_kernel4(cuFloatComplex *data, int nchan, int N, cuFloatComplex *gain) {

  int k;
  int j = (blockDim.x * blockIdx.x + threadIdx.x)*32;
  cuFloatComplex temp;

  for (k=0; k<32; k++) {
    data[j] = cuCmulf(data[j], gain[k]);
    data[j+N] = cuCmulf(data[j+N], gain[k+nchan]);

    temp = cuCsubf(data[j], data[j+N]);
    data[j+N] = cuCaddf(data[j], data[j+N]);
    data[j] = temp;
    
    j++;
  }
}

#endif

int main(int argc, char *argv[]) {
  int8_t *idata;
  int opt, tmp, ss, repeat, nsample, nthread, nblock;
  float dtime, *fdata;
  cuFloatComplex *cdata;
  cudaError_t status;
  cudaEvent_t start_exec, stop_exec, start_test, stop_test;
#ifdef PACK
  float hostthresh[3] = {-10,0,10}, *thresh;
#endif
#ifdef POLSWAP
  int i;
  cuFloatComplex *gain, *hostgain;
#endif


  nsample = 1024*1024*10/4; // Aprox 10 MB floating point data
  nthread = 512;
  repeat = 1;

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


  cudaEventCreate(&start_exec);
  cudaEventCreate(&stop_exec);
  cudaEventCreate(&start_test);
  cudaEventCreate(&stop_test);

  // Start total time event
  cudaEventRecord(start_exec, 0);

  nblock = nsample/nthread;
  nsample = nblock*nthread;   // Round

  printf("Number of samples = %d\n", nsample);
  printf("Number of repeats = %d\n", repeat);
  printf("Number of threads = %d\n", nthread);
  printf("Number of blocks = %d\n", nblock);
  printf("Processing approx %lu MB per iteration\n", nsample*sizeof(float)/1024/1024);

  // Allocate memory on the host
  status = cudaMalloc(&fdata, nsample*sizeof(float));
  if (status != cudaSuccess) {
    fprintf(stderr, "Error: cudaMalloc failed\n");
    return EXIT_FAILURE;
  }

  status = cudaMalloc(&idata, nsample*sizeof(int8_t));
  if (status != cudaSuccess) {
    fprintf(stderr, "Error: cudaMalloc failed\n");
    return EXIT_FAILURE;
  }

  status = cudaMalloc(&cdata, nsample*sizeof(cuFloatComplex));
  if (status != cudaSuccess) {
    fprintf(stderr, "Error: cudaMalloc failed\n");
    return EXIT_FAILURE;
  }

#ifdef PACK
  status = cudaMalloc(&thresh, 3*sizeof(float));
  if (status != cudaSuccess) {
    fprintf(stderr, "Error: cudaMalloc failed\n");
    return EXIT_FAILURE;
  }
  status = cudaMemcpy(thresh, hostthresh, 3*sizeof(float), cudaMemcpyHostToDevice);
  if (status != cudaSuccess) {
    fprintf(stderr, "Error: cudaMemcpy failed\n");
    return EXIT_FAILURE;
  }
#endif

#ifdef POLSWAP

  status = cudaMalloc(&gain, NSPEC*2*sizeof(cuFloatComplex));
  if (status != cudaSuccess) {
    fprintf(stderr, "Error: cudaMalloc failed\n");
    return EXIT_FAILURE;
  }
  status = cudaMallocHost(&hostgain, NSPEC*2*sizeof(cuFloatComplex));
  if (status != cudaSuccess) {
    fprintf(stderr, "Error: cudaMallocHost failed\n");
    return EXIT_FAILURE;
  }

  for (i=0; i<NSPEC*2; i++) {
    hostgain[i].x = sin(float(NSPEC)*i/M_PI);
    hostgain[i].y = cos(float(NSPEC)*i/M_PI);
  }

  status = cudaMemcpy(gain, hostgain, NSPEC*2*sizeof(cuFloatComplex), 
		      cudaMemcpyHostToDevice);
  if (status != cudaSuccess) {
    fprintf(stderr, "Error: cudaMemcpy failed\n");
    return EXIT_FAILURE;
  }

#endif

#ifdef UNPACK
  printf("\n======= 8 bit to float ========\n");
  printf("              |    Time     |    1 GHz     |   Realtime   |\n");

  // 1 channel
  preLaunchCheck();
  cudaEventRecord(start_test, 0);
  for (int i=0; i<repeat; i++) {
    unpack8bit_kernel<<<nblock,nthread>>>(fdata, idata);
  }
  cudaEventRecord(stop_test, 0);
  cudaEventSynchronize(stop_test);
  cudaEventElapsedTime(&dtime, start_test, stop_test);
  postLaunchCheck();
  printf("Simple 1 chan | %8.3f ms |  %8.3f ms | %8.1f MHz | \n", dtime, dtime*4e9/((float)nsample*repeat), 
	 (float)nsample*repeat/4/1e6/dtime*1000);

  // 1 channel - NPP
  nppsConvert_8s32f(idata, fdata, nsample); // Dummy call
  preLaunchCheck();
  cudaEventRecord(start_test, 0);
  for (int i=0; i<repeat; i++) {
    nppsConvert_8s32f(idata, fdata, nsample);
  }
  cudaEventRecord(stop_test, 0);
  cudaEventSynchronize(stop_test);
  cudaEventElapsedTime(&dtime, start_test, stop_test);
  postLaunchCheck();
  printf("NPP 1 chan    | %8.3f ms |  %8.3f ms | %8.1f MHz | \n", dtime, dtime*4e9/((float)nsample*repeat), 
	 (float)nsample*repeat/4/1e6/dtime*1000);

  // 2 channel
  preLaunchCheck();
  cudaEventRecord(start_test, 0);
  for (int i=0; i<repeat; i++) {
    unpack8bit_2chan_kernel<<<nblock/2,nthread>>>(fdata, idata, nsample/2);
  }
  cudaEventRecord(stop_test, 0);
  cudaEventSynchronize(stop_test);
  cudaEventElapsedTime(&dtime, start_test, stop_test);
  postLaunchCheck();

  printf("       2 chan | %8.3f ms |  %8.3f ms | %8.1f MHz | \n", dtime, dtime*4e9/((float)nsample*repeat), 
	 (float)nsample*repeat/4/1e6/dtime*1000);

  // 4 channel
  preLaunchCheck();
  cudaEventRecord(start_test, 0);
  for (int i=0; i<repeat; i++) {
    unpack8bit_4chan_kernel<<<nblock/4,nthread>>>(fdata, idata, nsample/4);
  }
  cudaEventRecord(stop_test, 0);
  cudaEventSynchronize(stop_test);
  cudaEventElapsedTime(&dtime, start_test, stop_test);
  postLaunchCheck();
  printf("       4 chan | %8.3f ms |  %8.3f ms | %8.1f MHz | \n", dtime, dtime*4e9/((float)nsample*repeat), 
	 (float)nsample*repeat/4/1e6/dtime*1000);

  // 8 channel
  preLaunchCheck();
  cudaEventRecord(start_test, 0);
  for (int i=0; i<repeat; i++) {
    unpack8bit_4chan_kernel<<<nblock/8,nthread>>>(fdata, idata, nsample/8);
  }
  cudaEventRecord(stop_test, 0);
  cudaEventSynchronize(stop_test);
  cudaEventElapsedTime(&dtime, start_test, stop_test);
  postLaunchCheck();
  printf("       8 chan | %8.3f ms |  %8.3f ms | %8.1f MHz | \n", dtime, dtime*4e9/((float)nsample*repeat), 
	 (float)nsample*repeat/4/1e6/dtime*1000);
#endif

#ifdef PACK
  printf("\n======= float to 2bit ========\n");
  printf("              |    Time     |    1 GHz     |   Realtime   |\n");


  // 1 channel
  preLaunchCheck();
  cudaEventRecord(start_test, 0);
  for (int i=0; i<repeat; i++) {
    pack2bit_kernel<<<nblock/4,nthread>>>(idata, fdata, thresh);
  }
  cudaEventRecord(stop_test, 0);
  cudaEventSynchronize(stop_test);
  cudaEventElapsedTime(&dtime, start_test, stop_test);
  postLaunchCheck();
  printf("Simple 1 chan | %8.3f ms |  %8.3f ms | %8.3f GHz | \n", dtime, dtime*2e9/((float)nsample*repeat), 
	 (float)nsample*repeat/2/1e9/dtime*1000);

  // 1 channel dual pol
  preLaunchCheck();
  cudaEventRecord(start_test, 0);
  for (int i=0; i<repeat; i++) {
    pack2bit_dualpol_kernel<<<nblock/4,nthread>>>(idata, fdata, thresh);
  }
  cudaEventRecord(stop_test, 0);
  cudaEventSynchronize(stop_test);
  cudaEventElapsedTime(&dtime, start_test, stop_test);
  postLaunchCheck();
  printf("Simple 2 pol  | %8.3f ms |  %8.3f ms | %8.3f GHz | \n", dtime, dtime*4e9/((float)nsample*repeat), 
	 (float)nsample*repeat/4/1e9/dtime*1000);

  // 32 channel dual pol
  preLaunchCheck();
  cudaEventRecord(start_test, 0);
  for (int i=0; i<repeat; i++) {
    pack2bit_dualpol_32chan_kernel<<<nblock/64,nthread>>>(idata, fdata, thresh);
  }
  cudaEventRecord(stop_test, 0);
  cudaEventSynchronize(stop_test);
  cudaEventElapsedTime(&dtime, start_test, stop_test);
  postLaunchCheck();
  printf("32chan dualpol| %8.3f ms |  %8.3f ms | %8.3f GHz | \n", dtime, dtime*4e9/((float)nsample*repeat), 
	 (float)nsample*repeat/4/1e9/dtime*1000);

#endif

#ifdef POLSWAP
  printf("\n======= Linear to Circular ========\n");
  printf("              |    Time     |    1 GHz     |   Realtime   |\n");

  preLaunchCheck();
  cudaEventRecord(start_test, 0);
  for (int i=0; i<repeat; i++) {
    linear2circular_kernel<<<nblock/2,nthread>>>(cdata, NSPEC, nsample/2, gain);
  }
  cudaEventRecord(stop_test, 0);
  cudaEventSynchronize(stop_test);
  cudaEventElapsedTime(&dtime, start_test, stop_test);
  postLaunchCheck();
  // TODO Check factors of 2 here - mostly likely wrong
  printf(" Simple       | %8.3f ms |  %8.3f ms | %8.1f MHz | \n", dtime, dtime*2e9/((float)nsample*repeat), 
	 (float)nsample*repeat/2/1e6/dtime*1000);

  preLaunchCheck();
  cudaEventRecord(start_test, 0);
  for (int i=0; i<repeat; i++) {
    linear2circular_kernel2<<<nblock/2, nthread, NSPEC*2*sizeof(cuFloatComplex)>>>(cdata, NSPEC, nsample/2, gain);
  }
  cudaEventRecord(stop_test, 0);
  cudaEventSynchronize(stop_test);
  cudaEventElapsedTime(&dtime, start_test, stop_test);
  postLaunchCheck();
  // TODO Check factors of 2 here - mostly likely wrong
  printf(" Shared Mem   | %8.3f ms |  %8.3f ms | %8.1f MHz | \n", dtime, dtime*2e9/((float)nsample*repeat), 
	 (float)nsample*repeat/2/1e6/dtime*1000);

  preLaunchCheck();
  cudaMemcpyToSymbol(cgain, hostgain, sizeof(cuFloatComplex)*NSPEC*2);

  cudaEventRecord(start_test, 0);
  for (int i=0; i<repeat; i++) {
    linear2circular_kernel3<<<nblock/2, nthread>>>(cdata, NSPEC, nsample/2);
  }
  cudaEventRecord(stop_test, 0);
  cudaEventSynchronize(stop_test);
  cudaEventElapsedTime(&dtime, start_test, stop_test);
  postLaunchCheck();
  // TODO Check factors of 2 here - mostly likely wrong
  printf(" Const Mem    | %8.3f ms |  %8.3f ms | %8.1f MHz | \n", dtime, dtime*2e9/((float)nsample*repeat), 
	 (float)nsample*repeat/2/1e6/dtime*1000);


  preLaunchCheck();
  cudaEventRecord(start_test, 0);
  for (int i=0; i<repeat; i++) {
    linear2circular_kernel4<<<nblock/64,nthread>>>(cdata, NSPEC, nsample/2, gain);
  }
  cudaEventRecord(stop_test, 0);
  cudaEventSynchronize(stop_test);
  cudaEventElapsedTime(&dtime, start_test, stop_test);
  postLaunchCheck();
  // TODO Check factors of 2 here - mostly likely wrong
  printf(" All value    | %8.3f ms |  %8.3f ms | %8.1f MHz | \n", dtime, dtime*2e9/((float)nsample*repeat), 
	 (float)nsample*repeat/2/1e6/dtime*1000);


#endif

  cudaEventRecord(stop_exec, 0);
  cudaEventSynchronize(stop_exec);
  cudaEventElapsedTime(&dtime, start_exec, stop_exec);
  printf("\nTotal executution time =  %.3f ms\n", dtime);


  // Free allocated memory
  cudaFree(idata);
  cudaFree(fdata);

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
