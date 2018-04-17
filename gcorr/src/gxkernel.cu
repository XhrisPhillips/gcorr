#include <cuComplex.h>
#include "gxkernel.h"


// Add complex number to another number 
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

// Divide a complex number by a constant
__host__ __device__ static __inline__ void cuCdivCf(cuFloatComplex *a, float b)
{
  (*a).x /= b;
  (*a).y /= b;
}

// Rotate inplace a complex number by theta (radians)
__host__ __device__ static __inline__ void cuRotatePhase (cuComplex *x, float theta)
{
  float cs, sn;
  sincosf(theta, &sn, &cs);
    
  float px = x->x * cs - x->y * sn; 
  float py = x->x * sn + x->y * cs;

  x->x = px;
  x->y = py;
  return;
}

// Rotate inplace a complex number by theta (radians)
__host__ __device__ static __inline__ void cuRotatePhase2 (cuComplex &x, float &sinA, float &cosA)
{
  float px = x.x * cosA - x.y * sinA; 
  float py = x.x * sinA + x.y * cosA;
  x.x = px;
  x.y = py;
  return;
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


/* Set fringe rotation vectors - dummy routine for now */
__global__ void setFringeRotation(float **rotVec) {
  size_t ifft = threadIdx.x + blockIdx.x * blockDim.x;
  size_t iant = blockIdx.y;

  rotVec[iant][ifft*2] = 1e-6;
  rotVec[iant][ifft*2+1] = 1e-12;
}


/* Fringe rotate a single antenna inplace, assuming dual pol data */

__global__ void FringeRotate(cuComplex **ant, float **rotVec) {
  // ant[0] pointer to pol A
  // ant[1] pointer to pol B
  // rotVec is an array of 2 values - initial phase and phase step per sample 

  int fftsize = blockDim.x * gridDim.x;
  size_t ichan = threadIdx.x + blockIdx.x * blockDim.x;
  size_t ifft = blockIdx.y;
  size_t iant = blockIdx.z;

  // phase and slope for this FFT
  float p0 = rotVec[iant][ifft*2];
  float p1 = rotVec[iant][ifft*2+1];
  float theta = p0 + ichan*p1;

  // Should precompute sin/cos
  cuRotatePhase(&ant[iant*2][ichan+ifft*fftsize], theta);
  cuRotatePhase(&ant[iant*2+1][ichan+ifft*fftsize], theta);
}

__global__ void FringeRotate2(cuComplex **ant, float **rotVec) {
  // ant[0] pointer to pol A
  // ant[1] pointer to pol B
  // rotVec is an array of 2 values - initial phase and phase step per sample 

  int fftsize = blockDim.x * gridDim.x;
  size_t ichan = threadIdx.x + blockIdx.x * blockDim.x;
  size_t ifft = blockIdx.y;
  size_t iant = blockIdx.z;

  // phase and slope for this FFT
  float p0 = rotVec[iant][ifft*2];
  float p1 = rotVec[iant][ifft*2+1];
  float theta = p0 + ichan*p1;

  // Should precompute sin/cos
  float sinT, cosT;
  __sincosf(theta, &sinT, &cosT);
  
  cuRotatePhase2(ant[iant*2][ichan+ifft*fftsize], sinT, cosT);
  cuRotatePhase2(ant[iant*2+1][ichan+ifft*fftsize], sinT, cosT);
}

//__constant__ float levels_2bit[4];

void init_2bitLevels() {
  //static const float HiMag = 3.3359;  // Optimal value
  //const float lut4level[4] = {-HiMag, -1.0, 1.0, HiMag};

  //gpuErrchk(cudaMemcpyToSymbol(levels_2bit, lut4level, sizeof(levels_2bit)));
}


/* Unpack 2bit real data in complex float, assuming 2 interleave channels 
   This is probably NOT suitable for the final system, just an initial place holder
   Specifically I think delay compersation needs to be done here
   Each thread unpacks 4 samples, 2 per channel (pol). Total number of threads should be
  a factor of 2 smaller than numbwe of time samples (4 than total # samples).
*/

__global__ void unpack2bit_2chan(cuComplex **dest, const int8_t *src, const int iant) {
  static const float HiMag = 3.3359;  // Optimal value
  const float levels_2bit[4] = {-HiMag, -1.0, 1.0, HiMag};
  const int a = iant*2;
  const size_t i = (blockDim.x * blockIdx.x + threadIdx.x);
  int j = i*2;

  dest[a][j] = make_cuFloatComplex(levels_2bit[src[i]&0x3], 0);
  dest[a+1][j] = make_cuFloatComplex(levels_2bit[(src[i]>>2)&0x3], 0);
  j++;
  dest[a][j] = make_cuFloatComplex(levels_2bit[(src[i]>>4)&0x3], 0);
  dest[a+1][j] = make_cuFloatComplex(levels_2bit[(src[i]>>6)&0x3], 0);
}




/* Cross correlate and accumulate nant antenna data

   ants is an array of array pointers for each telescope. There are nant*2 arrays (dual pol)
   Each antenna array has nchan frequency points, repeated XX times.
   accum contains the cross correlation values - there is nant*(nant-1)*2*4 values repeated XX times

*/


__global__ void CrossCorr(cuComplex **ants, cuComplex **accum, int nant, int nchunk) { 
  int nchan = blockDim.x * gridDim.x;
  size_t ichan = threadIdx.x + blockIdx.x * blockDim.x + blockIdx.y * nchan * nchunk * 2;
  int ochan = threadIdx.x + blockIdx.x * blockDim.x + blockIdx.y * nchan;

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
  size_t ichan = threadIdx.x + blockIdx.x * blockDim.x + blockIdx.y * nchan * nchunk * 2;
  const size_t ochan = threadIdx.x + blockIdx.x * blockDim.x + blockIdx.y * nchan;

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

__global__ void finaliseAccum(cuComplex **accum, int nant, int nchunk) { 

  int nchan = blockDim.x * gridDim.x;

  int ichan = (blockDim.x * blockIdx.x + threadIdx.x);
  int b = blockIdx.y+blockIdx.z*4;
  
  for (int i=1; i<nchunk; i++) {
    cuCaddIf(&accum[b][ichan], accum[b][ichan + i*nchan]);
  }
  cuCdivCf(&accum[b][ichan], nchunk);
}

__global__ void printArray(cuComplex *a) {
  int i = threadIdx.x;
  printf("%f%+fi\n", a[i].x, a[i].y);
}
__global__ void printArrayInt(int8_t *a) {
  int i = threadIdx.x;
  printf("%d\n", a[i]);
}
