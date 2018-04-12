#include <cuComplex.h>


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

// Rotate inplace a complex number by theta (radians)
__host__ __device__ static __inline__ void cuRotatePhase (cuComplex x, float theta)
{
  float cs, sn;
  sincosf(theta, &sn, &cs);
    
  float px = x->x * cs - x->y * sn; 
  float py = x->x * sn + x->y * cs;

  x->x = px;
  x->y = py;
  return;
}

/* Fringe rotate a single antenna inplace, assuming dual pol data */

__global__ void FringeRotate(cuComplex **ant, float **rotVec) {
  // ant[0] pointer to pol A
  // ant[a] pointer to pol B
  // rotVec is an array of 2 values - initial phase and phase step per sample 

  size_t ichan = threadIdx.x;
  site_t ifft = 0;  // FFT block number  NEED TO CALCULATE
  int fftsize  = blockIdx.x;

  float theta = rotVec[ifft][0]*ichan*rotVec[ifft][1];
  cuRotatePhase(ant[ifft*fftsize+ichan][0], theta);
  cuRotatePhase(ant[ifft*fftsize+ichan][1], theta);
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

