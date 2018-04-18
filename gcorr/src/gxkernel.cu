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
__global__ void setFringeRotation(float *rotVec) {
  size_t ifft = threadIdx.x + blockIdx.x * blockDim.x;
  size_t iant = blockIdx.y;
  int numffts = blockDim.x * gridDim.x;

  rotVec[iant*numffts*2 + ifft*2] = 1e-6;
  rotVec[iant*numffts*2 + ifft*2+1] = 1e-12;
}


/* Fringe rotate the data, using a linear phase slopre per input FFT. Assume 2 polarisations

   threads * gridDim.x is size of FFT
   grid.y   is number of FFTs
   grid.z is number of Antennas
 */

__global__ void FringeRotate(cuComplex *ant, float *rotVec) {
  int fftsize = blockDim.x * gridDim.x;
  size_t ichan = threadIdx.x + blockIdx.x * blockDim.x;
  size_t ifft = blockIdx.y;
  size_t iant = blockIdx.z;
  int numffts = gridDim.y;
  int subintsamples = numffts * fftsize;

  // phase and slope for this FFT
  float p0 = rotVec[iant*numffts*2 + ifft*2];
  float p1 = rotVec[iant*numffts*2 + ifft*2+1];
  float theta = p0 + ichan*p1;

  // Should precompute sin/cos
  cuRotatePhase(&ant[sampIdx(iant, 0, ichan+ifft*fftsize, subintsamples)], theta);
  cuRotatePhase(&ant[sampIdx(iant, 1, ichan+ifft*fftsize, subintsamples)], theta);
}

__global__ void FringeRotate2(cuComplex *ant, float *rotVec) {
  int fftsize = blockDim.x * gridDim.x;
  size_t ichan = threadIdx.x + blockIdx.x * blockDim.x;
  size_t ifft = blockIdx.y;
  size_t iant = blockIdx.z;
  int numffts = blockDim.y * gridDim.y;
  int subintsamples = numffts * fftsize * 2;

  // phase and slope for this FFT
  float p0 = rotVec[iant*numffts + ifft*2];
  float p1 = rotVec[iant*numffts + ifft*2+1];
  float theta = p0 + ichan*p1;

  float sinT, cosT;
  __sincosf(theta, &sinT, &cosT);
  cuRotatePhase2(ant[sampIdx(iant, 0, ichan+ifft*fftsize, subintsamples)], sinT, cosT);
  cuRotatePhase2(ant[sampIdx(iant, 1, ichan+ifft*fftsize, subintsamples)], sinT, cosT);
}

//__constant__ float levels_2bit[4];
__constant__ float kLevels_2bit[4];

void init_2bitLevels() {
  static const float HiMag = 3.3359;  // Optimal value
  const float lut4level[4] = {-HiMag, -1.0, 1.0, HiMag};

  gpuErrchk(cudaMemcpyToSymbol(kLevels_2bit, lut4level, sizeof(kLevels_2bit)));
}


/* Unpack 2bit real data in complex float, assuming 2 interleave channels 
   This is probably NOT suitable for the final system, just an initial place holder
   Specifically I think delay compersation needs to be done here
   Each thread unpacks 4 samples, 2 per channel (pol). Total number of threads should be
  a factor of 2 smaller than number of time samples (4x smakker than total # samples assuming 2 pols).
*/


__global__ void unpack2bit_2chan(cuComplex *dest, const int8_t *src) {
  static const float HiMag = 3.3359;  // Optimal value
  const float levels_2bit[4] = {-HiMag, -1.0, 1.0, HiMag};
  const size_t i = (blockDim.x * blockIdx.x + threadIdx.x);
  int subintsamples = 2 * blockDim.x * gridDim.x;
  int j = i*2;

  dest[j] = make_cuFloatComplex(levels_2bit[src[i]&0x3], 0);
  dest[subintsamples + j] = make_cuFloatComplex(levels_2bit[(src[i]>>2)&0x3], 0);
  j++;
  dest[j] = make_cuFloatComplex(levels_2bit[(src[i]>>4)&0x3], 0);
  dest[subintsamples + j] = make_cuFloatComplex(levels_2bit[(src[i]>>6)&0x3], 0);
}

/* Unpack 16bit complex data (assumed to be 2's complement) to complex float
   src data is assumed to have 2 channels (polarisations) interleaved.
   Number of threads equal to twice the number subints (separate thread per pol)
*/

__global__ void unpack8bitcomplex_2chan(cuComplex *dest, const int8_t *src) {
  const size_t i = (blockDim.x * blockIdx.x + threadIdx.x);
  int subintsamples = blockDim.x * gridDim.x/2;

  int ichan = i*2; // 2 bytes per complex sample
  int pol = i % 2;
  int ochan = i/2 + pol*subintsamples;

  dest[ochan] = make_cuFloatComplex(src[ichan], src[ichan+1]);
}

__global__ void unpack2bit_2chan_fast(cuComplex *dest, const int8_t *src) {
  // static const float HiMag = 3.3359;  // Optimal value
  // const float levels_2bit[4] = {-HiMag, -1.0, 1.0, HiMag};
  const size_t i = (blockDim.x * blockIdx.x + threadIdx.x);
  int subintsamples = 2 * blockDim.x * gridDim.x;
  int j = i*2;
  int8_t src_i = src[i]; // Here I am just loading src into local memory to 
                         // reduce the number of reads from global memory

  // I have just changed the order of the writes made to dest
  // In theory this should reduce the number of write operations made
  // I have also implemented the use of constant memory for the levels_2bit
  // array
  dest[j] = make_cuFloatComplex(kLevels_2bit[src_i&0x3], 0);
  dest[j+1] = make_cuFloatComplex(kLevels_2bit[(src_i>>4)&0x3], 0);

  dest[subintsamples + j] = make_cuFloatComplex(kLevels_2bit[(src_i>>2)&0x3], 0);
  dest[subintsamples + j + 1] = make_cuFloatComplex(kLevels_2bit[(src_i>>6)&0x3], 0);
}

/* Unpack 2bit real data in complex float, assuming 2 interleave channels 
   This is probably NOT suitable for the final system, just an initial place holder
   Specifically I think delay compersation needs to be done here
   Each thread unpacks 4 samples, 2 per channel (pol). Total number of threads should be
  a factor of 2 smaller than numbwe of time samples (4 than total # samples).
*/

__global__ void old_unpack2bit_2chan(cuComplex **dest, const int8_t *src, const int iant) {
  // DO NOT USE THIS FUNCTION
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


__global__ void CrossCorr(cuComplex *ants, cuComplex *accum, int nant, int nchunk) { 
  int nchan = blockDim.x * gridDim.x;
  size_t ichan = threadIdx.x + blockIdx.x * blockDim.x + blockIdx.y * nchan * nchunk * 2;
  int ochan = threadIdx.x + blockIdx.x * blockDim.x + blockIdx.y * nchan;
  int parallelAccum = blockDim.y * gridDim.y;
  int subintsamples = parallelAccum * nchan * nchunk * 2;

  int i, j, l, b;
  for (l=0; l<nchunk; l++) {
    b=0;
    for (i=0; i<nant-1; i++) {
      for (j=i+1; j<nant; j++) {
	cuCaddIf(&accum[accumIdx(b, 0, ochan, nchan*parallelAccum)],
	  cuCmulConjf(ants[antIdx(i, 0, ichan, subintsamples)], ants[antIdx(j, 0, ichan, subintsamples)]));
	cuCaddIf(&accum[accumIdx(b, 1, ochan, nchan*parallelAccum)],
	  cuCmulConjf(ants[antIdx(i, 0, ichan, subintsamples)], ants[antIdx(j, 1, ichan, subintsamples)]));
	cuCaddIf(&accum[accumIdx(b, 2, ochan, nchan*parallelAccum)],
	  cuCmulConjf(ants[antIdx(i, 1, ichan, subintsamples)], ants[antIdx(j, 0, ichan, subintsamples)]));
	cuCaddIf(&accum[accumIdx(b, 3, ochan, nchan*parallelAccum)],
	  cuCmulConjf(ants[antIdx(i, 1, ichan, subintsamples)], ants[antIdx(j, 1, ichan, subintsamples)]));
	b++;
      }
    }
    ichan += nchan*2;
  }
}

__global__ void CrossCorrShared(cuComplex *ants, cuComplex *accum, int nant, int nchunk) { 

  extern __shared__ cuComplex antShar[];
  
  int nchan = blockDim.x * gridDim.x;
  size_t ichan = threadIdx.x + blockIdx.x * blockDim.x + blockIdx.y * nchan * nchunk * 2;
  const size_t ochan = threadIdx.x + blockIdx.x * blockDim.x + blockIdx.y * nchan;
  int parallelAccum = blockDim.y * gridDim.y;
  int subintsamples = parallelAccum * nchan * nchunk * 2;

  int i, j, l, b;
  for (l=0; l<nchunk; l++) {
    if (threadIdx.x<nant*2) antShar[threadIdx.x] = ants[antIdx(threadIdx.x / 2, threadIdx.x % 2, ichan, subintsamples)];
    __syncthreads();

    b=0;
    for (i=0; i<nant-1; i++) {
      for (j=i+1; j<nant; j++) {
	cuCaddIf(&accum[accumIdx(b, 0, ochan, nchan*parallelAccum)], cuCmulConjf(antShar[i*2], antShar[j*2]));
	cuCaddIf(&accum[accumIdx(b, 1, ochan, nchan*parallelAccum)], cuCmulConjf(antShar[i*2], antShar[j*2+1]));
	cuCaddIf(&accum[accumIdx(b, 2, ochan, nchan*parallelAccum)], cuCmulConjf(antShar[i*2+1], antShar[j*2]));
	cuCaddIf(&accum[accumIdx(b, 3, ochan, nchan*parallelAccum)], cuCmulConjf(antShar[i*2+1], antShar[j*2+1]));
	b++;
      }
    }
    ichan += nchan*2;
  }
}

__global__ void finaliseAccum(cuComplex *accum, int parallelAccum) { 

  int nchan = blockDim.x * gridDim.x;
  int ichan = (blockDim.x * blockIdx.x + threadIdx.x);
  int prod = blockIdx.y;
  int b = blockIdx.z;

  for (int i=1; i<parallelAccum; i++) {
    cuCaddIf(&accum[accumIdx(b, prod, ichan, nchan*parallelAccum)],
      accum[accumIdx(b, prod, ichan + i*nchan, nchan*parallelAccum)]);
  }
  cuCdivCf(&accum[accumIdx(b, prod, ichan, nchan*parallelAccum)], parallelAccum);
}

// Launched with antenna indices in block .y and .z.
// Accumulates over first nchan entries in each fftwidth-wide block.
template <int npol>
__global__ void CrossCorrAccumHoriz(cuComplex *accum, const cuComplex *ants, int nant, int nfft, int nchan, int fftwidth) {
    int t = threadIdx.x+blockIdx.x*blockDim.x;
    if (t>=nchan) return;

    // input vector indices in block .y and .z
    int i = blockIdx.y;
    int j = blockIdx.z;
    j += i+1;

    if (i>=nant || j>=nant) return;

    // index into output vectors: = (j-i-1) + n-1 + ... + n-i
    int b = i*nant-i*(i+1)/2 + j-i-1;
    b *= npol*npol;

    int r = nfft*nchan;

    for (int pi = 0; pi<npol; ++pi) {
	for (int pj = 0; pj<npol; ++pj) {
	    const float2* iv = ants+(i*npol+pi)*r+t;
	    const float2* jv = ants+(j*npol+pj)*r+t;

	    float2 u = iv[0];
	    float2 v = jv[0];
	    float2 a;
	    a.x = u.x*v.x + u.y*v.y;
	    a.y = u.y*v.x - u.x*v.y;

	    for (int k = fftwidth; k<r; k += fftwidth) {
		u = iv[k];
		v = jv[k];

		a.x += u.x*v.x + u.y*v.y;
		a.y += u.y*v.x - u.x*v.y;
	    }

	    a.x /= nfft;
	    a.y /= nfft;
	    accum[b*nchan+t] = a;
	    ++b;
	}
    }
}

template __global__ void CrossCorrAccumHoriz<1>(cuComplex*, const cuComplex*, int, int, int, int);
template __global__ void CrossCorrAccumHoriz<2>(cuComplex*, const cuComplex*, int, int, int, int);

__global__ void printArray(cuComplex *a) {
  int i = threadIdx.x;
  printf("%f%+fi\n", a[i].x, a[i].y);
}
__global__ void printArrayInt(int8_t *a) {
  int i = threadIdx.x;
  printf("%d\n", a[i]);
}
