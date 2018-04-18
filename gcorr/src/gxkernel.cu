#include <cuComplex.h>
#include "gxkernel.h"
#include <math.h>

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

/* Calculate the starting fringe rotation phase and phase increment for each FFT of each antenna, and the fractional sample error */
__global__ void calculateDelaysAndPhases(double * gpuDelays, double lo, double sampletime, int fftsamples, int fftchannels, float * rotationPhaseInfo, int *sampleShifts, float* fractionalSampleDelays)
{
  size_t ifft = threadIdx.x + blockIdx.x * blockDim.x;
  size_t iant = blockIdx.y;
  int numffts = blockDim.x * gridDim.x;
  double meandelay, deltadelay, netdelaysamples_f, startphase;
  double d0, d1, d2, a, b;
  double * interpolator = &(gpuDelays[iant*4]);
  double filestartoffset = gpuDelays[iant*4+3];
  float fractionaldelay;
  int netdelaysamples;

  // evaluate the delay for the given FFT of the given antenna

  // calculate values at the beginning, middle, and end of this FFT
  d0 = interpolator[0]*ifft*ifft + interpolator[1]*ifft + interpolator[2];
  d1 = interpolator[0]*(ifft+0.5)*(ifft+0.5) + interpolator[1]*(ifft+0.5) + interpolator[2];
  d2 = interpolator[0]*(ifft+1.0)*(ifft+1.0) + interpolator[1]*(ifft+1.0) + interpolator[2];

  // use these to calculate a linear interpolator across the FFT, as well as a mean value
  a = d2-d0; //this is the delay gradient across this FFT
  b = d0 + (d1 - (a*0.5 + d0))/3.0; //this is the delay at the start of the FFT
  meandelay = a*0.5 + b; //this is the delay in the middle of the FFT
  deltadelay = b / fftsamples; // this is the change in delay per sample across this FFT window

  netdelaysamples_f = (meandelay - filestartoffset) / sampletime;
  netdelaysamples   = int(netdelaysamples_f + 0.5); //FIXME: do we ever get negative values here?

  // Save the integer number of sample shifts
  sampleShifts[iant*numffts + ifft] = netdelaysamples;

  // Save the fractional delay
  fractionaldelay = (float)(-(netdelaysamples_f - netdelaysamples)*2*M_PI/fftchannels);  // radians per FFT channel
  fractionalSampleDelays[iant*numffts + ifft] = fractionaldelay;

  // set the fringe rotation phase for the first sample of a given FFT of a given antenna
  startphase = b*lo;
  rotationPhaseInfo[iant*numffts*2 + ifft*2] = (float)(startphase - int(startphase))*2*M_PI;
  rotationPhaseInfo[iant*numffts*2 + ifft*2 + 1] = (float)(deltadelay * lo)*2*M_PI;
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
  int subintsamples = numffts * fftsize;

  // phase and slope for this FFT
  float p0 = rotVec[iant*numffts*2 + ifft*2];
  float p1 = rotVec[iant*numffts*2 + ifft*2+1];
  float theta = p0 + ichan*p1;

  float sinT, cosT;
  __sincosf(theta, &sinT, &cosT);
  cuRotatePhase2(ant[sampIdx(iant, 0, ichan+ifft*fftsize, subintsamples)], sinT, cosT);
  cuRotatePhase2(ant[sampIdx(iant, 1, ichan+ifft*fftsize, subintsamples)], sinT, cosT);
}

/* Apply fractional delay correction, inplace correction
   ant is data after FFT
   fractionalDelayValues is array of phase corrections - assumed to be nornalised to 
   bandwidth and number of channels

   Kernel Dimensions:

   threadIdx.x/blockInd.x give FFT channel number
   blockIdx.y is FFT number 
   blockIdx.z is antenna number 
*/

__global__ void FracSampleCorrection(cuComplex *ant, float *fractionalDelayValues,
				     int numchannels, int fftchannels, int numffts, int subintsamples) {
  size_t ichan = threadIdx.x + blockIdx.x * blockDim.x;
  size_t ifft = blockIdx.y;
  size_t iant = blockIdx.z;
  //int numffts = gridDim.y;
  //int subintsamples = numffts * fftsize;

  // phase and slope for this FFT
  float dslope = fractionalDelayValues[iant*numffts + ifft];
  float theta = ichan*dslope;

  cuRotatePhase(&ant[sampIdx(iant, 0, ichan+ifft*fftchannels, subintsamples)], theta);
  cuRotatePhase(&ant[sampIdx(iant, 1, ichan+ifft*fftchannels, subintsamples)], theta);
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

__global__ void unpack2bit_2chan_fast(cuComplex *dest, const int8_t *src, const int32_t *shifts) {
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

__global__ void finaliseAccum(cuComplex *accum, int parallelAccum, int nchunk) { 

  int nchan = blockDim.x * gridDim.x;
  int ichan = (blockDim.x * blockIdx.x + threadIdx.x);
  int prod = blockIdx.y;
  int b = blockIdx.z;

  for (int i=1; i<parallelAccum; i++) {
    cuCaddIf(&accum[accumIdx(b, prod, ichan, nchan*parallelAccum)],
      accum[accumIdx(b, prod, ichan + i*nchan, nchan*parallelAccum)]);
  }
  cuCdivCf(&accum[accumIdx(b, prod, ichan, nchan*parallelAccum)], parallelAccum*nchunk);
}

// Launched with antenna indices in block .y and .z.
// (Turns out having the pol loop in the kernel performs poorly!)
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

    int s = nfft*fftwidth;

    for (int pi = 0; pi<2; ++pi) {
	for (int pj = 0; pj<2; ++pj) {
	    const float2* iv = &ants[antIdx(i, pi, t, s)];
	    const float2* jv = &ants[antIdx(j, pj, t, s)];

	    float2 u = iv[0];
	    float2 v = jv[0];
	    float2 a;
	    a.x = u.x*v.x + u.y*v.y;
	    a.y = u.y*v.x - u.x*v.y;

	    for (int k = fftwidth; k<s; k += fftwidth) {
		u = iv[k];
		v = jv[k];

		a.x += u.x*v.x + u.y*v.y;
		a.y += u.y*v.x - u.x*v.y;
	    }

	    a.x /= nfft;
	    a.y /= nfft;
	    accum[accumIdx(b, pi*2+pj, t, nchan)] = a;
	}
    }
}

__global__ void CCAH2(cuComplex *accum, const cuComplex *ants, int nant, int nfft, int nchan, int fftwidth) {
    int t = threadIdx.x+blockIdx.x*blockDim.x;
    if (t>=nchan) return;

    // blockIdx.y: index of first vector (2*antennaindex+polindex)
    // blockIdx.z: index delta to second vector, minus 1.
    int ii = blockIdx.y;
    int ij = blockIdx.z;

    ij += ii+1;

    int ai = ii/2;
    int aj = ij/2;

    if (ai>=aj || ai>=nant || aj>=nant) return;

    int pi = ii%2;
    int pj = ij%2;

    // index into output vector blocks: = (j-i-1) + n-1 + ... + n-i
    int b = 4*(ai*nant-ai*(ai+1)/2 + aj-ai-1)+2*pi+pj;

    int s = nfft*fftwidth;

    const float2* iv = ants+ii*s+t;
    const float2* jv = ants+ij*s+t;

    float2 u = iv[0];
    float2 v = jv[0];
    float2 a;
    a.x = u.x*v.x + u.y*v.y;
    a.y = u.y*v.x - u.x*v.y;

    for (int k = fftwidth; k<s; k += fftwidth) {
        u = iv[k];
        v = jv[k];

        a.x += u.x*v.x + u.y*v.y;
        a.y += u.y*v.x - u.x*v.y;
    }

    a.x /= nfft;
    a.y /= nfft;
    accum[b*nchan+t] = a;
}

__global__ void printArray(cuComplex *a) {
  int i = threadIdx.x;
  printf("%f%+fi\n", a[i].x, a[i].y);
}
__global__ void printArrayInt(int8_t *a) {
  int i = threadIdx.x;
  printf("%d\n", a[i]);
}
