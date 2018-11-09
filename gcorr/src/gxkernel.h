
#ifndef _gxkernel
#define _gxkernel

//#define USEHALF

#include <stdio.h>

#ifdef USEHALF
#include <cuda_fp16.h>
#define COMPLEX half2
#define MAKECOMPLEX(x,y)   __floats2half2_rn(x,y)
#define HALF2FLOAT2(x)     __half22float2(x)
#define HALF2FLOAT(x)      __half2float(x)
#else
#define COMPLEX cuComplex
#define MAKECOMPLEX(x,y)  make_cuFloatComplex(x,y)
#define HALF2FLOAT2(x)    x
#define HALF2FLOAT(x)     x
#endif

__host__ __device__ static __inline__ int
sampIdx(int antenna, int pol, int sample, int stride)
{
  const int num_pols = 2;

  return (antenna * num_pols + pol) * stride + sample;
}

__host__ __device__ static __inline__ int
antIdx(int antenna, int pol, int channel, int stride)
{
  const int num_pols = 2;

  return (antenna * num_pols + pol) * stride + channel;
}

__host__ __device__ static __inline__ int
accumIdx(int baseline, int product, int channel, int stride)
{
  const int num_products = 4;

  return (baseline * num_products + product) * stride + channel;
}

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

void freeMem();

void init_2bitLevels();

__global__ void unpack2bit_2chan(cuComplex *dest, const int8_t *src);
__global__ void unpack8bitcomplex_2chan(cuComplex *dest, const int8_t *src, const int32_t *shifts, int fftsamples);
__global__ void unpack8bitcomplex_2chan_rotate(COMPLEX *dest, const int8_t *src, float *rotVec, const int32_t *shifts, int fftsamples);
__global__ void unpack2bit_2chan_fast(cuComplex *dest, const int8_t *src, const int32_t *shifts, int fftsamples);
__global__ void unpack2bit_2chan_rotate(COMPLEX *dest, const int8_t *src, float *rotVec, const int32_t *shifts, int fftsamples);
__global__ void old_unpack2bit_2chan(cuComplex **dest, const int8_t *src, const int iant);
__global__ void calculateDelaysAndPhases(double * gpuDelays, double lo, double sampletime, int fftsamples, int fftchannels, int samplegranularity, float * rotationPhaseInfo, int* sampleShifts, float* fractionalSampleDelays);
__global__ void setFringeRotation(float *rotVec);
__global__ void FringeRotate(cuComplex *ant, float *rotVec);
__global__ void FringeRotate2(cuComplex *ant, float *rotVec);
__global__ void FracSampleCorrection(COMPLEX *ant, float *fractionalDelayValues,
				     int numchannels, int fftchannels, int numffts, int subintsamples);
__global__ void CrossCorr(COMPLEX *ants, cuComplex *accum, int nant, int nchunk);
__global__ void CrossCorrShared(COMPLEX *ants, cuComplex *accum, int nant, int nchunk);
__global__ void finaliseAccum(cuComplex *accum, int parallelAccum, int nchunk);
__global__ void CrossCorrAccumHoriz(cuComplex *accum, const COMPLEX *ants, int nantxp, int nfft, int nchan, int fftwidth);
__global__ void CCAH2(cuComplex *accum, const COMPLEX *ants, int nant, int nfft, int nchan, int fftwidth);
__global__ void CCAH3(cuComplex *accum, const COMPLEX *ants, int nant, int nfft, int nchan, int fftwidth);
__global__ void printArray(cuComplex *a);
__global__ void printArrayInt(int8_t *a);

#endif
