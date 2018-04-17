
#ifndef _gxkernel
#define _gxkernel

#include <stdio.h>


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
__global__ void setFringeRotation(float *rotVec);
__global__ void FringeRotate(cuComplex *ant, float *rotVec);
__global__ void CrossCorr(cuComplex *ants, cuComplex *accum, int nant, int nchunk);
__global__ void CrossCorrShared(cuComplex *ants, cuComplex *accum, int nant, int nchunk);
__global__ void finaliseAccum(cuComplex *accum, int parallelAccum, int nchunk);
__global__ void printArray(cuComplex *a);
__global__ void printArrayInt(int8_t *a);

#endif
