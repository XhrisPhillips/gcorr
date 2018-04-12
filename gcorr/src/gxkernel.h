#ifndef _gxkernel
#define _gxkernel


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

#endif
