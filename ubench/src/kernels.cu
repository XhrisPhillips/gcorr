#include <complex>

#include "kernels.h"

using cfloat = std::complex<float>;

namespace kernel {

// Naive kernel: accumulates in global memory, no chunking.

__global__
void naive_accumulate(float2* out, const float2* data, int n, int r, int m) {
    int t = threadIdx.x+blockIdx.x*blockDim.x;
    if (t>=m) return;

    // first vector index
    int i = blockIdx.y;

    if (i>=n) return;

    // index into output vectors: = n-1 + ... + n-i
    int b = i*n-i*(i+1)/2;

    for (int k = 0; k<r; k += m) {
        int b2 = b;
	for (int j = i+1; j<n; ++j) {
	    float2* p = out+b2*m+t;
	    p->x = 0;
	    p->y = 0;
            ++b2;
        }
    }

    for (int k = 0; k<r; k += m) {
	float2 u = data[i*r+k+t];

        int b2 = b;
	for (int j = i+1; j<n; ++j) {
	    float2 v = data[j*r+k+t];

	    float2 z;
	    z.x = u.x*v.x + u.y*v.y;
	    z.y = u.y*v.x - u.x*v.y;

	    float2* p = out+b2*m+t;
	    p->x += z.x;
	    p->y += z.y;

            ++b2;
	}
    }
}

__global__
void simple_horiz_accumulate(float2* out, const float2* data, int n, int r, int m) {
    int t = threadIdx.x+blockIdx.x*blockDim.x;
    if (t>=m) return;

    // input vector indices in block .y and .z
    int i = blockIdx.y;
    int j = blockIdx.z;

    j += i+1;
    if (i>=n || j>=n) return;

    // index into output vectors: = (j-i-1) + n-1 + ... + n-i
    int b = i*n-i*(i+1)/2 + j-i-1;

    const float2* iv = data+i*r+t;
    const float2* jv = data+j*r+t;

    float2 u = iv[0];
    float2 v = jv[0];
    float2 a;
    a.x = u.x*v.x + u.y*v.y;
    a.y = u.y*v.x - u.x*v.y;

    for (int k = m; k<r; k += m) {
        u = iv[k];
        v = jv[k];

        a.x += u.x*v.x + u.y*v.y;
        a.y += u.y*v.x - u.x*v.y;
    }

    out[b*m+t] = a;
}

// Gcorr global memory chunked correlation.

__host__ __device__ static __inline__ int
accumIdx(int baseline, int channel, int stride)
{
    return baseline * stride + channel;
}

 __device__ static __inline__ int
antIdx(int antenna, int channel, int stride)
{
    return antenna * stride + channel;
}

__device__ __inline__ void cuCaddIf(float2 *a, float2 b)
{
  (*a).x += b.x;
  (*a).y += b.y;
}

__device__ __inline__ void cuCaddIfAtomic(float2 *a, float2 b)
{
  atomicAdd(&a->x, b.x);
  atomicAdd(&a->y, b.y);
}

// Multiply a complex number by the conjugate of the other
__host__ __device__ static __inline__ float2 cuCmulConjf (float2 x, float2 y)
{
    float2 prod;
    prod.x = x.x * y.x + x.y * y.y;
    prod.y = x.y * y.x - x.x * y.y;

    return prod;
}


__global__
void CrossCorr(const float2 *ants, float2 *accum, int nant, int nchunk) {
    int nchan = blockDim.x * gridDim.x;
    size_t ichan = threadIdx.x + blockIdx.x * blockDim.x + blockIdx.y * nchan * nchunk;
    int ochan = threadIdx.x + blockIdx.x * blockDim.x + blockIdx.y * nchan;
    int parallelAccum = blockDim.y * gridDim.y;
    int subintsamples = parallelAccum * nchan * nchunk;

    int i, j, l, b;
    for (l=0; l<nchunk; l++) {
        b=0;
        for (i=0; i<nant-1; i++) {
            for (j=i+1; j<nant; j++) {
                cuCaddIf(&accum[accumIdx(b, ochan, nchan*parallelAccum)],
                        cuCmulConjf(ants[antIdx(i, ichan, subintsamples)], ants[antIdx(j, ichan, subintsamples)]));
                b++;
            }
        }
        ichan += nchan;
    }
}

__device__
inline bool operator!=(float2 a, float2 b) { return a.x!=b.x || a.y!=b.y; }


__global__
void CrossCorrShared(const float2 *ants, float2 *accum, int nant, int nchunk) {
    extern __shared__ float2 antShar[];

    int nchan = blockDim.x * gridDim.x;
    size_t ichan = threadIdx.x + blockIdx.x * blockDim.x + blockIdx.y * nchan * nchunk;
    const size_t ochan = threadIdx.x + blockIdx.x * blockDim.x + blockIdx.y * nchan;
    int parallelAccum = blockDim.y * gridDim.y;
    int subintsamples = parallelAccum * nchan * nchunk;

    int i, j, l, b;
    for (l=0; l<nchunk; l++) {
        if (threadIdx.x<nant) antShar[threadIdx.x] = ants[antIdx(threadIdx.x, ichan, subintsamples)];
        __syncthreads();

        b=0;
        for (i=0; i<nant-1; i++) {
            for (j=i+1; j<nant; j++) {
                cuCaddIf(&accum[accumIdx(b, ochan, nchan*parallelAccum)], cuCmulConjf(antShar[i], antShar[j]));
                b++;
            }
        }
        ichan += nchan;
        __syncthreads();
    }
}

__global__
void finaliseAccum(float2* out, float2 *accum, int parallelAccum, int nchunk) {
    int nchan = blockDim.x * gridDim.x;
    int ichan = (blockDim.x * blockIdx.x + threadIdx.x);
    int b = blockIdx.z;

    out[b*nchan+ichan] = accum[accumIdx(b, ichan, nchan*parallelAccum)];
    for (int i=1; i<parallelAccum; i++) {
        cuCaddIf(&out[b*nchan+ichan], accum[accumIdx(b, ichan + i*nchan, nchan*parallelAccum)]);
    }
}

} // kernel

// simple implementations

cfloat* naive_accumulate::operator()() {
    unsigned block_width = 128;

    dim3 block(1+(m-1)/block_width, n-1, 1);
    kernel::naive_accumulate<<<block, block_width>>>((float2*)out, (const float2*)data, n, r, m);

    return out;
}

cfloat* simple_horiz_accumulate::operator()() {
    unsigned block_width = 64;

    dim3 block(1+(m-1)/block_width, n-1, n-1);
    kernel::simple_horiz_accumulate<<<block, block_width>>>((float2*)out, (const float2*)data, n, r, m);

    return out;
}

// gcorr using global mem:

void gcorr_global_accumulate::init() {
    // m corresponds to fftchannels; n to numantennas; r to subintsamples.
    int targetThreads = 50e4;
    parallelAccum = (int)ceil(targetThreads/m+1); // I suspect this has failure modes
    while (parallelAccum && (r/m) % parallelAccum) parallelAccum--;

    cudaMalloc(&baselineData, n*(n-1)/2*m*parallelAccum*sizeof(float2));
}

cfloat* gcorr_global_accumulate::operator()() {
    int corrThreads = 512;
    int blockchan = m/512;
    dim3 corrBlocks = dim3(blockchan, parallelAccum);
    int numffts = r/m;
    int nchunk = numffts / parallelAccum;

    cudaMemset(baselineData, 0, n*(n-1)/2*m*parallelAccum*sizeof(float2));
    kernel::CrossCorr<<<corrBlocks,corrThreads>>>((const float2*)data, (float2*)baselineData, n, nchunk);

    dim3 accumBlocks = dim3(blockchan, 1, n*(n-1)/2);
    kernel::finaliseAccum<<<accumBlocks,corrThreads>>>((float2*)out, (float2*)baselineData, parallelAccum, nchunk);

    return out;
}

gcorr_global_accumulate::~gcorr_global_accumulate() {
    cudaFree(baselineData);
}
// gcorr using shared mem:

void gcorr_shared_accumulate::init() {
    // m corresponds to fftchannels; n to numantennas; r to subintsamples.
    int targetThreads = 50e4;
    parallelAccum = (int)ceil(targetThreads/m+1); // I suspect this has failure modes
    while (parallelAccum && (r/m) % parallelAccum) parallelAccum--;

    cudaMalloc(&baselineData, n*(n-1)/2*m*parallelAccum*sizeof(float2));
}

cfloat* gcorr_shared_accumulate::operator()() {
    int corrThreads = 512;
    int blockchan = m/512;
    dim3 corrBlocks = dim3(blockchan, parallelAccum);
    int numffts = r/m;
    int nchunk = numffts / parallelAccum;

    float2* abuf;
    cudaMalloc(&abuf, n*sizeof(float2));

    cudaMemset(baselineData, 0, n*(n-1)/2*m*parallelAccum*sizeof(float2));
    kernel::CrossCorrShared<<<corrBlocks,corrThreads,n*sizeof(float2)>>>((const float2*)data, (float2*)baselineData, n, nchunk);

    dim3 accumBlocks = dim3(blockchan, 1, n*(n-1)/2);
    kernel::finaliseAccum<<<accumBlocks,corrThreads>>>((float2*)out, (float2*)baselineData, parallelAccum, nchunk);

    cudaFree(abuf);
    return out;
}

gcorr_shared_accumulate::~gcorr_shared_accumulate() {
    cudaFree(baselineData);
}

