#include <cstddef>
#include <vector>
#include <iostream>
#include <random>

#include <cuComplex.h>

#include "gtest.h"

#include "gpu_array.h"
#include "gxkernel.h"

using std::size_t;

template <typename Wrapped>
double run_kernel(int repeat_count, Wrapped fn) {
    cudaEvent_t ev[2];
    cudaEventCreate(&ev[0]);
    cudaEventCreate(&ev[1]);

    cudaEventRecord(ev[0]);
    for (int i = 0; i<repeat_count; ++i) fn();
    cudaEventRecord(ev[1]);
    cudaEventSynchronize(ev[1]);

    float ms = 0;
    cudaEventElapsedTime(&ms, ev[0], ev[1]);
    cudaEventDestroy(ev[0]);
    cudaEventDestroy(ev[1]);

    return ms/1000.0/(double)repeat_count;
}

inline int nblocks(int n, int width) {
    return n? 1+(n-1)/width: 0;
}

std::vector<float2> run_CrossCorrAccumHoriz(const std::vector<float2>& data, int nant, int nfft, int nchan, int fftwidth) {
    constexpr int npol = 2;

    int block_width = 128;
    dim3 ccblock(nblocks(nchan, block_width), nant-1, nant-1);

    size_t result_sz = nant*(nant-1)/2*npol*npol*nchan;

    gpu_array<float2> gpu_data(data);
    gpu_array<float2> gpu_result(result_sz);

    CrossCorrAccumHoriz<<<ccblock, block_width>>>(gpu_result.data(), gpu_data.data(), nant, nfft, nchan, fftwidth);
    return gpu_result;
}

double time_CrossCorrAccumHoriz(int repeat_count, const float2* gpu_data, int nant, int nfft, int nchan, int fftwidth) {
    constexpr int npol = 2;

    int block_width = 128;
    dim3 ccblock(nblocks(nchan, block_width), nant-1, nant-1);

    size_t result_sz = nant*(nant-1)/2*npol*npol*nchan;
    gpu_array<float2> gpu_result(result_sz);

    return run_kernel(repeat_count,
	[&]() {
	    CrossCorrAccumHoriz<<<ccblock, block_width>>>(gpu_result.data(), gpu_data, nant, nfft, nchan, fftwidth);
	});
}


std::vector<float2> run_CCAH2(const std::vector<float2>& data, int nant, int nfft, int nchan, int fftwidth) {
    constexpr int npol = 2;
    int nantxp = nant*npol;

    int block_width = 128;
    dim3 ccblock(nblocks(nchan, block_width), nantxp-1, nantxp-1);

    size_t result_sz = nant*(nant-1)/2*npol*npol*nchan;

    gpu_array<float2> gpu_data(data);
    gpu_array<float2> gpu_result(result_sz);

    CCAH2<<<ccblock, block_width>>>(gpu_result.data(), gpu_data.data(), nant, nfft, nchan, fftwidth);
    return gpu_result;
}

double time_CCAH2(int repeat_count, const float2* gpu_data, int nant, int nfft, int nchan, int fftwidth) {
    constexpr int npol = 2;
    int nantxp = nant*npol;

    int block_width = 128;
    dim3 ccblock(nblocks(nchan, block_width), nantxp-1, nantxp-1);

    size_t result_sz = nant*(nant-1)/2*npol*npol*nchan;
    gpu_array<float2> gpu_result(result_sz);

    return run_kernel(repeat_count,
	[&]() {
	    CCAH2<<<ccblock, block_width>>>(gpu_result.data(), gpu_data, nant, nfft, nchan, fftwidth);
	});
}


std::vector<float2> run_CrossCorr(const std::vector<float2>& data, int nant, int nfft, int nchan, int fftwidth) {
    int targetThreads = 50e4;
    int parallelAccum = (int)ceil(targetThreads/nchan+1);
    while (parallelAccum && nfft % parallelAccum) parallelAccum--;

    int block_width = 512;
    int blockx = nblocks(nchan, block_width);

    dim3 corrBlocks(blockx, parallelAccum);
    dim3 accumBlocks(blockx, 4, nant*(nant-1)/2);

    size_t result_sz = nant*(nant-1)*2*nchan;

    gpu_array<float2> gpu_data(data);
    gpu_array<float2> gpu_baselinedata(result_sz*parallelAccum);

    CrossCorr<<<corrBlocks, block_width>>>(gpu_data.data(), gpu_baselinedata.data(), nant, nfft/parallelAccum);
    finaliseAccum<<<accumBlocks, block_width>>>(gpu_baselinedata.data(), parallelAccum, nfft/parallelAccum);

    std::vector<float2> baselinedata(gpu_baselinedata);
    std::vector<float2> result(result_sz);

    int nvec = nant*(nant-1)*2;
    int rstride = nchan, bstride = nchan*parallelAccum;
    for (int i = 0; i<nvec; ++i) {
        std::copy(baselinedata.data()+i*bstride, baselinedata.data()+i*bstride+nchan, result.data()+i*rstride);
    }

    return result;
}

double time_CrossCorr(int repeat_count, const float2* gpu_data, int nant, int nfft, int nchan, int fftwidth) {
    int targetThreads = 50e4;
    int parallelAccum = (int)ceil(targetThreads/nchan+1);
    while (parallelAccum && nfft % parallelAccum) parallelAccum--;

    int block_width = 512;
    int blockx = nblocks(nchan, block_width);

    dim3 corrBlocks(blockx, parallelAccum);
    dim3 accumBlocks(blockx, 4, nant*(nant-1)/2);

    size_t result_sz = nant*(nant-1)*2*nchan;
    gpu_array<float2> gpu_baselinedata(result_sz*parallelAccum);

    return run_kernel(repeat_count,
	[&]() {
            CrossCorr<<<corrBlocks, block_width>>>((float2*)gpu_data, gpu_baselinedata.data(), nant, nfft/parallelAccum);
            finaliseAccum<<<accumBlocks, block_width>>>(gpu_baselinedata.data(), parallelAccum, nfft/parallelAccum);
        });
}
