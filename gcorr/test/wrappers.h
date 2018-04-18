#pragma once

#include <vector>
#include <cuComplex.h>

// unit test wrappers:

std::vector<float2> run_CrossCorrAccumHoriz(const std::vector<float2>& data, int nant, int nfft, int nchan, int fftwidth);
std::vector<float2> run_CCAH2(const std::vector<float2>& data, int nant, int nfft, int nchan, int fftwidth);
std::vector<float2> run_CrossCorr(const std::vector<float2>& data, int nant, int nfft, int nchan, int fftwidth);

// ubench wrappers:

double time_CrossCorrAccumHoriz(int repeat_count, const float2* gpu_data, int nant, int nfft, int nchan, int fftwidth);
double time_CCAH2(int repeat_count, const float2* gpu_data, int nant, int nfft, int nchan, int fftwidth);
double time_CrossCorr(int repeat_count, const float2* gpu_data, int nant, int nfft, int nchan, int fftwidth);
