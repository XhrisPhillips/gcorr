#include "benchmark/benchmark.h"

#include "wrappers.h"
#include "mkdata.h"
#include "gpu_array.h"

// crosscorr/accum benches

typedef double (*time_ccaccum_wrapper)(int, const float2*, int, int, int, int);

void run_bench_ccaccum(benchmark::State& state, time_ccaccum_wrapper wrapper) {
    constexpr int npol = 2;

    int nant = state.range(0);
    int nfft = state.range(1);
    int nchan = state.range(2);
    int fftwidth = state.range(3);

    size_t datasz = npol*nant*fftwidth*nfft;
    auto data = random_cfloat_data(datasz, -1.f, 1.f);
    gpu_array<float2> gpu_data(data);

    constexpr int inner_repeat = 1;

    for (auto _: state) {
        double seconds = wrapper(inner_repeat, gpu_data.data(), nant, nfft, nchan, fftwidth);
        state.SetIterationTime(seconds);
    }
}

void bench_ccaccum_args(benchmark::internal::Benchmark* b) {
    for (int nant = 2; nant<=20; ++nant) {
        int nfft = 3250;
        for (int nchan: {512, 1024}) {
            int fftwidth = nchan*2;
            std::vector<int64_t> args = {nant, nfft, nchan, fftwidth};
            b->Args(args)->UseManualTime()->Unit(benchmark::kMicrosecond);
        }
    }
}

void bench_CrossCorrAccumHoriz(benchmark::State& state) {
    run_bench_ccaccum(state, time_CrossCorrAccumHoriz);
}

BENCHMARK(bench_CrossCorrAccumHoriz)->Apply(bench_ccaccum_args);

void bench_CCAH2(benchmark::State& state) {
    run_bench_ccaccum(state, time_CCAH2);
}

BENCHMARK(bench_CCAH2)->Apply(bench_ccaccum_args);

void bench_CrossCorr(benchmark::State& state) {
    run_bench_ccaccum(state, time_CrossCorr);
}

BENCHMARK(bench_CrossCorr)->Apply(bench_ccaccum_args);

BENCHMARK_MAIN();


