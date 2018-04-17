#include <cassert>
#include <cfloat>
#include <complex>
#include <cmath>
#include <random>
#include <vector>
#include <stdexcept>
#include <string>
#include <algorithm>

#include <cuda_runtime.h>

#include "benchmark/benchmark.h"

using cfloat = std::complex<float>;
typedef void (*accum_fn)(cfloat*, const cfloat*, int, int, int);

void naive_accum(cfloat* out, const cfloat* data, int n, int m, int r);

// accum_fn(out, data, n, r, m):
//
// b = 0
// for i=0..n-1, j=i+1..n-1:
//    for s=0..r/m:
//        out[b+k] = data[i*r+k+m*s]*conj(data[j*r+k+m*s] for k=0..m-1
// 
//    ++b


template <typename Seq1, typename Seq2, typename V>
void assert_near(const Seq1& a, const Seq2& b, V err) {
    using std::begin;
    using std::end;

    auto ai = begin(a);
    auto ae = end(a);

    auto bi = begin(b);
    auto be = end(b);

    std::size_t i = 0;
    for (;ai!=ae && bi!=be; ++i, ++ai, ++bi) {
        auto delta = std::abs(*ai-*bi);
        if (delta>err) {
            throw std::runtime_error("sequences differ at index "+std::to_string(i)+" by "+std::to_string(delta));
        }
    }

    if (ai!=ae || bi!=be) {
        throw std::runtime_error("sequences differ in length");
    }
}

std::vector<cfloat> random_data(std::size_t n) {
    std::minstd_rand R;
    std::uniform_real_distribution<float> U(0,1);

    std::vector<cfloat> data(n);
    std::generate(data.begin(), data.end(), [&]() { return cfloat(U(R), U(R)); });
    return data;
}

std::vector<cfloat> run_accum(const std::vector<cfloat>& data, int n, int r, int m) {
    std::vector<cfloat> accum(m*n*(n-1)/2);

    cfloat* a = accum.data();
    for (int i = 0; i<n; ++i) {
        for (int j = 0; j<n; ++j) {
            const cfloat* u = data.data()+i*r;
            const cfloat* v = data.data()+j*r;
            for (int k = 0; k<r; k+=m) {
                for (int l = 0; l<m; ++l) {
                    a[i] += u[l]*std::conj(v[l]);
                }
            }
            a += m;
        }
    }
    return accum;
}

void harness_accum(benchmark::State& state, accum_fn f) {
    int n = state.range(0);
    int r = state.range(1);
    int m = state.range(2);

    std::vector<cfloat> data = random_data(n*r);
    std::vector<cfloat> expected = run_accum(data, n, r, m);
    std::vector<cfloat> result(expected.size());

    cudaEvent_t ev[2];
    cudaEventCreate(&ev[0]);
    cudaEventCreate(&ev[1]);

    void* gpu_data;
    std::size_t data_bytes = data.size()*sizeof(cfloat);
    cudaMalloc(&gpu_data, data_bytes);

    void* gpu_result;
    std::size_t result_bytes = result.size()*sizeof(cfloat);
    cudaMalloc(&gpu_result, result_bytes);

    cudaMemcpy(gpu_data, data.data(), data_bytes, cudaMemcpyHostToDevice);

    for (auto _: state) {
        cudaEventRecord(ev[0]);
        f((cfloat*)gpu_result, (const cfloat*)gpu_data, n, r, m);
        cudaEventRecord(ev[1]);
        cudaEventSynchronize(ev[1]);

        float ms = 0;
        cudaEventElapsedTime(&ms, ev[0], ev[1]);
        state.SetIterationTime(ms*1000.0);
    }

    cudaMemcpy(result.data(), gpu_result, result_bytes, cudaMemcpyDeviceToHost);

    const float fudge = 10; // (we could have cancellation errors from complex mul)
    float errbound = fudge*3*r/m*FLT_EPSILON;
    assert_near(expected, result, errbound);
}

void with_custom_args(benchmark::internal::Benchmark* b) {
    for (int n = 4; n<=12; ++n) {
        for (int m = 1625; m<=6500; m<<=1) {
            int r = 1024*m;

            std::vector<int64_t> args = {n, r, m};
            b->Args(args)->UseManualTime();
        }
    }
}

#define BENCH_ACCUM(fn)\
void wrap##fn(benchmark::State& state) { harness_accum(state, fn); }\
BENCHMARK(wrap##fn)->Apply(with_custom_args);

BENCH_ACCUM(naive_accum);

BENCHMARK_MAIN();
