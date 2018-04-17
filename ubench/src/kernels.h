#pragma once

#include <complex>

struct naive_accumulate {
    using cfloat = std::complex<float>;

    naive_accumulate(cfloat* out, const cfloat* data, int n, int r, int m):
        out(out), data(data), n(n), r(r), m(m) {}

    cfloat* operator()();

    cfloat* out;
    const cfloat* data;
    int n, r, m;
};

struct simple_horiz_accumulate {
    using cfloat = std::complex<float>;

    simple_horiz_accumulate(cfloat* out, const cfloat* data, int n, int r, int m):
        out(out), data(data), n(n), r(r), m(m) {}

    cfloat* operator()();

    cfloat* out;
    const cfloat* data;
    int n, r, m;
};

struct gcorr_global_accumulate {
    using cfloat = std::complex<float>;

    gcorr_global_accumulate(cfloat* out, const cfloat* data, int n, int r, int m):
        out(out), data(data), n(n), r(r), m(m)
    {
        init();
    }

    void init();
    cfloat* operator()();
    ~gcorr_global_accumulate();

    cfloat* out;
    const cfloat* data;
    int n, r, m;

    int parallelAccum = 0;
    cfloat* baselineData = nullptr;
};

