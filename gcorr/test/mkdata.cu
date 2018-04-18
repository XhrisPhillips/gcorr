#include <cstddef>
#include <vector>
#include <algorithm>
#include <random>

#include <cuComplex.h>

#include "mkdata.h"

// Use complex small magnitude ints to avoid rouding artefacts in testing.
std::vector<float2> random_cint_data(std::size_t n, int vmin, int vmax, int seed) {
    std::minstd_rand R(seed);
    std::uniform_int_distribution<int> U(vmin, vmax);

    std::vector<float2> data(n);
    std::generate(data.begin(), data.end(), [&]() {
	float2 f;
	f.x = U(R);
	f.y = U(R);
       	return f;
    });
    return data;
}

std::vector<float2> random_cfloat_data(std::size_t n, float vmin, float vmax, int seed) {
    std::minstd_rand R(seed);
    std::uniform_real_distribution<float> U(vmin, vmax);

    std::vector<float2> data(n);
    std::generate(data.begin(), data.end(), [&]() {
	float2 f;
	f.x = U(R);
	f.y = U(R);
       	return f;
    });
    return data;
}
