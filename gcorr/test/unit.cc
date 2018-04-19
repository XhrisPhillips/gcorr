#include <cstddef>
#include <vector>
#include <iostream>
#include <random>

#include <cuComplex.h>

#include "gtest.h"
#include "wrappers.h"
#include "mkdata.h"

using std::size_t;

bool operator==(float2 a, float2 b) {
    return a.x==b.x && a.y==b.y;
}

bool operator!=(float2 a, float2 b) {
    return !(a==b);
}

std::ostream& operator<<(std::ostream& o, float2 v) {
    return o << '(' << v.x << ',' << v.y << ')';
}

// Use complex small magnitude ints to avoid rouding artefacts in testing.
std::vector<float2> random_cint_data(size_t n, int seed = 12345) {
    std::minstd_rand R(seed);
    std::uniform_int_distribution<int> U(-5, 5);

    std::vector<float2> data(n);
    std::generate(data.begin(), data.end(), [&]() {
	float2 f;
	f.x = U(R);
	f.y = U(R);
       	return f;
    });
    return data;
}

template <int npol>
std::vector<float2> expected_crosscorraccum(const std::vector<float2>& data, int nant, int nfft, int nchan, int fftwidth) {
    size_t result_sz = nant*(nant-1)/2*npol*npol*nchan;
    std::vector<float2> result(result_sz);

    size_t stride = nfft*fftwidth;
    size_t b = 0;

    for (int i = 0; i<nant-1; ++i) {
	for (int j = i+1; j<nant; ++j) {
	    for (int pi = 0; pi<npol; ++pi) {
		for (int pj = 0; pj<npol; ++pj) {
		    for (int k = 0; k<nchan; ++k) {
			float2 a = {0.f, 0.f};
			for (int f = 0; f<nfft; ++f) {
			    float2 u = data[(pi+i*npol)*stride+k+f*fftwidth];
			    float2 v = data[(pj+j*npol)*stride+k+f*fftwidth];
			    a.x += u.x*v.x + u.y*v.y;
			    a.y += u.y*v.x - u.x*v.y;
			}
			a.x /= nfft;
			a.y /= nfft;
			result[b++] = a;
		    }
		}
	    }
	}
    }

    return result;
}

TEST(gxkernel, CrossCorrAccumHoriz) {
    constexpr int pol = 2;

    {
	int nant = 3;
	int fftwidth = 512;
	int nchan = 256;
	int nfft = 4;

        size_t datasz = pol*nant*fftwidth*nfft;
	auto data = random_cint_data(datasz, -5, 5);

	auto expected = expected_crosscorraccum<pol>(data, nant, nfft, nchan, fftwidth);
	auto result = run_CrossCorrAccumHoriz(data, nant, nfft, nchan, fftwidth);
	EXPECT_EQ(expected, result);
	for (int i=0; i<result.size(); ++i) {
	    ASSERT_EQ(expected[i], result[i]) << "unequal at i=" << i;
	}
    }
    {
	int nant = 5;
	int fftwidth = 1024;
	int nchan = 1024;
	int nfft = 100;

        size_t datasz = pol*nant*fftwidth*nfft;
	auto data = random_cint_data(datasz, -5, 5);

	auto expected = expected_crosscorraccum<pol>(data, nant, nfft, nchan, fftwidth);
	auto result = run_CrossCorrAccumHoriz(data, nant, nfft, nchan, fftwidth);
	EXPECT_EQ(expected, result);
    }
}

TEST(gxkernel, CCCAH2) {
    constexpr int pol = 2;

    {
	int nant = 3;
	int fftwidth = 512;
	int nchan = 256;
	int nfft = 4;

        size_t datasz = pol*nant*fftwidth*nfft;
	auto data = random_cint_data(datasz, -5, 5);

	auto expected = expected_crosscorraccum<pol>(data, nant, nfft, nchan, fftwidth);
	auto result = run_CCAH2(data, nant, nfft, nchan, fftwidth);
	EXPECT_EQ(expected, result);
	for (int i=0; i<result.size(); ++i) {
	    ASSERT_EQ(expected[i], result[i]) << "unequal at i=" << i;
	}
    }
    {
	int nant = 5;
	int fftwidth = 1024;
	int nchan = 1024;
	int nfft = 100;

        size_t datasz = pol*nant*fftwidth*nfft;
	auto data = random_cint_data(datasz, -5, 5);

	auto expected = expected_crosscorraccum<pol>(data, nant, nfft, nchan, fftwidth);
	auto result = run_CCAH2(data, nant, nfft, nchan, fftwidth);
	EXPECT_EQ(expected, result);
    }
}


TEST(gxkernel, CrossCorr) {
    // require nchan*2=fftwidth and nchan a multiple of 512.
    constexpr int pol = 2;

    {
	int nant = 3;
	int fftwidth = 1024;
	int nchan = 512;
	int nfft = 3000;

        size_t datasz = pol*nant*fftwidth*nfft;
	auto data = random_cint_data(datasz, -5, 5);

	auto expected = expected_crosscorraccum<pol>(data, nant, nfft, nchan, fftwidth);
	auto result = run_CrossCorr(data, nant, nfft, nchan, fftwidth);
	EXPECT_EQ(expected, result);
    }
}


int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}

