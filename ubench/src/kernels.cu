#include <complex>

namespace kernel {

__global__
void naive_accum(float2* out, const float2* data, int n, int r, int m) {
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
void sleep(unsigned long cycles) {
    auto t0 = clock64();
    while (clock64()-t0<cycles) ;
}

} // kernel

using cfloat = std::complex<float>;
void naive_accum(cfloat* out, const cfloat* data, int n, int r, int m) {
    unsigned block_width = 128;

    dim3 block(1+(m-1)/block_width, n-1, 1);
    kernel::naive_accum<<<block, block_width>>>((float2*)out, (const float2*)data, n, r, m);
}

void sleep_10ms(cfloat* out, const cfloat* data, int n, int r, int m) {
    cudaDeviceProp prop;
    int device;
    cudaGetDevice(&device);
    cudaGetDeviceProperties(&prop, device);
    unsigned long ncycle = 10*prop.clockRate;
    kernel::sleep<<<1,1>>>(ncycle);
}
