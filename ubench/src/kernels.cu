#include <complex>

namespace kernel {

__global__
void naive_accum(float2* out, const float2* data, int n, int m, int r) {
    int t = threadIdx.x+blockIdx.x*blockDim.x;
    if (t>=m) return;

    // first vector index
    int i = blockIdx.y;

    if (i>=n) return;

    // index into output vectors: = n-1 + ... + n-i
    int b = i*n-i*(i+1)/2;

    for (int k = 0; k<r; k += m) {
	float2 u = data[i*r+k+t];

	for (int j = i+1; j<n; ++j) {
	    float2 v = data[j*r+k+t];

	    float2 r;
	    r.x = u.x*v.x + u.y*v.y;
	    r.y = u.y*v.x - u.x*v.y;

	    float2* p = out+(b+j)*m+t;
	    p->x += r.x;
	    p->y += r.y;
	}
    }
}

} // kernel

using cfloat = std::complex<float>;
void naive_accum(cfloat* out, const cfloat* data, int n, int m, int r) {
    unsigned block_width = 128;

    dim3 block(1+(m-1)/block_width, n-1, 1);
    kernel::naive_accum<<<block, block_width>>>((float2*)out, (const float2*)data, n, m, r);
}
