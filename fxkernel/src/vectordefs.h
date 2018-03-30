/* Convenience constants */
#define TWO_PI                   2*IPP_PI
#define TINY                     0.00000000001

/* Types */
#define f32                      Ipp32f
#define f64                      Ipp64f
#define cf32                     Ipp32fc
#define cf64                     Ipp64fc
#define vecNoErr                 ippStsNoErr

/* Allocation of arrays */
#define vectorAlloc_f32(length)  ippsMalloc_32f(length)
#define vectorAlloc_f64(length)  ippsMalloc_64f(length)
#define vectorAlloc_cf32(length) ippsMalloc_32fc(length)
#define vectorAlloc_cf64(length) ippsMalloc_64fc(length)

/* De-allocation of arrays */
#define vectorFree(memptr)       ippsFree(memptr)

/* Vector functions, ordered alphabetically */
#define vectorAddC_f64_I(val, srcdest, length)                              ippsAddC_64f_I(val, srcdest, length)
#define vectorAddProduct_cf32(src1, src2, accumulator, length)              ippsAddProduct_32fc(src1, src2, accumulator, length)
#define vectorMul_cf32(src1, src2, dest, length)                            ippsMul_32fc(src1, src2, dest, length)
#define vectorMulC_f64(src, val, dest, length)                              ippsMulC_64f(src, val, dest, length)
