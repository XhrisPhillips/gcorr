/* Types */
#define f32                      Ipp32f
#define f64                      Ipp64f
#define cf32                     Ipp32fc
#define cf64                     Ipp64fc

/* Allocation of arrays */
#define vectorAlloc_f32(length)  ippsMalloc_32f(length)
#define vectorAlloc_f64(length)  ippsMalloc_64f(length)
#define vectorAlloc_cf32(length) ippsMalloc_32fc(length)
#define vectorAlloc_cf64(length) ippsMalloc_64fc(length)

/* De-allocation of arrays */
#define vectorFree(memptr)       ippsFree(memptr)

/* Vector functions, ordered alphabetically */
#define vectorAddProduct_cf32(src1, src2, accumulator, length)              ippsAddProduct_32fc(src1, src2, accumulator, length)
#define vectorMul_cf32(src1, src2, dest, length)                            ippsMul_32fc(src1, src2, dest, length)
