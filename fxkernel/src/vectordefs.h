#ifndef VECTORDEFS_H
#define VECTORDEFS_H


#include <math.h>

/* Convenience constants */
#define TWO_PI                   2*IPP_PI
#define TINY                     0.00000000001

/* Types */
#define u8                       Ipp8u
#define f32                      Ipp32f
#define f64                      Ipp64f
#define cf32                     Ipp32fc
#define cf64                     Ipp64fc
#define vecNoErr                 ippStsNoErr
#define vecStatus                IppStatus

/* Allocation of arrays */
#define vectorAlloc_u8(length)   ippsMalloc_8u(length)
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
#define vectorMulC_cf32(src, val, dest, length)                             ippsMulC_32fc(src, val, dest, length)
#define vectorMulC_f64(src, val, dest, length)                              ippsMulC_64f(src, val, dest, length)
#define vectorRealToComplex_f32(real, imag, complex, length)                ippsRealToCplx_32f(real, imag, complex, length)
#define vectorSinCos_f32(src, sin, cos, length)                             ippsSinCos_32f_A11(src, sin, cos, length)
//#define vectorSinCos_f32(src, sin, cos, length)                             genericSinCos_32f(src, sin, cos, length)

/* A generic version of SinCos, since it seems to have disappeared out of IPPS (probably in MKL?) */
inline vecStatus genericSinCos_32f(const f32 *src, f32 *sin, f32 *cos, int length)
{ for(int i=0;i<length;i++) sin[i]=sinf(src[i]);
  for(int i=0;i<length;i++) cos[i]=cosf(src[i]);
  return vecNoErr; }

#endif
