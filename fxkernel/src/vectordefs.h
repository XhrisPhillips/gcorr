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
#define vectorAdd_cf32_I(src, srcdest, length)                              ippsAdd_32fc_I(src, srcdest, length)
#define vectorAddC_f64_I(val, srcdest, length)                              ippsAddC_64f_I(val, srcdest, length)
#define vectorAddProduct_cf32(src1, src2, accumulator, length)              ippsAddProduct_32fc(src1, src2, accumulator, length)
#define vectorConj_cf32(src, dest, length)                                  ippsConj_32fc(src, dest, length)
#define vectorDivC_cf32_I(val, srcdest, length)                             ippsDivC_32fc_I(val, srcdest, length)
#define vectorMagnitude_cf32(src, dest, length)                             ippsMagnitude_32fc(src, dest, length)
#define vectorMul_cf32(src1, src2, dest, length)                            ippsMul_32fc(src1, src2, dest, length)
#define vectorMul_cf32_I(src, srcdest, length)                              ippsMul_32fc_I(src, srcdest, length)
#define vectorMulC_f32(src, val, dest, length)                              ippsMulC_32f(src, val, dest, length)
#define vectorMulC_cf32(src, val, dest, length)                             ippsMulC_32fc(src, val, dest, length)
#define vectorMulC_cf32_I(val, srcdest, length)                             ippsMulC_32fc_I(val, srcdest, length)
#define vectorMulC_f64(src, val, dest, length)                              ippsMulC_64f(src, val, dest, length)
#define vectorMulC_f64_I(val, srcdest, length)                              ippsMulC_64f_I(val, srcdest, length)
#define vectorPhase_cf32(src, dest, length)                                 ippsPhase_32fc(src, dest, length)
#define vectorRealToComplex_f32(real, imag, complex, length)                ippsRealToCplx_32f(real, imag, complex, length)
#define vectorSinCos_f32(src, sin, cos, length)                             ippsSinCos_32f_A11(src, sin, cos, length)
#define vectorZero_cf32(dest, length)                                       ippsZero_32fc(dest, length)

/* Copy Routines */

#define vectorCopy_u8(src, dest, length)                                    ippsCopy_8u(src, dest, length)
#define vectorCopy_s16(src, dest, length)                                   ippsCopy_16s(src, dest, length)
#define vectorCopy_s32(src, dest, length)                                   ippsCopy_32f((f32*)src, (f32*)dest, length)
#define vectorCopy_f32(src, dest, length)                                   ippsCopy_32f(src, dest, length)
#define vectorCopy_cf32(src, dest, length)                                  ippsCopy_32fc(src, dest, length)
#define vectorCopy_f64(src, dest, length)                                   ippsCopy_64f(src, dest, length)

/* FFT routines */
#define vecFFT_NoReNorm          IPP_FFT_NODIV_BY_ANY
#define vecFFT_ReNorm            IPP_FFT_DIV_INV_BY_N
#define vecAlgHintFast           ippAlgHintFast
#define vecAlgHintAccurate       ippAlgHintAccurate
#define vecFFTSpecC_cf32         IppsFFTSpec_C_32fc
#define vectorFFT_RtoC_f32(src, dest, fftspec, fftbuffer)                   ippsFFTFwd_RToCCS_32f(src, dest, fftspec, fftbuffer)
#define vectorFFT_CtoC_cf32(src, dest, fftspec, fftbuff)                    ippsFFTFwd_CToC_32fc(src, dest, fftspec, fftbuff)

#endif
