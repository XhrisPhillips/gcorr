#ifndef _BITCONVERSION_H
#define _BITCONVERSION_H

#include <ipps.h>

extern Ipp32f lookup[4];

void initlut();
void unpack2bit(Ipp8u *in, Ipp32f **out, int nchan, int n);
void unpack8bit(Ipp8s *in, Ipp32f **out, int nchan, int iscomplex, int n);
void unpack16bit(Ipp16s *in, Ipp32f **out, int nchan, int n);
void unpackFloat8(Ipp8u *in, Ipp32f **out, int nchan, int n);


int packBit2(Ipp32f **in, Ipp8u *out, int nchan, float mean, float stddev, int len);
int packBit8(Ipp32f **in, Ipp8s *out, int nchan, int iscomplex, float mean, float stddev, float target, int len);
int packBit16(Ipp32f **in, Ipp16s *out, int nchan, float mean, float stddev, float target, int len);
int packFloat8(Ipp32f **in, Ipp8u *out, int nchan, float mean, float stddev, float target, int len);

#endif
