#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <math.h>

#include "bitconversion.h"

Ipp32f lookup[4];
Ipp32f lookupFloat8[256];


typedef struct float8 {
  uint8_t mantisa : 4;
  uint8_t exponent : 3;
  uint8_t sign : 1;
} float8;


void initlut() {
  int i;
  float8 *f;
  
  /* Setup the lookup table */
  lookup[0] = -3.3359*4.73065987974663;  // Force RMS 10
  lookup[1] = -1.0*4.73065987974663;
  lookup[2] = +1.0*4.73065987974663;
  lookup[3] = +3.3359*4.73065987974663;

  for (i=0; i<256; i++) {
    f = (float8*)(&i);
    lookupFloat8[i] = f->mantisa;
    if (f->exponent>0) {
      lookupFloat8[i] +=16;
      lookupFloat8[i] *=pow(2,f->exponent-1);
    } 
    if (f->sign) lookupFloat8[i] *= -1;
  }
  return;
}

void unpack8bit(Ipp8s *in, Ipp32f **out, int nchan, int n) {
  IppStatus status;
  status =  ippsConvert_8s32f(in, out[0], n);
  if (status!=ippStsNoErr) {
    printf("ippsConvert_8s32f failed!\n");
    exit(1);
  }
  return;
}

void unpack16bit(Ipp16s *in, Ipp32f **out, int  nchan, int n) {
  IppStatus status;
  status =  ippsConvert_16s32f(in, out[0], n);
  if (status!=ippStsNoErr) {
    printf("ippsConvert_16s32f failed!\n");
    exit(1);
  }
  return;
}

void unpack2bit(Ipp8u *in, Ipp32f **out, int nchan, int n) {
  int i;
  if (nchan==1) {
    for (i=0; i<n/4; i++) {
      out[0][i*4]   = lookup[in[i]&0x3];
      out[0][i*4+1] = lookup[(in[i]>>2)&0x3];
      out[0][i*4+2] = lookup[(in[i]>>4)&0x3];
      out[0][i*4+3] = lookup[(in[i]>>6)&0x3];
    }
  } else if (nchan==2) {
    for (i=0; i<n/2; i++) {
      out[0][i*2]   = lookup[in[i]&0x3];
      out[1][i*2]   = lookup[(in[i]>>2)&0x3];
      out[0][i*2+1] = lookup[(in[i]>>4)&0x3];
      out[1][i*2+1] = lookup[(in[i]>>6)&0x3];
    }
  }
  return;
}


void unpackFloat8(Ipp8u *in, Ipp32f **out, int nchan,  int n) {
  int i;
  for (i=0; i<n; i++) {
    out[0][i] = lookupFloat8[in[i]];
  }
}

#define MAXPOS          3
#define SMALLPOS        2
#define SMALLNEG        1
#define MAXNEG          0

#define F2BIT(f,j) {					\
  if(f >= maxposThresh)  /* large positive */		\
    ch[j] = MAXPOS;					\
  else if(f <= maxnegThresh) /* large negative */	\
    ch[j] = MAXNEG;					\
  else if(f > mean)  /* small positive */		\
    ch[j] = SMALLPOS;					\
  else  /* small negative */				\
    ch[j] = SMALLNEG;					\
}


int packBit2(Ipp32f **in, Ipp8u *out, int nchan, float mean, float stddev, int len) {
  int i, j, ch[4];
  float maxposThresh, maxnegThresh;

  if (len*nchan%4!=0) {
    printf("Can only pack multiple of 4 samples!\n");
    return(1);
  }

  maxposThresh = mean+stddev*0.95;
  maxnegThresh = mean-stddev*0.95;

  j = 0;

  if (nchan==1) {
  
    for (i=0;i<len;) {
      F2BIT(in[0][i],0);
      i++;
      F2BIT(in[0][i],1);
      i++;
      F2BIT(in[0][i],2);
      i++;
      F2BIT(in[0][i],3);
      i++;
      out[j] = (ch[0])|((ch[1]<<2) )|((ch[2]<<4) )|((ch[3]<<6) );
      j++;
    }
  } else if (nchan==2) {
    for (i=0;i<len;) {
      F2BIT(in[0][i],0);
      F2BIT(in[1][i],1);
      i++;
      F2BIT(in[0][i],2);
      F2BIT(in[1][i],3);
      i++;
      out[j] = (ch[0])|((ch[1]<<2) )|((ch[2]<<4) )|((ch[3]<<6) );
      j++;
    }
  } else {
    fprintf(stderr, "Error: Do not support %d channels\n", nchan);
    exit(1);
  }
  return 0;
}


int packBit8(Ipp32f **in, Ipp8s *out, int nchan, int iscomplex, float mean, float stddev, float target, int len) {
  IppStatus status;

  for (int i=0; i<nchan; i++) {
  // Subtract mean and scale to target
  if (mean!=0.0) 
    ippsSubC_32f_I(mean, in[i], len);

  if (stddev!=0)
    ippsMulC_32f_I(target/stddev, in[i], len);
  }

  if (nchan==1) 
    status = ippsConvert_32f8s_Sfs(in[0], out, len, ippRndNear, 0);
  else if (nchan==2) {
    if (iscomplex) {
      status = ippStsNoErr;
      int o=0;
      for (int i=0;i<len;i+=2) {
	out[o++] = in[0][i];
	out[o++] = in[0][i+1];
	out[o++] = in[1][i];
	out[o++] = in[1][i+1];
      }
    } else {
      fprintf(stderr, "Error: Do not support %d channels 8bit\n", nchan);
      exit(1);
    }
  } else {
    fprintf(stderr, "Error: Do not support %d channels 8bit\n", nchan);
    exit(1);
  }
  
  if (status != ippStsNoErr) {
    fprintf(stderr, "Error calling ippsConvert_32f8s_Sfs\n");
  }
  return 0;
}

int packBit16(Ipp32f **in, Ipp16s *out, int nchan, float mean, float stddev, float target, int len) {
  IppStatus status;

  for (int i=0; i<nchan; i++) {
    // Subtract mean and scale to target
    if (mean!=0.0) 
      ippsSubC_32f_I(mean, in[i], len);

    if (stddev!=0)
      ippsMulC_32f_I(target/stddev, in[i], len);
  }

  if (nchan==1) {
    status = ippsConvert_32f16s_Sfs(in[0], out, len, ippRndNear, 0);
  } else {
    fprintf(stderr, "Error: Do not support %d channels\n", nchan);
    exit(1);
  }
  if (status != ippStsNoErr) {
    
  }
  return 0;
}

int packFloat8(Ipp32f **in, Ipp8u *out, int  nchan, float mean, float stddev, float target, int len) {
  int i;
  Ipp32f f;
  float8 *f8;
  

  for (int i=0; i<nchan; i++) {
    // Subtract mean and scale to target
    if (mean!=0.0) 
      ippsSubC_32f_I(mean, in[i], len);

    if (stddev!=0)
      ippsMulC_32f_I(target/stddev, in[i], len);
  }

  if (nchan==1) {
    for (i=0; i<len; i++) {
      f = fabs(in[0][i]);
      out[i] = 0;
      f8 = (float8*)&out[i];
	if (f>1984) {
	  f8->exponent=7;
	  f8->mantisa=15;
	} else {
	  while (f>=31.5) {
	    f /= 2;
	    f8->exponent++;
	  }
	  if (f>15.5) {
	    f -= 16;
	    f8->exponent++;
	  }
	  f8->mantisa = lrintf(f);
	}
	if (in[0][i]<0) f8->sign = 1;
    }
  } else {
    fprintf(stderr, "Error: Do not support %d channels\n", nchan);
    exit(1);
  }
  return 0;
}
