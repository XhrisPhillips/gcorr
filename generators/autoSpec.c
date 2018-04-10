#ifdef __APPLE__

#define OSX

#define OPENOPTIONS O_RDONLY

#else

#define LINUX
#define _LARGEFILE_SOURCE 
#define _LARGEFILE64_SOURCE
#define _FILE_OFFSET_BITS 64

#define OPENOPTIONS O_RDONLY|O_LARGEFILE

#endif

#include <stdlib.h>
#include <stdio.h>
#include <stdint.h>
#include <math.h>
#include <string.h>
#include <ctype.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#include <getopt.h>
#include <limits.h>
#include <time.h>
#include <sys/time.h>
#include <cpgplot.h>
#include <complex.h>
#include <math.h>
#include <sys/ipc.h>  /* for shared memory */
#include <sys/shm.h>

#include <ipps.h>
#include <ippvm.h>
#include <ippcore.h>

#include "bitconversion.h"

void setSpecname (char *filename, char *outfile);

void makespectrum(Ipp64f *spectrum, Ipp32fc *out, int n);
void doplot(int npoint, float *xvals, float *plotspec, char *outfile, int noplot);

Ipp32f *temp32f;
Ipp64f *temp64f;

#define BUFSIZE 16

#define MAXSTR 1024

double tim(void) {
  struct timeval tv;
  double t;

  gettimeofday(&tv, NULL);
  t = (double)tv.tv_sec + (double)tv.tv_usec/1000000.0;

  return t;
}

#define IPPMALLOC(var,type,n)	    \
  var = ippsMalloc_ ## type(n);     \
  if (var==NULL) {                  \
    fprintf(stderr, "Error allocating %d %s elements for %s\n", n, #type, #var); \
    exit(EXIT_FAILURE);             \
  }                                 \

int main (int argc, char * const argv[]) {
  int i, file, s, nread, opt, tmp, thisread, dobreak;
  int nfft, bytesperfft;
  char msg[MAXSTR+1], *filename, *outfile;
  Ipp8u  *buf;
  off_t off, offset;
  Ipp32fc *out;
  Ipp64f *spectrum, sumStdDev, sumMean;
  Ipp32f *plotspec, *in=NULL, *xvals, mean, stdDev;
  double t0, t1, tt, tA, tB;
  IppStatus status;

  IppsDFTSpec_R_32f *spec;
  IppsDFTSpec_C_32fc *specC;

  int bits=0;                   /* Number of bits/sample */
  int bandwidth=0;              /* Observing bandwidth */
  int iscomplex = 0;            /* Is sampling complex */
  int isfloat = 0;              /* Is data IEEE floating point */
  int skip=0;                   /* Number of bytes to skip at start of file */
  int npoint=512;               /* Number of spectral channels */
  int noplot=0;                 /* Don't plot with pgplot */
  int bufsize=BUFSIZE;          /* Size of chunks to proccess at once */
  char pgdev[MAXSTR+1] = "/xs"; /* Pgplot device to use */
  char *specfile = NULL;        /* Output filename spectrum */

  struct option options[] = {
    {"npoint", 1, 0, 'n'},
    {"nbits", 1, 0, 'b'},
    {"bits", 1, 0, 'b'},
    {"skip", 1, 0, 's'},
    {"bandwidth", 1, 0, 'w'},
    {"bufsize", 1, 0, 'm'},
    {"complex", 0, 0, 'c'},
    {"float", 0, 0, 'f'},
    {"noplot", 0, 0, 'B'},
    {"specfile", 1, 0, 'S'},
    {"device", 1, 0, 'd'},
    {"pgdev", 1, 0, 'd'},
    {"help", 0, 0, 'h'},
    {0, 0, 0, 0}
  };

  filename = NULL;

  /* Avoid compiler complaints */
  bytesperfft = 0;

  nfft = 0;
  /* Read command line options */

#define CASEINT(ch,var)                                     \
  case ch:						    \
    s = sscanf(optarg, "%d", &tmp);	          	    \
    if (s!=1)				        	    \
      fprintf(stderr, "Bad %s option %s\n", #var, optarg);  \
    else                                                    \
      var = tmp;                                            \
    break

#define CASEFLOAT(ch,var)                                   \
  case ch:						    \
    s = sscanf(optarg, "%f", &ftmp);		            \
    if (s!=1)				        	    \
      fprintf(stderr, "Bad %s option %s\n", #var, optarg);  \
    else                                                    \
      var = ftmp;                                           \
    break


  while (1) {
    opt = getopt_long_only(argc, argv, "b:w:d:N:s:y:n:c:C:Vpvo1:2:S:", 
			   options, NULL);
    if (opt==EOF) break;

    switch (opt) {
    
      CASEINT('b', bits);
      CASEINT('w', bandwidth);
      CASEINT('s', skip);
      CASEINT('n', npoint);
      CASEINT('m', bufsize);
  
    case 'd':
      if (strlen(optarg)>MAXSTR) {
	fprintf(stderr, "PGDEV option %s too long\n", optarg);
	return 0;
      }
      strcpy(pgdev, optarg);
      break;

    case 'S':
      if (strlen(optarg)>MAXSTR) {
	fprintf(stderr, " -specfile option %s too long\n", optarg);
	return 0;
      }
      specfile = strdup(optarg);
      break;

    case 'c':
      iscomplex = 1;
      break;

    case 'f':
      isfloat = 1;
      break;

    case 'B':
      noplot = 1;
      break;

    case 'h':
      printf("Usage: autoSpec [options]\n");
      printf("  -C/-chan <n>        Channel to correlate (can specify "
	     "multiple times)\n");
      printf("  -n/-npoint <n>      # spectral channels\n");
      printf("  -N/-init <val>      Number of ffts to average per integration\n");
      printf("  -device <pgdev>     Pgplot device to plot to\n");
      printf("  -s/-skip <n>        Skip this many bytes at start of file\n");
      printf("  -h/-help            This list\n");
      return(1);
      break;

    case '?':
    default:
      break;
    }
  }

  if (isfloat && bits==0) bits = 32;

  if (bits==0) {
    fprintf(stderr, "Must pass -bits\n");
    exit(1);
  }
  
  bufsize *= 1024*1024;

  /* Initiate plotting arrays */
  IPPMALLOC(xvals, 32f, npoint);
  IPPMALLOC(plotspec, 32f, npoint);

  IPPMALLOC(temp64f, 64f, npoint);
  IPPMALLOC(temp32f, 32f, npoint);

  if (specfile==NULL) {
    outfile = (char *)malloc(MAXSTR+1);
    if (outfile==NULL) {
      perror("Allocating memory");
      exit(1);
    }
  } else {
    outfile = specfile;
  }

  initlut();
  
  tt = 0; /* Total time */

  // TODO Check file passed!
  filename = argv[optind];

  file = open(filename, OPENOPTIONS);
  if (file==-1) {
    sprintf(msg, "Failed to open input file (%s)", filename);
    perror(msg);
    return(1);
  }
  fprintf(stderr, "%s\n", filename);

  bytesperfft = npoint*2*bits; // Actually bits
  if (bytesperfft % 8) { // Implicit assumption input data is an exact number of bytes
    fprintf(stderr, "Unsupported combination of %d spectral points and %d bits\n", npoint, bits);
    exit(1);
  }
  bytesperfft /= 8;
  printf("Bytes per fft = %d\n", bytesperfft);

  bufsize = (bufsize/bytesperfft)*bytesperfft;
  if (bufsize==0) {
    bufsize = bytesperfft;
  }
  printf("Using bufsize of %d bytes\n", bufsize);

  IPPMALLOC(buf, 8u, bufsize);
  if (!isfloat || bits!=32) { IPPMALLOC(in, 32f, npoint*2);}
  IPPMALLOC(out, 32fc, npoint+1); 
  IPPMALLOC(spectrum, 64f, npoint);
  IPPMALLOC(plotspec, 32f, npoint);
  status = ippsZero_64f(spectrum, npoint);
  if (status!=ippStsNoErr) {
    printf("ippsZero_64f failed (%d: %s)!\n", status, ippGetStatusString(status));
    exit(1);
  }

  int sizeDFTSpec, sizeDFTInitBuf, wbufsize;
  Ipp8u *workbuf;

  if (iscomplex) {
    status = ippsDFTGetSize_C_32fc(npoint, IPP_FFT_NODIV_BY_ANY, ippAlgHintAccurate, &sizeDFTSpec, &sizeDFTInitBuf, &wbufsize);
    if (status!=ippStsNoErr) {
      printf("ippsDFTGetSize_C_32fc failed (%d: %s)!\n", status, ippGetStatusString(status));
    }
  
    specC = (IppsDFTSpec_C_32fc*)ippsMalloc_8u(sizeDFTSpec);
    if (specC==NULL) {
      fprintf(stderr, "ippsMalloc failed\n");
      exit(1);
    }
    if (sizeDFTInitBuf>0) {
      IPPMALLOC(workbuf,8u, sizeDFTInitBuf);
    } else {
      workbuf = 0;
    }
  
    status = ippsDFTInit_C_32fc(npoint, IPP_FFT_NODIV_BY_ANY, ippAlgHintAccurate, specC, workbuf);
    if (status!=ippStsNoErr) {
      printf("ippsDFTInit_C_32fc failed (%d: %s)!\n", status, ippGetStatusString(status));
      exit(1);
    }
    if (workbuf) ippFree(workbuf);

  } else {
    status = ippsDFTGetSize_R_32f(npoint*2, IPP_FFT_NODIV_BY_ANY, ippAlgHintAccurate, &sizeDFTSpec, &sizeDFTInitBuf, &wbufsize);
    if (status!=ippStsNoErr) {
      printf("ippsDFTGetSize_R_32f failed (%d: %s)!\n", status, ippGetStatusString(status));
    }
  
    spec = (IppsDFTSpec_R_32f*)ippsMalloc_8u(sizeDFTSpec);
    if (spec==NULL) {
      fprintf(stderr, "ippsMalloc failed\n");
      exit(1);
    }
    if (sizeDFTInitBuf>0) {
      IPPMALLOC(workbuf,8u, sizeDFTInitBuf);
    } else {
      workbuf = 0;
    }
  
    status = ippsDFTInit_R_32f(npoint*2, IPP_FFT_NODIV_BY_ANY, ippAlgHintAccurate, spec, workbuf);
    if (status!=ippStsNoErr) {
      printf("ippsDFTInit_R_32f failed (%d: %s)!\n", status, ippGetStatusString(status));
      exit(1);
    }
    if (workbuf) ippFree(workbuf);
  }
  if (wbufsize>0) {
    IPPMALLOC(workbuf, 8u, wbufsize);
  } else {
    workbuf = 0;
  }

  if (skip>0) {
    offset = lseek(file, skip, SEEK_CUR);
    if (offset==-1) {
      sprintf(msg, "Trying to skip %d bytes", skip);
      perror(msg);
      return(1);
    }
  }


  nread = 0;
  dobreak = 0;
  sumStdDev = 0;
  sumMean = 0;
  uint64_t totalread = 0;

  tA = tim();
  
  while (1) {
    // Fill the buffer
    nread = 0;
    while (nread < bufsize) { 
      // Try and read a little more
	  
      thisread = read(file, buf+nread, bufsize-nread);
      if (thisread==0) {  // EOF
	if (totalread==0) {
	  fprintf(stderr, "No data read from %s\n", filename);
	  dobreak = 1;
	}
	break;
      } else if (nread==-1) {
	perror("Error reading file");
	close(file);
	return(1);
      }
      nread += thisread;
      totalread += thisread;
    }
    if (dobreak) break;
    
    if (nread < bytesperfft) break;
    if (nread%bytesperfft != 0) { /* Need to read multiple of lags */
	  /* Data may be packed with multiple samples/byte or mutiple 
	     bytes/sample */
      int shift;
      shift = nread % bytesperfft;
      printf("DEBUG: Need to shift %d bytes\n", shift);
	  
      off = lseek(file, -shift, SEEK_CUR); 
      if (off==-1) {
	perror(msg);
	return(1);
      }
      nread -= shift;
    }

    t0 = tim();

    /* Copy data into "in" array, fft then accumulate */
    /* No need to convert if already in float format */
    for (i=0; i<nread/bytesperfft; i++) {
      if (!isfloat) {
	if (bits==2) {
	  unpack2bit(&buf[bytesperfft*i], in, npoint*2);
	} else if (bits==8) {
	  unpack8bit((Ipp8s*)&buf[bytesperfft*i], in, npoint*2);
	} else if (bits==16) {
	  unpack16bit((Ipp16s*)&buf[bytesperfft*i], in, npoint*2);
	} else {
	  fprintf(stderr, "Error: Do not support %d bits\n", bits);
	}
      } else {
	if (bits==32) {
	  in = (Ipp32f*)(&buf[bytesperfft*i]);
	} else if (bits==8) {
	  unpackFloat8((Ipp8u*)&buf[bytesperfft*i], in, npoint*2);
	} else {
	  fprintf(stderr, "Error: Do not support floating point %d bits\n", bits);
	}
      }

      status = ippsMeanStdDev_32f(in, npoint*2, &mean, &stdDev, ippAlgHintFast);
      sumStdDev += stdDev;
      sumMean += mean;

      if (iscomplex) {
	status = ippsDFTFwd_CToC_32fc((Ipp32fc*)in, out, specC, workbuf);
	if (status!=ippStsNoErr) {
	  printf("ippsDFTFwd_CToC_32fc failed (%d: %s)!\n", status, ippGetStatusString(status));
	  exit(1);
	}
      } else {
	status = ippsDFTFwd_RToCCS_32f(in, (Ipp32f*)out, spec, workbuf);
	if (status!=ippStsNoErr) {
	  printf("ippsDFTFwd_RToCCS_32f failed (%d: %s)!\n", status, ippGetStatusString(status));
	  exit(1);
	}
      }
      
      makespectrum(spectrum,out,npoint);
      nfft++;
    }
    t1 = tim();
    tt += t1-t0;
  }
  close(file);
  
  tB = tim();

  for (i=0; i<npoint; i++) {
    if (bandwidth!=0) {
      xvals[i] = (float)i/(float)npoint*(float)bandwidth;
    } else {
      xvals[i] = i;
    }
  }

  status = ippsConvert_64f32f(spectrum, plotspec, npoint);
  if (status!=ippStsNoErr) {
    printf("ippsConvert_64f32f failed (%d: %s)!\n", status, ippGetStatusString(status));
    exit(1);
  }

  status = ippsMulC_32f_I(1.0/(nfft*npoint*2*M_PI), plotspec, npoint);
  
  if (specfile==NULL) setSpecname(filename, outfile);


  if (!noplot) {
    if (cpgbeg(0,pgdev,1,1)!=1) {
      fprintf(stderr, "Error calling PGBEGIN");
      return(1);
    }
  }
  doplot(npoint, xvals, plotspec, outfile, noplot);

  printf("Integration took %.1f sec\n", tB-tA);
  printf("Total computational time= %0.1f seconds for %d ffts\n",  tt, nfft);
  printf("StdDev = %.6f, mean=%.6f\n", sumStdDev/nfft, mean/nfft);

  ippsFree(buf);
  if (!isfloat || bits!=32) ippsFree(in);
  ippsFree(out);

  if (!noplot) cpgend();

  return(1);
}

void doplot(int npoint, float *xvals, float *plotspec, char *outfile, int noplot) {
  int i;
  float max, min, delta;
  FILE *os = 0;

  max = plotspec[0];
  min = max;

  if (!noplot) {
    for (i=1; i<npoint; i++) {
      if (plotspec[i]>max) 
	max = plotspec[i];
      else if (plotspec[i]<min) 
	min = plotspec[i];
    }

    delta = (max-min)*0.05;
    min -= delta/2;
    max += delta;

    cpgsci(7);
    cpgbbuf();
    cpgenv(xvals[0], xvals[npoint-1], min, max, 0, 0);
    cpglab("Freq", "Amp", "");
    cpgsci(2);
    cpgline(npoint, xvals, plotspec);

    cpgebuf();
  }

  if (outfile!=NULL) {
    os=fopen(outfile,"w");
    if (os==NULL) {
      fprintf(stderr, "Error opening output file %s\n", outfile);
      outfile = NULL;
    }

    for (i=0;i<npoint;i++) {
      fprintf(os,"%.8f  %f\n", xvals[i], plotspec[i]);
    }
    fclose(os);
  }
}

void setSpecname (char *filename, char *outfile) {
  // Remove leading path if present
  char *cptr = rindex(filename, '/');
  if (cptr==NULL) cptr = filename;
  strcpy(outfile, filename);

  // Remove trailing prefix, if present
  cptr = rindex(outfile, '.');
  if (cptr!=NULL) *cptr = 0;
  strcat(outfile, ".spec");
}



void makespectrum(Ipp64f *spectrum, Ipp32fc *out, int n) {
  IppStatus status;

  status = ippsPowerSpectr_32fc(out, temp32f, n);
  if (status!=ippStsNoErr) {
    printf("ippsPowerSpectr_32fc failed!\n");
    exit(1);
  }
  status = ippsConvert_32f64f(temp32f, temp64f, n);
  if (status!=ippStsNoErr) {
    printf("ippsConvert_32f64f failed!\n");
    exit(1);
  }
  
  status = ippsAdd_64f_I(temp64f, spectrum, n);
  if (status!=ippStsNoErr) {
    printf("ippsAdd_64f failed!\n");
    exit(1);
  }
}
