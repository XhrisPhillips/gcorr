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
void doplot(int npoint, float *xvals, Ipp64f **spectrum, int nchan, char *outfile, int noplot);

Ipp32f *temp32f;
Ipp64f *temp64f;
Ipp32f *plotspec;

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
  int i, j, file, s, nread, opt, tmp, thisread, dobreak;
  int nfft, bytesperfft;
  char msg[MAXSTR+1], *filename, *outfile;
  Ipp8u  *buf;
  off_t off, offset;
  Ipp32fc *out;
  Ipp64f **spectrum, sumStdDev, sumMean;
  Ipp32f **in, *xvals, mean, stdDev;
  double t0, t1, tt, tA, tB;
  IppStatus status;

  IppsDFTSpec_R_32f *spec;
  IppsDFTSpec_C_32fc *specC;

  int bits=0;                   /* Number of bits/sample */
  int channels=1;               /* Number of IF channels */
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
    {"channels", 1, 0, 'C'},
    {"nchan", 1, 0, 'C'},
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
      CASEINT('C', channels);
  
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

  int nchan=channels;

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
  bufsize *= nchan;
  IPPMALLOC(buf, 8u, bufsize);

 in = malloc(nchan*sizeof(Ipp32f*));
  if (!isfloat || bits!=32) {
    for (i=0;i<nchan;i++) {IPPMALLOC(in[i], 32f, npoint*2)};
  }
  IPPMALLOC(out, 32fc, npoint+1);
  spectrum = malloc(nchan*sizeof(Ipp64f*));
  for (i=0;i<nchan;i++) {
    IPPMALLOC(spectrum[i], 64f, npoint);
    status = ippsZero_64f(spectrum[i], npoint);
    if (status!=ippStsNoErr) {
      printf("ippsZero_64f failed (%d: %s)!\n", status, ippGetStatusString(status));
      exit(1);
    }
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
    
    if (nread < bytesperfft*nchan) break;
    if (nread%(bytesperfft*nchan) != 0) { /* Need to read multiple of lags */
	  /* Data may be packed with multiple samples/byte or mutiple 
	     bytes/sample */
      int shift;
      shift = nread % (bytesperfft*nchan);
	  
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
    for (i=0; i<nread/(bytesperfft*nchan); i++) {
      if (!isfloat) {
	if (bits==2) {
	  unpack2bit(&buf[bytesperfft*i], in, nchan, npoint*2);
	} else if (bits==8) {
	  unpack8bit((Ipp8s*)&buf[bytesperfft*i], in, nchan, npoint*2);
	} else if (bits==16) {
	  unpack16bit((Ipp16s*)&buf[bytesperfft*i], in, nchan, npoint*2);
	} else {
	  fprintf(stderr, "Error: Do not support %d bits\n", bits);
	}
      } else {
	if (bits==32) {
	  in[0] = (Ipp32f*)(&buf[bytesperfft*i]);
	} else if (bits==8) {
	  unpackFloat8((Ipp8u*)&buf[bytesperfft*i], in, nchan, npoint*2);
	} else {
	  fprintf(stderr, "Error: Do not support floating point %d bits\n", bits);
	}
      }
      // Loop over channel
      for (j=0; j<nchan; j++) {
	status = ippsMeanStdDev_32f(in[j], npoint*2, &mean, &stdDev, ippAlgHintFast);
	sumStdDev += stdDev;
	sumMean += mean;

	if (iscomplex) {
	  status = ippsDFTFwd_CToC_32fc((Ipp32fc*)in[j], out, specC, workbuf);
	  if (status!=ippStsNoErr) {
	    printf("ippsDFTFwd_CToC_32fc failed (%d: %s)!\n", status, ippGetStatusString(status));
	    exit(1);
	  }
	} else {
	  status = ippsDFTFwd_RToCCS_32f(in[j], (Ipp32f*)out, spec, workbuf);
	  if (status!=ippStsNoErr) {
	    printf("ippsDFTFwd_RToCCS_32f failed (%d: %s)!\n", status, ippGetStatusString(status));
	    exit(1);
	  }
	}
      
	makespectrum(spectrum[j],out,npoint);
      }
      nfft++;
    }
    t1 = tim();
    tt += t1-t0;
  }
  close(file);
  
  tB = tim();

  // Normalise
  for (i=0; i<nchan; i++) {
    status = ippsMulC_64f_I(1.0/(nfft*npoint*2*M_PI), spectrum[i], npoint);
  }

  for (i=0; i<npoint; i++) {
    if (bandwidth!=0) {
      xvals[i] = (float)i/(float)npoint*(float)bandwidth;
    } else {
      xvals[i] = i;
    }
  }


  
  if (specfile==NULL) setSpecname(filename, outfile);


  if (!noplot) {
    if (cpgbeg(0,pgdev,1,nchan)!=1) {
      fprintf(stderr, "Error calling PGBEGIN");
      return(1);
    }
  }
  doplot(npoint, xvals, spectrum, nchan, outfile, noplot);

  printf("Integration took %.1f sec\n", tB-tA);
  printf("Total computational time= %0.1f seconds for %d ffts\n",  tt, nfft);
  printf("StdDev = %.6f, mean=%.6f\n", sumStdDev/nfft, mean/nfft);

  ippsFree(buf);
  if (!isfloat || bits!=32) {
    for (int i=0; i<nchan;i++) {
      ippsFree(in[i]);
      ippsFree(spectrum[i]);
    };
    free(in);
    free(spectrum);
  }
  ippsFree(out);
  ippsFree(xvals);
  ippsFree(plotspec);
  ippsFree(temp64f);
  ippsFree(temp32f);
  
  if (!noplot) cpgend();

  return(1);
}

void doplot(int npoint, float *xvals, Ipp64f **spectrum, int nchan, char *outfile, int noplot) {
  int i, j;
  float max, min, delta;
  FILE *os = 0;
  IppStatus status;

  max = spectrum[0][0];
  min = max;

  if (!noplot) {
    for (i=0; i<nchan; i++) {
      for (j=0; j<npoint; j++) {
      if (spectrum[i][j]>max) 
	max = spectrum[i][j];
      else if (spectrum[i][j]<min) 
	min = spectrum[i][j];
      }
    }
    delta = (max-min)*0.05;
    min -= delta/2;
    max += delta;

    for (i=0; i<nchan; i++) {
      cpgsci(7);
      cpgbbuf();
      cpgenv(xvals[0], xvals[npoint-1], min, max, 0, 0);
      cpglab("Freq", "Amp", "");
      cpgsci(2);

      status = ippsConvert_64f32f(spectrum[i], plotspec, npoint);
      if (status!=ippStsNoErr) {
	printf("ippsConvert_64f32f failed (%d: %s)!\n", status, ippGetStatusString(status));
      exit(1);
      }
      cpgline(npoint, xvals, plotspec);
      cpgebuf();
    }
  }

  if (outfile!=NULL) {
    os=fopen(outfile,"w");
    if (os==NULL) {
      fprintf(stderr, "Error opening output file %s\n", outfile);
      outfile = NULL;
    }

    for (i=0;i<npoint;i++) {
      fprintf(os, "%.8f ", xvals[i]);
      for (j=0;j<nchan;j++) fprintf(os, " %f", plotspec[i]);
      fprintf(os, "\n");
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
