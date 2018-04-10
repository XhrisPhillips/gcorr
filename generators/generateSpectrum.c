#ifdef __APPLE__

#define OSX

#define OPENREADOPTIONS O_RDONLY
#define OPENWRITEOPTIONS O_WRONLY|O_CREAT|O_TRUNC

#else

#define LINUX
#define _LARGEFILE_SOURCE 
#define _LARGEFILE64_SOURCE
#define _FILE_OFFSET_BITS 64

#define OPENREADOPTIONS O_RDONLY|O_LARGEFILE
#define OPENWRITEOPTIONS O_WRONLY|O_CREAT|O_TRUNC|O_LARGEFILE
#include <sys/stat.h>    /* for S_* modes on some flavors */

#endif

#include <stdlib.h>
#include <stdio.h>
#include <stdint.h>
#include <unistd.h>
#include <string.h>
#include <errno.h>
#include <math.h>
#include <fcntl.h>
#include <sys/time.h>
#include <time.h>
#include <getopt.h>

#include <ippcore.h>
#include <ipps.h>

#include "bitconversion.h"

#define DEFAULTTAP 128

IppsRandGaussState_32f *pRandGaussState, *pRandGaussState2;
Ipp32f *scratch;
Ipp32f phase, phase2;
Ipp32f *dly;
Ipp8u *buf;
IppsFIRSpec_32f *pSpec;
IppsFIRSpec_32fc *pcSpec;

#define SEED 48573

double currentmjd();
void mjd2cal(double mjd, int *day, int *month, int *year, double *ut);

void dayno2cal (int dayno, int year, int *day, int *month);
double cal2mjd(int day, int month, int year);
double tm2mjd(struct tm date);

void generateData(Ipp32f **data, int nchan, int nsamp, int iscomplex, int nobandpass,
		  int noise, int bandwidth, float tone, float amp,
		  float tone2, float amp2, float *mean, float *stdDev);

#define MAXSTR        255
#define BUFSIZE       128  // MB
#define MAXPOS          3
#define SMALLPOS        2
#define SMALLNEG        1
#define MAXNEG          0

typedef enum {FLOAT, FLOAT8, INT} data_type;

#define IPPMALLOC(var,type,n)	                               \
  var = ippsMalloc_ ## type(n);                                \
  if (var==NULL) {                                             \
    fprintf(stderr, "Error allocating memory for %s\n", #var); \
    exit(EXIT_FAILURE);                                        \
  }                                                            \

int main (int argc, char * const argv[]) {
  char *filename, msg[MAXSTR];
  int i, status, outfile, opt, tmp, nbuf, outsize;
  float **data, ftmp, *stdDev, *mean;
  Ipp8u *outdata;
  Ipp64f *taps64;
  Ipp32f *taps;
  Ipp32fc *tapsC;
  ssize_t nr;

  int memsize = BUFSIZE;
  int nbits = 0;
  
  data_type outData = INT;
  int bandwidth = 64;
  int channels = 1;
  int ntap = DEFAULTTAP;
  int iscomplex = 0;
  int nobandpass = 0;
  int noise = 0;
  int year = -1;
  int month = -1;
  int day = -1;
  int dayno = -1;
  double mjd = -1;
  float tone = 10;    // MHz
  float tone2 = 0.0;  // MHz
  float amp = 0.1;
  float amp2 = 0.0;
  float duration = 0; // Seconds
  char *timestr = NULL;

  struct option options[] = {
    {"bandwidth", 1, 0, 'w'},
    {"channels", 1, 0, 'C'},
    {"day", 1, 0, 'd'},
    {"dayno", 1, 0, 'D'},
    {"month", 1, 0, 'm'},
    {"mjd", 1, 0, 'M'},
    {"year", 1, 0, 'y'},
    {"time", 1, 0, 't'},
    {"duration", 1, 0, 'l'},
    {"amp", 1, 0, 'a'},
    {"amp2", 1, 0, 'A'},
    {"tone", 1, 0, 'T'},
    {"tone2", 1, 0, '2'},
    {"ntap", 1, 0, 'x'},
    {"bufsize", 1, 0, 'B'},
    {"nbits", 1, 0, 'b'},
    {"bits", 1, 0, 'b'},
    {"complex", 0, 0, 'c'},
    {"nobandpass", 0, 0, 'N'},
    {"noise", 0, 0, 'n'},
    {"float", 0, 0, 'f'},
    {"help", 0, 0, 'h'},
    {0, 0, 0, 0}
  };

  srand(time(NULL));
  
  /* Read command line options */
  while (1) {
    opt = getopt_long_only(argc, argv, "w:B:xb:d:m:M:y:t:n:c:T:hF:", options, NULL);
    if (opt==EOF) break;
    
#define CASEINT(ch,var)                                     \
  case ch:						    \
    status = sscanf(optarg, "%d", &tmp);		    \
    if (status!=1)					    \
      fprintf(stderr, "Bad %s option %s\n", #var, optarg);  \
    else                                                    \
      var = tmp;                                            \
    break

#define CASEFLOAT(ch,var)                                   \
  case ch:						    \
    status = sscanf(optarg, "%f", &ftmp);		    \
    if (status!=1)					    \
      fprintf(stderr, "Bad %s option %s\n", #var, optarg);  \
    else                                                    \
      var = ftmp;                                           \
    break
    
    switch (opt) {

      CASEINT('w', bandwidth);
      CASEINT('d', day);
      CASEINT('D', dayno);
      CASEINT('m', month);
      CASEINT('M', mjd);
      CASEINT('y', year);
      CASEINT('b', nbits);
      CASEINT('x', ntap);
      CASEINT('C', channels);
      CASEFLOAT('B', memsize);
      CASEFLOAT('l', duration);
      CASEFLOAT('T', tone);
      CASEFLOAT('2', tone2);
      CASEFLOAT('a', amp);
      CASEFLOAT('A', amp2);

      case 't':
 	timestr = strdup(optarg);
	break;
	
      case 'c':
	iscomplex = 1;
	break;

      case 'N':
	nobandpass = 1;
	break;

      case 'n':
	noise = 1;
	break;

      case 'f':
	outData = FLOAT;
	break;

      case 'h':
	printf("Usage: generateSpectrum [options]\n");
	printf("  -bandwidth <BANWIDTH>     Channel bandwidth in MHz (64)\n");
	printf("  -N/-nbits <N>             Number of bits/sample (default 2)\n");
	printf("  -C/-channels <N>          Number of if channels (default 1)\n");
	printf("  -t/-duration <DURATION>   Length of output, in seconds\n");
	printf("  -T/-tone <TONE>           Frequency (MHz) of tone to insert\n");
	printf("  -ntap <TAPS>              Number of taps for FIR filter to create band shape\n");
	printf("  -day <DAY>                Day of month of start time (now)\n");
	printf("  -month <MONTH>            Month of start time (now)\n");
	printf("  -dayno <DAYNO>            Day of year of start time (now)\n");
	printf("  -year <YEAR>              Year of start time (now)\n");
	printf("  -time <HH:MM:SS>          Year of start time (now)\n");
	printf("  -mjd <MJD>                MJD of start time\n");
	return(1);
	break;
      
    case '?':
    default:
      break;
    }
  }

  int nchan = channels;

  if (argc==optind) {
    filename = strdup("Test.vdf");
  } else {
    filename = strdup(argv[optind]);
  }

  // Set time
  double thismjd = currentmjd();
  int thisday, thismonth, thisyear;
  double ut;
  mjd2cal(thismjd, &thisday, &thismonth, &thisyear, &ut);

  if (year==-1) year = thisyear;
  if (day==-1) day = thisday;
  if (month==-1) month = thismonth;
  if (dayno!=-1) dayno2cal(dayno, year, &day, &month);

  if (timestr!=NULL) {
    int hour, min, sec;
    status = sscanf(timestr, "%2d:%2d:%2d", &hour, &min, &sec);
    if (status==0) {
      fprintf(stderr, "Warning: Could not parse %s (%d)\n", timestr, status);
    } else {
      ut = ((sec/60.0+min)/60.0+hour)/24.0;
    }
  }

  mjd = cal2mjd(day, month, year)+ut;

  memsize *= 1024*1024;

  if (outData==FLOAT) {
    if (nbits==0)
      nbits=32;
    else if (nbits==8) {
      outData=FLOAT8;
    } else {
      fprintf(stderr, "Error: Do not support %d bit floats\n", nbits);
    }
  } else if (nbits==0) {
    nbits=2;
    printf("BITS=%d\n", nbits);
  }
  
  int cfact = 1;
  if (iscomplex) cfact = 2;

  // Frame size and number of sampler/frame
  int completesample = nbits*cfact*nchan;
  int sampleperbyte = 8/completesample;  // Output byte
  if (sampleperbyte==0) sampleperbyte=1;
  if (outData==FLOAT) sampleperbyte = 1; // Dummy value

  uint64_t samplerate = bandwidth*1e6*2/cfact;

  // memsize needs to be integral number of complete samples
  // bufsamples is number of time samples/buffer
  int bufsamples = memsize/(sizeof(float)*cfact*nchan);
  bufsamples = (bufsamples/sampleperbyte)*sampleperbyte;

  if (duration==0) { // Just create BUFSIZE bytes
    nbuf = 1;
  } else {
    // Make sure buffer divides nicely into expected duration
    uint64_t totalsamples = samplerate * duration;
    while (totalsamples % bufsamples) {
      bufsamples -= sampleperbyte;
    }
    if (bufsamples==0) {
      fprintf(stderr, "Could not figure out bufsize\n");
      exit(1);
    }
    nbuf = totalsamples/bufsamples;
  }

  if (tone2>0 && amp2==0) amp2=amp;
  
  memsize = bufsamples*sizeof(float)*cfact;

  data = malloc(nchan*sizeof(float*));
  for (int i=0; i<nchan; i++) {
    IPPMALLOC(data[i], 32f, bufsamples*cfact);
  }

  if (outData==FLOAT) { // Just write floats directly
    outdata = (Ipp8u*)data;
    outsize = memsize;
  } else {
    outsize = ((uint64_t)bufsamples*completesample)/8; // bytes
    IPPMALLOC(outdata, 8u, outsize);
  }
  
  // Initialise random number generator
  int pRandGaussStateSize;
  ippsRandGaussGetSize_32f(&pRandGaussStateSize);
  pRandGaussState = (IppsRandGaussState_32f *)ippsMalloc_8u(pRandGaussStateSize);
  ippsRandGaussInit_32f(pRandGaussState, 0.0, 1.0, time(NULL));  // Mean 0.0, RMS=1
  if (noise) {
    pRandGaussState2 = (IppsRandGaussState_32f *)ippsMalloc_8u(pRandGaussStateSize);
    ippsRandGaussInit_32f(pRandGaussState2, 0.0, amp, SEED);  // Mean 0.0, RMS=amp
  }

  IPPMALLOC(scratch, 32f, bufsamples*cfact);
  phase = phase2 = 0;
  int specSize;
  int bufsize, bufsize2;

  if (!nobandpass) {
    // Initialise FIR filter
    IPPMALLOC(taps64,64f, ntap);
    IPPMALLOC(taps, 32f, ntap);
    IPPMALLOC(tapsC, 32fc, ntap);
    IPPMALLOC(dly, 32f, (ntap-1)*cfact); 
    ippsZero_32f(dly, (ntap-1)*cfact);
    
    status = ippsFIRGenGetBufferSize(ntap, &bufsize);
    if (status != ippStsNoErr) {
      fprintf(stderr, "Error calling ippsFIRGenGetBufferSize (%s)\n", ippGetStatusString(status));
      exit(1);
    }
    IPPMALLOC(buf, 8u, bufsize);
    status = ippsFIRGenBandpass_64f(0.02, 0.48, taps64, ntap, ippWinHamming, ippTrue, buf);
    if (status != ippStsNoErr) {
      fprintf(stderr, "Error generating tap coefficients (%s)\n", ippGetStatusString(status));
      exit(1);
    }
    ippsConvert_64f32f(taps64, taps, ntap);

    // Real FIR filter
    status = ippsFIRSRGetSize (ntap, ipp32f, &specSize, &bufsize);
    if (status != ippStsNoErr) {
      fprintf(stderr, "Error Getting filter initialisation sizes (%s)\n", ippGetStatusString(status));
      exit(1);
    }
    pSpec = (IppsFIRSpec_32f*)ippsMalloc_8u(specSize);
    status = ippsFIRSRInit_32f(taps, ntap, ippAlgAuto, pSpec);
    if (status != ippStsNoErr) {
      fprintf(stderr, "Error Initialising filter (%s)\n", ippGetStatusString(status));
      exit(1);
    }

    status = ippsFIRGenHighpass_64f(0.02, taps64, ntap, ippWinHamming, ippTrue, buf);
    if (status != ippStsNoErr) {
      fprintf(stderr, "Error generating tap coefficients (%s)\n", ippGetStatusString(status));
      exit(1);
    }
    for (i=0; i<ntap; i++) {
      tapsC[i].re = taps64[i];
      tapsC[i].im = 0;
    }
    ippsFree(taps64);
    ippsFree(buf);
  
    // Complex FIR Filter
    status = ippsFIRSRGetSize (ntap, ipp32fc, &specSize, &bufsize2);
    if (status != ippStsNoErr) {
      fprintf(stderr, "Error Getting filter initialisation sizes (%s)\n", ippGetStatusString(status));
      exit(1);
    }
    pcSpec = (IppsFIRSpec_32fc*)ippsMalloc_8u(specSize);
    status = ippsFIRSRInit_32fc(tapsC, ntap, ippAlgAuto, pcSpec);
    if (status != ippStsNoErr) {
      fprintf(stderr, "Error Initialising filter (%s)\n", ippGetStatusString(status));
      exit(1);
    }

    if (bufsize2 > bufsize) bufsize = bufsize2;
    IPPMALLOC(buf, 8u, bufsize);
  }

  mean = malloc(sizeof(float)*nchan);
  stdDev = malloc(sizeof(float)*nchan);
  
  outfile = open(filename, OPENWRITEOPTIONS, S_IRWXU|S_IRWXG|S_IRWXO); 
  if (outfile==-1) {
    sprintf(msg, "Failed to open output (%s)", filename);
    perror(msg);
    exit(1);
  }
  printf("Writing %s\n", filename);

  while (nbuf>0) {
    
    generateData(data, nchan, bufsamples, iscomplex, nobandpass, noise, bandwidth, tone, amp, tone2, amp2, mean, stdDev);

    for (int i=1;i<nchan;i++) {
      mean[0] += mean[i];
      stdDev[0] += stdDev[i];
    }
    mean[0] /= nchan;
    stdDev[0] /= nchan;

    if (outData==INT) {
      printf("BITS=%d\n", nbits);
      if (nbits==2) {
	status = packBit2(data, outdata, nchan, mean[0], stdDev[0], bufsamples*cfact);
	if (status) exit(1);
      } else if (nbits==8) {
	status = packBit8(data, (Ipp8s*)outdata, nchan, mean[0], 1.0, 10, bufsamples*cfact);
      } else if (nbits==16) {
	status = packBit16(data, (Ipp16s*)outdata, nchan, mean[0], 1.0, 10, bufsamples*cfact);
      } else {
	printf("Unsupported number of bits\n");
	exit(1);
      }
    } else if (outData==FLOAT8) {
      status = packFloat8(data, outdata, nchan, mean[0], 1.0, 10, bufsamples*cfact);
    } else { // Float
      status = ippsMulC_32f_I(10.0, data[0], bufsamples*cfact);
    }

    nr = write(outfile, outdata, outsize); 
    if (nr == -1) {
      sprintf(msg, "Writing to %s:", filename);
      perror(msg);
      exit(1);
    } else if (nr != outsize) {
      printf("Error: Partial write to %s\n", filename);
      exit(1);
    }
    nbuf--;
  }

  close(outfile);

  printf("CLosed outfile\n");

  free(filename);
  if (timestr!=NULL) free(timestr);
  for (i=0; i<nchan; i++) ippsFree(data[i]);
  free(data);
  if (outData!=FLOAT) ippsFree(outdata);
  ippsFree(pRandGaussState);
  if (noise) ippsFree(pRandGaussState2);
  ippsFree(scratch);
  if (!nobandpass) {
    ippsFree(taps);
    ippsFree(tapsC);
    ippsFree(dly);
    ippsFree(buf);
    ippsFree(pSpec);
    ippsFree(pcSpec);
  }
  return(0);
}
  
double currentmjd () {
  struct tm *tim;
  struct timeval currenttime;
  setenv("TZ", "", 1); /* Force mktime to return gmt not local time */
  tzset();
  gettimeofday(&currenttime, NULL);
  tim = localtime(&currenttime.tv_sec);
  return(tm2mjd(*tim)+(currenttime.tv_usec/1.0e6)/(24.0*60.0*60.0));
}

void mjd2cal(double mjd, int *day, int *month, int *year, double *ut) {
  int jd, temp1, temp2;

  *ut = fmod(mjd,1.0);

  if (*ut<0.0) {
    *ut += 1.0;
    mjd -= 1;
  }

  jd = (int)floor(mjd + 2400001);

  // Do some rather cryptic calculations
  
  temp1 = 4*(jd+((6*(((4*jd-17918)/146097)))/4+1)/2-37);
  temp2 = 10*(((temp1-237)%1461)/4)+5;

  *year = temp1/1461-4712;
  *month =((temp2/306+2)%12)+1;
  *day = (temp2%306)/10+1;
}


int leap (int year) {
  return (((!(year%4))&&(year%100))||(!(year%400)));
}

static int days[] = {31,28,31,30,31,30,31,31,30,31,30,31};

void dayno2cal (int dayno, int year, int *day, int *month) {
  int end;

  if (leap(year)) {
    days[1] = 29;
  } else {
    days[1] = 28;
  }

  *month = 0;
  end = days[*month];
  while (dayno>end) {
    (*month)++;
    end+= days[*month];
  }
  end -= days[*month];
  *day = dayno - end;
  (*month)++;

  return;
}

double cal2mjd(int day, int month, int year) {
  int m, y, c, x1, x2, x3;

  if (month <= 2) {
    m = month+9;
    y = year-1;
  } else {
    m = month-3;
    y = year;
  }

  c = y/100;
  y = y-c*100;

  x1 = 146097*c/4;
  x2 = 1461*y/4;
  x3 = (153*m+2)/5;

  return(x1+x2+x3+day-678882);
}

double tm2mjd(struct tm date) {
  int y, c;
  double dayfrac;

  if (date.tm_mon < 2) {
    y = date.tm_mon+1900-1;
  } else {
    y = date.tm_year+1900;
  }

  c = y/100;
  y = y-c*100;

  dayfrac = ((date.tm_hour*60.0+date.tm_min)*60.0+date.tm_sec)/(60.0*60.0*24.0);

  return(cal2mjd(date.tm_mday, date.tm_mon+1, date.tm_year+1900)+dayfrac);
}

void generateData(Ipp32f **data, int nchan, int nsamp, int iscomplex, int nobandpass,
		  int noise, int bandwidth, float tone, float amp, float tone2,
		  float amp2, float *mean, float *stdDev) {
  int cfact;
  IppStatus status;
  if (iscomplex)
    cfact = 2;
  else
    cfact = 1;
  

  status = ippStsNoErr;

  for (int i=0; i<nchan; i++) {
    mean[i] = 0;
    stdDev[i] = 0;

    status = ippsRandGauss_32f(data[i], nsamp*cfact, pRandGaussState);
    if (status!=ippStsNoErr) {
      fprintf(stderr, "Error generating Gaussian noise (%s)\n", ippGetStatusString(status));
      exit(1);
    }

    if (amp>0.0) {
      if (noise) {
	status = ippsRandGauss_32f(scratch, nsamp*cfact, pRandGaussState2);
	if (status==ippStsNoErr) status = ippsAdd_32f_I(scratch, data[i], nsamp*cfact);
	if (status!=ippStsNoErr) {
	  fprintf(stderr, "Error generating Gaussian noise2 (%s)\n", ippGetStatusString(status));
	  exit(1);
	}
      } else {
	if (iscomplex) {
	  status = ippsTone_32fc((Ipp32fc*)scratch, nsamp, sqrt(amp), tone/bandwidth, &phase, ippAlgHintFast);
	} else {
	  status = ippsTone_32f(scratch, nsamp, sqrt(amp), tone/(bandwidth*2), &phase, ippAlgHintFast);
	}
	if (status==ippStsNoErr) status = ippsAdd_32f_I(scratch, data[i], nsamp*cfact);
	if (status!=ippStsNoErr) {
	  fprintf(stderr, "Error generating tone (%s)\n", ippGetStatusString(status));
	  exit(1);
	}
      }

      if (amp2>0 && tone2!=0.0) {
	if (iscomplex) {
	  status = ippsTone_32fc((Ipp32fc*)scratch, nsamp, sqrt(amp2), tone2/bandwidth, &phase2, ippAlgHintFast);
	} else {
	  status = ippsTone_32f(scratch, nsamp, sqrt(amp2), tone2/(bandwidth*2), &phase2, ippAlgHintFast);
	}
	if (status==ippStsNoErr) status = ippsAdd_32f_I(scratch, data[i], nsamp*cfact);
	if (status!=ippStsNoErr) {
	  fprintf(stderr, "Error generating second tone (%s)\n", ippGetStatusString(status));
	  exit(1);
	}
      }
    }

    if (!nobandpass) {
      if (iscomplex) {
	ippsFIRSR_32fc((Ipp32fc*)data[i], (Ipp32fc*)data[i], nsamp, pcSpec, (Ipp32fc*)dly, (Ipp32fc*)dly,  buf);
      } else {
	ippsFIRSR_32f(data[i], data[i], nsamp, pSpec, dly, dly, buf);
      }
    }
    status = ippsMeanStdDev_32f(data[i], nsamp*cfact, mean, stdDev, ippAlgHintFast);
  }
}
