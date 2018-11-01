#include <iostream>
#include <fstream>
#include <sstream>
#include <iomanip>
#include <string>
#include <vector>
#include <cstdlib>
#include <cstring>
#include <stdint.h>
#include <complex>
#include <argp.h>
#include <cuda.h>
#include <curand.h>
#include <cuComplex.h>
#include <cufft.h>
#include <complex>
#include <cmath>

#include "common.h"

#define NTHREADS 256

using std::string;
using std::cout;
using std::cerr;
using std::endl;
using std::vector;
using std::conj;
using std::abs;
using std::complex;

const char *argp_program_version = "validate_xcorr 1.0";
static char doc[] = "validate_xcorr -- Compare various GPU xcorr engines to C version";
static char args_doc[] = "configuration_file";

#define BUFSIZE 256

/* Our command line options */
static struct argp_option options[] = {
  { "gpu", 'g', "GPU", 0, "Select specific GPU"},
  { 0 }
};

struct arguments {
  int gpu_select;
  char configfile[BUFSIZE];
};


#define CURAND_CALL(x) {__curand_call((x), __FILE__, __LINE__); }
inline void __curand_call(curandStatus_t code, const char *file, int line) {
  if (code != CURAND_STATUS_SUCCESS) {  
    fprintf(stderr, "Curand error (%d) at %s:%d\n", code, file, line);  
    exit(EXIT_FAILURE); 
  }
}



/* The option parser */
static error_t parse_opt(int key, char *arg, struct argp_state *state) {
  struct arguments *arguments = (struct arguments *)state->input;

  switch (key) {
  case 'g':
    arguments->gpu_select = atoi(arg);
    break;
  case ARGP_KEY_END:
    if (strlen(arguments->configfile) == 0) {
      argp_usage(state);
      exit(0);
    }
    break;
  default:
    // Assume this is the config file.
    if (arg != NULL) {
       if (strlen(arg) > 0) {
       	  strncpy(arguments->configfile, arg, BUFSIZE);
       }
    }
  }
  return 0;
}

/* The argp parser */
static struct argp argp = { options, parse_opt, args_doc, doc };


#include "gxkernel.h"

void allocDataGPU(cuComplex **channelisedData, cuComplex **baselineData, 
		  int numantenna, int subintsamples,  int nchan, int nPol, int parallelAccum )
//, int nbit, int numffts)
{
  unsigned long long GPUalloc = 0;

  // FFT output
  gpuErrchk(cudaMalloc(channelisedData, numantenna*nPol*subintsamples*sizeof(cuComplex)));
  GPUalloc += numantenna*nPol*subintsamples*sizeof(cuComplex);

  // Baseline visibilities
  int nbaseline = numantenna*(numantenna-1)/2;  // Dont include autocorrelations
  int polProds =1;
  if (nPol>1) polProds = 4;
  
  cout << "Alloc " << nchan*parallelAccum << " complex output values per baseline" << endl;
  gpuErrchk(cudaMalloc(baselineData, nbaseline*polProds*nchan*parallelAccum*sizeof(cuComplex)));
  GPUalloc += nbaseline*polProds*nchan*parallelAccum*sizeof(cuComplex);

  cout << "Allocated " << GPUalloc/1e6 << " Mb on GPU" << endl;
}

void allocHostData(complex<float> **data, complex<float> **baselineData, complex<float> **gpuBaseline,
		   int numantenna, int subintsamples,  int nchan, int nPol, int parallelAccum) {

  *data = new complex<float>[numantenna*nPol*subintsamples];
  // Check for success

  int nbaseline = numantenna*(numantenna-1)/2;
  int polProds =1;
  if (nPol>1) polProds = 4;
  
  *baselineData = new complex<float>[nbaseline*polProds*nchan*parallelAccum];
  *gpuBaseline = new complex<float>[nbaseline*polProds*nchan*parallelAccum];
  
}

void cpuXcorr(complex<float> *inputData, complex<float> *outputData, int numantennas, int nPol,
	      int numchannels, int numffts, bool isComplex) {
  int b; // Baseline #
  int polProd; // 1 or 4
  int cfactor;  // Input real data will only populate half FFT output with useful data

  if (nPol==1)
    polProd = 1;
  else
    polProd = 4;
  if (isComplex)
    cfactor = 1;
  else
    cfactor = 2;

  int nbaseline = numantennas*(numantennas-1)/2;
  for (b=0; b<nbaseline; b++) {
    for (int p=0; p<polProd; p++) {
      for (int c=0; c<numchannels; c++) {
	outputData[(b*polProd+p)*numchannels+c] = 0;
      }
    }
  }

  int antBlock = numffts*nPol*numchannels*cfactor;  // Total number of samples for an antenna
  int polBlock = numchannels*cfactor*numffts;
  

  //cout << "**AntBlock= " << antBlock << endl;
    
  b = 0;
  for (int a1=0; a1<numantennas-1; a1++) {
    for (int a2=a1+1; a2<numantennas; a2++) {
      for (int f=0; f<numffts; f++) {
	if (polProd==1) {
	  for (int c=0; c<numchannels; c++) {
	    outputData[b*numchannels + c] += inputData[a1*antBlock + f*numchannels*cfactor + c]
	      * conj(inputData[a2*antBlock + f*numchannels*cfactor + c]);
	  }
	} else { // Dual pol, 4 products
	  for (int c=0; c<numchannels; c++) {
	    if (c==0 && a1==0 && a2==1) {
	      //cout << "*** " << antBlock << "* " << a1*antBlock + f*numchannels*cfactor*nPol + c << "  " << a2*antBlock + f*numchannels*cfactor*nPol + c << endl;
	    }
	    outputData[b*4*numchannels + c]     += inputData[a1*antBlock + f*numchannels*cfactor + c] * conj(inputData[a2*antBlock + f*numchannels*cfactor + c]);
	    outputData[(b*4+1)*numchannels + c] += inputData[a1*antBlock + f*numchannels*cfactor + c] * conj(inputData[a2*antBlock + f*numchannels*cfactor + polBlock + c]);
	    outputData[(b*4+2)*numchannels + c] += inputData[a1*antBlock + f*numchannels*cfactor + polBlock + c] * conj(inputData[a2*antBlock + f*numchannels*cfactor + c]);
	    outputData[(b*4+3)*numchannels + c] += inputData[a1*antBlock + f*numchannels*cfactor + polBlock + c] * conj(inputData[a2*antBlock + f*numchannels*cfactor + polBlock + c]);
	  }
	}
      }
      b++;
    }
  }

  for (b=0; b<nbaseline; b++) {
    for (int p=0; p<polProd; p++) {
      for (int c=0; c<numchannels; c++) {
	outputData[(b*polProd+p)*numchannels+c] /= numffts;
      }
    }
  }
}

void printBaseline(complex<float> *baselineData, int numantennas, int nPol, int numchannels) {
  int b; // Baseline #
  int polProd; // 1 or 4

  if (nPol==1)
    polProd = 1;
  else
    polProd = 4;

  int nbaseline = numantennas*(numantennas-1)/2;
  for (b=0; b<nbaseline; b++) {
    for (int p=0; p<polProd; p++) {
      cout<<b+1<<" "<<p+1<<": ";
      for (int c=0; c<10; c++) {
	cout << " " << baselineData[(b*polProd+p)*numchannels+c];
      }
      cout<<endl;
    }
  }
}

void compareBaseline(complex<float> *data1, complex<float> *data2, int numantennas, int nPol, int numchannels) {
  int polProd; // 1 or 4
  float diff, maxDiff = 0;
  double sum = 0;
  
  
  if (nPol==1)
    polProd = 1;
  else
    polProd = 4;

  int nbaseline = numantennas*(numantennas-1)/2;

  int nval = nbaseline * polProd * numchannels;

  for (int i=0; i<nval; i++) {
    diff = abs(data1[i] - data2[i]);
    if (diff>maxDiff) maxDiff = diff;
    sum += diff;
  }
  printf("Maximum difference = %.4f\n", maxDiff);
  printf("Average difference = %.4f\n", sum/nval);

}

inline float carg(const cuComplex& z) {return atan2(cuCimagf(z), cuCrealf(z));} // polar angle

int main(int argc, char *argv[])
{
  // variables for the test
  char *configfile;
  int cfactor;
  int nPol, polProd;
  int samplegranularity; /**< how many time samples per byte.  If >1, then our fractional sample error can be >0.5 samples */
  double ** delays; /**< delay polynomial for each antenna.  delay is in seconds, time is in units of FFT duration */
  double * antfileoffsets; /**< offset from each the nominal start time of the integration for each antenna data file.  
                                In units of seconds. */
  int numchannels, numantennas, nbaseline, numffts, nbit;
  double lo, bandwidth, sampletime, subinttime;
  bool iscomplex;
  vector<string> antennas, antFiles;
  curandGenerator_t gen;

  complex <float> *channelisedData_h, *baselineData_h, *gpubaselineData_h;
  cuComplex *channelisedData, *baselineData;
  
  // Read in the command line arguments.
  struct arguments arguments;
  arguments.gpu_select = -1;
  arguments.configfile[0] = 0;
  argp_parse(&argp, argc, argv, 0, 0, &arguments);

  if (strlen(arguments.configfile) > 0) {
    configfile = arguments.configfile;
  }
  printf("reading configuration file %s\n", arguments.configfile);
  
  if (arguments.gpu_select>0) {
    printf("Using GPU %d\n", arguments.gpu_select);
    
    int deviceCount;
    cudaGetDeviceCount(&deviceCount);
    if (arguments.gpu_select>=deviceCount) {
      fprintf(stderr, "Error: Selected GPU (%d) too high for number of GPU (%d)!\n",
	      arguments.gpu_select, deviceCount);
      exit(1);
    }
    //cudaDeviceProp deviceProperties;
    //cudaGetDeviceProperties(&deviceProperties, arguments.gpu_select);  // Check it is available
    cudaSetDevice(arguments.gpu_select);
  }

  init_2bitLevels();

  // load up the test input data and delays from the configfile
  parseConfig(configfile, nbit, nPol, iscomplex, numchannels, numantennas, lo, bandwidth, numffts, antennas, antFiles, &delays, &antfileoffsets);

  samplegranularity = 8 / (nbit * nPol);
  if (samplegranularity < 1)
  {
    samplegranularity = 1;
  }
  nbaseline = numantennas*(numantennas-1)/2; // Dont include autos
  if (iscomplex) {
    cfactor = 1;
  } else{
    cfactor = 2; // If real data FFT size twice size of number of frequecy channels
  }
  polProd = 1;
  if (nPol>1) polProd = 4;
  
  int fftsamples = numchannels*cfactor;
  int subintsamples = numffts*fftsamples;  // Number of time samples - need to factor # channels (pols) also
  cout << "Subintsamples= " << subintsamples << endl;

  sampletime = 1.0/bandwidth;
  if (!iscomplex) sampletime /= 2.0; 
  subinttime = subintsamples*sampletime;
  cout << "Subint = " << subinttime*1000 << " msec" << endl;

  // CrossCorr
  int corrThreads, blockchan;
  if (numchannels<=512) {
    corrThreads = numchannels;
    blockchan = 1;
  } else {
    corrThreads = 512;
    blockchan = numchannels/512;
  }

  int targetThreads = 50e4;  // This seems a *lot*
  int parallelAccum = (int)ceil(targetThreads/numchannels+1); // I suspect this has failure modes
  //cout << "Initial parallelAccum=" << parallelAccum << endl;
  while (parallelAccum && numffts % parallelAccum) parallelAccum--;
  if (parallelAccum==0) {
    cerr << "Error: Could not determine block size for Cross Correlation" << endl;
    exit(1);
  }

  int nchunk = numffts / parallelAccum;
  dim3 corrBlocks = dim3(blockchan, parallelAccum);
  cout << "Corr Threads:  " << corrThreads << " " << blockchan << ":" << parallelAccum << "/" << nchunk << endl;
  // Final Cross Corr accumulation
  dim3 accumBlocks = dim3(blockchan, 4, nbaseline);
  
  cout << "Allocate Memory" << endl;
  // Allocate space in the buffers for the data and the delays
  allocHostData(&channelisedData_h, &baselineData_h, &gpubaselineData_h, numantennas, subintsamples, numchannels, nPol, parallelAccum);

  // Allocate space on the GPU
  allocDataGPU(&channelisedData, &baselineData, numantennas, subintsamples, numchannels, nPol, parallelAccum);

  // Fill GPU data with Gaussian noise
  CURAND_CALL(curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT));
  CURAND_CALL(curandSetPseudoRandomGeneratorSeed(gen, time(NULL)));
  CURAND_CALL(curandGenerateNormal(gen, (float*)channelisedData, numantennas*nPol*subintsamples*2, 0.0, 10.0)); 
  CURAND_CALL(curandDestroyGenerator(gen));

  // Copy data to back to CPU
  cout << "Copy data to GPU" << endl;
  gpuErrchk(cudaMemcpy(channelisedData_h, channelisedData, numantennas*nPol*subintsamples*sizeof(cuComplex),
		       cudaMemcpyDeviceToHost));

  cpuXcorr(channelisedData_h, baselineData_h, numantennas, nPol, numchannels, numffts, iscomplex);

  cout << "***CPU***"<<endl;
  printBaseline(baselineData_h, numantennas, nPol, numchannels);
  
  cout << "***CrossCorr***" << endl;
  // Zero output data
  gpuErrchk(cudaMemset(baselineData, 0, nbaseline*4*numchannels*sizeof(cuComplex)*parallelAccum));
  CrossCorr<<<corrBlocks,corrThreads>>>(channelisedData, baselineData, numantennas, nchunk);
  CudaCheckError();
  finaliseAccum<<<accumBlocks,corrThreads>>>(baselineData, parallelAccum, nchunk);
  CudaCheckError();

  gpuErrchk(cudaMemcpy(gpubaselineData_h, baselineData, nbaseline*polProd*numchannels*sizeof(cuComplex),
		       cudaMemcpyDeviceToHost));
  printBaseline(gpubaselineData_h, numantennas, nPol, numchannels);
  compareBaseline(baselineData_h, gpubaselineData_h, numantennas, nPol, numchannels);

  cout << "***CrossCorrAccumHoriz***" << endl;
  int ccblock_width = 128;
  dim3 ccblock(1+(numchannels-1)/ccblock_width, numantennas-1, numantennas-1);
  CrossCorrAccumHoriz<<<ccblock, ccblock_width>>>(baselineData, channelisedData, numantennas, numffts, numchannels, fftsamples);
  CudaCheckError();

  gpuErrchk(cudaMemcpy(gpubaselineData_h, baselineData, nbaseline*polProd*numchannels*sizeof(cuComplex),
		       cudaMemcpyDeviceToHost));
  //printBaseline(gpubaselineData_h, numantennas, nPol, numchannels);
  compareBaseline(baselineData_h, gpubaselineData_h, numantennas, nPol, numchannels);

  cout << "***CCAH2***"<<endl;
  
  int nantxp = numantennas*2;
  // Zero output data
  gpuErrchk(cudaMemset(baselineData, 0, nbaseline*4*numchannels*sizeof(cuComplex)*parallelAccum));
  dim3 ccblock2(1+(numchannels-1)/ccblock_width, nantxp-1, nantxp-1);
  CCAH2<<<ccblock2, ccblock_width>>>(baselineData, channelisedData, numantennas, numffts, numchannels, fftsamples);
  cudaDeviceSynchronize();

  // Copy baseline data to back to CPU
  gpuErrchk(cudaMemcpy(gpubaselineData_h, baselineData, nbaseline*polProd*numchannels*sizeof(cuComplex),
		       cudaMemcpyDeviceToHost));

  //printBaseline(gpubaselineData_h, numantennas, nPol, numchannels);
  compareBaseline(baselineData_h, gpubaselineData_h, numantennas, nPol, numchannels);

  cout << "**CCAH3**" << endl;

  // Zero output data
  gpuErrchk(cudaMemset(baselineData, 0, nbaseline*4*numchannels*sizeof(cuComplex)*parallelAccum));
  nantxp = numantennas;
  dim3 ccblock3(1+(numchannels-1)/ccblock_width, nantxp-1, nantxp-1);
  CCAH3<<<ccblock3, ccblock_width>>>(baselineData, channelisedData, numantennas, numffts, numchannels, fftsamples);
  cudaDeviceSynchronize();

  // Copy baseline data to back to CPU
  gpuErrchk(cudaMemcpy(gpubaselineData_h, baselineData, nbaseline*polProd*numchannels*sizeof(cuComplex),
		       cudaMemcpyDeviceToHost));

  compareBaseline(baselineData_h, gpubaselineData_h, numantennas, nPol, numchannels);

  //printBaseline(gpubaselineData_h, numantennas, nPol, numchannels);

  cudaDeviceReset();


  // Free memory
  //  for (i=0; i<numantennas; i++)
  //{
  //  delete(inputdata[i]);
  //  delete(delays[i]);
  //}
  //delete(inputdata);
  //delete(delays);
}
