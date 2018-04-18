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

#include <cuComplex.h>
#include <cufft.h>

#include "common.h"

#define NTHREADS 256

using std::string;
using std::cout;
using std::cerr;
using std::endl;
using std::vector;

const char *argp_program_version = "testgpukernel 1.0";
static char doc[] = "testgpukernel -- testing operation of the GPU correlator code";
static char args_doc[] = "configuration_file";

#define BUFSIZE 256

/* Our command line options */
static struct argp_option options[] = {
  { "loops", 'n', "NLOOPS", 0, "run the code N times in a loop" },
  { "binary", 'b', 0, 0, "output binary instead of default text" },
  { 0 }
};

struct arguments {
  int output_binary;
  int nloops;
  char configfile[BUFSIZE];
};

/* The option parser */
static error_t parse_opt(int key, char *arg, struct argp_state *state) {
  struct arguments *arguments = (struct arguments *)state->input;

  switch (key) {
  case 'b':
    arguments->output_binary = 1;
    break;
  case 'n':
    arguments->nloops = atoi(arg);
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

void allocDataGPU(int8_t ***packedData, cuComplex **unpackedData,
		  cuComplex **channelisedData, cuComplex **baselineData, 
		  float **rotationPhaseInfo, float **fractionalSampleDelays, int ** sampleShifts, 
                  double ** gpuDelays, int numantenna, int subintsamples, int nbit, 
                  int nPol, bool iscomplex, int nchan, int numffts, int parallelAccum)
{
  unsigned long long GPUalloc = 0;

  int packedBytes = subintsamples*nbit*nPol/8;
  *packedData = new int8_t*[numantenna];
  
  for (int i=0; i<numantenna; i++) {
    gpuErrchk(cudaMalloc(&(*packedData)[i], packedBytes));
    GPUalloc += packedBytes;
  }

  // Unpacked data
  gpuErrchk(cudaMalloc(unpackedData, numantenna*nPol*subintsamples*sizeof(cuComplex)));
  GPUalloc += numantenna*nPol*subintsamples*sizeof(cuComplex);
  
  // FFT output
  gpuErrchk(cudaMalloc(channelisedData, numantenna*nPol*subintsamples*sizeof(cuComplex)));
  GPUalloc += numantenna*nPol*subintsamples*sizeof(cuComplex);

  // Baseline visibilities
  int nbaseline = numantenna*(numantenna-1)/2;
  if (!iscomplex) subintsamples /= 2;
  cout << "Alloc " << nchan*parallelAccum << " complex output values per baseline" << endl;
  gpuErrchk(cudaMalloc(baselineData, nbaseline*4*nchan*parallelAccum*sizeof(cuComplex)));
  GPUalloc += nbaseline*4*nchan*parallelAccum*sizeof(cuComplex);

  // Fringe rotation vector (will contain starting phase and phase increment for every FFT of every antenna)
  gpuErrchk(cudaMalloc(rotationPhaseInfo, numantenna*numffts*2*sizeof(float)));
  GPUalloc += numantenna*numffts*2*sizeof(float);

  // Fractional sample delay vector (will contain midpoint fractional sample delay [in units of radians per channel!] 
  // for every FFT of every antenna)
  gpuErrchk(cudaMalloc(fractionalSampleDelays, numantenna*numffts*sizeof(float)));
  GPUalloc += numantenna*numffts*sizeof(float);

  // Sample shifts vector (will contain the integer sample shift relative to nominal FFT start for every FFT of every antenna)
  gpuErrchk(cudaMalloc(sampleShifts, numantenna*numffts*sizeof(int)));
  GPUalloc += numantenna*numffts*sizeof(int);

  // Delay information vectors
  gpuErrchk(cudaMalloc(gpuDelays, numantenna*4*sizeof(double)));
  GPUalloc += numantenna*4*sizeof(double);
  
  cout << "Allocated " << GPUalloc/1e6 << " Mb on GPU" << endl;
}

inline float carg(const cuComplex& z) {return atan2(cuCimagf(z), cuCrealf(z));} // polar angle

void saveVisibilities(const char *outfile, cuComplex *baselines, int nbaseline, int nchan, int stride, double bandwidth) {
  cuComplex **vis;
  std::ofstream fvis(outfile);

  // Copy final visibilities back to CPU
  vis = new cuComplex*[nbaseline*4];
  for (int i=0; i<nbaseline*4; i++) {
    vis[i] = new cuComplex[nchan];
    gpuErrchk(cudaMemcpy(vis[i], &baselines[i*stride], nchan*sizeof(cuComplex), cudaMemcpyDeviceToHost));
  }
  
  for (int c=0; c<nchan; c++) {
    fvis << std::setw(5) << c << " " << std::setw(11) << std::fixed << std::setprecision(6) << (c+0.5)/nchan*bandwidth/1e6;
    fvis  << std::setprecision(5);
    for (int i=0; i<nbaseline*4; i++) {
      fvis << " " << std::setw(11) << vis[i][c].x << " " << std::setw(11) << vis[i][c].y;
      fvis << " " << std::setw(11) << cuCabsf(vis[i][c]) << " " << std::setw(11) << carg(vis[i][c]);
    }
    fvis << std::endl;
  }
  fvis.close();
  
  for (int i=0;i<nbaseline*4;i++) {
    delete [] vis[i];
  }
  delete [] vis;
}

int main(int argc, char *argv[])
{
  // variables for the test
  char *configfile;
  int subintbytes, status, cfactor;
  int nPol;
  uint8_t ** inputdata;
  double ** delays; /**< delay polynomial for each antenna.  delay is in seconds, time is in units of FFT duration */
  double * antfileoffsets; /**< offset from each the nominal start time of the integration for each antenna data file.  
                                In units of seconds. */
  int numchannels, numantennas, nbaseline, numffts, nbit;
  double lo, bandwidth, sampletime, subinttime;
  bool iscomplex;
  vector<string> antennas, antFiles;
  vector<std::ifstream *> antStream;

  int8_t **packedData;
  float *rotationPhaseInfo;
  float *fractionalSampleDelays;
  int *sampleShifts;
  double *gpuDelays;
  cuComplex *unpackedData, *channelisedData, *baselineData;
  cufftHandle plan;
  cudaEvent_t start_exec, stop_exec;
  
  // Read in the command line arguments.
  struct arguments arguments;
  arguments.nloops = 1;
  arguments.output_binary = 0;
  arguments.configfile[0] = 0;
  argp_parse(&argp, argc, argv, 0, 0, &arguments);

  if (strlen(arguments.configfile) > 0) {
    configfile = arguments.configfile;
  }
  printf("reading configuration file %s\n", arguments.configfile);
  printf("running %d loops\n", arguments.nloops);
  printf("will output %s data\n", (arguments.output_binary == 0) ? "text" : "binary");

  cudaEventCreate(&start_exec);
  cudaEventCreate(&stop_exec);
  
  init_2bitLevels();

  // load up the test input data and delays from the configfile
  parseConfig(configfile, nbit, nPol, iscomplex, numchannels, numantennas, lo, bandwidth, numffts, antennas, antFiles, &delays, &antfileoffsets);

  nbaseline = numantennas*(numantennas-1)/2;
  if (iscomplex) {
    cfactor = 1;
  } else{
    cfactor = 2; // If real data FFT size twice size of number of frequecy channels
  }

  int fftchannels = numchannels*cfactor;
  int subintsamples = numffts*fftchannels;  // Number of time samples - need to factor # channels (pols) also
  cout << "Subintsamples= " << subintsamples << endl;

  sampletime = 1.0/bandwidth;
  if (!iscomplex) sampletime /= 2.0; 
  subinttime = subintsamples*sampletime;
  cout << "Subint = " << subinttime*1000 << " msec" << endl;

  // Setup threads and blocks for the various kernels
  // Unpack
  int unpackThreads = NTHREADS;
  int unpackBlocks  = subintsamples/nPol/unpackThreads;
  if (unpackThreads*unpackBlocks*nPol!=subintsamples) {
    cerr << "Error: <<" << unpackBlocks << "," << unpackThreads << ">> inconsistent with " << subintsamples << " samples for unpack kernel" << endl;
  }

  // Fringe Rotate
  int fringeThreads, blockchan;
  if (fftchannels<=NTHREADS) {
    fringeThreads = fftchannels;
    blockchan = 1;
  } else {
    fringeThreads = NTHREADS;
    blockchan = fftchannels/NTHREADS;
    if (fftchannels%NTHREADS) {
      cerr << "Error: NTHREADS not divisible into fftchannels" << endl;
      exit(1);
    }
  }
  dim3 fringeBlocks = dim3(blockchan, numffts, numantennas);

  // Fractional Delay
  int fracDelayThreads;
  if (numchannels<=NTHREADS) {
    fracDelayThreads = numchannels;
    blockchan = 1;
  } else {
    fracDelayThreads = NTHREADS;
    blockchan = numchannels/NTHREADS;
    if (numchannels%NTHREADS) {
      cerr << "Error: NTHREADS not divisible into fftchannels" << endl;
      exit(1);
    }
  }
  dim3 fracDelayBlocks = dim3(blockchan, numffts, numantennas);

  // CrossCorr
  int targetThreads = 50e4;  // This seems a *lot*
  int corrThreads;
  if (numchannels<=512) {
    corrThreads = numchannels;
    blockchan = 1;
  } else {
    corrThreads = 512;
    blockchan = numchannels/512;
  }
  int parallelAccum = (int)ceil(targetThreads/numchannels+1); // I suspect this has failure modes
  cout << "Initial parallelAccum=" << parallelAccum << endl;
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
  allocDataHost(&inputdata, numantennas, numchannels, numffts, nbit, nPol, iscomplex, subintbytes);

  // Allocate space on the GPU
  allocDataGPU(&packedData, &unpackedData, &channelisedData,
	       &baselineData, &rotationPhaseInfo, &fractionalSampleDelays, &sampleShifts,
               &gpuDelays, numantennas, subintsamples,
	       nbit, nPol, iscomplex, numchannels, numffts, parallelAccum);

  for (int i=0; i<numantennas; i++) {
    antStream.push_back(new std::ifstream(antFiles[i].c_str(), std::ios::binary));
  }

  // Configure CUFFT
  if (cufftPlan1d(&plan, fftchannels, CUFFT_C2C, 2*numantennas*numffts) != CUFFT_SUCCESS) {
    cout << "CUFFT error: Plan creation failed" << endl;
    return(0);
  }
  
  status = readdata(subintbytes, antStream, inputdata);
  if (status) exit(1);

  // Copy data to GPU
  cout << "Copy data to GPU" << endl;
  for (int i=0; i<numantennas; i++) {
    gpuErrchk(cudaMemcpy(packedData[i], inputdata[i], subintbytes, cudaMemcpyHostToDevice)); 
  }

  // Copy delays to GPU
  cout << "Copy delays to GPU" << endl;
  for (int i=0; i<numantennas; i++) {
    gpuErrchk(cudaMemcpy(&(gpuDelays[i*4]), delays[i], 3*sizeof(double), cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(&(gpuDelays[i*4+3]), &(antfileoffsets[i]), sizeof(double), cudaMemcpyHostToDevice));
  }

  // Check that the number of FFTs is a valid number
  if (numffts%8)
  {
    cerr << "Error: numffts must be divisible by 8" << endl;
    exit(1);
  }
  // Set the number of blocks for fringe rotation (and fractional sample delay?)
  dim3 FringeSetblocks = dim3(8, numantennas);

  // Record the start time
  cudaEventRecord(start_exec, 0);
  for (int l=0; l<arguments.nloops; l++)
  {
    // Use the delays to calculate fringe rotation phases and fractional sample delays for each FFT //
    calculateDelaysAndPhases<<<FringeSetblocks, numffts/8>>>(gpuDelays, lo, sampletime, fftchannels, numchannels, rotationPhaseInfo, 
                                                             sampleShifts, fractionalSampleDelays);
    CudaCheckError();

    // Unpack the data
    //cout << "Unpack data" << endl;
    for (int i=0; i<numantennas; i++) {
      unpack2bit_2chan_fast<<<unpackBlocks,unpackThreads>>>(&unpackedData[2*i*subintsamples], packedData[i], &(sampleShifts[numffts*i]));
      CudaCheckError();
    }

    /*// Fringe Rotate //
    //cout << "Fringe Rotate" << endl;
    setFringeRotation<<<FringeSetblocks, numffts/8>>>(rotationPhaseInfo);
    CudaCheckError();*/

    FringeRotate<<<fringeBlocks,fringeThreads>>>(unpackedData, rotationPhaseInfo);
    CudaCheckError();
  
    // FFT
    //cout << "FFT data" << endl;
    if (cufftExecC2C(plan, unpackedData, channelisedData, CUFFT_FORWARD) != CUFFT_SUCCESS) {
      cout << "CUFFT error: ExecC2C Forward failed" << endl;
      return(0);
    }

    // Fractional Delay Correction
    //FracSampleCorrection<<<fracDelayBlocks,fracDelayThreads>>>(channelisedData, fractionalDelayValues, numchannels, fftchannels, numffts, subintsamples);
    //CudaCheckError();
    
    // Cross correlate
    gpuErrchk(cudaMemset(baselineData, 0, nbaseline*4*numchannels*parallelAccum*sizeof(cuComplex)));

#if 0
    cout << "Cross correlate" << endl;
    CrossCorr<<<corrBlocks,corrThreads>>>(channelisedData, baselineData, numantennas, nchunk);
    CudaCheckError();
    // cout << "Finalise" << endl;
    finaliseAccum<<<accumBlocks,corrThreads>>>(baselineData, parallelAccum, nchunk);
    CudaCheckError();
#else
    int ccblock_width = 128;
    dim3 ccblock(1+(numchannels-1)/ccblock_width, numantennas-1, numantennas-1);
    CrossCorrAccumHoriz<2><<<ccblock, ccblock_width>>>(baselineData, channelisedData, numantennas, numffts, numchannels, fftchannels);
#endif
  }
  
  float dtime;
  cudaEventRecord(stop_exec, 0);
  cudaEventSynchronize(stop_exec);
  cudaEventElapsedTime(&dtime, start_exec, stop_exec);

  cout << "Total execution time for " << arguments.nloops << " loops =  " <<  dtime << " ms" << endl;

#if 0
  saveVisibilities("vis.out", baselineData, nbaseline, numchannels, parallelAccum*numchannels, bandwidth);
#else
  saveVisibilities("vis.out", baselineData, nbaseline, numchannels, numchannels, bandwidth);
#endif

  cudaDeviceSynchronize();
  cudaDeviceReset();

  // Calculate the elapsed time

  // Free memory
  //  for (i=0; i<numantennas; i++)
  //{
  //  delete(inputdata[i]);
  //  delete(delays[i]);
  //}
  //delete(inputdata);
  //delete(delays);
}
