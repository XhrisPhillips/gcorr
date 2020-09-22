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
#include "gxkernel.h"

#ifdef USEHALF
#include <cufftXt.h>
#endif

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
				       { "gpu", 'g', "GPU", 0, "Select specific GPU"},
				       { 0 }
};

struct arguments {
  int output_binary;
  int nloops;
  int gpu_select;
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

int kNumStreams = 2;

void allocDataGPU(int8_t ****packedData, COMPLEX ***unpackedData,
		  COMPLEX ***channelisedData, cuComplex ***baselineData, 
		  float ***rotationPhaseInfo, float ***fractionalSampleDelays, int ***sampleShifts, 
		  double ***gpuDelays, int numantenna, int subintsamples, int nbit, int nPol, 
		  bool iscomplex, int nchan, int numffts, int parallelAccum, int num_streams, int num_packed_bytes) {

  unsigned long long GPUalloc = 0;

  *packedData = new int8_t**[num_streams];
  *unpackedData = new COMPLEX*[num_streams];
  *channelisedData = new COMPLEX*[num_streams];
  *baselineData = new cuComplex*[num_streams];
  *rotationPhaseInfo = new float*[num_streams];
  *fractionalSampleDelays = new float*[num_streams];
  *sampleShifts = new int*[num_streams];
  *gpuDelays = new double*[num_streams];

  // Unpacked data
  cout << "Alloc " << nchan*parallelAccum << " complex output values per baseline" << endl;
  for (int s=0; s<num_streams; s++) {
    (*packedData)[s] = new int8_t*[numantenna];
    for (int i=0; i<numantenna; i++) {
      gpuErrchk(cudaMalloc(&((*packedData)[s][i]), num_packed_bytes));
      GPUalloc += num_packed_bytes;
    }

    gpuErrchk(cudaMalloc(&(*unpackedData)[s], numantenna*nPol*subintsamples*sizeof(cuComplex)));
    GPUalloc += numantenna*nPol*subintsamples*sizeof(cuComplex);
  
    // FFT output
    gpuErrchk(cudaMalloc(&(*channelisedData)[s], numantenna*nPol*subintsamples*sizeof(cuComplex)));
    GPUalloc += numantenna*nPol*subintsamples*sizeof(cuComplex);

    // Baseline visibilities
    int nbaseline = numantenna*(numantenna-1)/2;
    if (!iscomplex) subintsamples /= 2;
    gpuErrchk(cudaMalloc(&(*baselineData)[s], nbaseline*4*nchan*parallelAccum*sizeof(cuComplex)));
    GPUalloc += nbaseline*4*nchan*parallelAccum*sizeof(cuComplex);

    // Fringe rotation vector (will contain starting phase and phase increment for every FFT of every antenna)
    gpuErrchk(cudaMalloc(&(*rotationPhaseInfo)[s], numantenna*numffts*2*sizeof(float)));
    GPUalloc += numantenna*numffts*2*sizeof(float);

    // Fractional sample delay vector (will contain midpoint fractional sample delay [in units of radians per channel!] 
    // for every FFT of every antenna)
    gpuErrchk(cudaMalloc(&(*fractionalSampleDelays)[s], numantenna*numffts*sizeof(float)));
    GPUalloc += numantenna*numffts*sizeof(float);

    // Sample shifts vector (will contain the integer sample shift relative to nominal FFT start for every FFT of every antenna)
    gpuErrchk(cudaMalloc(&(*sampleShifts)[s], numantenna*numffts*sizeof(int)));
    GPUalloc += numantenna*numffts*sizeof(int);

    // Delay information vectors
    gpuErrchk(cudaMalloc(&(*gpuDelays)[s], numantenna*4*sizeof(double)));
    GPUalloc += numantenna*4*sizeof(double);
  }
  
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
  int numkernelexecutions, executionsperthread;
  int nPol;
  int samplegranularity; /**< how many time samples per byte.  If >1, then our fractional sample error can be >0.5 samples */
  uint8_t ** inputdata;
  double ** delays; /**< delay polynomial for each antenna.  delay is in seconds, time is in units of FFT duration */
  double * antfileoffsets; /**< offset from each the nominal start time of the integration for each antenna data file.  
			      In units of seconds. */
  int numchannels, numantennas, nbaseline, numffts, nbit;
  double lo, bandwidth, sampletime, subinttime;
  bool iscomplex;
  vector<string> antennas, antFiles;
  vector<std::ifstream *> antStream;

  int8_t ***packedData;
  float **rotationPhaseInfo;
  float **fractionalSampleDelays;
  int32_t **sampleShifts;
  double **gpuDelays;
  COMPLEX **unpackedData, **channelisedData;
  cuComplex **baselineData;
  cufftHandle plan[kNumStreams];
  cudaEvent_t start_exec, stop_exec;
  
  // Read in the command line arguments.
  struct arguments arguments;
  arguments.nloops = 10;
  arguments.output_binary = 0;
  arguments.gpu_select = -1;
  arguments.configfile[0] = 0;
  argp_parse(&argp, argc, argv, 0, 0, &arguments);

  if (strlen(arguments.configfile) > 0) {
    configfile = arguments.configfile;
  }
  printf("reading configuration file %s\n", arguments.configfile);
  printf("running %d loops\n", arguments.nloops);
  printf("will output %s data\n", (arguments.output_binary == 0) ? "text" : "binary");

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

  cudaEventCreate(&start_exec);
  cudaEventCreate(&stop_exec);
  
  init_2bitLevels();

  // load up the test input data and delays from the configfile
  parseConfig(configfile, nbit, nPol, iscomplex, numchannels, numantennas, lo, bandwidth, numffts, antennas, antFiles, &delays, &antfileoffsets);
  nPol = 2;

  samplegranularity = 8 / (nbit * nPol);
  if (samplegranularity < 1)
    {
      samplegranularity = 1;
    }
  nbaseline = numantennas*(numantennas-1)/2;
  if (iscomplex) {
    cfactor = 1;
  } else{
    cfactor = 2; // If real data FFT size twice size of number of frequecy channels
  }

  int fftsamples = numchannels*cfactor;
  int subintsamples = numffts*fftsamples;  // Number of time samples - need to factor # channels (pols) also
  cout << "Subintsamples= " << subintsamples << endl;

  sampletime = 1.0/bandwidth;
  if (!iscomplex) sampletime /= 2.0; 
  subinttime = subintsamples*sampletime;
  cout << "Subint = " << subinttime*1000 << " msec" << endl;

  // Setup threads and blocks for the various kernels
  // Unpack
  int unpackThreads;
  if (nbit==2 && !iscomplex)
    {
      numkernelexecutions = fftsamples/2;
    }
  else if (nbit==8 && iscomplex)
    {
      numkernelexecutions = fftsamples*2;
    } 
  else
    {
      cerr << "Error: Unsupported number if bits/complex (" << nbit << "/" << iscomplex << ")" << endl;
      exit(1);
    }

  if (numkernelexecutions<=NTHREADS) {
    unpackThreads = numkernelexecutions;
    executionsperthread = 1;
  } else {
    unpackThreads = NTHREADS;
    executionsperthread = numkernelexecutions/(NTHREADS);
    if (numkernelexecutions%(NTHREADS)) {
      cerr << "Error: NTHREADS not divisible into number of kernel executions for unpack" << endl;
      exit(1);
    }
  }
  dim3 unpackBlocks = dim3(executionsperthread, numffts);

  /*
    int unpackThreads = NTHREADS;
    int unpackBlocks;
    if (nbit==2 && !iscomplex) {
    unpackBlocks = subintsamples/2/unpackThreads; // 2 time samples/byte
    if (unpackThreads*unpackBlocks*nPol!=subintsamples) {
    cerr << "Error: <<" << unpackBlocks << "," << unpackThreads << ">> inconsistent with " << subintsamples << " samples for unpack kernel" << endl;
    exit(1);
    }
    } else if (nbit==8 && iscomplex) {
    unpackBlocks = subintsamples*nPol/unpackThreads; // Each pol separately
    if (unpackThreads*unpackBlocks!=subintsamples*nPol) {
    cerr << "Error: <<" << unpackBlocks << "," << unpackThreads << ">> inconsistent with " << subintsamples << " samples for unpack kernel" << endl;
    exit(1);
    }
    } else {
    cerr << "Error: Unsupported number if bits/complex (" << nbit << "/" << iscomplex << ")" << endl;
    exit(1);
    }*/
  
  // Set the number of blocks for when calculating the delay and phase info
  int delayPhaseThreads;
  numkernelexecutions = numffts;

  if (numkernelexecutions<=NTHREADS) {
    delayPhaseThreads = numkernelexecutions;
    executionsperthread = 1;
  } else {
    delayPhaseThreads = NTHREADS;
    executionsperthread = numkernelexecutions/NTHREADS;
    if (numkernelexecutions%NTHREADS) {
      cerr << "Error: NTHREADS not divisible into numkernelexecutions for delay and phase calculations" << endl;
      exit(1);
    }
  }
  dim3 delayPhaseBlocks = dim3(executionsperthread, numantennas);

#if 0
  // Fringe Rotate
  int fringeThreads;
  numkernelexecutions = fftsamples;

  if (numkernelexecutions<=NTHREADS) {
    fringeThreads = numkernelexecutions;
    executionsperthread = 1;
  } else {
    fringeThreads = NTHREADS;
    executionsperthread = numkernelexecutions/NTHREADS;
    if (numkernelexecutions%NTHREADS) {
      cerr << "Error: NTHREADS not divisible into numkernelexecutions for fringe rotation" << endl;
      exit(1);
    }
  }
  dim3 fringeBlocks = dim3(executionsperthread, numffts, numantennas);
#endif
  
  // Fractional Delay
  int fracDelayThreads;
  numkernelexecutions = numchannels;

  if (numkernelexecutions<=NTHREADS) {
    fracDelayThreads = numkernelexecutions;
    executionsperthread = 1;
  } else {
    fracDelayThreads = NTHREADS;
    executionsperthread = numkernelexecutions/NTHREADS;
    if (numkernelexecutions%NTHREADS) {
      cerr << "Error: NTHREADS not divisible into numkernelexecutions for fractional sample correction" << endl;
      exit(1);
    }
  }
  dim3 fracDelayBlocks = dim3(executionsperthread, numffts, numantennas);

#if 0
  // CrossCorr
  int corrThreads;
  if (numchannels<=512) {
    corrThreads = numchannels;
    blockchan = 1;
  } else {
    corrThreads = 512;
    blockchan = numchannels/512;
  }
#endif
  int targetThreads = 50e4;  // This seems a *lot*
  int parallelAccum = (int)ceil(targetThreads/numchannels+1); // I suspect this has failure modes
  //cout << "Initial parallelAccum=" << parallelAccum << endl;
  while (parallelAccum && numffts % parallelAccum) parallelAccum--;
  if (parallelAccum==0) {
    cerr << "Error: Could not determine block size for Cross Correlation" << endl;
    exit(1);
  }
#if 0
  int nchunk = numffts / parallelAccum;
  dim3 corrBlocks = dim3(blockchan, parallelAccum);
  cout << "Corr Threads:  " << corrThreads << " " << blockchan << ":" << parallelAccum << "/" << nchunk << endl;
  // Final Cross Corr accumulation
  dim3 accumBlocks = dim3(blockchan, 4, nbaseline);
#endif
  
  cout << "Allocate Memory" << endl;
  // Allocate space in the buffers for the data and the delays
  printf("Allocating host data\n");
  allocDataHost(&inputdata, numantennas, numchannels, numffts, nbit, nPol, iscomplex, subintbytes);

  // Allocate space on the GPU
  printf("Allocating GPU data\n");
  allocDataGPU(&packedData, &unpackedData, &channelisedData,
	       &baselineData, &rotationPhaseInfo, &fractionalSampleDelays, &sampleShifts,
               &gpuDelays, numantennas, subintsamples,
	       nbit, nPol, iscomplex, numchannels, numffts, parallelAccum, kNumStreams, subintbytes);

  for (int i=0; i<numantennas; i++)
    {
      antStream.push_back(new std::ifstream(antFiles[i].c_str(), std::ios::binary));
      if (!antStream.back()->good())
	{
	  cerr << "Problem with file " << antFiles[i] << " - does it exist?" << endl;
	}
    }

  // Configure CUFFT
#ifdef USEHALF

  long long int n[1] = { fftsamples };
  size_t workSize[1] = { 0 };
  for (int s=0; s<kNumStreams; s++) {
    if (cufftCreate(&plan[s]) != CUFFT_SUCCESS) {
      cout << "CUFFT error: Handle creation failed" << endl;
      return(0);
    }
    if (cufftXtMakePlanMany(plan[s], 1, n, NULL, 0, 0, CUDA_C_16F, NULL, 0, 0, CUDA_C_16F, nPol*numantennas*numffts, workSize, CUDA_C_16F) != CUFFT_SUCCESS) {
      cout << "CUFFT error: Plan creation failed" << endl;
      return(0);
    }
  }
#else
  for (int s=0; s<kNumStreams; s++) {
    if (cufftPlan1d(&plan[s], fftsamples, CUFFT_C2C, nPol*numantennas*numffts) != CUFFT_SUCCESS) {
      cout << "CUFFT error: Plan creation failed" << endl;
      return(0);
    }
  }
#endif
  
  cout << "Reading data" << endl;
  status = readdata(subintbytes, antStream, inputdata);
  if (status) exit(1);
  init_2bitLevels();

  // Initialise CUDA streams
  cout << "Initialising CUDA streams" << endl;
  cudaStream_t streams[kNumStreams];
  for (int s=0; s<kNumStreams; s++)
    gpuErrchk(cudaStreamCreate(&streams[s]));

  // Check that the number of FFTs is a valid number
  if (numffts%8)
    {
      cerr << "Error: numffts must be divisible by 8" << endl;
      exit(1);
    }

  // Record the start time
  cudaEventRecord(start_exec, 0);
  cout << "Entering loop" << endl;
  for (int l=0; l<arguments.nloops; l++)
    {
      int stream = l % kNumStreams;

      // Copy data to GPU
      // cout << "Copy data to GPU" << endl;
      for (int i=0; i<numantennas; i++) {
	gpuErrchk(cudaMemcpyAsync(packedData[stream][i], inputdata[i], subintbytes, cudaMemcpyHostToDevice, streams[stream])); 
      }
    
      // Copy delays to GPU
      // cout << "Copy delays to GPU" << endl;
      for (int i=0; i<numantennas; i++) {
	gpuErrchk(cudaMemcpyAsync(&(gpuDelays[stream][i*4]), delays[i], 3*sizeof(double), cudaMemcpyHostToDevice, streams[stream]));
	gpuErrchk(cudaMemcpyAsync(&(gpuDelays[stream][i*4+3]), &(antfileoffsets[i]), sizeof(double), cudaMemcpyHostToDevice, streams[stream]));
      }

      int previous_stream = (stream-1);
      if (previous_stream < 0)
	previous_stream += kNumStreams;
    
      // Wait for the previous stream to complete all of its compute
      // In testing it was observed that overlapping compute kernels
      // reduced performance
      cudaStreamSynchronize(streams[previous_stream]);

      // Use the delays to calculate fringe rotation phases and fractional sample delays for each FFT //

      calculateDelaysAndPhases<<<delayPhaseBlocks, delayPhaseThreads,0,streams[stream]>>>(gpuDelays[stream], lo, sampletime, fftsamples, numchannels, 
											  samplegranularity, rotationPhaseInfo[stream], 
											  sampleShifts[stream], fractionalSampleDelays[stream]);
      CudaCheckError();

      // Unpack the data
      //cout << "Unpack data" << endl;
      for (int i=0; i<numantennas; i++) {
	if (nbit==2 && !iscomplex) {
	  unpack2bit_2chan_rotate<<<unpackBlocks,unpackThreads,0,streams[stream]>>>(&unpackedData[stream][2*i*subintsamples], packedData[stream][i], &rotationPhaseInfo[stream][i*numffts*2], &(sampleShifts[stream][numffts*i]), fftsamples);
	} else if (nbit==8 && iscomplex) {
	  unpack8bitcomplex_2chan_rotate<<<unpackBlocks,unpackThreads,0,streams[stream]>>>(&unpackedData[stream][2*i*subintsamples], packedData[stream][i], &rotationPhaseInfo[stream][i*numffts*2], &(sampleShifts[stream][numffts*i]), fftsamples);
	}
	CudaCheckError();
      }
  
      // FFT

      cufftSetStream(plan[stream], streams[stream]);
#ifdef USEHALF
      if (cufftXtExec(plan[stream], unpackedData[stream], channelisedData[stream], CUFFT_FORWARD) != CUFFT_SUCCESS) {
	cout << "CUFFT error: XtExec Forward failed" << endl;
	return(0);
      }
#else
      if (cufftExecC2C(plan[stream], unpackedData[stream], channelisedData[stream], CUFFT_FORWARD) != CUFFT_SUCCESS) {
	cout << "CUFFT error: ExecC2C Forward failed" << endl;
	return(0);
      }
#endif
      // Fractional Delay Correction
      FracSampleCorrection<<<fracDelayBlocks,fracDelayThreads>>>(channelisedData[stream], fractionalSampleDelays[stream], numchannels, fftsamples, numffts, subintsamples);
      CudaCheckError();
    
      // Cross correlate

      gpuErrchk(cudaMemsetAsync(baselineData[stream], 0, nbaseline*4*numchannels*sizeof(cuComplex), streams[stream]));


      int ccblock_width = 128;
      //int nantxp = numantennas*2;
      int nantxp = numantennas;
      dim3 ccblock(1+(numchannels-1)/ccblock_width, nantxp-1, nantxp-1);
      CCAH3<<<ccblock, ccblock_width, 0, streams[stream]>>>(baselineData[stream], channelisedData[stream], numantennas, numffts, numchannels, fftsamples);
    }
  
  float dtime;
  cudaEventRecord(stop_exec, 0);
  cudaEventSynchronize(stop_exec);
  cudaEventElapsedTime(&dtime, start_exec, stop_exec);

  cout << "Total execution time for " << arguments.nloops << " loops =  " <<  dtime << " ms" << endl;

  float rate = (float)subintsamples * numantennas * (2./cfactor) * nPol * nbit *arguments.nloops /(dtime/1000.)/1e9;
  cout << "Processed " << subinttime*arguments.nloops << " sec of data (" << rate << " Gbps)" << endl;

  // Write out the first streams data, this will need to be thought about more carefully
  saveVisibilities("vis.out", baselineData[0], nbaseline, numchannels, numchannels, bandwidth);

  cudaDeviceSynchronize();
  cudaDeviceReset();

 
}
