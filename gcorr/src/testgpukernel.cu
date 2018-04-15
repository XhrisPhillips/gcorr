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

//#include <chrono>  // for high_resolution_clock

#include <cuComplex.h>
#include <cufft.h>

using std::string;
using std::cout;
using std::cerr;
using std::endl;
using std::vector;

#include "gxkernel.h"

void allocDataGPU(int8_t ***packedData, cuComplex ***unpackedData, cuComplex ***unpackedData_h,
		  cuComplex ***channelisedData, cuComplex ***channelisedData_h, cuComplex ***baselineData, cuComplex ***baselineData_h,
		  int numantenna, int subintsamples, int nbit, int nPol, bool iscomplex, int nchan, int parallelAccum) {
  
  unsigned long long GPUalloc = 0;

  int packedBytes = subintsamples*nbit*nPol/8;
  *packedData = new int8_t*[numantenna];
  for (int i=0; i<numantenna; i++) {
    gpuErrchk(cudaMalloc(&(*packedData)[i], packedBytes));
    GPUalloc += packedBytes;
  }

  // Unpacked data
  cuComplex **unpacked = new cuComplex*[numantenna*nPol];
  for (int i=0; i<numantenna*nPol; i++) {
    gpuErrchk(cudaMalloc(&unpacked[i], subintsamples*sizeof(cuComplex)));
    GPUalloc += subintsamples*sizeof(cuComplex);
  }
  gpuErrchk(cudaMalloc(unpackedData, numantenna*nPol*sizeof(cuComplex*)));
  gpuErrchk(cudaMemcpy(*unpackedData, unpacked, numantenna*nPol*sizeof(cuComplex*), cudaMemcpyHostToDevice));
  *unpackedData_h = unpacked;
  
  // FFT output
  cuComplex **channelised = new cuComplex*[numantenna*nPol];
  for (int i=0; i<numantenna*nPol; i++) {
    gpuErrchk(cudaMalloc(&channelised[i], subintsamples*sizeof(cuComplex)));
    GPUalloc += subintsamples*sizeof(cuComplex);
  }
  gpuErrchk(cudaMalloc(channelisedData, numantenna*nPol*sizeof(cuComplex*)));
  gpuErrchk(cudaMemcpy(*channelisedData, channelised, numantenna*nPol*sizeof(cuComplex*), cudaMemcpyHostToDevice));
  *channelisedData_h = channelised;

  // Baseline visibilities
  int nbaseline = numantenna*(numantenna-1)/2;
  if (!iscomplex) subintsamples /= 2;
  cuComplex **baseline_h = new cuComplex*[nbaseline*4*sizeof(cuComplex*)];
  cout << "Alloc " << nchan*parallelAccum << " complex output values per baseline" << endl;
  for (int i=0; i<nbaseline*4; i++) {
    gpuErrchk(cudaMalloc(&baseline_h[i], nchan*parallelAccum*sizeof(cuComplex)));
    GPUalloc += nchan*parallelAccum*sizeof(cuComplex);
  }
  gpuErrchk(cudaMalloc(baselineData, nbaseline*4*sizeof(cuComplex*)));
  gpuErrchk(cudaMemcpy((*baselineData), baseline_h, nbaseline*4*sizeof(cuComplex*), cudaMemcpyHostToDevice));
  *baselineData_h = baseline_h;
  
  cout << "Allocated " << GPUalloc/1e6 << " Mb on GPU" << endl;
}

void allocDataHost(uint8_t ***data, int numantenna, int subintsamples, int nbit, int nPol, bool iscomplex, int &subintbytes)
{
  int i;

  subintbytes = subintsamples*nbit*nPol/8;  // Watch 31bit overflow
  cout << "Allocating " << subintbytes/1024/1024 << " MB per antenna per subint" << endl;
  cout << "           " << subintbytes * numantenna / 1024 / 1024 << " MB total" << endl;


  *data = new uint8_t*[numantenna];
  for (i=0; i<numantenna; i++)
  {
    (*data)[i] = new uint8_t[subintbytes];  // SHOULD BE PINNED
  }
}

void parseConfig(char *config, int &nbit, bool &iscomplex, int &nchan, int &nant, double &lo, double &bandwidth,
		 int &numffts, vector<string>& antenna, vector<string>& antFiles, double ***delays) {

  std::ifstream fconfig(config);

  string line;
  int anttoread = 0;
  int iant = 0;
  while (std::getline(fconfig, line)) {
    std::istringstream iss(line);
    string keyword;
    if (!(iss >> keyword)) {
      cerr << "Error: Could not parse \"" << line << "\"" << endl;
      std::exit(1);
    }
    if (anttoread) {
      string thisfile;
      iss >> thisfile;
      antenna.push_back(keyword);
      antFiles.push_back(thisfile);
      (*delays)[iant] = new double[3]; //assume we're going to read a second-order polynomial for each antenna, d = a*t^2 + b*t + c, t in units of FFT windows, d in seconds
      for (int i=0;i<3;i++) {
	iss >> (*delays)[iant][i];  // Error checking needed
      }
      iant++;
      anttoread--;
    } else if (strcasecmp(keyword.c_str(), "COMPLEX")==0) {
      iss >> iscomplex; // Should error check
    } else if (strcasecmp(keyword.c_str(), "NBIT")==0) {
      iss >> nbit; // Should error check
    } else if (strcasecmp(keyword.c_str(), "NCHAN")==0) {
      iss >> nchan; // Should error check
    } else if (strcasecmp(keyword.c_str(), "LO")==0) {
      iss >> lo; // Should error check
    } else if (strcasecmp(keyword.c_str(), "BANDWIDTH")==0) {
      iss >> bandwidth; // Should error check
    } else if (strcasecmp(keyword.c_str(), "NUMFFTS")==0) {
      iss >> numffts; // Should error check
    } else if (strcasecmp(keyword.c_str(), "NANT")==0) {
      iss >> nant; // Should error check
      *delays = new double*[nant]; // Alloc memory for delay buffer
      anttoread = nant;
      iant = 0;
    } else {
      std::cerr << "Error: Unknown keyword \"" << keyword << "\"" << endl;
    }
  }
}

int readdata(int bytestoread, vector<std::ifstream*> &antStream, uint8_t **inputdata) {
  for (int i=0; i<antStream.size(); i++) {
    antStream[i]->read((char*)inputdata[i], bytestoread);
    if (! *(antStream[i])) {
      if (antStream[i]->eof())    {
    	return(2);
      } else {
    	cerr << "Error: Problem reading data" << endl;
    	return(1);
      }
    }
  }
  return(0);
}

inline float carg(const cuComplex& z) {return atan2(cuCimagf(z), cuCrealf(z));} // polar angle

void saveVisibilities(const char * outfile, cuComplex **baselines, int nbaseline, int nchan, double bandwidth) {
  cuComplex **vis;
  std::ofstream fvis(outfile);

  // Copy final visibilities back to CPU
  vis = new cuComplex*[nbaseline*4];
  for (int i=0; i<nbaseline*4; i++) {
    vis[i] = new cuComplex[nchan];
    gpuErrchk(cudaMemcpy(vis[i], baselines[i], nchan*sizeof(cuComplex), cudaMemcpyDeviceToHost));
  }
  
  for (int c=0; c<nchan; c++) {
    fvis << std::setw(5) << c << " " << std::setw(11) << std::fixed << std::setprecision(6) << (c+0.5)/nchan*bandwidth/1e6;
    fvis  << std::setprecision(5);
    for (int i=0; i<2*4; i++) {
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
  uint8_t ** inputdata;
  double ** delays;
  int numchannels, numantennas, nbaseline, numffts, nbit;
  double lo, bandwidth;
  bool iscomplex;
  vector<string> antennas, antFiles;
  vector<std::ifstream *> antStream;

  int8_t **packedData;
  cuComplex **unpackedData, **unpackedData_h, **channelisedData, **channelisedData_h, **baselineData, **baselineData_h;
  cufftHandle plan;
  
  if (argc!=2) {
    cout << "Usage:  testfxkernel <config>\n" << endl;
    exit(1);
  }

  configfile = argv[1];

  void init_2bitLevels();

  // load up the test input data and delays from the configfile
  parseConfig(configfile, nbit, iscomplex, numchannels, numantennas, lo, bandwidth, numffts, antennas, antFiles, &delays);

  nbaseline = numantennas*(numantennas-1)/2;
  int nPol = 2;
  if (iscomplex) {
    cfactor = 1;
  } else{
    cfactor = 2; // If real data FFT size twice size of number of frequecy channels
  }

  int fftchannels = numchannels*cfactor;
  int subintsamples = numffts*fftchannels;  // Number of time samples - need to factor # channels (pols) also
  cout << "Subintsamples= " << subintsamples << endl;

  // Setup threads and blocks for the various kernels
  // Unpack
  int unpackThreads = 512;
  int unpackBlocks  = subintsamples/nPol/unpackThreads;
  if (unpackThreads*unpackBlocks*nPol!=subintsamples) {
    cerr << "Error: <<" << unpackBlocks << "," << unpackThreads << ">> inconsistent with " << subintsamples << " samples for unpack kernel" << endl;
  }

  // CrossCorr
  int targetThreads = 50e4;  // This seems a *lot*
  int blockchan;
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
  allocDataHost(&inputdata, numantennas, subintsamples, nbit, nPol, iscomplex, subintbytes);

  // Allocate space on the GPU
  allocDataGPU(&packedData, &unpackedData, &unpackedData_h, &channelisedData, &channelisedData_h,
	       &baselineData, &baselineData_h, numantennas, subintsamples,
	       nbit, nPol, iscomplex, numchannels, parallelAccum);

  for (int i=0; i<numantennas; i++) {
    antStream.push_back(new std::ifstream(antFiles[i].c_str(), std::ios::binary));
  }

  // Configure CUFFT
  if (cufftPlan1d(&plan, fftchannels, CUFFT_C2C, numffts) != CUFFT_SUCCESS) {
    cout << "CUFFT error: Plan creation failed" << endl;
    return(0);
  }
  
  // One loop for now
  status = readdata(subintbytes, antStream, inputdata);
  if (status) exit(1);

  // Copy data to GPU
  cout << "Copy data to GPU" << endl;
  for (int i=0; i<numantennas; i++) {
    gpuErrchk(cudaMemcpy(packedData[i], inputdata[i], subintbytes, cudaMemcpyHostToDevice)); 
  }

  // Set the delays //

  // Unpack the data
  cout << "Unpack data" << endl;
  for (int i=0; i<numantennas; i++) {
    unpack2bit_2chan<<<unpackBlocks,unpackThreads>>>(&unpackedData[i*2], packedData[i]);
    CudaCheckError();
  }

  // Fringe Rotate //
  
  // FFT
  cout << "FFT data" << endl;
  for (int i=0; i<numantennas*2; i++) {
    if (cufftExecC2C(plan, unpackedData_h[i], channelisedData_h[i], CUFFT_FORWARD) != CUFFT_SUCCESS) {
      cout << "CUFFT error: ExecC2C Forward failed" << endl;
      return(0);
    }
  }

  // Cross correlate
  for (int i=0; i<nbaseline*4; i++) {
    gpuErrchk(cudaMemset(baselineData_h[i], 0, numchannels*parallelAccum*sizeof(cuComplex)));
  }
  cout << "Cross correlate" << endl;
  CrossCorr<<<corrBlocks,corrThreads>>>(channelisedData, baselineData, numantennas, nchunk);
  //CrossCorrShared<<<blocks,threads>>>(antData, baseline, nant, nchunk);
  CudaCheckError();
  
  //finaliseAccum<<<accumBlocks,corrThreads>>>(baselineData, numantennas, nchunk);
  //CudaCheckError();

  
  saveVisibilities("vis.out", baselineData_h, nbaseline, numchannels, bandwidth);

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
