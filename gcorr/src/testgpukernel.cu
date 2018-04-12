#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <cstdlib>
#include <cstring>
#include <chrono>  // for high_resolution_clock

using std::string;
using std::cout;
using std::cerr;
using std::endl;
using std::vector;

#include "gxkernel.h"

void allocDataGPU(int8_t ***packedData, cuComplex ***unPackedData, cuConplex ***channelisedData,  int numantenna, int numchannels, int numffts, int nbit, bool iscomplex) {
  int i, cfactor;
  unsigned long long GPUalloc = 0;

  int nPol = 2;
  if (iscomplex) {
    cfactor = 1;
  } else{
    cfactor = 2; // If real data FFT size twice size of number of frequecy channels
  }

  int subintsamples = numffts*numchannels*cfactor;
  
  int packedBytes = subintsamples*nbit*nPol/8;
  *packedData = new int8_t*[numantenna];
  for (int i=0; i<numantenna; i++) {
    gpuErrchk(cudaMalloc(&(*packedData)[i], packedBytes));
    GPUalloc += packedBytes;
  }
  // CJP This may not be needed
  //  status = cudaMalloc(&packedData_d, numantenna*sizeof(int8_t*));  
  //if (status != cudaSuccess) {
  //  fprintf(stderr, "Error: cudaMalloc failed\n");
  //  return EXIT_FAILURE;
  //}
  //gpuErrchk(cudaMemcpy(packedData_d, packedData, nant*sizeof(int8_t*), cudaMemcpyHostToDevice));  // Need to get pointers onto GPU

  unpackedData = new cuComplex*[numantenna*2];
  for (int i=0; i<numantenna*2; i++) {
    gpuErrchk(cudaMalloc(&unpackedData[i], subintsamples*sizeof(cuComplex)));
    GPUalloc += subintsamples*sizeof(cuComplex);
  }

  channelisedData = new cuComplex*[numantenna*2];
  for (int i=0; i<numantenna*2; i++) {
    gpuErrchk(cudaMalloc(&channelisedData[i], subintsamples*sizeof(cuComplex)));
    GPUalloc += subintsamples*sizeof(cuComplex);
  }

  cout << "Allocted " << GPUalloc/1e6 << " Mb on GPU" << endl;
}


void allocDataHost(u8 ***data, int numantenna, int numchannels, int numffts, int nbit, bool iscomplex, int &subintbytes)
{
  int i, cfactor;

  int iscomplex = 0;
  int nPol = 2;
  if (iscomplex)
  {
    cfactor = 1;
  }
  else
  {
    cfactor = 2; // If real data FFT size twice size of number of frequecy channels
  }
  
  subintbytes = numchannels*cfactor*numffts*nbit/8*nPol;
  cout << "Allocating " << subintbytes/1024/1024 << " MB per antenna per subint" << endl;
  cout << "          " << subintbytes * numantenna / 1024 / 1024 << " MB total" << endl;


  *data = new u8*[numantenna];
  for (i=0; i<numantenna; i++)
  {
    (*data)[i] = new u8[subintbytes];  // SHOULD BE PINNED
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

int readdata(int bytestoread, vector<std::ifstream*> &antStream, u8 **inputdata) {
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


int main(int argc, char *argv[])
{
  // variables for the test
  char *configfile;
  int i, subintbytes, status;
  u8 ** inputdata;
  double ** delays;
  int numchannels, numantennas, numffts, nbit;
  double lo, bandwidth;
  bool iscomplex;
  vector<string> antennas, antFiles;
  vector<std::ifstream *> antStream;

  int8_t **packedData;
  cuComplex **unPackedData, **channelisedData;
  cufftHandle plan;
  
  if (argc!=2) {
    cout << "Usage:  testfxkernel <config>\n" << endl;
    exit(1);
  }

  int fftchannels;
  bool iscomplex=false;

  configfile = argv[1];

  // load up the test input data and delays from the configfile
  parseConfig(configfile, nbit, iscomplex, numchannels, numantennas, lo, bandwidth, numffts, antennas, antFiles, &delays);
  fftchannels = numchannels;
  if (!iscomplex) fftchannels *=2;

  // Allocate space in the buffers for the data and the delays
  allocDataHost(&inputdata, numantennas, numchannels, numffts, nbit, iscomplex, subintbytes);

  // Allocate space on the GPU
  allocDataGPU(&packedData, &unPaxckedData, &channelisedData, numantennas, numchannels, numffts, nbit, iscomplex);

  for (int i=0; i<numantennas; i++) {
    antStream.push_back(new std::ifstream(antFiles[i].c_str(), std::ios::binary));
  }

  // Configure CUFFT

  if (cufftPlan1d(&plan, fftchannels, CUFFT_C2C, numffts) != CUFFT_SUCCESS) {
    cout << "CUFFT error: Plan creation failed" << endl;
    return;
  }	
  
  // One loop for now

  status = readdata(subintbytes, antStream, inputdata);
  if (status) exit(1);

  // Copy data to GPU

  for (int i=0; i<numantenna; i++) {
    gpuErrchk(cudaMemcpy(packedData[i], inputdata[i], subintbytes, cudaMemcpyHostToDevice)); 
  }

  // Set the delays

  // UNPACK

  // FFT
  for (int i=0; i<numantenna*2; i++) {
    if (cufftExecC2C(plan, unpackedData[i], channelisedData[i], CUFFT_FORWARD) != CUFFT_SUCCESS) {
      cout << "CUFFT error: ExecC2C Forward failed" << endl;
      return;
    }
  }

  // Cross correlate

  // Accumulate Visibilities
  
  //fxkernel.saveVisibilities("vis.out");

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
