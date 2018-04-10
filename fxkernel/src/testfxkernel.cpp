#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <strings.h>
#include <string.h>

using std::string;
using std::cout;
using std::cerr;
using std::endl;

#include "fxkernel.h"
#include "vectordefs.h"

void allocData(u8 ***data, double *** delays, int numantenna, int numchannels, int numffts, int nbit, int nsubint)
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
  
  int bytespersubint = numchannels*cfactor*numffts*nbit/8*nPol;
  cout << "Allocating " << bytespersubint/1024/1024 << " MB per antenna per subint" << endl;
  cout << "          " << bytespersubint * numantenna * nsubint / 1024 / 1024 << " MB total" << endl;


  *data = new u8*[numantenna];
  *delays = new double*[numantenna];
  for (i=0; i<numantenna; i++)
  {
    (*data)[i] = new u8[bytespersubint*nsubint];
    (*delays)[i] = new double[3]; //assume we're going to read a second-order polynomial for each antenna
  }
}

void parseConfig(char *config, int &nbit, bool &iscomplex, int &nchan, int &nant, double &lo, double &bandwidth,
		 char ***antenna, char ***antFiles ) {

  std::ifstream fconfig(config);

  string line;
  int anttoread = 0;
  int iant = 0;
  while (std::getline(fconfig, line)) {
    std::istringstream iss(line);
    string keyword;
    if (!(iss >> keyword)) {
      cerr << "Error: Could not parse \"" << line << "\"" << endl;
      exit(1);
    }
    if (anttoread) {
      (*antenna)[iant] = new char[keyword.length()+1];
      strcpy((*antenna)[iant],keyword.c_str());
      string thisfile;
      iss >> thisfile;
      (*antFiles)[iant] = new char[thisfile.length()+1];
      strcpy((*antFiles)[iant],thisfile.c_str());
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
    } else if (strcasecmp(keyword.c_str(), "NANT")==0) {
      iss >> nant; // Should error check
      *antenna = new char*[nant];
      *antFiles = new char*[nant];
      anttoread = nant;
      iant = 0;
    } else {
      std::cerr << "Error: Unknown keyword \"" << keyword << "\"" << endl;
    }
  }
}


int main(int argc, char *argv[])
{
  // variables for the test
  char *configfile;
  int i;
  u8 ** inputdata;
  double ** delays;
  int numchannels, numantennas, numffts, nbit, nsubint;
  double lo, bandwidth;
  bool iscomplex;
  char **antennas, **antFiles;

  if (argc!=2) {
    cout << "Usage:  testfxkernel <config>\n" << endl;
    exit(1);
  }

  configfile = argv[1];

  parseConfig(configfile, nbit, iscomplex, numchannels, numantennas, lo, bandwidth, &antennas, &antFiles);

  cout << "Got COMPLEX " << iscomplex << endl;
  cout << "Got NBIT " << nbit << endl;
  cout << "Got NCHAN " << numchannels << endl;
  cout << "Got LO " << lo << endl;
  cout << "Got BANDWIDTH" << bandwidth << endl;
  cout << "Got NANT " << numantennas << endl;
  for (int i=0;i<numantennas;i++) {
    cout << "  " << antennas[i] << ":" << antFiles[i] << endl;
  }
    

  
  exit(1);
  // Set the inputs we'll use as a test
  // Current values would give a subint of 100ms, which is fairly reasonable
  //nbit = 2;
  //numchannels = 1024;
  //numantennas = 6;
  numffts = 3125;
  //lo = 1650000000.0;
  //bandwidth = 32000000.0;
  nsubint = 10;

  // Allocate space in the buffers for the data and the delays
  allocData(&inputdata, &delays, numantennas, numchannels, numffts, nbit, nsubint);

  // load up the test input data from somewhere

  // Load up the delays from somewhere - these should be a 2nd order polynomial per antenna
  // with the x value being in units of FFTs.


  // create the FxKernel
  // We could also create multiple FxKernels to test parallelisation in a simple/lazy way
  FxKernel fxkernel = FxKernel(numantennas, numchannels, numffts, nbit, lo, bandwidth);

  for (i=0; i<nsubint; i++) {

    // Set the input data and the delays
    fxkernel.setInputData(inputdata);
    fxkernel.setDelays(delays);

    // Checkpoint for timing
  
    // Run the processing
    fxkernel.process();
  }

  // Calculate the elapsed time

  // Free memory
  for (i=0; i<numantennas; i++)
  {
    delete(inputdata[i]);
    delete(delays[i]);
  }
  delete(inputdata);
  delete(delays);
}
