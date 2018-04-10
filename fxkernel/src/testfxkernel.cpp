#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <cstdlib>
#include <cstring>

using std::string;
using std::cout;
using std::cerr;
using std::endl;
using std::vector;

#include "fxkernel.h"
#include "vectordefs.h"

void allocData(u8 ***data, double *** delays, int numantenna, int numchannels, int numffts, int nbit, int &subintbytes)
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
  *delays = new double*[numantenna];
  for (i=0; i<numantenna; i++)
  {
    (*data)[i] = new u8[subintbytes];
    (*delays)[i] = new double[3]; //assume we're going to read a second-order polynomial for each antenna
  }
}

void parseConfig(char *config, int &nbit, bool &iscomplex, int &nchan, int &nant, double &lo, double &bandwidth,
		 int &numffts, vector<string>& antenna, vector<string>& antFiles ) {

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
      anttoread = nant;
      iant = 0;
    } else {
      std::cerr << "Error: Unknown keyword \"" << keyword << "\"" << endl;
    }
  }
}


int readdata(int bytestoread, vector<std::ifstream*> &antStream, u8 **inputdata) {
  for (int i=0; i<antStream.size(); i++) {
    (*antStream[i]).read((char*)inputdata[i], bytestoread);
    if (! *antStream[i]) {
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
  int i, subintbytes;
  u8 ** inputdata;
  double ** delays;
  int numchannels, numantennas, numffts, nbit;
  double lo, bandwidth;
  bool iscomplex;
  vector<string> antennas, antFiles;
  vector<std::ifstream *> antStream;

  if (argc!=2) {
    cout << "Usage:  testfxkernel <config>\n" << endl;
    exit(1);
  }

  configfile = argv[1];

  parseConfig(configfile, nbit, iscomplex, numchannels, numantennas, lo, bandwidth, numffts, antennas, antFiles);

  cout << "Got COMPLEX " << iscomplex << endl;
  cout << "Got NBIT " << nbit << endl;
  cout << "Got NCHAN " << numchannels << endl;
  cout << "Got LO " << lo << endl;
  cout << "Got BANDWIDTH" << bandwidth << endl;
  cout << "Got NANT " << numantennas << endl;
  for (int i=0;i<numantennas;i++) {
    cout << "  " << antennas[i] << ":" << antFiles[i] << endl;
  }
    
  // Allocate space in the buffers for the data and the delays
  allocData(&inputdata, &delays, numantennas, numchannels, numffts, nbit, subintbytes);

  //openFiles(antennas, antFiles, antStream);
  for (int i=0; i<numantennas; i++) {
    std::ifstream thisfile(antFiles[i].c_str(), std::ios::binary);
    antStream.push_back(&thisfile);
  }

  // load up the test input data from somewhere

  // Load up the delays from somewhere - these should be a 2nd order polynomial per antenna
  // with the x value being in units of FFTs.


  // create the FxKernel
  // We could also create multiple FxKernels to test parallelisation in a simple/lazy way
  FxKernel fxkernel = FxKernel(numantennas, numchannels, numffts, nbit, lo, bandwidth);

  fxkernel.setInputData(inputdata);

  int status;
  while (1) {
    status = readdata(subintbytes, antStream, inputdata);
    if (status) break;

    // Set the delays
    fxkernel.setDelays(delays);

    // Checkpoint for timing
  
    // Run the processing
    fxkernel.process();

    // Do somethung with subint
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
