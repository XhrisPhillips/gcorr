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

#include "fxkernel.h"
#include "vectordefs.h"

/** 
 * @file testfxkernel.cpp
 * @brief A test harness for fxkernel
 *   
 * This test harness reads a brief and simple config file, then creates a single
 * FxKernel object and sets it up with the input data and delays, and then
 * runs process (to process a single subintegration).  Timing information is gathered
 * and written out along with the visibilities from this sub-integration.
 */

/**
 * Allocates the space for the raw (packed, quantised) voltage data.
 * @param data the data array to be allocated
 * @param numantenna the number of antennas
 * @param numchannels the number of channels to be produced in the correlation
 * @param numffts the number of FFTs that will be processed in one subintegration
 * @param nbit the number of bits per sample
 * @param subintbytes the number of bytes allocated in total (set in this function)
 */
void allocData(u8 ***data, int numantenna, int numchannels, int numffts, int nbit, int &subintbytes)
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
    (*data)[i] = new u8[subintbytes];
  }
}

/**
 * Parses the config file and sets a bunch of necessary parameters
 * @param configfilename the data array to be allocated
 * @param nbit the number of bits per sample (set in this function)
 * @param iscomplex whether the data is complex (true) or real (false).  Set in this function.
 * @param nchan the number of channels to be produced in the correlation (set in this function).
 * @param nant the number of antennas (set in this function).
 * @param lo the local oscillator frequency in Hz (set in this function).
 * @param bandwidth the bandwidth in Hz (set in this function).
 * @param numffts the number of FFTs that will be processed in one subintegration (set in this function)
 * @param antenna the name of each antenna (set in this function)
 * @param antFiles the filename for each antenna's raw data (set in this function)
 * @param delays the polynomial delay for each antenna: 2nd order, in seconds, time unit in FFT intervals, +ve delay value goes in the opposite direction to wallclock time [same sense as DiFX delays].  (set in this function)
 * @param antfileoffsets the offset of the start time of each antenna's file from the nominal start of this subintegration (in seconds, set in this function).
 */
void parseConfig(char *configfilename, int &nbit, bool &iscomplex, int &nchan, int &nant, double &lo, double &bandwidth,
		 int &numffts, vector<string>& antenna, vector<string>& antFiles, double ***delays, double ** antfileoffsets) {

  std::ifstream fconfig(configfilename);

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
      iss >> (*antfileoffsets)[iant]; // Error checking needed
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
      *antfileoffsets = new double[nant]; // Alloc memory for antenna file offsets
      anttoread = nant;
      iant = 0;
    } else {
      std::cerr << "Error: Unknown keyword \"" << keyword << "\"" << endl;
    }
  }
}

/**
 * Allocates the space for the raw (packed, quantised) voltage data.
 * @param bytestoread the number of bytes to read in from the file for each antenna.
 * @param antStream the file streams for each antenna
 * @param inputdata the buffers to fill with data from the file
 * @return 0 for success, positive for an error.
 */
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

/**
 * Main function that actually runs the show
 * @return 0 for success, positive for an error.
 */
int main(int argc, char *argv[])
{
  // variables for the test
  char *configfile; /**< The filename of the config file */
  int subintbytes; /**< The number of bytes of raw data to be read in per antenna for one subintegration */
  u8 ** inputdata; /**< the input data [numstations][subintbytes] */
  double ** delays; /**< delay polynomial for each antenna.  delay is in seconds, time is in units of FFT duration */
  double * antfileoffsets; /**< offset from each the nominal start time of the integration for each antenna data file.  
                                In units of seconds. */
  int numchannels; /**< The number of channels that will be produced by the FFT */
  int numantennas; /**< The number of antennas in this correlation */
  int numffts; /**< The number of FFTs to be processed in this subintegration */
  int nbit; /**< The number of bits per sample of the quantised data */
  double lo; /**< The local oscillator frequency, in Hz */
  double bandwidth; /**< The bandwidth, in Hz */
  bool iscomplex; /**< Is the data complex or not */
  vector<string> antennas; /**< the names of the antennas */
  vector<string> antFiles; /**< the data files for each antenna */
  vector<std::ifstream *> antStream; /**< a file stream for each antenna */
  int i, status;

  // check invocation, get the config file name
  if (argc!=2) {
    cout << "Usage:  testfxkernel <config>\n" << endl;
    exit(1);
  }
  configfile = argv[1];

  // load up the test input data and delays from the configfile
  parseConfig(configfile, nbit, iscomplex, numchannels, numantennas, lo, bandwidth, numffts, antennas, antFiles, &delays, &antfileoffsets);

  // spit out a little bit of info
  cout << "Got COMPLEX " << iscomplex << endl;
  cout << "Got NBIT " << nbit << endl;
  cout << "Got NCHAN " << numchannels << endl;
  cout << "Got LO " << lo << endl;
  cout << "Got BANDWIDTH " << bandwidth << endl;
  cout << "Got NUMFFTS " << numffts << endl;
  cout << "Got NANT " << numantennas << endl;
  for (int i=0;i<numantennas;i++) {
    cout << "  " << antennas[i] << ":" << antFiles[i] << endl;
  }
    
  // Allocate space in the buffers for the data and the delays
  allocData(&inputdata, numantennas, numchannels, numffts, nbit, subintbytes);

  //openFiles(antennas, antFiles, antStream);
  for (int i=0; i<numantennas; i++) {
    antStream.push_back(new std::ifstream(antFiles[i].c_str(), std::ios::binary));
  }

  // create the FxKernel
  // We could also create multiple FxKernels to test parallelisation in a simple/lazy way
  FxKernel fxkernel = FxKernel(numantennas, numchannels, numffts, nbit, lo, bandwidth);

  // Give the fxkernel its pointer to the input data
  fxkernel.setInputData(inputdata);

  // Read in the voltage data: just one loop for now
  status = readdata(subintbytes, antStream, inputdata);
  if (status) exit(1);

  // Set the delays
  fxkernel.setDelays(delays, antfileoffsets);

  // Checkpoint for timing
  auto starttime = std::chrono::high_resolution_clock::now();
  std::time_t time_now_t = std::chrono::system_clock::to_time_t(starttime);
  string starttimestring = std::ctime(&time_now_t);
  starttimestring.pop_back();
  
  // Run the processing
  fxkernel.process();

  // Calculate the elapsed time
  auto diff = std::chrono::high_resolution_clock::now() - starttime;
  auto t1 = std::chrono::duration_cast<std::chrono::nanoseconds>(diff);

  // Save the visibilities into a dumb ascii file
  fxkernel.saveVisibilities("vis.out", t1.count(), starttimestring);

  // Free memory
  for (i=0; i<numantennas; i++)
  {
    delete(inputdata[i]);
    delete(delays[i]);
  }
  delete(inputdata);
  delete(delays);
}
