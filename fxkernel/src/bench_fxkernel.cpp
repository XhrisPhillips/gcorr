#include <chrono>  // for high_resolution_clock
#include "fxkernel.h"
#include "common.h"
#include "vectordefs.h"

/** 
 * @file testfxkernel.cpp
 * @brief A test harness for fxkernel
 * @see FxKernel
 *   
 * This test harness reads a brief and simple config file, then creates a single
 * FxKernel object and sets it up with the input data and delays, and then
 * runs process (to process a single subintegration).  Timing information is gathered
 * and written out along with the visibilities from this sub-integration.
 */

/**
 * Main function that actually runs the show
 * @return 0 for success, positive for an error.
 */


#define SEED 48573

void  initData(u8 **inputdata, int numantennas, int nbit, int subintbytes, bool iscomplex) {
  u8 *buf=NULL;
  int pRandGaussStateSize;
  ippsRandGaussGetSize_8u(&pRandGaussStateSize);
  IppsRandGaussState_8u *pRandGaussState = (IppsRandGaussState_8u *)ippsMalloc_8u(pRandGaussStateSize);
  ippsRandGaussInit_8u(pRandGaussState, 127, 10, SEED);

  if (nbit==2) {
    buf = ippsMalloc_8u(subintbytes*4); // 4 samples per byte
  }

  
  
  for (int n=0; n<numantennas; n++) {
    if (nbit==8) 
      ippsRandGauss_8u(inputdata[n], subintbytes, pRandGaussState);
    else {
#define MAXPOS (127+10)
#define MAXNEG (127-10)
#define MEAN 127
      
      ippsRandGauss_8u(buf, subintbytes*4, pRandGaussState);
      for (int i=0; i<subintbytes; i++) {
	u8 byte[4];
	for (int j=0; j<4; j++) {
	  u8 x = buf[i*4+j];
	  if (x >= MAXPOS)
	    byte[j] = 3;
	  else if (x <= MAXNEG)
	    byte[j] = 0;
	  else if (x > MEAN)
	    byte[j] = 2;
	  else
	    byte[j] = 1;
	}
	inputdata[n][i] = byte[0] | byte[1]<<2 | byte[2]<<4 | byte[3]<<6;
      }
    }
  }

  if (buf!=NULL) ippsFree(buf);
  ippsFree(pRandGaussState);
}

void *fxbench(void *arg) {
  int tId;

  tId = *((int*)arg);

}

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
  int nPol; /**< The number of polarisations in the data (1 or 2) */
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

  int nloop = 10;

  // load up the test input data and delays from the configfile
  parseConfig(configfile, nbit, nPol, iscomplex, numchannels, numantennas, lo, bandwidth, numffts, antennas, antFiles, &delays, &antfileoffsets);

  // spit out a little bit of info
  cout << "Got COMPLEX " << iscomplex << endl;
  cout << "Got NBIT " << nbit << endl;
  cout << "Got NPOL " << nPol << endl;
  cout << "Got NCHAN " << numchannels << endl;
  cout << "Got LO " << lo << endl;
  cout << "Got BANDWIDTH " << bandwidth << endl;
  cout << "Got NUMFFTS " << numffts << endl;
  cout << "Got NANT " << numantennas << endl;
  for (int i=0;i<numantennas;i++) {
    cout << "  " << antennas[i] << ":" << antFiles[i] << endl;
  }

  float sampletime = 1.0/(2.0*bandwidth);
  int fftchannels = numchannels*2;
  if (iscomplex) {
    sampletime *= 2;
    fftchannels /= 2;
  }
  float subintTime = sampletime*fftchannels*numffts*1000.0;
  
  cout << "Subint time is " << subintTime << " msec" << std::endl;
  cout << "Processing " << subintTime*nloop/1000 << " sec " << endl;
  
  // Allocate space in the buffers for the data and the delays
  allocDataHost(&inputdata, numantennas, numchannels, numffts, nbit, nPol, iscomplex, subintbytes);

  cout << "Initialising data to random values" << endl;
  initData(inputdata, numantennas, nbit, subintbytes, iscomplex);

  // create the FxKernel
  // We could also create multiple FxKernels to test parallelisation in a simple/lazy way
  FxKernel fxkernel = FxKernel(numantennas, numchannels, numffts, nbit, lo, bandwidth);

  // Give the fxkernel its pointer to the input data
  fxkernel.setInputData(inputdata);

  // Set the delays
  fxkernel.setDelays(delays, antfileoffsets);

  // Checkpoint for timing
  auto starttime = std::chrono::high_resolution_clock::now();
  std::time_t time_now_t = std::chrono::system_clock::to_time_t(std::chrono::system_clock::now());
  string starttimestring = std::ctime(&time_now_t);
  starttimestring.pop_back();

  for (int i=0; i<nloop; i++) {
    // Run the processing
    fxkernel.process();
  }
  
  // Calculate the elapsed time
  auto diff = std::chrono::high_resolution_clock::now() - starttime;
  auto t1 = std::chrono::duration_cast<std::chrono::milliseconds>(diff);

  std::cout << "Run time was " << t1.count() << " milliseconds" << endl;

  // Free memory
  for (i=0; i<numantennas; i++)
  {
    delete(inputdata[i]);
    delete(delays[i]);
  }
  delete(inputdata);
  delete(delays);
}
