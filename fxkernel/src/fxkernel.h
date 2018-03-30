#include <ipps.h>
#include "vectordefs.h"

class FxKernel{
public:
  FxKernel(int nant, int nchan, int nfft, double localosc, double bw);
  ~FxKernel();
  void setInputData(char ** idata);
  void setDelays(double ** d);
  void process();

private:
  /* Method to unpack the coarsely quantised input data to complex floats */
  void unpack(char * inputdata, cf32 ** unpacked);

  // input data array
  char ** inputdata;

  // unpacked data
  cf32 *** unpacked;

  // output data array
  cf32 *** visibilities;

  // delay polynomial for each antenna
  double ** delays;

  // internal arrays
  cf32 ** rotator1;
  cf32 ** rotator2;
  cf32 *** channelised;
  cf32 ** fracsamp1;
  cf32 ** fracsamp2;

  // other constants
  int numantennas;
  int numchannels;
  int numffts;  // i.e., the length of a subint
  double lo; // in Hz
  double bandwidth; // in Hz
};
