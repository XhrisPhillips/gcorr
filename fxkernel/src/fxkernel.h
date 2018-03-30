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

  /* Method to fringe rotate the unpacked data in place */
  void fringerotate(cf32 ** unpacked, f64 delay1, f64 delay2);

  // input data array
  char ** inputdata;

  // unpacked data
  cf32 *** unpacked;

  // output data array
  cf32 *** visibilities;

  // delay polynomial for each antenna
  double ** delays;

  // internal arrays
  f64 * subtoff;
  f64 * subtval;
  f64 * subxoff;
  f64 * subxval;
  f64 * subphase;
  f32 * subarg;
  f32 * subsin;
  f32 * subcos;

  f64 * steptoff;
  f64 * steptval;
  f64 * stepxoff;
  f64 * stepxval;
  f64 * stepphase;
  f32 * steparg;
  f32 * stepsin;
  f32 * stepcos;
  cf32 * stepcplx;
  cf32 * complexrotator;
  cf32 *** channelised;
  cf32 ** fracsamp1;
  cf32 ** fracsamp2;

  // other constants
  int numantennas;
  int numchannels;
  int numffts;  // i.e., the length of a subint
  int stridesize; // used for the time-saving complex multiplications
  double lo; // in Hz
  double bandwidth; // in Hz
};
