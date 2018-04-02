#include <ipps.h>
#include <ippvm.h>
#include <ippcore.h>
#include "vectordefs.h"

class FxKernel{
public:
  FxKernel(int nant, int nchan, int nfft, double localosc, double bw);
  ~FxKernel();
  void setInputData(u8 ** idata);
  void setDelays(double ** d);
  void process();

private:
  /* Method to unpack the coarsely quantised input data to complex floats */
  void unpack(u8 * inputdata, cf32 ** unpacked, int offset);

  /* Method to fringe rotate the unpacked data in place */
  void fringerotate(cf32 ** unpacked, f64 delay1, f64 delay2);

  /* Method to channelised (FFT) the data, not in place */
  void dofft(cf32 ** unpacked, cf32 ** channelised);

  /* Method to calculate complex conjugate of thje channelised data */
  void conjChannels(cf32 ** channelised, cf32 ** conjchannels);
  
  // input data array
  u8 ** inputdata;

  // unpacked data
  cf32 *** unpacked;

  // channelised data, and conjugated values
  cf32 *** channelised;
  cf32 *** conjchannels;

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
  f32 * subchannelfreqs;

  f64 * steptoff;
  f64 * steptval;
  f64 * stepxoff;
  f64 * stepxval;
  f64 * stepphase;
  f32 * steparg;
  f32 * stepsin;
  f32 * stepcos;
  f32 * stepchannelfreqs;
  cf32 * stepcplx;
  cf32 * complexrotator;
  cf32 ** fracsamp1;
  cf32 ** fracsamp2;

  // FFTs
  u8 * fftbuffer;
  vecFFTSpecC_cf32 * pFFTSpecC;


  // other constants
  int numantennas;
  int numchannels;
  int numffts;  // i.e., the length of a subint
  int fftchannels; //2*nchan, since we're assuming real data
  int stridesize; // used for the time-saving complex multiplications
  double lofreq; // in Hz
  double bandwidth; // in Hz
  double sampletime; //in seconds
  bool fractionalLoFreq; //if true, means we need to do an extra multiplication in fringe rotation phase calculation
  bool iscomplex;  // Is the original data real or complex voltages
  int cfact;
};
