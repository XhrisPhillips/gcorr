#include <ipps.h>
#include <ippvm.h>
#include <ippcore.h>
#include "vectordefs.h"

class FxKernel{
public:
  FxKernel(int nant, int nchan, int nfft, int numbits, double localosc, double bw);
  ~FxKernel();
  void setInputData(u8 ** idata);
  void setDelays(double ** d);
  void process();
  void accumulate(cf32 *** odata);

private:
  /* Method to unpack the coarsely quantised input data to complex floats */
  void unpack(u8 * inputdata, cf32 ** unpacked, int offset);

  /* Method to get the station delay for a given station for a given FFT */
  void getStationDelay(int antenna, int fftindex, double & meandelay, double a, double b);

  /* Method to fringe rotate the unpacked data in place */
  void fringerotate(cf32 ** unpacked, f64 a, f64 b);

  /* Method to channelised (FFT) the data, not in place */
  void dofft(cf32 ** unpacked, cf32 ** channelised);

  /* Method to calculate complex conjugate of the channelised data */
  void conjChannels(cf32 ** channelised, cf32 ** conjchannels);

  /* Method to correct fractional sample delay of the channelised data in-place */
  void fracSampleCorrect(cf32 ** channelised, f64 fracdelay);
  
  // input data array
  u8 ** inputdata;

  // unpacked data (fringe rotation is performed in-place here)
  cf32 *** unpacked;

  // channelised data, and conjugated values
  cf32 *** channelised;
  cf32 *** conjchannels;

  // output data array
  cf32 *** visibilities;

  // delay polynomial for each antenna.  Referenced to the first sample of the block of data.
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

  f32 * subfracsamparg;
  f32 * subfracsampsin;
  f32 * subfracsampcos;
  f32 * subchannelfreqs;

  f32 * stepfracsamparg;
  f32 * stepfracsampsin;
  f32 * stepfracsampcos;
  cf32 * stepfracsampcplx;
  f32 * stepchannelfreqs;
  cf32 * fracsamprotator;

  // FFTs
  u8 * fftbuffer;
  vecFFTSpecC_cf32 * pFFTSpecC;

  // other constants
  int numantennas;
  int numchannels;
  int nbits;  // Number of bits for voltage samples
  int numffts;  // i.e., the length of a subint
  int fftchannels; // 2*nchan for real data, nchan for complex
  int nbaselines; // Number of baselines (nant*(nant-1)/2)
  int stridesize; // used for the time-saving complex multiplications
  int substridesize; // used for the time-saving complex multiplications.  Equal to stridesize for complex data, or 2x stride size for real data
  double lofreq; // in Hz
  double bandwidth; // in Hz
  double sampletime; //in seconds
  bool fractionalLoFreq; //if true, means we need to do an extra multiplication in fringe rotation phase calculation
  bool iscomplex;  // Is the original data real or complex voltages
  int cfact;
};

