#include <string>
#include <ipps.h>
#include <ippvm.h>
#include <ippcore.h>
#include "vectordefs.h"

class FxKernel{
public:
  FxKernel(int nant, int nchan, int nfft, int numbits, double localosc, double bw);
  ~FxKernel();
  void setInputData(u8 ** idata);
  void setDelays(double ** d, double * f);
  void process();
  void accumulate(cf32 *** odata);
  void saveVisibilities(const char * outfile, int runtimens, std::string starttimestring);

private:
  /**
   * Method to unpack the coarsely quantised input data to complex floats
   * @param inputdata an array of packed voltage data of length 1 byte 
   * @param unpacked an array of unpacked voltage data; complex float with 32 bit real and 32 bit imaginary
   * @param offset the offset from the start time of the data in number of samples
   */
  void unpack(u8 * inputdata, cf32 ** unpacked, int offset);

  /**
   * Method to get the station delay for a given station for a given FFT
   * @param antenna the current antenna being processed
   * @param fftindex index of FFT within one subint you want to process
   * @param meandelay the required time delay at the midpoint of the FFT interval in seconds
   * @param a gradiant of delay with time; seconds per FFT interval
   * @param b delay in seconds at the start of FFT interval
   */
  void getStationDelay(int antenna, int fftindex, double & meandelay, double & a, double & b);

  /**
   * Method to fringe rotate the unpacked data in place
   * @param unpacked array of unpacked voltage data (complex 32bit float)
   * @param a gradiant of delay with time; seconds per FFT interval
   * @param b delay in seconds at the start of FFT interval
   */
  void fringerotate(cf32 ** unpacked, f64 a, f64 b);

  /**
   * Method to channelise (FFT) the data, not in place
   * @param unpacked array of unpacked voltage data (complex 32bit float)
   * @param array containing FFTed (channelised) unpacked data array (complex 32bit float)
   */
  void dofft(cf32 ** unpacked, cf32 ** channelised);

  /**
   * Method to calculate complex conjugate of the channelised data
   * @param channelised array containing FFTed (channelised) unpacked data array (complex 32bit float)
   * @param complex conjugate of channelised data array
   */
  void conjChannels(cf32 ** channelised, cf32 ** conjchannels);

  /**
   * Method to correct fractional sample delay of the channelised data in-place
   * @param channelised channelised array containing FFTed (channelised) unpacked data array (complex 32bit float)
   * @param fracdelay resisdual delay between the desired delay and course integer delay correction 
   */
  void fracSampleCorrect(cf32 ** channelised, f64 fracdelay);
  
  u8 ** inputdata; /**< input data array */

  cf32 *** unpacked; /**< unpacked data (fringe rotation is performed in-place here) */

  cf32 *** channelised; /**< channelised data */
  cf32 *** conjchannels; /**< conjugated values */

  // Validity/weights
  bool *antValid; /**< checks if there is good data for the given antenna at a given time (if yes, it will accumulate; if not, if will leave this data out) */
  int *baselineCount; /**< counter incrementing number of baselines with successfully obtained cross correlations and accumulations */
  
  cf32 *** visibilities; /**< output data array */

  double ** delays; /**< delay polynomial for each antenna. Put in the time in units of FFT lengths since start of subintegration, get back delay in seconds. */

  double * filestartoffsets; /**< Offset for each antenna file from the nominal start time of the subintegration. In seconds. */

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
  int numantennas; /**< number of antennas in the dataset */
  int numchannels; /**< the number of channels to create during channelisation; each channel is a unique frequency point */
  int nbits;  /**< Number of bits for voltage samples */
  int numffts;  /**< the number of FFTs computed; i.e., the length of a subint */
  int fftchannels; /**< length of an FFT; 2*nchan for real data, and nchan for complex */
  int nbaselines; /**< Number of baselines (nant*(nant-1)/2) */
  int stridesize; /**< used for the time-saving complex multiplications */
  int substridesize; /**< used for the time-saving complex multiplications. Equal to stridesize for complex data, or 2x stridesize for real data */
  double lofreq; /**< local oscillator frequency; in Hz */
  double bandwidth; /**< in Hz */
  double sampletime; /**< 1/(2*bandwidth); in seconds */
  bool fractionalLoFreq; /**< if true, means we need to do an extra multiplication in fringe rotation phase calculation */
  bool iscomplex;  /**< Is the original data real or complex voltages? */
  int cfact; /**< "complex factor"; either 1 (for real data) or 2 (for complex data); determines length of substridesize (twice the stridesize for real data [i.e. need 2N samples for nchan] and equal to the stridesize for complex data [need N samples for nchan]) */
};

