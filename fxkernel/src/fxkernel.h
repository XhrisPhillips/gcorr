#include <string>
#include <ipps.h>
#include <ippvm.h>
#include <ippcore.h>
#include "vectordefs.h"

/** 
 * @class FxKernel
 * @brief The class representing a kernel that will process radio interferometry data in an FX fashion to produce visibilities.
 *   
 * FxKernel encapsulates all the processing required to turn sampled voltage data from N datastreams into N*(N-1)/2 baselines
 * of visibility data. It operates on a single upper sideband, dual polarisation, with interleaved data samples where the headers have been
 * stripped out.
 */
class FxKernel{
public:
  /**
   * Constructor for the FxKernel object
   * @param nant the number of antennas (set in this function).
   * @param nchan the number of channels to be produced in the correlation 
   * @param nfft the number of FFTs that will be processed in one subintegration 
   * @param nbit the number of bits per sample
   * @param localosc the local oscillator frequency in Hz 
   * @param bw the bandwidth in Hz 
   */
  FxKernel(int nant, int nchan, int nfft, int nbit, double localosc, double bw);

  /**
   * Destructor for the FxKernel
   */
  ~FxKernel();

  /**
   * Set the input (packed, quantised) data that will be processed
   * @param idata arrays of packed, quantised voltage data (1 per antenna, containing data for the whole subintegration)
   */
  void setInputData(u8 ** idata);

  /**
   * Set the delays and file start time information for each antenna
   * @param d The delay polynomial info (one 2nd order polynomial per antenna
   * @param f The offset in start time for each antenna relative to the start time of the subintegration at the array reference position
   */
  void setDelays(double ** d, double * f);

  /**
   * Process one subintegration, turning voltages into visibilities
   */
  void process();

  /**
   * Accumulate the subintegration results into a visibility vector
   * @param odata visibility arrays [baseline][stokes][channel] that the subintegration results from this FxKernel will be accumulated into
   */
  void accumulate(cf32 *** odata);

  /**
   * Write the subintegration results out into a file
   * @param outfile The output file name
   * @param runtimens The duration that the processing run took, in nanoseconds
   * @param starttimestring A human-readable string that contains the time at which this run was started
   */
  void saveVisibilities(const char * outfile, int runtimens, std::string starttimestring);

private:
  /**
   * Method to unpack the coarsely quantised input data to complex floats
   * @param inputdata an array of packed voltage data per station
   * @param unpacked an array of unpacked voltage data; complex float with 32 bit real and 32 bit imaginary
   * @param offset the requested offset into the packed data from the start time of the data in number of samples
   */
  void unpack(u8 * inputdata, cf32 ** unpacked, int offset);

  /**
   * Method to get the station delay information for a given station for a given FFT
   * @param antenna the current antenna being processed
   * @param fftindex index of FFT within one subint you want to process
   * @param meandelay the required time delay at the midpoint of the FFT interval in seconds
   * @param a gradient of delay with time; seconds per FFT interval
   * @param b delay in seconds at the start of FFT interval
   */
  void getStationDelay(int antenna, int fftindex, double & meandelay, double & a, double & b);

  /**
   * Method to fringe rotate the unpacked data in place
   * @param unpacked array of unpacked voltage data (complex 32bit float)
   * @param a gradient of delay with time; seconds per FFT interval
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

