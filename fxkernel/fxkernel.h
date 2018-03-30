class FxKernel{
public:
  FxKernel(int nant, int nchan, int nfft, double localosc, double bw);
  ~FxKernel();
  setInputData(char ** idata);
  setDelays(double ** delays);
  process();

private:
  // input data array
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
}

