// Some initial pseudocode and thoughts from Adam

// FxKernel will operate on a single subband (dual pol), upper sideband, for the duration of one subintegration
// Suggest we fix to use 2 bit real data, in 2's complement?  i.e., assume that the data itself has headers stripped etc
//   - Ideally we would generate this by stripping out real data from e.g. VDIF files
// Will we use pthread parallelisation to do time division multiplexing like in DiFX, or use something else?
// Suggest that we force the nchan to be a power of 2 to make the striding complex multiplication for phase rotations easy
FxKernel::FxKernel(int nant, int nchan, int nfft, double localosc, double bw)
  : numantennas(nant), numchannels(numchan), numffts(nfft), lo(localosc), bandwidth(bw)
{
  // allocate the various arrays
  unpacked = new float**[nant];
  for(int i=0;i<nant;i++)
  {
    unpacked[i] = new float*[2];
    for(int j=0;j<2;j++)
    {
      unpacked[i][j] = vectorAlloc_cf32(2*nchan);
    }
  }

  // also the fringe rotation arrays, fractional sample correction arrays, etc etc
}

FxKernel::~FxKernel()
{}

FxKernel::setInputData(char ** idata)
{}

FxKernel::setDelays(double ** delays)
{}

FxKernel::process()
{
  // for(number of FFTs)...
  //
  //   unpack
  //
  //   fringe rotate
  //
  //   fractional sample correct
  //
  //   cross multiply + accumultae
}
