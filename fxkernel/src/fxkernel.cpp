// Some initial pseudocode and thoughts from Adam
#include "fxkernel.h"

// FxKernel will operate on a single subband (dual pol), upper sideband, for the duration of one subintegration
// Suggest we fix to use 2 bit real data, in 2's complement?  i.e., assume that the data itself has headers stripped etc
//   - Ideally we would generate this by stripping out real data from e.g. VDIF files
// Will we use pthread parallelisation to do time division multiplexing like in DiFX, or use something else?
// Suggest that we force the nchan to be a power of 2 to make the striding complex multiplication for phase rotations easy
// I'm also suggesting that we unpack directly to a complex array, to make the subsequent fringe rotation easier
FxKernel::FxKernel(int nant, int nchan, int nfft, double localosc, double bw)
  : numantennas(nant), numchannels(nchan), numffts(nfft), lo(localosc), bandwidth(bw)
{
  // allocate the various arrays
  unpacked = new cf32**[nant];
  for(int i=0;i<nant;i++)
  {
    unpacked[i] = new cf32*[2];
    for(int j=0;j<2;j++)
    {
      unpacked[i][j] = vectorAlloc_cf32(2*nchan);
    }
  }

  // also the fringe rotation arrays, fractional sample correction arrays, etc etc
}

FxKernel::~FxKernel()
{
  //de-allocate the internal arrays
  for(int i=0;i<numantennas;i++)
  {
    for(int j=0;j<2;j++)
    {
      vectorFree(unpacked[i][j]);
    }
    delete [] unpacked[i];
  }
  delete [] unpacked;


}

void FxKernel::setInputData(char ** idata)
{
  inputdata = idata;
}

void FxKernel::setDelays(double ** d)
{
  delays = d;
}

void FxKernel::process()
{
  // for(number of FFTs)... (parallelised via pthreads?)
  for(int i=0;i<numffts;i++)
  {
    // unpack
    for(int j=0;j<numantennas;j++)
    {
      unpack(inputdata[j], unpacked[j]);
    }
  
    // fringe rotate
    // fractional sample correct
    // cross multiply + accumultae
  }
}

void FxKernel::unpack(char * inputdata, cf32 ** unpacked)
{}
