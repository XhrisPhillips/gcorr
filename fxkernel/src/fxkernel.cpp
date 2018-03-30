// Some initial pseudocode and thoughts from Adam
#include "fxkernel.h"
#include "math.h"
#include <stdio.h>

// FxKernel will operate on a single subband (dual pol), upper sideband, for the duration of one subintegration
// Suggest we fix to use 2 bit real data, in 2's complement?  i.e., assume that the data itself has headers stripped etc
//   - Ideally we would generate this by stripping out real data from e.g. VDIF files
// Will we use pthread parallelisation to do time division multiplexing like in DiFX, or use something else? Or simply run multiple, 
// completely independent instances of FxKernel (probably easier)?
// Suggest that we force the nchan to be a power of 2 to make the striding complex multiplication for phase rotations easy
// I'm also suggesting that we unpack directly to a complex array, to make the subsequent fringe rotation easier
FxKernel::FxKernel(int nant, int nchan, int nfft, double localosc, double bw)
  : numantennas(nant), numchannels(nchan), numffts(nfft), lo(localosc), bandwidth(bw)
{
  // Figure out the array stride size
  stridesize = (int)sqrt(nchan);
  if(stridesize*stridesize != nchan)
  {
    printf("Please choose a number of channels that is a square\n");
    exit(1);
  }

  // allocate the unpacked array
  unpacked = new cf32**[nant];
  for(int i=0;i<nant;i++)
  {
    unpacked[i] = new cf32*[2];
    for(int j=0;j<2;j++)
    {
      unpacked[i][j] = vectorAlloc_cf32(2*nchan);
    }
  }

  //allocate the arrays for holding the fringe rotation vectors
  subtoff  = vectorAlloc_f64(stridesize);
  subtval  = vectorAlloc_f64(stridesize);
  subxoff  = vectorAlloc_f64(stridesize);
  subxval  = vectorAlloc_f64(stridesize);
  subphase = vectorAlloc_f64(stridesize);
  subarg   = vectorAlloc_f32(stridesize);
  subsin   = vectorAlloc_f32(stridesize);
  subcos   = vectorAlloc_f32(stridesize);
  steptoff  = vectorAlloc_f64(stridesize);
  steptval  = vectorAlloc_f64(stridesize);
  stepxoff  = vectorAlloc_f64(stridesize);
  stepxval  = vectorAlloc_f64(stridesize);
  stepphase = vectorAlloc_f64(stridesize);
  steparg   = vectorAlloc_f32(stridesize);
  stepsin   = vectorAlloc_f32(stridesize);
  stepcos   = vectorAlloc_f32(stridesize);
  stepcplx  = vectorAlloc_cf32(stridesize);
  complexrotator = vectorAlloc_cf32(2*nchan);

  // also the FFT'd array, fractional sample correction arrays, etc etc
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

  vectorFree(subtoff);
  vectorFree(subtval);
  vectorFree(subxoff);
  vectorFree(subxval);
  vectorFree(subphase);
  vectorFree(subarg);
  vectorFree(subsin);
  vectorFree(subcos);
  vectorFree(steptoff);
  vectorFree(steptval);
  vectorFree(stepxoff);
  vectorFree(stepxval);
  vectorFree(stepphase);
  vectorFree(steparg);
  vectorFree(stepsin);
  vectorFree(stepcos);
  vectorFree(stepcplx);
  vectorFree(complexrotator);
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
    // do station-based processing for each antenna in turn
    for(int j=0;j<numantennas;j++)
    {
      // unpack
      unpack(inputdata[j], unpacked[j]);
  
      // fringe rotate
      fringerotate(unpacked[j], delays[j][i], delays[j][i+1]);

      // Channelise
    
      // Fractional sample correct
      
    }

    // then do the baseline based processing
    for(int j=0;j<numantennas-1;j++)
    {
      for(int k=j+1;k<numantennas;k++)
      {
        // cross multiply + accumultae

      }
    }
  }
}

void FxKernel::unpack(char * inputdata, cf32 ** unpacked)
{}

void FxKernel::fringerotate(cf32 ** unpacked, f64 delay1, f64 delay2)
{}
