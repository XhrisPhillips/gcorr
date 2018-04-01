// Some initial pseudocode and thoughts from Adam
#include "fxkernel.h"
#include "math.h"
#include <stdio.h>
#include <cstdlib>
#include <iostream>


// FxKernel will operate on a single subband (dual pol), upper sideband, for the duration of one subintegration
// Suggest we fix to use 2 bit real data, in 2's complement?  i.e., assume that the data itself has headers stripped etc
//   - Ideally we would generate this by stripping out real data from e.g. VDIF files
// Will we use pthread parallelisation to do time division multiplexing like in DiFX, or use something else? Or simply run multiple, 
// completely independent instances of FxKernel (probably easier)?
// Suggest that we force the nchan to be a power of 2 to make the striding complex multiplication for phase rotations easy
// I'm also suggesting that we unpack directly to a complex array, to make the subsequent fringe rotation easier
FxKernel::FxKernel(int nant, int nchan, int nfft, double lo, double bw)
  : numantennas(nant), numchannels(nchan), fftchannels(2*nchan), numffts(nfft), lofreq(lo), bandwidth(bw), sampletime(1.0/(2.0*bw))
{
  iscomplex = 0; // Allow for further generalisation later
  if (iscomplex)
    cfact = 2;
  else
    cfact = 1;
  
  // Figure out the array stride size
  stridesize = (int)sqrt(nchan);
  if(stridesize*stridesize != nchan)
  {
    std::cerr << "Please choose a number of channels that is a square" << std::endl;
    exit(1);
  }

  // check if LO frequency has a fractional component
  fractionalLoFreq = false;
  if(lofreq - int(lofreq) > TINY)
  {
    fractionalLoFreq = true;
  }

  int fftchannels = cfact*nchan;
  
  // allocate the unpacked array
  unpacked = new cf32**[nant];
  for(int i=0;i<nant;i++)
  {
    unpacked[i] = new cf32*[2];
    for(int j=0;j<2;j++)
    {
      unpacked[i][j] = vectorAlloc_cf32(fftchannels);
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
  subchannelfreqs = vectorAlloc_f32(stridesize);
  stepchannelfreqs = vectorAlloc_f32(stridesize);
  complexrotator = vectorAlloc_cf32(fftchannels);

  // populate the fringe rotation arrays that can be pre-populated
  for(int i=0;i<stridesize;i++) 
  {
    subxoff[i] = (double(i)/double(fftchannels));
    subtoff[i] = i*sampletime;
    stepxoff[i] = double(i*stridesize)/double(fftchannels);
    steptoff[i] = i*stridesize*sampletime;
    subchannelfreqs[i] = (float)((TWO_PI*(i)*bandwidth)/numchannels);
    stepchannelfreqs[i] = (float)((TWO_PI*i*stridesize*bandwidth)/numchannels);
  }

  // Allocate memory for FFT'ed data and initialised FFT
  int order = 0;
  while((fftchannels) >> order != 1)
    order++;

  channelised = new cf32**[nant];
  for(int i=0;i<nant;i++)
  {
    channelised[i] = new cf32*[2];
    for(int j=0;j<2;j++)
    {
      channelised[i][j] = vectorAlloc_cf32(fftchannels);
    }
  }
  
  int sizeFFTSpec, sizeFFTInitBuf, wbufsize;
  u8 *fftInitBuf, *fftSpecBuf;
  ippsFFTGetSize_C_32fc(order, vecFFT_NoReNorm, vecAlgHintFast, &sizeFFTSpec, &sizeFFTInitBuf, &wbufsize);
  fftSpecBuf = ippsMalloc_8u(sizeFFTSpec);
  fftInitBuf = ippsMalloc_8u(sizeFFTInitBuf);
  fftbuffer = ippsMalloc_8u(wbufsize);
  // Initialize
  ippsFFTInit_C_32fc(&pFFTSpecC, order, vecFFT_NoReNorm, vecAlgHintFast, fftSpecBuf, fftInitBuf);
  if (fftInitBuf) ippFree(fftInitBuf);
  
  // also fractional sample correction arrays, etc etc
}

FxKernel::~FxKernel()
{
  //de-allocate the internal arrays
  for(int i=0;i<numantennas;i++)
  {
    for(int j=0;j<2;j++)
    {
      vectorFree(unpacked[i][j]);
      vectorFree(channelised[i][j]);
    }
    delete [] unpacked[i];
    delete [] channelised[i];
  }
  delete [] unpacked;
  delete [] channelised;

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
  vectorFree(subchannelfreqs);
  vectorFree(stepchannelfreqs);
  vectorFree(complexrotator);
}

void FxKernel::setInputData(u8 ** idata)
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
      // Obviously this needs some coarse delay correction to be added! i.e., &(inputdata[j][someoffset])
      unpack(inputdata[j], unpacked[j]);
  
      // fringe rotate
      fringerotate(unpacked[j], delays[j][i], delays[j][i+1]);

      // Channelise
      dofft(unpacked[j], channelised[j]);
    
      // Fractional sample correct
      
    }

    // then do the baseline based processing
    for(int j=0;j<numantennas-1;j++)
    {
      for(int k=j+1;k<numantennas;k++)
      {
        // cross multiply + accumulate

      }
    }
  }
}

void FxKernel::unpack(u8 * inputdata, cf32 ** unpacked)
{}

void FxKernel::fringerotate(cf32 ** unpacked, f64 delay1, f64 delay2)
{
  double a, b; //coefficients for the linear approximation of delay across this FFT
  int integerdelay;
  int status;

  // calculate a and b (should really provide 3 delays to do this, or better yet just provide a and b to the function)
 
  // subtract off any integer delay present
  integerdelay = static_cast<int>(b);
  b -= integerdelay;

  // Fill in the delay values, using a and b and the precomputeed offsets
  status = vectorMulC_f64(subxoff, a, subxval, stridesize);
  if(status != vecNoErr)
    fprintf(stderr, "Error in linearinterpolate, subval multiplication\n");
  status = vectorMulC_f64(stepxoff, a, stepxval, stridesize);
  if(status != vecNoErr)
    fprintf(stderr, "Error in linearinterpolate, stepval multiplication\n");
  status = vectorAddC_f64_I(b, subxval, stridesize);
  if(status != vecNoErr)
    fprintf(stderr, "Error in linearinterpolate, subval addition!!!\n");

  // Turn delay into turns of phase by multiplying by the lo
  status = vectorMulC_f64(subxval, lofreq, subphase, stridesize);
  if(status != vecNoErr)
    fprintf(stderr, "Error in linearinterpolate lofreq sub multiplication!!!\n");
  status = vectorMulC_f64(stepxval, lofreq, stepphase, stridesize);
  if(status != vecNoErr)
    fprintf(stderr, "Error in linearinterpolate lofreq step multiplication!!!\n");
  if(fractionalLoFreq) 
  {
    status = vectorAddC_f64_I((lofreq-int(lofreq))*double(integerdelay), subphase, stridesize);
    if(status != vecNoErr)
      fprintf(stderr, "Error in linearinterpolate lofreq non-integer freq addition!!!\n");
  }

  // Convert turns of phase into radians and bound into [0,2pi), then take sin/cos and assemble rotator vector
  for(int i=0;i<stridesize;i++) {
    subarg[i] = -TWO_PI*(subphase[i] - int(subphase[i]));
    steparg[i] = -TWO_PI*(stepphase[i] - int(stepphase[i]));
  }
  status = vectorSinCos_f32(subarg, subsin, subcos, stridesize);
  if(status != vecNoErr)
    fprintf(stderr, "Error in sin/cos of sub rotate argument!!!\n");
  status = vectorSinCos_f32(steparg, stepsin, stepcos, stridesize);
  if(status != vecNoErr)
    fprintf(stderr, "Error in sin/cos of step rotate argument!!!\n");
  status = vectorRealToComplex_f32(subcos, subsin, complexrotator, stridesize);
  if(status != vecNoErr)
    fprintf(stderr, "Error assembling sub into complex!!!\n");
  status = vectorRealToComplex_f32(stepcos, stepsin, stepcplx, stridesize);
  if(status != vecNoErr)
    fprintf(stderr, "Error assembling step into complex!!!\n");
  for(int i=1;i<stridesize;i++) {
    status = vectorMulC_cf32(complexrotator, stepcplx[i], &complexrotator[i*stridesize], stridesize);
    if(status != vecNoErr)
      fprintf(stderr, "Error doing the time-saving complex multiplication!!!\n");
  }
}

void FxKernel::dofft(cf32 ** unpacked, cf32 ** channelised) {
  // Do a single FFT on the 2 pols for a single antenna
  vecStatus status;
  
  
  for (int i=0; i<2; i++) {
    status = vectorFFT_CtoC_cf32(unpacked[i], channelised[i], pFFTSpecC, fftbuffer);
    if(status != vecNoErr) {
      std::cerr << "Error calling FFT" << std::endl;
      exit(1);
    }
  }


}
