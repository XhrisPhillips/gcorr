#include <iostream>

#include "fxkernel.h"
#include "vectordefs.h"

void allocData(u8 ***data, int numantenna, int numchannels, int numffts, int nbit, int nsubint) {
  int i, cfactor;

  int iscomplex = 0;
  int nPol = 2;
  if (iscomplex) {
    cfactor = 1;
  } else {
    cfactor = 2; // If real data FFT size twice size of number of frequecy channels
  }
  
  int bytespersubint = numchannels*cfactor*numffts*nbit/8*nPol;
  std::cout << "Allocating " << bytespersubint/1024/1024 << " MB per antenna per subint" << std::endl;
  std::cout << "          " << bytespersubint * numantenna * nsubint / 1024 / 1024 << " MB total" << std::endl;


  *data = new u8*[numantenna];
  for (i=0; i<numantenna; i++) {
    (*data)[i] = new u8[bytespersubint*nsubint];
  }
}


int main(int argc, char *argv[])
{
  // variables for the test
  int i;
  u8 ** inputdata;
  double ** delays;
  int numchannels, numantennas, numffts, nbit, nsubint;
  double lo, bandwidth;

  // load up the test input data from somewhere
  
  // Load up the delays from somewhere

  // Set the inputs we'll use as a test
  // Current values would give a subint of 100ms, which is fairly reasonable
  nbit = 2;
  numchannels = 1024;
  numantennas = 6;
  numffts = 3125;
  lo = 1650000000.0;
  bandwidth = 32000000.0;
  nsubint = 10;

  allocData(&inputdata, numantennas, numchannels, numffts, nbit, nsubint);

  // create the FxKernel
  // We could also create multiple FxKernels to test parallelisation in a simple/lazy way
  FxKernel fxkernel = FxKernel(numantennas, numchannels, numffts, lo, bandwidth);

  for (i=0; i<nsubint; i++) {

    // Set the input data and the delays
    fxkernel.setInputData(inputdata);
    fxkernel.setDelays(delays);

    // Checkpoint for timing
  
    // Run the processing
    fxkernel.process();
  }

  // Calculate the elapsed time

  // Free memory
   for (i=0; i<numantennas; i++) {
     delete(inputdata[i]);
  }
   delete(inputdata);
}
