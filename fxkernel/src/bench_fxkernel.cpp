#include <chrono>  // for high_resolution_clock
#include <pthread.h>
#include "fxkernel.h"
#include "common.h"
#include "vectordefs.h"

#include <getopt.h>

/** 
 * @file testfxkernel.cpp
 * @brief A test harness for fxkernel
 * @see FxKernel
 *   
 * This test harness reads a brief and simple config file, then creates a single
 * FxKernel object and sets it up with the input data and delays, and then
 * runs process (to process a single subintegration).  Timing information is gathered
 * and written out along with the visibilities from this sub-integration.
 */

/**
 * Main function that actually runs the show
 * @return 0 for success, positive for an error.
 */


#define MAXTHREAD  64


#define SEED 48573


u8 ** inputdata;  // Unpacked voltage data. Global as each thread will grab a copy
int subintbytes;  // # Bytes/antenna in above array
int numantennas;  // Number of antennas
int nbit;         // The number of bits per sample of the quantised data
int numffts;      // The number of FFTs to be processed in this subintegration
int numchannels;  // The number of channels that will be produced by the FFT
double lo;        // The local oscillator frequency, in Hz
double bandwidth; // The bandwidth, in Hz
double **delays;    // Delay polynomial for each antenna.  delay is in seconds, time is in units of FFT duration
double * antfileoffsets; // Not used
int nloop = 10;   //  Number of times to loop

pthread_mutex_t childready_mutex[MAXTHREAD];
pthread_mutex_t start_mutex = PTHREAD_MUTEX_INITIALIZER;


void  initData(u8 **inputdata, int numantennas, int nbit, int subintbytes, bool iscomplex) {
  u8 *buf=NULL;
  int pRandGaussStateSize;
  ippsRandGaussGetSize_8u(&pRandGaussStateSize);
  IppsRandGaussState_8u *pRandGaussState = (IppsRandGaussState_8u *)ippsMalloc_8u(pRandGaussStateSize);
  ippsRandGaussInit_8u(pRandGaussState, 127, 10, SEED);

  if (nbit==2) {
    buf = ippsMalloc_8u(subintbytes*4); // 4 samples per byte
  }

  for (int n=0; n<numantennas; n++) {
    if (nbit==8) 
      ippsRandGauss_8u(inputdata[n], subintbytes, pRandGaussState);
    else {
#define MAXPOS (127+10)
#define MAXNEG (127-10)
#define MEAN 127
      
      ippsRandGauss_8u(buf, subintbytes*4, pRandGaussState);
      for (int i=0; i<subintbytes; i++) {
	u8 byte[4];
	for (int j=0; j<4; j++) {
	  u8 x = buf[i*4+j];
	  if (x >= MAXPOS)
	    byte[j] = 3;
	  else if (x <= MAXNEG)
	    byte[j] = 0;
	  else if (x > MEAN)
	    byte[j] = 2;
	  else
	    byte[j] = 1;
	}
	inputdata[n][i] = byte[0] | byte[1]<<2 | byte[2]<<4 | byte[3]<<6;
      }
    }
  }

  if (buf!=NULL) ippsFree(buf);
  ippsFree(pRandGaussState);
}

void *fxbench(void *arg) {
  int tId;

  tId = *((int*)arg);

  // Make a copy if the input data

  u8 **data = (u8**)malloc(sizeof(u8*)*numantennas);
  for (int i=0; i<numantennas; i++) {
    data[i] = vectorAlloc_u8(subintbytes);
    vectorCopy_u8(inputdata[i], data[i], subintbytes);
  }

  // create the FxKernel
  FxKernel fxkernel = FxKernel(numantennas, numchannels, numffts, nbit, lo, bandwidth);

  // Give the fxkernel its pointer to the input data
  fxkernel.setInputData(data);

  // Set the delays
  fxkernel.setDelays(delays, antfileoffsets);

  // Wait until other threads are setup

  // Tell parent we are ready
  pthread_mutex_unlock(&childready_mutex[tId]);

  // Wait until parent tells us to go 
  pthread_mutex_lock(&start_mutex);
  pthread_mutex_unlock(&start_mutex);

  for (int i=0; i<nloop; i++) {
    // Run the processing
    fxkernel.process();
  }

  for (int i=0; i<numantennas; i++) {
    vectorFree(data[i]);
  }

  pthread_exit(NULL);
}

int main(int argc, char *argv[])
{
  int tid[MAXTHREAD];
  char *configfile; /**< The filename of the config file */
  int nPol; /**< The number of polarisations in the data (1 or 2) */
  bool iscomplex; /**< Is the data complex or not */
  vector<string> antennas; /**< the names of the antennas */
  vector<string> antFiles; /**< the data files for each antenna */
  vector<std::ifstream *> antStream; /**< a file stream for each antenna */
  int i, status, tmp, s;
  pthread_t fxthread[MAXTHREAD];
  int nthread = 1;


  struct option options[] = {
    {"nloop", 1, 0, 'n'},
    {"threads", 1, 0, 't'},
    {"help", 0, 0, 'h'},
    {0, 0, 0, 0}
  };

#define CASEINT(ch,var)                                     \
  case ch:                                                  \
    s = sscanf(optarg, "%d", &tmp);                         \
    if (s!=1)                                               \
      fprintf(stderr, "Bad %s option %s\n", #var, optarg);  \
    else                                                    \
      var = tmp;                                            \
    break
  
  while (1) {
    int opt = getopt_long_only(argc, argv, "n:t:", options, NULL);
    if (opt==EOF) break;

    switch (opt) {
    
      CASEINT('n', nloop);
      CASEINT('t', nthread);

  case 'h':
      printf("Usage: bench_fxkernel [options] <config> \n");
      printf("  -n/-nloop <n>      Number of loops to process\n");
      printf("  -t/-threads <t>    Number threads to launch\n");
      return(1);
      break;

    case '?':
    default:
      break;
    }
  }



      
  // check invocation, get the config file name
  if (argc != optind+1) {
    cout << "Usage:  bench_fxkernel <config>\n" << endl;
    exit(1);
  }
  configfile = argv[optind];

  // load up the test input data and delays from the configfile
  parseConfig(configfile, nbit, nPol, iscomplex, numchannels, numantennas, lo, bandwidth, numffts, antennas, antFiles, &delays, &antfileoffsets);

  // spit out a little bit of info
  cout << "Got COMPLEX " << iscomplex << endl;
  cout << "Got NBIT " << nbit << endl;
  cout << "Got NPOL " << nPol << endl;
  cout << "Got NCHAN " << numchannels << endl;
  cout << "Got LO " << lo << endl;
  cout << "Got BANDWIDTH " << bandwidth << endl;
  cout << "Got NUMFFTS " << numffts << endl;
  cout << "Got NANT " << numantennas << endl;

  float sampletime = 1.0/(2.0*bandwidth);
  int fftchannels = numchannels*2;
  if (iscomplex) {
    sampletime *= 2;
    fftchannels /= 2;
  }
  float subintTime = sampletime*fftchannels*numffts*1000.0;
  
  cout << "Subint time is " << subintTime << " msec" << std::endl;
  cout << "Processing " << subintTime*nloop*nthread/1000 << " sec " << endl;
  
  // Allocate space in the buffers for the data and the delays
  allocDataHost(&inputdata, numantennas, numchannels, numffts, nbit, nPol, iscomplex, subintbytes);

  cout << "Initialising data to random values" << endl;
  initData(inputdata, numantennas, nbit, subintbytes, iscomplex);

  cout << "Launching Threads" << endl;
  pthread_mutex_lock(&start_mutex);

  for (i=0; i<nthread; i++) {
    tid[i] = i;
    pthread_mutex_init(&childready_mutex[i], NULL);
    pthread_mutex_lock(&childready_mutex[i]);

    status = pthread_create(&fxthread[i], NULL, fxbench, (void *)&tid[i]);
    if (status) {
      cout << "Failed to start fftbench thread " << i << "(" << status << ")" << endl;
      exit(1);    
    }
  }

  // Wait until all threads are ready
  for (i=0; i<nthread; i++) {
    pthread_mutex_lock(&childready_mutex[i]);
  }

  // Start threads
  pthread_mutex_unlock(&start_mutex);
  cout << "Go" << endl;
  
  // Checkpoint for timing
  auto starttime = std::chrono::high_resolution_clock::now();
  std::time_t time_now_t = std::chrono::system_clock::to_time_t(std::chrono::system_clock::now());
  string starttimestring = std::ctime(&time_now_t);
  starttimestring.pop_back();

  // Wait on all threads"
  for (i=0; i<nthread; i++) {
    status = pthread_join(fxthread[i], NULL);
    if (status) {
      printf("Error waiting for thread %d (%d)\n", i, status);
      perror("");
    }
  }
  
  // Calculate the elapsed time
  auto diff = std::chrono::high_resolution_clock::now() - starttime;
  auto t1 = std::chrono::duration_cast<std::chrono::milliseconds>(diff);
  auto f_secs = std::chrono::duration_cast<std::chrono::duration<float>>(diff);

  std::cout << "Run time was " << t1.count() << " milliseconds" << endl;

  uint64_t totalBits = (uint64_t)subintbytes * nloop * nthread * 8;

  std::cout << "    " << totalBits/f_secs.count()/1e6 << " Mbps" << endl;

  // Free memory
  for (i=0; i<numantennas; i++)
  {
    delete(inputdata[i]);
    delete(delays[i]);
  }
  delete(inputdata);
  delete(delays);
}
