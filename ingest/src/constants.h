#ifdef __cplusplus
extern "C" {
#endif
#ifndef __CONSTANTS_H
#define __CONSTANTS_H

#define MAX_STRLEN 1024
  
#define FULL_BANDWIDTH 128  // MHz
#define BANDWIDTH_PER_STREAM 64 // MHz
#define PERIOD 1
  
#define NAHCN (FULL_BANDWIDTH/BANDWIDTH_PER_STREAM)
#define NPOL  2 // two pols
#define NPOL_PER_STREAM 1
#define NCHAN_PER_STREAM 1
#define NSTREAM (NCHAN*NPOL/(NPOL_PER_STREAM*NCHAN_PER_STREAM))
  
#define NBYTE_PER_DATA  1 // 8-bits per data
#define NDATA_PER_POL   2 // Complex

#define SAMPLE_RESOLUTION 1.0L
#define NSAMPLE_PER_FRAME 2048
#define FRAME_RESOLUTION  (SAMPLE_RESOLUTION * NSAMPLE_PER_FRAME)
#define NFRAME_PER_STREAM_PER_PERIOD (1E6*FULL_BANDWIDTH*PERIOD*NPOL/(NSAMPLE_PER_FRAME*NSTREAM)) // Critical sampled
  
  // VDIF_HEADER_BYTES in vdifio.h

#define SECDAY 86400.0
#define MJD1970 40587.0
  
#endif
#ifdef __cplusplus
} 
#endif
