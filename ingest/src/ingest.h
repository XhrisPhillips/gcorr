#ifdef __cplusplus
extern "C" {
#endif
#ifndef __INGEST_H
#define __INGEST_H

#include <dirent.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
  
#include "vdifio.h"
  
#include "dada_hdu.h"
#include "dada_def.h"
#include "ipcio.h"
#include "ascii_header.h"
#include "daemon.h"
#include "futils.h"
  
#include "constants.h"
#include "log.h"
  
  typedef struct conf_t
  {
    FILE *log_file;
    char file_name[MAX_STRLEN];
    key_t key;
    double center_frequency;

    dada_hdu_t *hdu;
    ipcbuf_t *data_block, *header_block;
    
    char source[MAX_STRLEN];
    char ra[MAX_STRLEN];
    char dec[MAX_STRLEN];
    
    int affinity;
    int cpu;
    char dada_header_template[MAX_STRLEN];
    
    uint64_t nframe_per_stream_rbuf;
    
    char runtime_directory[MAX_STRLEN];
    
    uint64_t picoseconds0;
    
    double bandwidth;
    double frequecy_resolution;
    
    double mjd0;
    char utc0[MAX_STRLEN];
    
    uint64_t days_from_1970;
    
    uint64_t seconds_from_epoch;
    uint64_t frame_in_period;

    uint64_t curbuf_size;
  }conf_t;
  
  void usage();
  int reset_configuration(conf_t *conf);
  int parse_configuration(int argc, char *argv[], conf_t *conf);
  int verify_configuration(conf_t *conf);
  int initialize_log(conf_t *conf);
  int decode_reference_time(conf_t *conf);
  int initialize_hdu_write(conf_t *conf);
  int destroy_hdu_write(conf_t conf);
  int register_dada_header(conf_t *conf);
  int ingest_file(conf_t *conf);
  
#endif
#ifdef __cplusplus
} 
#endif
