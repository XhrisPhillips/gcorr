#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif

#include "ingest.h"

void usage()
{
  fprintf(stdout,
	  "ingest_file - read VDIF format data from a single data file and put the data into a PSRDADA ring buffer\n"
	  "              assume that the file keeps VDIF data frames with headers\n"
	  "              the header of PSRDADA keeps the reference information of the data file\n"
	  "              the data in the PSRDADA ring buffer does not keep VDIF headers\n"	  
	  "\n"
	  "Usage: ingest_file [options]\n"
	  " -a The input file name with full directory name \n"
	  " -b The output ring buffer key\n"
	  " -c The center frequency of the data\n"
	  " -d The reference informaton of the current file, EPOCH_SECOND_DATAFRAME\n"
	  " -e The source information, NAME_RA_DEC\n"
	  " -f The name of the DADA header template\n"
	  " -g The number of data frames in each temp buffer of each stream\n"
	  " -h Show help\n"
	  " -i The number of data frames in each buffer block of each stream\n"
	  " -j Runtime directory\n"
	  );
}

int reset_configuration(conf_t *conf)
{
  return EXIT_SUCCESS;
}

int parse_configuration(int argc, char *argv[], conf_t *conf)
{
  int arg;
  
  while((arg=getopt(argc,argv,"a:b:c:d:e:f:g:hi:j:k:")) != -1) {
    // Parse arguments
    switch(arg) {
    case 'h':
      usage();
      exit(EXIT_FAILURE);
      
    case 'a':	  	  
      if(sscanf(optarg, "%s", conf->file_name) != 1) {
	fprintf(stderr, "INGEST_FILE_ERROR: Could not get input file name from %s, which happens at \"%s\", line [%d], has to abort.\n", optarg, __FILE__, __LINE__);
	usage();	      
	exit(EXIT_FAILURE);
      }
      break;
      
    case 'b':	  	  
      if(sscanf(optarg, "%x", &conf->key) != 1) {
	fprintf(stderr, "INGEST_FILE_ERROR: Could not parse key from %s, which happens at \"%s\", line [%d], has to abort.\n", optarg, __FILE__, __LINE__);
	usage();	      
	exit(EXIT_FAILURE);
      }
      break;
      
    case 'c':	  	  
      if(sscanf(optarg, "%lf", &conf->center_frequency) != 1) {
	fprintf(stderr, "INGEST_FILE_ERROR: Could not parse key from %s, which happens at \"%s\", line [%d], has to abort.\n", optarg, __FILE__, __LINE__);
	usage();	      
	exit(EXIT_FAILURE);
      }
      break;
      
    case 'd':
      if(sscanf(optarg, "%"SCNu64"_%"SCNu64"_%"SCNu64"", &conf->days_from_1970, &conf->seconds_from_epoch, &conf->frame_in_period) != 3) {
	fprintf(stderr, "INGEST_FILE_ERROR: Could not parse reference information from %s, which happens at \"%s\", line [%d], has to abort.\n", optarg, __FILE__, __LINE__);
	usage();	      
	exit(EXIT_FAILURE);
      }
      break;
      
    case 'e':
      if(sscanf(optarg, "%s_%s_%s", conf->source, conf->ra, conf->dec) != 3) {
	fprintf(stderr, "INGEST_FILE_ERROR: Could not parse reference information from %s, which happens at \"%s\", line [%d], has to abort.\n", optarg, __FILE__, __LINE__);
	usage();	      
	exit(EXIT_FAILURE);
      }
      break;
      
    case 'f':
      if(sscanf(optarg, "%s", conf->dada_header_template) != 1) {
	fprintf(stderr, "INGEST_FILE_ERROR: Could not parse DADA header template name from %s, which happens at \"%s\", line [%d], has to abort.\n", optarg, __FILE__, __LINE__);
	usage();	      
	exit(EXIT_FAILURE);
      }
      break;
      
    case 'g':
      if(sscanf(optarg, "%"SCNu64"", &conf->nframe_per_stream_rbuf) != 1) {
	fprintf(stderr, "INGEST_FILE_ERROR: Could not parse number of frames per stream for ring buffer block from %s, which happens at \"%s\", line [%d], has to abort.\n", optarg, __FILE__, __LINE__);
	usage();	      
	exit(EXIT_FAILURE);
      }
      break;
      
    case 'i':
      if(sscanf(optarg, "%"SCNu64"", &conf->nframe_per_stream_tbuf) != 1) {
	fprintf(stderr, "INGEST_FILE_ERROR: Could not parse number of frames per stream for temp buffer from %s, which happens at \"%s\", line [%d], has to abort.\n", optarg, __FILE__, __LINE__);
	usage();	      
	exit(EXIT_FAILURE);
      }
      break;
      
    case 'j':
      if(sscanf(optarg, "%s", conf->runtime_directory) != 1) {
	fprintf(stderr, "INGEST_FILE_ERROR: Could not parse directory from %s, which happens at \"%s\", line [%d], has to abort.\n", optarg, __FILE__, __LINE__);
	usage();	      
	exit(EXIT_FAILURE);
      }
      break;	  
    }
  }
  
  return EXIT_SUCCESS;
}


int verify_configuration(conf_t *conf)
{
  return EXIT_SUCCESS;
}

int initialize_log(conf_t *conf)
{
  char log_fname[MAX_STRLEN] = {'\0'};
  
  DIR* dir = opendir(conf->runtime_directory); // First to check if the directory exists
  if(dir)
    closedir(dir);
  else{
    fprintf(stderr, "INGEST_FILE_ERROR: Failed to open %s with opendir or it does not exist, which happens at which happens at \"%s\", line [%d], has to abort\n", conf->runtime_directory, __FILE__, __LINE__);
    exit(EXIT_FAILURE);
  }
  sprintf(log_fname, "%s/ingest_file.log", conf->runtime_directory);  // Open the log file
  conf->log_file = log_open(log_fname, "ab+");
  if(conf->log_file == NULL){
    fprintf(stderr, "INGEST_FILE_ERROR: Can not open log file %s, which happends at \"%s\", line [%d], has to abort\n", log_fname, __FILE__, __LINE__);
    exit(EXIT_FAILURE);
  }
  log_add(conf->log_file, "INFO", 1,  "INGEST_FILE START");

  return EXIT_SUCCESS;
}

int decode_reference_time(conf_t *conf)
{
  time_t seconds_from_1970;
  
  seconds_from_1970 = floor(FRAME_RESOLUTION*conf->frame_in_period) + conf->seconds_from_epoch + SECDAY*conf->days_from_1970;
  conf->picoseconds0 = 1E12*(FRAME_RESOLUTION*conf->frame_in_period - floor(FRAME_RESOLUTION*conf->frame_in_period));
  conf->mjd0 = seconds_from_1970 / SECDAY + MJD1970;                           // Float MJD start time without fraction second
  strftime (conf->utc0, MAX_STRLEN, DADA_TIMESTR, gmtime(&seconds_from_1970)); // String start time without fraction second
  
  return EXIT_SUCCESS;
}

int initialize_hdu_write(conf_t *conf)
{
  conf->hdu = dada_hdu_create(NULL);
  dada_hdu_set_key(conf->hdu, conf->key);
  if(dada_hdu_connect(conf->hdu) < 0) { 
    log_add(conf->log_file, "ERR", 1,  "Can not connect to hdu, which happens at \"%s\", line [%d], has to abort", __FILE__, __LINE__);
    fprintf(stderr, "INGEST_FILE_ERROR: Can not connect to hdu, which happens at \"%s\", line [%d], has to abort.\n", __FILE__, __LINE__);
    
    dada_hdu_destroy(conf->hdu);
    
    log_close(conf->log_file);
    exit(EXIT_FAILURE);    
  }
  
  if(dada_hdu_lock_write(conf->hdu) < 0) {
    log_add(conf->log_file, "ERR", 1,  "Error locking HDU, which happens at \"%s\", line [%d], has to abort", __FILE__, __LINE__);
    fprintf(stderr, "INGEST_FILE_ERROR: Error locking HDU, which happens at \"%s\", line [%d], has to abort.\n", __FILE__, __LINE__);
    
    dada_hdu_unlock_write(conf->hdu);
    dada_hdu_destroy(conf->hdu);
    
    log_close(conf->log_file);
    exit(EXIT_FAILURE);
  }
  
  conf->data_block   = (ipcbuf_t *)(conf->hdu->data_block);
  conf->header_block = (ipcbuf_t *)(conf->hdu->header_block);
  
  return EXIT_SUCCESS;
}

int register_dada_header(conf_t *conf)
{
  char *hdr_buf = NULL;
  hdr_buf = ipcbuf_get_next_write(conf->header_block);
  if(!hdr_buf){
    log_add(conf->log_file, "ERR", 1,  "Error getting header_buf, which happens at \"%s\", line [%d], has to abort", __FILE__, __LINE__);
    fprintf(stderr, "INGEST_FILE_ERROR: Error getting header_buf, which happens at \"%s\", line [%d], has to abort.\n", __FILE__, __LINE__);
    
    destroy_hdu_write(*conf);
    log_close(conf->log_file);
    exit(EXIT_FAILURE);
  }
  if(!conf->dada_header_template){
    log_add(conf->log_file, "ERR", 1,  "Please specify header file, which happens at \"%s\", line [%d], has to abort", __FILE__, __LINE__);
    fprintf(stderr, "INGEST_FILE_ERROR: Please specify header file, which happens at \"%s\", line [%d], has to abort.\n", __FILE__, __LINE__);
    
    destroy_hdu_write(*conf);
    log_close(conf->log_file);
    exit(EXIT_FAILURE);
  }  
  if(fileread(conf->dada_header_template, hdr_buf, DADA_DEFAULT_HEADER_SIZE) < 0){
    log_add(conf->log_file, "ERR", 1,  "Error reading header file, which happens at \"%s\", line [%d], has to abort", __FILE__, __LINE__);
    fprintf(stderr, "INGEST_FILE_ERROR: Error reading header file, which happens at \"%s\", line [%d], has to abort.\n", __FILE__, __LINE__);
    
    destroy_hdu_write(*conf);
    log_close(conf->log_file);
    exit(EXIT_FAILURE);
  }
  
  if(ascii_header_set(hdr_buf, "UTC_START", "%s", conf->utc0) < 0){
    log_add(conf->log_file, "ERR", 1,  "Error setting UTC_START, which happens at \"%s\", line [%d], has to abort", __FILE__, __LINE__);
    fprintf(stderr, "INGEST_FILE_ERROR: Error setting UTC_START, which happens at \"%s\", line [%d], has to abort.\n", __FILE__, __LINE__);
    
    destroy_hdu_write(*conf);
    log_close(conf->log_file);
    exit(EXIT_FAILURE);
  }
  log_add(conf->log_file, "INFO", 1,  "UTC_START to DADA header is %s", conf->utc0);
  
  if(ascii_header_set(hdr_buf, "MJD_START", "%.15f", conf->mjd0) < 0){
    log_add(conf->log_file, "ERR", 1,  "Error setting MJD_START, which happens at \"%s\", line [%d], has to abort", __FILE__, __LINE__);
    fprintf(stderr, "INGEST_FILE_ERROR: Error setting MJD_START, which happens at \"%s\", line [%d], has to abort.\n", __FILE__, __LINE__);
    
    destroy_hdu_write(*conf);
    log_close(conf->log_file);
    exit(EXIT_FAILURE);
  }
  log_add(conf->log_file, "INFO", 1,  "MJD_START to DADA header is %f", conf->mjd0);
  
  if(ascii_header_set(hdr_buf, "PICOSECONDS", "%"PRIu64"", conf->picoseconds0) < 0){
    log_add(conf->log_file, "ERR", 1,  "Error setting PICOSECONDS, which happens at \"%s\", line [%d], has to abort", __FILE__, __LINE__);
    fprintf(stderr, "INGEST_FILE_ERROR: Error setting PICOSECONDS, which happens at \"%s\", line [%d], has to abort.\n", __FILE__, __LINE__);
    
    destroy_hdu_write(*conf);
    log_close(conf->log_file);
    exit(EXIT_FAILURE);
  }
  log_add(conf->log_file, "INFO", 1,  "PICOSECONDS to DADA header is %"PRIu64"", conf->picoseconds0);
    
  /* donot set header parameters anymore */
  if(ipcbuf_mark_filled(conf->header_block, DADA_DEFAULT_HEADER_SIZE) < 0)
    {
      log_add(conf->log_file, "ERR", 1,  "Error header_fill, which happens at \"%s\", line [%d], has to abort", __FILE__, __LINE__);
      fprintf(stderr, "INGEST_FILE_ERROR: Error header_fill, which happens at \"%s\", line [%d], has to abort.\n", __FILE__, __LINE__);

      destroy_hdu_write(*conf);
      log_close(conf->log_file);
      exit(EXIT_FAILURE);
    }

  return EXIT_SUCCESS;
}

int destroy_hdu_write(conf_t conf)
{
  if(conf.data_block){
    dada_hdu_unlock_write(conf.hdu);
    dada_hdu_destroy(conf.hdu); // it has disconnect
  }
  
  return EXIT_SUCCESS;
}
