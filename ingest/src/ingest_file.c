#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif

#include "ingest.h"

// Data rate is about 3.8Gbps

int main(int argc, char *argv[])
{
  /* Initialization */
  int narg = 21;
  conf_t conf;

  if (argc != narg) {
    usage();
    exit(EXIT_FAILURE);
  }
 
  reset_configuration(&conf); // Reset configuration 
  parse_configuration(argc, argv, &conf); // Parse configuration from command line
  verify_configuration(&conf); // Verify the configuration

  initialize_log(&conf);        // Initialize log interface
  decode_reference_time(&conf); // Get the reference time ready
  initialize_hdu_write(&conf);  // Get the HDU ready
  register_dada_header(&conf);  // Setup header DADA

  ingest_file(&conf);  // Read file and write data to ring buffer
  
  destroy_hdu_write(conf); // Destroy HDU
  
  return EXIT_SUCCESS;
}
