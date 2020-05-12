#include "common.h"

void allocDataHost(uint8_t ***data, int numantenna, int numchannels, int numffts, int nbit, int nPol, bool iscomplex, int &subintbytes)
{
  int i, cfactor;

  if (iscomplex)
  {
    cfactor = 1;
  }
  else
  {
    cfactor = 2; // If real data FFT size twice size of number of frequecy channels
  }
  
  subintbytes = numchannels*cfactor*(numffts+1)*nbit/8*nPol;
  cout << "Allocating " << subintbytes/1024/1024 << " MB per antenna per subint" << endl;
  cout << "          " << subintbytes * numantenna / 1024 / 1024 << " MB total" << endl;

  *data = new uint8_t*[numantenna];
  for (int a=0; a<numantenna; a++){
#ifdef USING_CUDA

#else
    (*data)[a] =  vectorAlloc_u8(subintbytes);
    if ((*data)[a]==NULL) {
      cerr << "Unable to allocate " << subintbytes << " bytes. Quitting" << endl;
      std::exit(1);
    }
#endif
  }
}

int readdata(int bytestoread, vector<std::ifstream*> &antStream, uint8_t **inputdata) 
{
  for (int i=0; i<antStream.size(); i++) {
    antStream[i]->read((char*)inputdata[i], bytestoread);
    if (! *(antStream[i])) {
      if (antStream[i]->eof())    {
        return(2);
      } else {
        cerr << "Error: Problem reading data" << endl;
        return(1);
      }
    }
  }
  return(0);
}

void parseConfig(char *configfilename, int &nbit, int & nPol, bool &iscomplex, int &nchan, int &nant, double &lo, double &bandwidth,
		 int &numffts, vector<string>& antenna, vector<string>& antFiles, double ***delays, double ** antfileoffsets) 
{
  std::ifstream fconfig(configfilename);

  string line;
  int anttoread = 0;
  int iant = 0;

  //set some defaults
  nPol = 2;
  iscomplex = 0;
  nbit = 2;

  // read the config file
  while (std::getline(fconfig, line))
  {
    std::istringstream iss(line);
    string keyword;
    if (!(iss >> keyword)) {
      cerr << "Error: Could not parse \"" << line << "\"" << endl;
      std::exit(1);
    }
    if (anttoread)
    {
      string thisfile;
      iss >> thisfile;
      antenna.push_back(keyword);
      antFiles.push_back(thisfile);
      (*delays)[iant] = new double[3]; //assume we're going to read a second-order polynomial for each antenna, d = a*t^2 + b*t + c, t in units of FFT windows, d in seconds
      for (int i=0;i<3;i++) {
	iss >> (*delays)[iant][i];  // Error checking needed
      }
      iss >> (*antfileoffsets)[iant]; // Error checking needed
      iant++;
      anttoread--;
    } else if (strcasecmp(keyword.c_str(), "COMPLEX")==0) {
      iss >> iscomplex; // Should error check
    } else if (strcasecmp(keyword.c_str(), "NBIT")==0) {
      iss >> nbit; // Should error check
    } else if (strcasecmp(keyword.c_str(), "NPOL")==0) {
      iss >> nPol; // Should error check
    } else if (strcasecmp(keyword.c_str(), "NCHAN")==0) {
      iss >> nchan; // Should error check
    } else if (strcasecmp(keyword.c_str(), "LO")==0) {
      iss >> lo; // Should error check
    } else if (strcasecmp(keyword.c_str(), "BANDWIDTH")==0) {
      iss >> bandwidth; // Should error check
    } else if (strcasecmp(keyword.c_str(), "NUMFFTS")==0) {
      iss >> numffts; // Should error check
    } else if (strcasecmp(keyword.c_str(), "NANT")==0) {
      iss >> nant; // Should error check
      *delays = new double*[nant]; // Alloc memory for delay buffer
      *antfileoffsets = new double[nant]; // Alloc memory for antenna file offsets
      anttoread = nant;
      iant = 0;
    } else {
      std::cerr << "Error: Unknown keyword \"" << keyword << "\"" << endl;
    }
  }
}
