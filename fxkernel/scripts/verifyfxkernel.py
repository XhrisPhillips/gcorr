#!/usr/bin/env python
import os,sys,argparse,pandas
import numpy as np

class Config:
    def __init__(self, f):
        with open(f) as inp:
            lines = inp.readlines()
            self.config = {}
            for line in lines:
                self.parseline(line)

    def parseline(self, line):
        splitline = line.split()
        if len(splitline) == 2:
            keyword = splitline[0]
            value = splitline[1]
            if keyword in ["COMPLEX","NBIT","NPOL","NCHAN","LO","BANDWIDTH","NUMFFTS","NANT"]:
                self.config[keyword] = int(value)
            else:
                self.config[keyword] = value
        elif len(splitline) > 2:
            self.config[line.split()[0]] = line.split()[1:]

class Visibility:
    def __init__(self, f, config):
        with open(f) as inp:
            self.data = pandas.read_csv(inp, header=None, delim_whitespace=True)
            self.config = config

# Main code
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compare output from fxkernel and/or gcorr for numerical correctness")
    parser.add_argument("--config", dest="config", type=str, default="", help="Config file with number of channels, antennas etc")
    parser.add_argument("--maxepsilon", type=float, default=1e-7, help="The maximum deviation of the two files, measured as a fraction of the median visibility")
    parser.add_argument("visfiles", type=str, nargs=2, help="The vis files to compare (a reference and a newly created one")
    args = parser.parse_args()
    
    # Check for file existence
    if not os.path.exists(args.config):
        parser.error("Config file {0} doesn't exist".format(args.config))
    if not os.path.exists(args.visfiles[0]):
        parser.error("Reference dataset {0} doesn't exist".format(visfiles[0]))
    if not os.path.exists(args.visfiles[1]):
        parser.error("Reference dataset {0} doesn't exist".format(visfiles[1]))

    # Create config and visibilty objects
    conf = Config(args.config)
    vis1 = Visibility(args.visfiles[0], conf)
    vis2 = Visibility(args.visfiles[1], conf)

    # Check that the shape matches
    if not vis1.data.shape == vis2.data.shape:
        print("Visibility shapes don't match: {0}, {1}".format(vis1.data.shape, vis2.data.shape))
        sys.exit()

    # Get the difference
    nbaselines = (conf.config["NANT"]*(conf.config["NANT"]-1))//2
    nstokes = conf.config["NPOL"]*conf.config["NPOL"]
    diff = vis1.data - vis2.data
    overallmaxdiff = -1
    failed = False
    for baseline in range(nbaselines):
        for stokes in range(nstokes):
            rediffmax = np.max(diff[1 + (baseline*nstokes + stokes)*4])
            imdiffmax = np.max(diff[1 + (baseline*nstokes + stokes)*4 + 1])
            remedian  = np.median(vis1.data[1 + (baseline*nstokes + stokes)*4])
            immedian  = np.median(vis1.data[1 + (baseline*nstokes + stokes)*4 + 1])
            if rediffmax / remedian > args.maxepsilon or imdiffmax / immedian > args.maxepsilon:
                print("FAIL: Baseline {0} pol product {1} has a max fractional difference of {2} + {3} j".format(baseline, stokes, rediffmax / remedian, imdiffmax / immedian))
                failed = True
            if rediffmax > overallmaxdiff:
                overallmaxdiff = rediffmax
            if imdiffmax > overallmaxdiff:
                overallmaxdiff = imdiffmax
    if not failed:
        print("PASS: maximum absolute difference in real or imag on any baseline was", overallmaxdiff)
