#!/usr/bin/env python

"""
Script to run bigcat udp2db pipeline with given configurations in YAML file.

Copyright (C) CSIRO 2020
"""

import coloredlogs
import logging
import argparse
import yaml
from os import path
from execute import ExecuteCommand

__author__ = "Xinping Deng <xinping.deng@csiro.au>"

VDIF_HDRSIZE = 32

class PipelineError(Exception):
    pass

class Pipeline(object):
    '''
    Class to define all function and behaviours 
    to run pipeline with different configurations
    '''
    def __init__(self, values):
        self._execution   = values.execution
        yaml_fname        = values.config[0]

        # Get log setup
        self._log = logging.getLogger(__name__)
        #self._log = logging.getLogger(None)
        #self._log = logging.getLogger("root")

        # Get configuration from yaml file
        with open(yaml_fname) as f:
            self._yaml_dict = yaml.load(f, yaml.Loader)
            
        ## Reset at the very beginning
        #self._log.info("Reset at the beginning")
        #self._reset()
        
        # Default we do not have ring buffer
        self._has_db = False
        
        # Record all execution instances
        # just in case we need to kill them all
        self._execution_instances = []
        
        # DB part has no seperate function
        # create ring buffer is part of initial
        self._log.info("Parse 'basic' keys from YAML file")
        
        # Not shared with other functions for sure
        app = self._yaml_dict["basic"]["app"]
        self._log.debug("app is {}".format(app))
        
        # Shared configurations for sure
        self._key    = self._yaml_dict["basic"]["key"]
        self._writer = self._yaml_dict["basic"]["writer"]
        self._reader = self._yaml_dict["basic"]["reader"]
        
        nreader = 1 
        if type(self._reader) is list:
            nreader = len(self._reader) # Not shared for sure
        self._log.debug("reader list is {}".format(self._reader))
        self._log.debug("writer list is {}".format(self._writer))
        self._log.debug("We have {} readers".format(nreader))
                
        # May shared
        self._bandwidth = int(self._yaml_dict["basic"]["bandwidth"])
        self._bits      = int(self._yaml_dict["basic"]["bits"])
        self._nchan     = int(self._yaml_dict["basic"]["nchan"])
        self._nthread   = int(self._yaml_dict["basic"]["nthread"])
        self._framesize = int(self._yaml_dict["basic"]["framesize"])
        
        self._log.debug("bandwidth is {} MHz".format(self._bandwidth))
        self._log.debug("bits per sample is {}".format(self._bits))
        self._log.debug("nchan is {}".format(self._nchan))
        self._log.debug("nthread is {}".format(self._nthread))
        self._log.debug("proposed framesize with vdif header is {} bytes".format(self._framesize))

        # build dada_db command line
        bytes_per_sec = 2E6*self._bits*self._nchan*self._bandwidth/8
        datasize = self._framesize - VDIF_HDRSIZE # Remove header for the calculation latter
        while datasize > 0:
            if(bytes_per_sec%datasize == 0):
                break
            datasize = datasize - 8

        # Figure out nframe from block length
        self._log.debug("1 seconds data per channel/thread per buffer block")
        nframe = int(bytes_per_sec/datasize)
        self._log.debug("{} frames per channel/thread per buffer block".format(nframe))
        
        blksz = self._nthread*nframe*datasize
        command = "{} -k {} -r {} -b {}".format(app,
                                                self._key,
                                                nreader, 
                                                blksz)
        self._log.info("Create ring buffer as '{}'".format(command))
        execution_instance = ExecuteCommand(command,
                                            self._execution)
        self._execution_instances.append(execution_instance)
        execution_instance.returncode_callbacks.add(self._returncode_handle)
        execution_instance.stdout_callbacks.add(self._stdout_handle)
        execution_instance.finish()
        
        # Now we have ring buffer
        self._has_db = True

    @classmethod
    def from_args(cls, values):
        '''
        Build a pipeline from command line arguments
        '''
        return Pipeline(values)
    
    def run(self):
        # We do not initialise all values at the startup, 
        # instead, we do that at seperate functions to
        # make sure that the dependence of these functions is meet
        
        # When the pipeline object initialized,
        # all tasks initialized and wait to finish
                
        # Attach all readers to the ring buffer
        self._run_reader()

        
        # Attach writer to the ring buffer
        self._run_writer()
        
        # Wait all applications finish
        self._sync_executions()
        
    def _run_reader(self):
        if self._reader:
            if "dbdisk" in self._reader:
                self._dbdisk()            

    def _run_writer(self):
        if self._writer:
            if "udp2db" in self._writer:
                self._udp2db()

    def _udp2db(self):
        # Build udp2db command line
        self._log.info("Parse 'udp2db' keys from YAML")
        
        # Not shared for sure
        app  = self._yaml_dict["udp2db"]["app"]
        host = self._yaml_dict["udp2db"]["host"]
        port = self._yaml_dict["udp2db"]["port"]
        duration = self._yaml_dict["udp2db"]["duration"]
        window   = self._yaml_dict["udp2db"]["window"]
        timeout  = self._yaml_dict["udp2db"]["timeout"]
        template = self._yaml_dict["udp2db"]["template"]
        reuse    = self._yaml_dict["udp2db"]["reuse"]
        copy     = self._yaml_dict["udp2db"]["copy"]

        # sod to 0 if there is no reader
        if self._reader == None:
            sod = 0
        else:
            sod = 1
        # taskset -c 0 ./udp2db -d 10 -M 128 -H 10.17.4.1 -p 10000 -w 10 -F 4096 -n 1 -b 16 -T 1 -t 1 -r -N 1024 -k dada -s -D psrdada_bigcat.txt -c
        command = ("{} -d {} -M {}"
                   " -H {} -p {} -w {} -F {} -n {}"
                   " -b {} -T {} -t {} -k {} -D {}").format(app, duration, self._bandwidth,
                                                            host, port, window, self._framesize, self._nchan,
                                                            self._bits, self._nthread, timeout,
                                                            self._key, template)
        
        if reuse:
            command = command + " -r "
        if sod:
            command = command + " -s"
        if copy:
            command = command + " -c"
            
        self._log.info("Run udp2db as '{}'".format(command))
        execution_instance = ExecuteCommand(command,
                                            self._execution)
        self._execution_instances.append(execution_instance)
        execution_instance.returncode_callbacks.add(self._returncode_handle)
        execution_instance.stdout_callbacks.add(self._stdout_handle)
        
    def _dbdisk(self):
        self._log.info("Parse 'dbdisk' keys from YAML")
        app       = self._yaml_dict["dbdisk"]["app"]
        directory = self._yaml_dict["dbdisk"]["directory"]

        # Build dbdisk command line
        command = "{} -k {} -D {}".format(app,
                                          self._key,
                                          directory)
        
        self._log.info("Run dbdisk as '{}'".format(command))
        execution_instance = ExecuteCommand(command,
                                            self._execution)
        self._execution_instances.append(execution_instance)
        execution_instance.returncode_callbacks.add(self._returncode_handle)
        execution_instance.stdout_callbacks.add(self._stdout_handle)
        
    def _sync_executions(self):
        # Wait all executions finish
        self._log.info("To check if we have launch failure")
        failed_launch = False
        for execution_instance in self._execution_instances:
            failed_launch = (failed_launch or execution_instance.failed_launch)

        if failed_launch:
            self._log.info("Have to terminate all applications "
                     "as launch failure happened")
            self._terminate_executions()
        else:
            self._log.info("Wait all executions to finish")
            for execution_instance in self._execution_instances:
                execution_instance.finish()
            
        if(self._has_db):
            self._log.info("Destroy ring buffer '{}'".format(self._key))
            command = ("dada_db -d -k {} ").format(self._key)
            execution_instance = ExecuteCommand(command,
                                                self._execution)
            self._execution_instances.append(execution_instance)
            execution_instance.returncode_callbacks.add(self._returncode_handle)
            execution_instance.stdout_callbacks.add(self._stdout_handle)
            execution_instance.finish()

        #self._log.info("Reset everything at the end")
        #self._reset()

    def _reset(self):
        commands = ["ipcrm -a",
                    "pkill -9 -f dada_db",
                    "pkill -9 -f dada_diskdb",
                    "pkill -9 -f dada_dbdisk",
                    "pkill -9 -f udp2db"
        ]
        execution_instances = []
        for command in commands:
            #print (command)
            self._log.debug("cleanup with {}".format(command))
            execution_instances.append(
                ExecuteCommand(command, self._execution, "y"))
        for execution_instance in execution_instances:
            # Wait until the reset is done
            execution_instance.finish()

        # ring buffer is deleted by now
        self._has_db = False
        
    def _returncode_handle(self, returncode, callback):
        # Rely on stdout_handle to print all information
        if self._execution:
            if returncode:
                self._log.error(returncode)
                
                self._log.error("Terminate all execution instances "
                          "when error happens")
                self._terminate_executions()
                
                #self._log.error("Reset when error happens")
                #self._reset()
                
                raise PipelineError(returncode)

    def _stdout_handle(self, stdout, callback):
        if self._execution:
            self._log.debug(stdout)

    def _terminate_executions(self):
        for execution_instance in self._execution_instances:
            execution_instance.terminate()

def _main():
    parser = argparse.ArgumentParser(
        description='To run the bigcat udp2db pipeline')
    parser.add_argument('-c', '--config', type=str, nargs='+',
                        help='YAML file to provide configurations')
    parser.add_argument('-e', '--execution', action='store_true',
                        help='Execution or not')
    parser.add_argument('-v', '--verbose', action='store_true',
                        help='Be verbose')
    parser.set_defaults(execution=False)
    parser.set_defaults(verbose=False)
    values = parser.parse_args()

    ## Setup logger
    logging.basicConfig(filename='{}.log'.format(__name__))
    log = logging.getLogger(__name__)
    #log = logging.getLogger("root")
    if values.verbose:
        coloredlogs.install(
            fmt="[ %(levelname)s\t- %(asctime)s - %(name)s - %(filename)s:%(lineno)s] %(message)s",
            level='DEBUG')
    else:            
        coloredlogs.install(
            fmt="[ %(levelname)s\t- %(asctime)s - %(name)s - %(filename)s:%(lineno)s] %(message)s",
            level='INFO')

    pipeline = Pipeline.from_args(values)
    pipeline.run()

if __name__ == "__main__":
    # pipeline -c config.txt -v -e
    # taskset -c 10 vlbi_fake -vdif  -H 10.17.4.1 -p 10000 -udp 10000 -d 10000 -novtp -bandwidth 128 -sleep 4.3 -b 16 -nchan 1 -complex -nthread 1
    _main()
