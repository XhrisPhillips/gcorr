#!/usr/bin/env python

"""
Script to run CRACO pipeline with different mode. 

Copyright (C) CSIRO 2020
"""

import coloredlogs
import logging
import shlex
from subprocess import PIPE, Popen, check_output
import threading
from fcntl import fcntl, F_GETFL, F_SETFL
from os import O_NONBLOCK

__author__ = "Xinping Deng <xinping.deng@csiro.au>"

class ExecuteCommand(object):
    '''
    A class to execute command line as a string. 
    It monitor stdout, stderr and returncode with callback functions.
    It also logs all stdout and stderr with debug level.    
    '''
    def __init__(self,
                 command,
                 execution     = True,
                 popup         = None,
                 process_index = None):

        # Get log setup
        self._log = logging.getLogger(__name__)
        #self._log = logging.getLogger(None)
        
        # stdout here includes stdout and stderr
        # some applications do use stderr as stdout
        self.stdout_callbacks     = set()
        self.returncode_callbacks = set()
        
        self.failed_launch = False
        self._stdout       = None
        self._returncode   = None
        
        # Command line
        self._command = command
        self._executable_command = shlex.split(self._command)

        # Useful when we need given options
        self._popup = popup
        
        # To see if we execution the command for real
        self._execution = execution

        # Useful when we have multi pipelines running in parallel
        self._process_index = process_index

        # Setup monitor threads 
        self._monitor_threads = []

        # Setup for force termination of executable
        self._terminate_event = threading.Event()
        self._terminate_event.clear()

        self._process = None
        if self._execution:
            try:
                self._process = Popen(self._executable_command,
                                      stdout=PIPE,
                                      stderr=PIPE,
                                      stdin=PIPE,
                                      bufsize=1,
                                      universal_newlines=True)
                
                if(self._popup): # To write pop up information to the pipe
                    self._process.communicate(input=self._popup)[0]
                else:
                    flags = fcntl(self._process.stdout, F_GETFL)  # Noblock
                    fcntl(self._process.stdout, F_SETFL, flags | O_NONBLOCK)
                    flags = fcntl(self._process.stderr, F_GETFL)
                    fcntl(self._process.stderr, F_SETFL, flags | O_NONBLOCK)
                
            except Exception as error:
                self.failed_launch = True
                self.returncode = self._command + "; RETURNCODE is: ' 1'"
                self._log.exception("Error while launching command: "
                                    "{} with error "
                                    "{}".format(self._command, error)) 
                
            # Start monitors
            self._monitor_threads.append(
                threading.Thread(target=self._process_monitor))

            for thread in self._monitor_threads:
                thread.start()

    def __del__(self):
        class_name = self.__class__.__name__

    def finish(self):
        if self._execution:
            for thread in self._monitor_threads:
                thread.join()

    def terminate(self):
        if self._execution:
            if self._process != None and \
               self._process.poll() == None:
                self._terminate_event.set()
                self._process.terminate()

    def stdout_notify(self):
        for callback in self.stdout_callbacks:
            callback(self._stdout, self)

    @property
    def stdout(self):
        return self._stdout

    @stdout.setter
    def stdout(self, value):
        self._stdout = value
        self.stdout_notify()

    def returncode_notify(self):
        for callback in self.returncode_callbacks:
            callback(self._returncode, self)

    @property
    def returncode(self):
        return self._returncode

    @returncode.setter
    def returncode(self, value):
        self._returncode = value
        self.returncode_notify()

    def _process_monitor(self):
        if self._execution:
            while (self._process != None and \
                   self._process.poll() == None) and \
                  (not self._terminate_event.is_set()):
                try:
                    stdout = self._process.stdout.readline().rstrip("\n\r")
                    if stdout != "":
                        if self._process_index != None:
                            self.stdout = "'" + self._command + "' " +\
                                          stdout + \
                                          "; PROCESS_INDEX is " + \
                                          str(self._process_index)
                        else:
                            self.stdout = "'" + self._command + "' " + stdout
                except:
                    pass

                try:
                    stderr = self._process.stderr.readline().rstrip("\n\r")
                    if stderr != "":
                        if self._process_index != None:
                            self.stdout = "'" + self._command + "' " +\
                                          stderr + \
                                          "; PROCESS_INDEX is " + \
                                          str(self._process_index)
                        else:
                            self.stdout = "'" + self._command + "' " + stderr
                except:
                    pass

            if self._process != None and \
               self._process.returncode and \
               (not self._terminate_event.is_set()):
                self.returncode = "'" + self._command + "' " +\
                    "; RETURNCODE is here: " +\
                    str(self._process.returncode)
