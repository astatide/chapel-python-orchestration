# This is a Python implementation of our Valkyrie.
# There are reasons for doing it in Chapel or Python.
# This ideally makes it a lot easier for me, but hey.
import zmq
import argparse
import re
import importlib
import yaml
import os
import time
import sys

import mist

mist.chpl_setup()

class valkyrieExecutor():

    def __init__(self):
        # blah blah blah
        # Sets up the ZMQ port and such.
        parser = argparse.ArgumentParser(description='Python Valkyrie; let us slay.')
        parser.add_argument('--zmqPort', metavar='-zp',
                            help='the ZMQ port')
        parser.add_argument('--id',
                            help='the Valkyrie ID')
        parser.add_argument('--config', default='ygg.yml',
                            help='the config file')

        args = parser.parse_args()
        self.port = args.zmqPort
        self.id = args.id
        # NOT WORKING OH WELL
        with open(args.config, 'r') as f:
            self.config = yaml.load(f)

        self.task = 0

        self.functions = {
            'RETURN_STATUS': self.returnStatus,
            'SET_TASK': self.setTask,
            'RECEIVE_AND_PROCESS_DELTA': self.receiveAndProcessDelta,
            'SET_ID': self.setId,
            'SHUTDOWN': self.shutdown,
            'MOVE': self.move,
            'SET_TIME': self.setTime}

        self.model = None

        self.setConfig()

    def receiveAndProcessDelta(self, m):
        start = time.time()
        deme = m['i']
        print(m['delta'])
        self.model.expressDelta(m['delta'])
        final_val, final_state = self.model.evaluate.eval_tartarus(self.model.model, deme)
        end = time.time()
        print("Valkyrie ID: " + self.id + " " + "Returning score: " + str(final_val))
        m['COMMAND'] = m['command']['RECEIVE_SCORE']
        m['r'] = final_val
        m['s'] = final_state
        self.returnStatus(m)


    def shutdown(self, m):
        return 0

    def move(self, m):
        return 0

    def setTime(self, m):
        return 0


    def connectToPort(self):
        self.zContext = zmq.Context()
        self.s = self.zContext.socket(zmq.PAIR)
        self.s.connect(self.port)

    def setConfig(self):
        # here, we set the configuration options.

        # Do a little string parsing to import the module.
        sys.path.insert(0, '../' + self.config['valkyrie']['model'].replace(os.path.basename(self.config['valkyrie']['model']), ''))
        self.modelModule = importlib.import_module(os.path.basename(self.config['valkyrie']['model']).replace('.py', ''))

        # instantiate model.  Probably enforce this structure to some degree.
        self.model = self.modelModule.yggdrasilModel()
        self.model.build_model()


    def mainLoop(self):
        # blah blah blah!
        while True:
            mist.run();
            #self.process(self.convertMsgToDict(msg))

    def process(self, m):

        return 0

if __name__ == '__main__':
    v = valkyrieExecutor()
    #v.connectToPort()
    mist.createValkyrie(v.port)
    #v.model = yggdrasilModel()
    #v.model.model = loki.build_model()
    v.mainLoop()
