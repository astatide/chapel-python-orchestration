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
import yggdrasil

class valkyrieExecutor():

    def __init__(self):
        # blah blah blah
        # Sets up the ZMQ port and such.
        parser = argparse.ArgumentParser(description='Python Valkyrie; let us slay.')
        parser.add_argument('--zmqPort', metavar='-zp',
                            help='the ZMQ port')
        parser.add_argument('--config', default='ygg.yml',
                            help='the config file')
        args = parser.parse_args()

        with open(args.config, 'r') as f:
            self.config = yaml.load(f)

        self.port = args.zmqPort
        self.id = None
        self.task = 0
        self.comm = yggdrasil.valkyrieCommunicator(str.encode(self.port))

        self.model = None
        self.setConfig()

    def receiveAndProcessDelta(self, delta):
        print(delta)
        self.model.expressDelta(delta)
        deme = 0
        final_val, final_state = self.model.evaluate.eval_tartarus(self.model.model, deme)
        #print("Valkyrie ID: " + self.id + " " + "Returning score: " + str(final_val))
        return (final_val, final_state)

    def shutdown(self, m):
        return 0

    def move(self, m):
        return 0

    def setTime(self, m):
        return 0


    def setConfig(self):
        # here, we set the configuration options.
        # Do a little string parsing to import the module.
        sys.path.insert(0, '../' + self.config['valkyrie']['model'].replace(os.path.basename(self.config['valkyrie']['model']), ''))
        self.modelModule = importlib.import_module(os.path.basename(self.config['valkyrie']['model']).replace('.py', ''))

        # instantiate model.  Probably enforce this structure to some degree.
        self.model = self.modelModule.yggdrasilModel()
        self.model.build_model()


    def run(self):
        while True:
            self.comm.run()
            self.process(self.comm.command)

    def process(self, command):
        print(command)
        if command == b"SET_TASK":
            self.task = 0
            self.comm.send(0, 0)
        if command == b"SET_ID":
            self.task = 0
            self.comm.send(0, 0)
        if command == b"RECEIVE_AND_PROCESS_DELTA":
            score, novelty = self.receiveAndProcessDelta(self.comm.delta)
            self.comm.returnScore(score, novelty)

        return 0

if __name__ == '__main__':
    v = valkyrieExecutor()
    v.run()
