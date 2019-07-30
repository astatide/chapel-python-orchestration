# This is a Python implementation of our Valkyrie.
# There are reasons for doing it in Chapel or Python.
# This ideally makes it a lot easier for me, but hey.
import zmq
import argparse

class valkyrieExecutor():

    def __init__(self):
        # blah blah blah
        # Sets up the ZMQ port and such.
        parser = argparse.ArgumentParser(description='Python Valkyrie; let us slay.')
        parser.add_argument('--zmqPort', metavar='-zp',
                            help='the ZMQ port')
        parser.add_argument('--id',
                            help='the Valkyrie ID')

        args = parser.parse_args()
        self.port = args.zmqPort
        self.id = args.id

        self.task = 0

        self.functions = {
            'RETURN_STATUS': self.returnStatus,
            'SET_TASK': self.setTask,
            'RECEIVE_AND_PROCESS_DELTA': self.receiveAndProcessDelta,
            'SET_ID': self.setId,
            'SHUTDOWN': self.shutdown,
            'MOVE': self.move,
            'SET_TIME': self.setTime}

    def returnStatus(self, m):
        print("attempting to return the status")
        self.s.send_string(self.convertDictToMsg(m), encoding='ascii')
        print("Msg sent!  Maybe!")
        #self.s.send_string("BLAH", encoding='ascii')
        #return 0

    def returnOkay(self, m):
        m['COMMAND'] = m['command']['RETURN_STATUS']
        m['i'] = m['status']['OK']
        self.returnStatus(m)


    def setTask(self, m):
        print("Setting task to", m['i'])
        self.task = m['i']
        self.returnOkay(m)


    def receiveAndProcessDelta(self, m):
        return 0

    def setId(self, m):
        self.id = m['s']
        self.returnOkay(m)

    def shutdown(self, m):
        return 0

    def move(self, m):
        return 0

    def setTime(self, m):
        return 0


    def convertMsgToDict(self, msg):
        import re
        # (STATUS = 3, COMMAND = 1, s = , i = 1, r = 0.0, exists = 0, status = (OK = 0, ERROR = 1, IGNORED = 2, SEALED = 3), command = (RETURN_STATUS = 0, SET_TASK = 1, RECEIVE_AND_PROCESS_DELTA = 2, RECEIVE_SCORE = 3, SET_ID = 4, SHUTDOWN = 5, MOVE = 6, SET_TIME = 7))
        #dict = eval(msg.replace("(", "{'").replace(')',"}").replace(' = ', "': ").replace(': ,', ': None,',).replace(', ', ", '")) #.replace("s' : ", "s' : '").replace(", 'i'", "', 'i'"))
        dict = msg.replace("(", "{").replace(')',"}").replace(' = ', "': ").replace(': ,', ': None,',).replace(', ', ", '").replace("STATUS", "'STATUS").replace("OK", "'OK").replace("RETURN_'STATUS", "'RETURN_STATUS") #.replace("s' : ", "s' : '").replace(", 'i'", "', 'i'"))
        #print(dict)
        dict = dict.replace("'s': ", "'s': '").replace(", 'i'", "', 'i'")
        dict = dict.replace('=',':')
        # '{SIZE:2,TO:"0002-GENE-800e3a36-a44b-55af-1f44-41ef92d3a0e1",FROM:"0002-GENE-0e9db6bf-a348-e958-da5c-526ce134e7ad",{'6876917637838023797,1.0},{'5058546265420500405,1.0}}'
        for i in ['SIZE', 'TO', 'FROM']:
            dict = dict.replace(i, "'" + i + "'")

        # actually, just grab the string
        p = re.compile("( 's': )(.+)(, 'i')")
        s = p.findall(dict)
        #if s is not None:
            #for i in s:
                #print(i)
                #s = s[6:-5]
                #print("s is not none!")
                # 1 is the valk id in lots of cases.  Or the actual string!
                #print(i[1])
        s = s[0][1]
        print(s)
        # we get rid of it ultimately, but.
        dict = p.sub(" 'i'", dict)
        if False:
            # Grab the delta using regex
            p = re.compile("\{([-?[0-9]+),([^}]+)\}")
            # {'STATUS': 3, 'COMMAND': 2, 's': '{'SIZE':2,'TO':"0002-GENE-cfe7363e-c939-766a-7730-7f8064cd323a",'FROM':"0002-GENE-cf2e2add-38e4-c000-8f83-1115ad003982",{8875112097686481946,1.0},{3451562649136780615,1.0}}', 'i': 0, 'r': 0.0, 'exists': 0, 'status': {'OK': 0, 'ERROR': 1, 'IGNORED': 2, 'SEALED': 3}, 'command': {'RETURN_STATUS': 0, 'SET_TASK': 1, 'RECEIVE_AND_PROCESS_DELTA': 2, 'RECEIVE_SCORE': 3, 'SET_ID': 4, 'SHUTDOWN': 5, 'MOVE': 6, 'SET_TIME': 7}}
            m = p.findall(dict) # this holds the delta!
            for i in m:
                dict = p.sub("", dict)
            # dict = dict.replace(",{",",'DELTA': '{").replace("}}'", "}'}")
            # this removes too many commas.
            p = re.compile("(,)+\1{1,}")
            m = p.findall(dict) # this holds the delta!
            for i in m:
                dict = p.sub("", dict)
            dict = dict.replace('",}','"}')
            print(dict)
            print(m)
        dict = eval(dict)
        dict['s'] = s
        print(dict)
        #print(dict)
        #dict = eval(msg.replace("(", "{'").replace(')',"}").replace(' = ', "' : ").replace(', ', ", '").replace("s' : ", "s' : '").replace(", 'i'", "', 'i'").replace(": '',", ': None,'))

        return dict

    def convertDictToMsg(self, dict):
        msg = str(dict).replace("{'", "(").replace("}",')').replace("': ", ' = ').replace(", '",', ').replace('= None', '= ',)
        return msg

    def connectToPort(self):
        self.zContext = zmq.Context()
        self.s = self.zContext.socket(zmq.PAIR)
        self.s.connect(self.port)

    def mainLoop(self):
        # blah blah blah!
        while True:
            msg = self.s.recv_string()
            self.process(self.convertMsgToDict(msg))

    def process(self, m):
        # these are our instruction keys
        status = m['status']
        command = m['command']
        # these are our actual instructions!
        COMMAND = m['COMMAND']
        STATUS = m['STATUS']
        for c in command.keys():
            if command[c] == COMMAND:
                print("EXECUTING COMMAND ", c)
                self.functions[c](m)


        return 0

if __name__ == '__main__':
    v = valkyrieExecutor()
    v.connectToPort()
    v.mainLoop()
