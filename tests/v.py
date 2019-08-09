import yggdrasil
import argparse

parser = argparse.ArgumentParser(description='Python Valkyrie; let us slay.')
parser.add_argument('--zmqPort', metavar='-zp',
                    help='the ZMQ port')
args = parser.parse_args()
port = args.zmqPort
v = yggdrasil.valkyrieCommunicator(str.encode(port))

def startup():
  # We receive two tasks from yggdrasil before we begin to run.
  v.run()
  print(v.command)
  v.send(0, 0)
  v.run()
  print(v.command)
  v.send(0, 0)


def run():
  v.run()
  print(v.command)
  print(v.delta)


startup()
run()