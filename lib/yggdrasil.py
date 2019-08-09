import mist

# This is the python companion class to our valkyrie.
class valkyrieCommunicator():
  def __init__(self, port=b'iamaport'):
    mist.chpl_setup()
    mist.createValkyrie(port)
    self.__delta__ = {}
    self.__instructions__ = {}

  @property
  def delta(self):
    s = []
    d = []
    # we need to rebuild this basically every time, so hey.
    self.__delta__.clear()
    for seeds in mist.__seeds__():
      s.append(seeds)
    for deltas in mist.__delta__():
      d.append(deltas)
    for i in range(0, len(s)):
      self.__delta__[s[i]] = d[i]
    return self.__delta__

  @property
  def command(self):
    # if our dictionary is empty, fill it.
    if self.__instructions__ == {}:
      # fill
      instruction = ''
      i = 0
      while instruction != b'__END_GETINSTRUCTIONS__':
        instruction = mist.__getInstructions__()
        if instruction != b'__END_GETINSTRUCTIONS__':
          self.__instructions__[i] = instruction
          i += 1
    return self.__instructions__[mist.getCurrentCommand()]

  def run(self):
    mist.receiveInstructions();

  def send(self, comm, val):
    # send a command back.
    if type(comm) != str:
      mist.returnStatus(comm, val)
    else:
      command = 0
      for i in self.__instructions__.keys():
          if str.encode(comm) == self.__instructions__[i]:
            command = i
      mist.returnStatus(command, val)

  def returnScore(self, score, novelty):
    mist.returnScore(score, novelty)