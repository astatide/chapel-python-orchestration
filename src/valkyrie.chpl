// We're shifting the valkyrie to be a separate execution unit.
// So that's what this is.  We'll need to actually compile it separately,
// then call it later.

//use propagator;
use genes;
use uuid;
use messaging;
use Time;
use Reflection;

var VUUID = new owned uuid.UUID();
VUUID.UUID4();

config var vSize: int = 1;

// As we have our tree of life, so too do we have winged badasses who choose
// who lives and who dies.
// (this is a class for a worker object)

class valkyrieHandler : msgHandler {

  var log: shared ygglog.YggdrasilLogging();
  var takeAnyPath: bool = false;
  var moved: bool = false;
  var canMove: bool = false;
  var currentNode: string;
  // Is there anything else I need?  Who knows!
  var priorityNodes: domain(string);
  var nPriorityNodesProcessed: int;
  var moveSuccessful: int = 0;
  var moveFailed: int = 1;

  var currentTask: int;
  var nMoves: int;
  var nProcessed: int;
  var gen: int;
  var currentLocale: string;

  var UUIDP = new owned uuid.UUID();
  var id = '%04i'.format(here.id) + '-VALK-' + UUIDP.UUID4();

  var yh = new ygglog.yggHeader();

  var __score__ : real;

  var isRunning: bool = false;


  proc moveToRoot() {
    // Zero out the matrix, return the root id.
    //this.matrixValues = 0;
    this.currentNode = 'root';
  }

  proc init() {
    // basically, the inheritance isn't working as I would have expected.
    // see https://github.com/chapel-lang/chapel/issues/8232
    super.init(1);
    this.complete();
    this.log = new shared ygglog.YggdrasilLogging();
    this.log.currentDebugLevel = 0;
  }

  proc init(n: int) {
    // basically, the inheritance isn't working as I would have expected.
    // see https://github.com/chapel-lang/chapel/issues/8232
    super.init(n);
    this.complete();
    this.log = new shared ygglog.YggdrasilLogging();
    this.log.currentDebugLevel = 0;
  }

  override proc PROCESS(m: msg, i: int) {
    // overriden from the messaging class
    this.log.debug("MESSAGE RECEIVED:", m : string, hstring = this.header);
    select m.COMMAND {
      when this.command.RECEIVE_SCORE {
        m.open(this.__score__);
      }
    }
  }

  inline proc score {
    var s = this.__score__;
    this.__score__ = 0;
    return s;
  }

  proc setSendTo() {
    this.yh.sendTo = "L" + here.id : string + "-V" + this.currentTask;
  }

  proc sendToFile {
    return 'EVOCAP----' + this.id + '----' + this.currentTask : string + '----';
  }

  proc header {
    //return ' '.join(this.sendToFile, 'V', '%05i'.format(this.currentTask) : string, 'M', '%05i'.format(this.nMoves), 'G', '%05i'.format(this.gen));
    this.yh.header = 'VALKYRIE';
    this.yh.id = this.id;
    this.yh.currentLocale = this.currentLocale;
    this.yh.currentTask = this.currentTask;
    return this.yh;
  }

  iter logo {
    var lorder: domain(int);
    var logo: [lorder] string;
    logo[0] = ' ▄█    █▄     ▄████████  ▄█          ▄█   ▄█▄ ▄██   ▄      ▄████████  ▄█     ▄████████ ';
    logo[1] = '███    ███   ███    ███ ███         ███ ▄███▀ ███   ██▄   ███    ███ ███    ███    ███ ';
    logo[2] = '███    ███   ███    ███ ███         ███▐██▀   ███▄▄▄███   ███    ███ ███▌   ███    █▀  ';
    logo[3] = '███    ███   ███    ███ ███        ▄█████▀    ▀▀▀▀▀▀███  ▄███▄▄▄▄██▀ ███▌  ▄███▄▄▄     ';
    logo[4] = '███    ███ ▀███████████ ███       ▀▀█████▄    ▄██   ███ ▀▀███▀▀▀▀▀   ███▌ ▀▀███▀▀▀     ';
    logo[5] = '███    ███   ███    ███ ███         ███▐██▄   ███   ███ ▀███████████ ███    ███    █▄  ';
    logo[6] = '███    ███   ███    ███ ███▌    ▄   ███ ▀███▄ ███   ███   ███    ███ ███    ███    ███ ';
    logo[7] = '▀██████▀    ███    █▀  █████▄▄██   ███   ▀█▀  ▀█████▀    ███    ███ █▀     ██████████ ';
    logo[8] = '                       ▀           ▀                     ███    ███                   ';
    logo[9] = 'VALKYRIE %s on locale %i, running task %i'.format(this.id, here.id, this.currentTask);
    for i in 0..9 {
      yield logo[i];
    }
  }

  iter processNodeOld(ref node: shared genes.GeneNode, delta : genes.deltaRecord) {
    var oldNode = this.currentNode;
    var score: real;
    var deme: int;
    var dNew = delta;
    this.currentNode = node.id;
    this.moved = true;
    this.nProcessed += 1;
    var adjustValkyrie: bool = true;

    // should work for multiple demes, now.
    for d in node.returnDemes() {
      this.log.log('Starting work for ID:', node.id: string, 'on deme #', deme : string, hstring=this.header);
      this.log.log("Attempting to run Python on seed ID", node.id : string, 'DELTA:', delta : string, hstring=this.header);
      assert(!delta.seeds.isEmpty());
      if adjustValkyrie {
        dNew = delta;
        adjustValkyrie = false;
      } else {
        dNew = new genes.deltaRecord();
      }
      var newMsg = new messaging.msg(dNew);
      this.log.log("ID", node.id : string, 'MSG:', newMsg : string, hstring=this.header);
      newMsg.i = d;
      this.log.log("DEME SET:", d : string, hstring=this.header);
      newMsg.COMMAND = this.command.RECEIVE_DELTA;
      this.log.log("Sending the following msg:", newMsg : string, hstring=this.header);
      this.SEND(newMsg);
      this.log.log("Message & delta sent; awaiting response", hstring=this.header);
      var m = this.RECV();
      writeln("MSG: ", m : string);
      score = m.r;
      // we want to receive the novelty, now.
      m = this.RECV();
      deme = d;
      //var s = "[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 1, 0, 0, 0, 0]";
      // This is for the novelty, which we are assuming, currently, comes back as a string of an array of ints.
      // this is not a great general assumption!  But whatever.
      //var s = m['s'];
      //s = s.replace("[","");
      //s = s.replace("]","");
      //for i in s.split(", ") {
      //  node.novelty[deme].push_back(i : int);
      //}
      //writeln('NOVEL: ', m.s : string);
      //node.novelty[deme] = m.s;
      this.log.log('SCORE FOR', node.id : string, 'IS', score : string, hstring=this.header);
      //this.log.log('NOVELTY FOR', node.id : string, 'IS', node.novelty[deme] : string, hstring=this.header);
      node.setDemeScore(deme, score);
      node.setValkyrie(this.id, this.nProcessed);
      yield (score, deme);
    }
  }
}

  iter processNode(ref node: shared genes.GeneNode, delta : genes.deltaRecord) {
    // we should control this from the easier to program Python end.
    // ergo, we should enter a loop that receives and processes data 
    // until it's told to stop.
    this.isRunning = true;
    while isRunning {
      this.receiveMessage();
    }
  }


    var oldNode = this.currentNode;
    var score: real;
    var deme: int;
    var dNew = delta;
    this.currentNode = node.id;
    this.moved = true;
    this.nProcessed += 1;
    var adjustValkyrie: bool = true;

    // should work for multiple demes, now.
    for d in node.returnDemes() {
      this.log.log('Starting work for ID:', node.id: string, 'on deme #', deme : string, hstring=this.header);
      this.log.log("Attempting to run Python on seed ID", node.id : string, 'DELTA:', delta : string, hstring=this.header);
      assert(!delta.seeds.isEmpty());
      if adjustValkyrie {
        dNew = delta;
        adjustValkyrie = false;
      } else {
        dNew = new genes.deltaRecord();
      }
      var newMsg = new messaging.msg(dNew);
      this.log.log("ID", node.id : string, 'MSG:', newMsg : string, hstring=this.header);
      newMsg.i = d;
      this.log.log("DEME SET:", d : string, hstring=this.header);
      newMsg.COMMAND = this.command.RECEIVE_DELTA;
      this.log.log("Sending the following msg:", newMsg : string, hstring=this.header);
      this.SEND(newMsg);
      this.log.log("Message & delta sent; awaiting response", hstring=this.header);
      var m = this.RECV();
      writeln("MSG: ", m : string);
      score = m.r;
      // we want to receive the novelty, now.
      m = this.RECV();
      deme = d;
      //var s = "[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 1, 0, 0, 0, 0]";
      // This is for the novelty, which we are assuming, currently, comes back as a string of an array of ints.
      // this is not a great general assumption!  But whatever.
      //var s = m['s'];
      //s = s.replace("[","");
      //s = s.replace("]","");
      //for i in s.split(", ") {
      //  node.novelty[deme].push_back(i : int);
      //}
      //writeln('NOVEL: ', m.s : string);
      //node.novelty[deme] = m.s;
      this.log.log('SCORE FOR', node.id : string, 'IS', score : string, hstring=this.header);
      //this.log.log('NOVELTY FOR', node.id : string, 'IS', node.novelty[deme] : string, hstring=this.header);
      node.setDemeScore(deme, score);
      node.setValkyrie(this.id, this.nProcessed);
      yield (score, deme);
    }
  }
}


class valkyrieExecutor: msgHandler {
  //var matrixValues: [0..vSize] c_double;
  var takeAnyPath: bool = false;
  var moved: bool = false;
  var canMove: bool = false;
  var currentNode: string;
  // Is there anything else I need?  Who knows!
  var priorityNodes: domain(string);
  var nPriorityNodesProcessed: int;
  var moveSuccessful: int = 0;
  var moveFailed: int = 1;

  var currentTask: int;
  var nMoves: int;
  var nProcessed: int;
  var gen: int;

  var id = VUUID.UUID4();

  var yh = new ygglog.yggHeader();

  //var gj = new owned gjallarbru.Gjallarbru();


  var newDelta: genes.deltaRecord;

  proc init(n: int) {
    // basically, the inheritance isn't working as I would have expected.
    // see https://github.com/chapel-lang/chapel/issues/8232
    super.init(n);
    this.size = n;
    this.complete();
    //gj.pInit();
  }

  proc header {
    //return ' '.join(this.sendToFile, 'V', '%05i'.format(this.currentTask) : string, 'M', '%05i'.format(this.nMoves), 'G', '%05i'.format(this.gen));
    this.yh.header = 'VALKYRIE';
    this.yh.id = this.id;
    this.yh.currentTask = this.currentTask;
    return this.yh;
  }

  proc moveToRoot() {
    // Zero out the matrix, return the root id.
    //this.matrixValues = 0;
    this.currentNode = 'root';
  }

  // Assume we're given a delta object so that we may express it.
  proc move(delta: genes.deltaRecord) {
    //delta.express(this.matrixValues);
    this.currentNode = delta.to;
    return this.moveSuccessful;
  }

  proc run() throws {
    // so, this is the function that will listen for an input and go from there.
    // basically, we want to sit at the read point... and then do something with
    // the input.
    // spawn the Python business.
    // python is initialized.  Yay.
    var writeOnce: bool = true;
    var t: Timer;
    this.heart.start();
    while true {
      t.start();
      // basically, while we're able to read in a record...
      // ... we pretty much read and process.
      this.receiveMessage();
      t.stop();
      if t.elapsed(TimeUnits.milliseconds) < 100 {
        sleep((100 - t.elapsed(TimeUnits.milliseconds)), TimeUnits.milliseconds);
      }
      if this.heart.elapsed() > this.heartbeat {
        // exit when you're done.
        //throw new owned Error();
      }
    }
    //gj.final();
  }

  // this is implemented as part of the messaging class.
  // Don't forget that override!
  override proc PROCESS(m: msg, i: int) {
    // overriden from the messaging class
    writeln("STARTING TO PROCESS");
    writeln(m : string);
    this.heart.clear();
    select m.COMMAND {
      when this.command.SET_ID do {
        m.open(this.id);
        var newMsg = new messaging.msg(0);
        newMsg.STATUS = this.status.OK;
        SEND(newMsg);
      }
      when this.command.SET_TIME do {
        //m.open(this.id);
        //var newMsg = new messaging.msg(0);
        //newMsg.STATUS = this.status.OK;
        //SEND(newMsg);
      }
      when this.command.SHUTDOWN do {
        exit(0);
        var newMsg = new messaging.msg(0);
        newMsg.STATUS = this.status.OK;
        SEND(newMsg);
      }
      when this.command.SET_TASK do {
        m.open(this.currentTask);
        var newMsg = new messaging.msg(0);
        newMsg.STATUS = this.status.OK;
        SEND(newMsg);
      }
      when this.command.RECEIVE_AND_PROCESS_DELTA do {
        var delta: genes.deltaRecord;
        m.open(delta);
        this.move(delta);
        //var score: real = gj.lockAndRun(this.matrixValues, this.currentTask, m.i, hstring=this.header);
        var score: real;
        writeln("score in valkyrie: " + score : string);
        var newMsg = new messaging.msg(score);
        //newMsg.s = "";
        newMsg.r = score;
        newMsg.COMMAND = this.command.RECEIVE_SCORE;
        writeln("what is our msg?: " + newMsg : string);
        SEND(newMsg);
      }
      when this.command.MOVE do {
        var delta: genes.deltaRecord;
        m.open(delta);
        this.move(delta);
        var newMsg = new messaging.msg(status.OK);
        newMsg.COMMAND = this.command.RETURN_STATUS;
        writeln("what is our msg?: " + newMsg : string);
        SEND(newMsg);
      }
    }
    writeln("VALKYRIE PROCESSED MSG");
    stdout.flush();
  }
}

class valkyrieExecutorPythonLib: valkyrieExecutor {

  var delta: genes.deltaRecord;

  var currentCommand: int;
  var receivedMessage: bool = false;
  var m: msg;

  proc init(n: int) {
    // basically, the inheritance isn't working as I would have expected.
    // see https://github.com/chapel-lang/chapel/issues/8232
    super.init(n);
    this.size = n;
    this.complete();
    //gj.pInit();
  }

  override proc run() throws {
    // basically, while we're able to read in a record...
    // ... we pretty much read and process.
    this.receiveMessage();
  }

  override proc PROCESS(m: msg, i: int) {
    // overriden from the messaging class
    this.m = msg;
    writeln("STARTING TO PROCESS");
    writeln(m : string);
    receivedMessage = true;
    this.currentCommand = m.COMMAND;
    select m.COMMAND {
      when this.command.SET_ID do {
        m.open(this.id);
      }
      when this.command.SET_TIME do {
      }
      when this.command.SHUTDOWN do {
        exit(0);
      }
      when this.command.SET_TASK do {
        m.open(this.currentTask);
      }
      when this.command.RECEIVE_AND_PROCESS_DELTA do {
        m.open(this.delta);
      }
    }
    writeln("VALKYRIE PROCESSED MSG");
    writeln(this.delta : string);
    writeln(m : string);
    stdout.flush();
  }
}

var v: shared valkyrieExecutorPythonLib;
// this is the command array.
var cArray: [1..0] c_string;
var instructionsCompiled: bool = false;
var instructionNumber: int = 0;
param n = numFields(messaging.commandRecord);

export proc createValkyrie(port: c_string) {
  // get the information necessary.  We need a currentTask, for instance.
  v = new shared valkyrie.valkyrieExecutorPythonLib(1);
  writeln('VALKYRIE %s on locale %i, running task %i : port %s'.format(v.id, here.id, v.currentTask, port : string));
  v.initRecvSocket(1, port : string);
  writeln('VALKYRIE %s on locale %i, ports initialized'.format(v.id, here.id));
}

export proc receiveInstructions() {
  writeln("running!");
  v.run();
}

export proc __delta__() : [] real {
  
  var delta: [0..v.delta.seeds.size-1] real;
  var i: int = 0;
  for j in v.delta.seeds {
    delta[i] = v.delta.delta[j];
    i += 1;
  }
  return delta;
}

export proc __seeds__() : [] int {
  
  var seeds: [0..v.delta.seeds.size-1] int;
  var i: int = 0;
  for j in v.delta.seeds {
    seeds[i] = j;
    i += 1;
  }
  return seeds;
}

export proc getCurrentCommand() : int {
  
  if v.receivedMessage {
    v.receivedMessage = false;
    return v.currentCommand;
  }
  return 0;
}

export proc __getInstructions__(): c_string {
  //var s = new messaging.statusRecord();
  //var c = new messaging.commandRecord();
  if !instructionsCompiled {
    for param i in 1..n {
      cArray.push_back(getFieldName(messaging.commandRecord, i) : c_string);
    }
    instructionsCompiled = true;
  }
  if instructionNumber < n {
    instructionNumber += 1;
    return cArray[instructionNumber];
  } else {
    return "__END_GETINSTRUCTIONS__" : c_string;
  }
}

export proc returnScore(score : real, novelty: [] real) {
  // we'll send in a novelty vector, cause why not?
  var newMsg = new messaging.msg(score);
  newMsg.r = score;
  newMsg.COMMAND = v.command.RECEIVE_SCORE;
  v.SEND(newMsg);
  //for i in novelty {
  //  newMsg.v.push_back(i);
  //}
  newMsg.COMMAND = v.command.RECEIVE_NOVELTY;
  v.SEND(newMsg);
}

export proc returnStatus(command : int, status : int) {
  // we'll send in a novelty vector, cause why not?
  var newMsg = new messaging.msg(0);
  newMsg.STATUS = status;
  newMsg.COMMAND = command;
  v.SEND(newMsg);
} 

export proc send(command: int, status: int, data: ?T) {
  // variable stuff for variable sets.
  var newMsg = new messaging.msg(0);
  newMsg.STATUS = status;
  newMsg.COMMAND = command;
  select T {
    when isArrayType(T) do with (ref newMsg) {
      // if this is an array, we want to send it one value at a time.
      // so basically loop and send until we're done.
      newMsg
    }
    when isStringType(T) do {

    }
    when isRecordType(T) do {

    }
  }
}