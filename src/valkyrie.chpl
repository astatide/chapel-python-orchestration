// We're shifting the valkyrie to be a separate execution unit.
// So that's what this is.  We'll need to actually compile it separately,
// then call it later.

use propagator;
use genes;
use uuid;
use messaging;
use Time;

var VUUID = new owned uuid.UUID();
VUUID.UUID4();

config var vSize: int;

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
  var id = UUIDP.UUID4();

  var yh = new ygglog.yggHeader();

  var __score__ : real;


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
    this.log.currentDebugLevel = debug;
  }

  proc init(n: int) {
    // basically, the inheritance isn't working as I would have expected.
    // see https://github.com/chapel-lang/chapel/issues/8232
    super.init(n);
    this.complete();
    this.log = new shared ygglog.YggdrasilLogging();
    this.log.currentDebugLevel = debug;
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

  proc processNode(ref node: shared genes.GeneNode, delta : genes.deltaRecord) {
    var oldNode = this.currentNode;
    var score: real;
    var deme: int;
    this.currentNode = node.id;
    this.moved = true;
    this.nProcessed += 1;

    for d in node.returnDemes() {
      this.log.log('Starting work for ID:', node.id: string, 'on deme #', deme : string, hstring=this.header);
      this.log.log("Attempting to run Python on seed ID", node.id : string, hstring=this.header);
      var newMsg = new messaging.msg(delta);
      newMsg.i = d;
      newMsg.COMMAND = this.command.RECEIVE_AND_PROCESS_DELTA;
      this.log.debug("Sending the following msg:", newMsg : string, hstring=this.header);
      this.SEND(newMsg);
      this.log.debug("Message & delta sent; awaiting instructions", hstring=this.header);
      var m = this.RECV();
      score = m.r;
      deme = d;
      this.log.log('SCORE FOR', node.id : string, 'IS', score : string, hstring=this.header);
      node.setDemeScore(deme, score);
    }
    node.setValkyrie(this.id, this.nProcessed);
    return (score, deme);
  }
}


class valkyrieExecutor: msgHandler {
  var matrixValues: [0..vSize] c_double;
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

  var gj = new owned gjallarbru.Gjallarbru();


  var newDelta: genes.deltaRecord;

  proc init(n: int) {
    // basically, the inheritance isn't working as I would have expected.
    // see https://github.com/chapel-lang/chapel/issues/8232
    super.init(n);
    this.size = n;
    this.complete();
    gj.pInit();
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
    this.matrixValues = 0;
    this.currentNode = 'root';
  }

  // Assume we're given a delta object so that we may express it.
  proc move(delta: genes.deltaRecord) {
    delta.express(this.matrixValues);
    this.currentNode = delta.to;
    if unitTestMode {
      if this.matrixValues[0] != this.currentNode : real {
        // in unit test mode, we set up our IDs and matrix values such that
        // every value of the matrix should be equal to ID.
        // in that event, return a failure code.
        return this.moveFailed;
      } else {
        return this.moveSuccessful;
      }
    }
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
    gj.final();
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
        var score: real = gj.lockAndRun(this.matrixValues, this.currentTask, m.i, hstring=this.header);
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
