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

config var recvPort: string;
config var sendPort: string;
config var vSize: int;

//var lf = open('test.log' : string, iomode.cwr);
//var c = lf.writer(kind=ionative);
//var z = lf.reader(kind=ionative);

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
  //var messageReceived: atomic bool = false;
  // reading and writing channels.
  //var cout: channel(true,iokind.dynamic,true);
  //var cin: channel(false,iokind.dynamic,true);

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

  proc sendToFile {
    return 'EVOCAP----' + this.id + '----' + this.currentTask : string + '----';
  }

  proc header {
    //return ' '.join(this.sendToFile, 'V', '%05i'.format(this.currentTask) : string, 'M', '%05i'.format(this.nMoves), 'G', '%05i'.format(this.gen));
    this.yh.header = 'VALKYRIE';
    this.yh.id = this.id;
    this.yh.currentTask = this.currentTask;
    return this.yh;
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
      if writeOnce {
        //writeOnce = false;
        //c.writeln("we're in the loop yo");
        //c.flush();
      }
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
      when messaging.command.SET_ID do {
        m.open(this.id);
        var newMsg = new messaging.msg(0);
        newMsg.STATUS = messaging.status.OK;
        SEND(newMsg);
      }
      when messaging.command.SET_TIME do {
        //m.open(this.id);
        //var newMsg = new messaging.msg(0);
        //newMsg.STATUS = messaging.status.OK;
        //SEND(newMsg);
      }
      when messaging.command.SHUTDOWN do {
        exit(0);
        var newMsg = new messaging.msg(0);
        newMsg.STATUS = messaging.status.OK;
        SEND(newMsg);
      }
      when messaging.command.SET_TASK do {
        m.open(this.currentTask);
        var newMsg = new messaging.msg(0);
        newMsg.STATUS = messaging.status.OK;
        SEND(newMsg);
      }
      when messaging.command.RECEIVE_AND_PROCESS_DELTA do {
        var delta: genes.deltaRecord;
        m.open(delta);
        this.move(delta);
        var score: real = gj.lockAndRun(this.matrixValues, this.currentTask, m.i, hstring=this.header);
        writeln("score in valkyrie: " + score : string);
        var newMsg = new messaging.msg(score);
        //newMsg.s = "";
        newMsg.r = score;
        newMsg.COMMAND = messaging.command.RECEIVE_SCORE;
        writeln("what is our msg?: " + newMsg : string);
        SEND(newMsg);
      }
      when messaging.command.MOVE do {
        var delta: genes.deltaRecord;
        m.open(delta);
        this.move(delta);
        var newMsg = new messaging.msg(status.OK);
        newMsg.COMMAND = messaging.command.RETURN_STATUS;
        writeln("what is our msg?: " + newMsg : string);
        SEND(newMsg);
      }
    }
    writeln("VALKYRIE PROCESSED MSG");
    stdout.flush();
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
}

proc main {
  // first, register the sigint handler.
  writeln("VALKYRIE on locale %i, spawned.".format(here.id));
  extern proc signal(sigNum : c_int, handler : c_fn_ptr) : c_fn_ptr;
  proc handler(x : int) : void {
      //ragnarok.setShutdown();
  }
  // Capturing sigint.
  signal(2, c_ptrTo(handler));

  // get the information necessary.  We need a currentTask, for instance.
  var v = new shared valkyrieExecutor(1);
  //writeln("VALKYRIE on locale %i, spawned.".format(here.id));
  //v.id = 1;
  //stdout.writeln("Started!");
  //writeln("received port!");
  //for iL in v.logo {
  //  writeln(iL);
  //}
  writeln('VALKYRIE %s on locale %i, running task %i : recvPort %s, sendPort %s'.format(v.id, here.id, v.currentTask, recvPort, sendPort));


  v.initRecvSocket(1, recvPort);
  //v.initPrevSendSocket(1, sendPort);
  writeln('VALKYRIE %s on locale %i, ports initialized'.format(v.id, here.id));
  //c.write("So, I've got all that stuff set");
  //c.write(v.sendPorts[1], " ", v.recvPorts[1]);
  //c.flush();
  //writeln("Hey, so, I can be chatty now, yeah?");
  //stdout.flush();
  //writeln("Hey, am I working?");
  // redirect normal stdout;
  //stdout = lf.writer();
  //writeln(5);
  //v.OK();
  v.run();
}
