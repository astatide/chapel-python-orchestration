// We're shifting the valkyrie to be a separate execution unit.
// So that's what this is.  We'll need to actually compile it separately,
// then call it later.

use propagator;
use genes;
use uuid;
use messaging;

var VUUID = new owned uuid.UUID();
VUUID.UUID4();

class valkyrieExecutor: msgHandler {
  var matrixValues: [0..propagator.mSize] c_double;
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
    this.size = n;
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

  proc run() {
    // so, this is the function that will listen for an input and go from there.
    // basically, we want to sit at the read point... and then do something with
    // the input.
    // spawn the Python business.
    gj.pInit();
    // python is initialized.  Yay.
    while true {
      // basically, while we're able to read in a record...
      // ... we pretty much read and process.
      this.receiveMessage();
    }
    gj.final();
  }

  // this is implemented as part of the messaging class.
  // Don't forget that override!
  override proc PROCESS(m: msg, i: int) {
    // This is a stub class.  Those inheriting it must
    // handle it themselves.
    // does chapel have case/switch?  Hmmmm.
    if m.COMMAND == messaging.command.RETURN_STATUS {
      SEND(this.STATUS);
    } else if m.COMMAND == messaging.command.SET_TASK {
      RECV(this.currentTask);
    } else if m.COMMAND == messaging.command.RECEIVE_AND_PROCESS_DELTA {
      var delta: genes.deltaRecord;
      RECV(delta);
      this.move(delta);
      var score: real = gj.lockAndRun(this.matrixValues, this.currentTask, hstring=this.header);
      // now, return the score.
      var newMsg: messaging.msg;
      newMsg.COMMAND = messaging.command.RECEIVE_SCORE;
      SEND(newMsg);
      SEND(score);
    } else {
      SEND_STATUS(messaging.status.IGNORED);
    }
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
    logo[9] = 'VALKYRIE %s on locale %i, running task %i'.format(this.id, 0, this.currentTask);
    for i in 0..9 {
      yield logo[i];
    }
  }
}

proc main {
  // first, register the sigint handler.
  extern proc signal(sigNum : c_int, handler : c_fn_ptr) : c_fn_ptr;
  proc handler(x : int) : void {
      //ragnarok.setShutdown();
  }
  // Capturing sigint.
  signal(2, c_ptrTo(handler));

  // get the information necessary.  We need a currentTask, for instance.
  var v = new owned valkyrieExecutor(1);
  v.setChannels(stdin, stdout);
  // redirect normal stdout;
  var lf = open('test.log' : string, iomode.cwr);
  stdout = lf.writer();
  //writeln(5);
  //v.OK();
  v.run();
}
