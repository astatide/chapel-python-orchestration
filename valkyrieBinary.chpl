// We're shifting the valkyrie to be a separate execution unit.
// So that's what this is.  We'll need to actually compile it separately,
// then call it later.

use propagator;
use genes;
use uuid;
use messaging;
use Time;
use valkyrie;

var VUUID = new owned uuid.UUID();
VUUID.UUID4();

config var recvPort: string;
config var sendPort: string;

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
  var v = new shared valkyrie.valkyrieExecutor(1);
  writeln('VALKYRIE %s on locale %i, running task %i : recvPort %s, sendPort %s'.format(v.id, here.id, v.currentTask, recvPort, sendPort));

  v.initRecvSocket(1, recvPort);

  writeln('VALKYRIE %s on locale %i, ports initialized'.format(v.id, here.id));

  v.run();
}
