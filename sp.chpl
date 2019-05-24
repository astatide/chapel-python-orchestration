

use Spawn;
use Random;
use messaging;
use uuid;
use rng;
use genes;
use network;
use propagator;
use spinlock;
use genes;

class A {
  proc blah {}
}


class vSpawner: msgHandler {

  proc init(n: int) {
    // basically, the inheritance isn't working as I would have expected.
    // see https://github.com/chapel-lang/chapel/issues/8232
    this.size = n;
  }
  proc run() {
    var i: int;
    var d = new deltaRecord();
    writeln("Spawning valkyrie");
    var vp = spawn(["./valkyrie"], stdout=PIPE, stdin=PIPE);
    // this SHOULD be good?
    this.setChannels(vp.stdout, vp.stdin);
    while true {
      //vp.stdin.writeln("blooo");
      //var b: int = 12;
      //vp.stdout.read(b);
      //writeln(b : string);
      d += (123, 2);
      d += (54345, 4);
      var newMsg: messaging.msg;
      newMsg.COMMAND = messaging.command.SET_TASK;
      writeln(newMsg);
      writeln("Attempting to send message");
      SEND(newMsg);
      writeln("Message sent");
      // setting the task, now.
      SEND(1);
    }
  }
}

var vs = new owned vSpawner(1);
vs.run();
