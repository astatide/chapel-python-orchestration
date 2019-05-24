

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
    d += (123, 2);
    d += (54345, 4);
    writeln("Spawning valkyrie");
    var vp = spawn(["./valkyrie"], stdout=PIPE, stdin=PIPE);
    // this SHOULD be good?
    this.setChannels(vp.stdout, vp.stdin);
    /*
    var c: channel(true,iokind.dynamic,true);
    var z: channel(false,iokind.dynamic,true);
    var lf = open('test.log' : string, iomode.cwr);
    c = lf.writer();
    z = lf.reader();
    c.write(d);
    c.flush();
    //var l: genes.deltaRecord;
    var l = z.read(genes.deltaRecord);
    writeln("This should be a delta");
    writeln(d);
    writeln(l);
    */
    while true {
      //vp.stdin.writeln("blooo");
      //var b: int = 12;
      //vp.stdout.read(b);
      //writeln(b : string);
      var newMsg: messaging.msg;
      newMsg.COMMAND = messaging.command.SET_TASK;
      writeln(newMsg);
      writeln("Attempting to send message");
      SEND(newMsg);
      writeln("Message sent");
      // setting the task, now.
      SEND(1);
      newMsg.COMMAND = messaging.command.RECEIVE_AND_PROCESS_DELTA;
      writeln(newMsg);
      writeln("Attempting to run TF");
      SEND(newMsg);
      writeln("Message sent; sending delta");
      writeln(d);
      SEND(d);
      writeln("delta sent; awaiting instructions");
      RECV(newMsg);
      var score: real;
      RECV(score);
      writeln(score);
      vp.wait();
    }
  }
}

var vs = new owned vSpawner(1);
vs.run();
