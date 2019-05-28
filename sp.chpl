

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
    var recvPort: string;
    d += (123, 2);
    d += (54345, 4);
    writeln("Spawning valkyrie");
    var vp = spawn(["./valkyrie"], stdout=PIPE, stdin=PIPE);
    //var vp = spawn(["./valkyrie"], stdin=PIPE);
    // this SHOULD be good?
    //vp.stdout.readln(recvPort);
    //writeln(recvPort);
    writeln("Valkyrie spawned; initializing sockets");
    //this.setChannels(vp.stdout, vp.stdin);
    this.initSendSocket(1);
    vp.stdin.writeln(this.sendPorts[1]);
    // don't forget to flush the connection
    vp.stdin.flush();
    vp.stdout.readln(recvPort);
    //writeln(recvPort);
    //vp.stdout.readln(recvPort);
    this.initRecvSocket(1, recvPort);
    writeln(this.sendPorts[1], " ", this.recvPorts[1]);
    writeln(this.recvSocket[1]);


    begin {
      // start dumping the stdout.
      var l: string;
      while true {
        vp.stdout.readline(l);
        if (l != "") {
          writeln(l);
        }
      }
    }


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
    var newMsg = new messaging.msg(1);
    newMsg.COMMAND = messaging.command.SET_TASK;
    writeln(newMsg);
    writeln("Attempting to send message");
    SEND(newMsg);
    writeln("Message sent");
    // setting the task, now.
    while true {
      //vp.stdin.writeln("blooo");
      //var b: int = 12;
      //vp.stdout.read(b);
      //writeln(b : string);
      newMsg = new messaging.msg(d);
      newMsg.COMMAND = messaging.command.RECEIVE_AND_PROCESS_DELTA;
      writeln(newMsg);
      writeln("Attempting to run TF");
      SEND(newMsg);
      writeln("Message & delta sent; awaiting instructions");
      RECV(newMsg);
      writeln("Message received; awaiting score");
      var score: real;
      newMsg.open(score);
      writeln(score);
      //vp.wait();
    }
  }
}

var vs = new owned vSpawner(1);
vs.run();
