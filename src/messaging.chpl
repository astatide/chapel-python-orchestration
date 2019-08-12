// This is just a source file for our messages.
// It's essentially a self executing record.

use genes;
use ZMQ;
use ygglog;
use Spawn;
use Time;

extern proc chpl_nodeName(): c_string;
config const awaitResponse = false;
config const heartBeat: int = 100;
config const yieldOnWait: bool = false;
config const trueSleep: bool = true;


record statusRecord {
  var OK: int = 0;
  var ERROR: int = 1;
  var IGNORED: int = 2;
  var SEALED: int = 3;
}

record dataTypeRecord {
  var isString: int = 0;
  var isVector: int = 1;
  var isReal: int = 2;
  var isInt: int = 3;
}

record commandRecord {
  var IDLE: int = 0;
  var SET_TASK: int = 1;
  var RECEIVE_DELTA: int = 2;
  var RECEIVE_NOVELTY: int = 3;
  var RECEIVE_SCORE: int = 4;
  var RECEIVE_AUX: int = 5;

  // we can't use ZMQ for these, so we send them in.
  var START_RECEIVE_VECTOR: int = 6;
  var RECEIVE_VECTOR: int = 7;
  var STOP_RECEIVE_VECTOR: int = 8;

  // these are the actual process commands.
  var APPLY_DELTA: int = 9;
  var CHANGE_DEME: int = 10;
  var RUN: int = 11;

  var SET_ID: int = 12;
}

record msg {
  // there are two types of records; commands and results.
  var STATUS: int;
  var COMMAND: int;
  var s: string;
  var i: int;
  var r: real;
  var __dtype__: int;
  // behold, a vector of reals.
  //var v: [1..0] real;
  var exists: int = 0;
  // this.complete() means we complete!
  var status: statusRecord;
  var command: commandRecord;
  var dtype: dataTypeRecord;

  proc init() {}

  proc init(s: string) {
    this.complete();
    //this.TYPE = this.ISSTR;
    this.s = s;
    this.STATUS = status.SEALED;
  }
  proc init(i: int) {
    this.complete();
    //this.TYPE = this.ISINT;
    this.i = i;
    this.STATUS = status.SEALED;

  }
  proc init(r: real) {
    this.complete();
    //this.TYPE = this.ISREL;
    this.r = r;
    this.STATUS = status.SEALED;

  }
  proc init(d: genes.deltaRecord) {
    this.complete();
    //this.TYPE = this.ISDEL;
    this.s = d : string;
    this.STATUS = status.SEALED;

  }

  proc open(ref ret) {
    // use this to open the message.
    select ret.type {
      when int do ret = this.i;
      when string do ret = this.s;
      when real do ret = this.r;
      when genes.deltaRecord do {
        var d: genes.deltaRecord;
        var lf = openmem();
        var c = lf.writer();
        var z = lf.reader();
        c.writeln(this.s);
        c.flush();
        d = z.readln(genes.deltaRecord);
        ret = d;
      }
    }
    return this.r;
  }
}

class msgHandler {

  //var channelsOpened: [filesOpened] channel(true,iokind.dynamic,true);

  var size: int = 1;
  // this is for single receivers/senders.
  var mId: int = 1;

  var fin: [0..size] channel(false,iokind.dynamic,true);
  var fout: [0..size] channel(true,iokind.dynamic,true);
  var STATUS: int;
  var context: Context;

  var ports: [0..size] string;

  var socket: [0..size] Socket;

  // not in use yet.
  var blocking: bool = true;

  var msgQueue: [1..0] msg;

  var heartbeat: int = heartBeat;
  var heart =  new Time.Timer();

  var status: statusRecord;
  var command: commandRecord;

  proc init(n: int) {
    this.size = n;
  }

  proc initSendSocket(i: int) {
    this.socket[i] = this.context.socket(ZMQ.PAIR);
    this.socket[i].yieldOnWait = yieldOnWait;
    this.socket[i].trueSleep = trueSleep;
    //this.socket[i].setsockopt(ZMQ.ZMQ_HEARTBEAT_IVL, 2000);
    //this.socket[i].setsockopt(ZMQ.ZMQ_HEARTBEAT_TIMEOUT, 2000);
    //this.socket[i].setsockopt(ZMQ.ZMQ_HEARTBEAT_TTL, 6000);
    //this.socket[i].setsockopt(ZMQ.HEARTBEAT_IVL, 20);
    //this.socket[i].setsockopt(ZMQ.HEARTBEAT_TIMEOUT, 20);
    //this.socket[i].setsockopt(ZMQ.HEARTBEAT_TTL, 60);
    this.socket[i].bind("tcp://*:*");
    this.ports[i] = this.socket[i].getLastEndpoint().replace("0.0.0.0",chpl_nodeName():string);
  }

  proc initRecvSocket(i: int, port: string) {
    // so, we're going to set up and use a random port.
    this.socket[i] = this.context.socket(ZMQ.PAIR);
    this.socket[i].yieldOnWait = yieldOnWait;
    this.socket[i].trueSleep = trueSleep;
    //this.socket[i].setsockopt(ZMQ.ZMQ_HEARTBEAT_IVL, 2000);
    //this.socket[i].setsockopt(ZMQ.ZMQ_HEARTBEAT_TIMEOUT, 2000);
    //this.socket[i].setsockopt(ZMQ.ZMQ_HEARTBEAT_TTL, 6000);
    //this.socket[i].setsockopt(ZMQ.HEARTBEAT_IVL, 20);
    //this.socket[i].setsockopt(ZMQ.HEARTBEAT_TIMEOUT, 20);
    //this.socket[i].setsockopt(ZMQ.HEARTBEAT_TTL, 60);
    this.socket[i].connect(port);
    this.ports[i] = port;
  }

  proc __receiveMessage__(i: int) {
    this.__PROCESS__(this.RECV(i), i);
  }

  // convenient handler function

  proc receiveMessage(i: int) { this.__receiveMessage__(i); }
  proc receiveMessage() { this.__receiveMessage__(this.mId); }

  // these are stubs the children should implement, if necessary.

  proc __PROCESS__(m: msg, i: int) { this.PROCESS(m, i); }
  proc __PROCESS__(m: msg) { this.PROCESS(m); }

  proc PROCESS(m: msg, i: int) { }
  proc PROCESS(m: msg) { }

  // these are functions for sending.  Essentially, all listen functions
  // must receive a status, or they are blocked.

  proc __SEND__(m: msg, i: int) {
    // send the message!
    this.socket[i].send(m);
    return true;
  }

  proc SEND(m: msg, i: int) { return this.__SEND__(m, i); }
  proc SEND(m: msg) { return this.__SEND__(m, this.mId); }

  // these are functions for receiving.  Essentially, all listen functions
  // must receive a status, or they are blocked.

  proc __RECV__(i: int) {
    // receive the message!
    var m: msg;
    var s: string;
    m.exists = -1;
    m = this.socket[i].recv(msg);
    //s = this.socket[i].recv(string);
    //var lf = openmem();
    //var c = lf.writer();
    //var z = lf.reader();
    //c.writeln(s : string);
    //c.flush();
    //c.close();
    //m = z.readln(msg);
    //writeln("MSG in MESSAGING: ", m : string);
    return m;
    //while m.exists == -1 do chpl_task_yield();
    //return m;
  }

  proc RECV(i: int) { return this.__RECV__(i); }
  proc RECV() { return this.__RECV__(this.mId); }

  proc valhalla(i: int, vId: string = '', vLog = new shared ygglog.YggdrasilLogging(), vstring = new ygglog.yggHeader()) {
    // ha ha, cause Valkyries are in Valhalla, get it?  Get it?
    // ... no?
    // set up a ZMQ client/server
    var iM: int = i; //+(maxValkyries*here.id);
    vLog.log("Initializing sockets", hstring=vstring);
    this.initSendSocket(iM);
    //this.initUnlinkedRecvSocket(iM);

    vLog.log("Spawning Valkyrie", hstring=vstring);
    var vp: subprocess(kind=iokind.dynamic, locking=true);
    vp = spawn(["./v.sh", this.ports[iM], this.ports[iM]], stdout=FORWARD, stderr=FORWARD, stdin=FORWARD, locking=true);

    vLog.log("SPAWN COMMAND:", "./valkyrie", "--recvPort", this.ports[iM], "--sendPort", this.ports[iM], hstring=vstring);

    var newMsg = new messaging.msg(i);
    newMsg.COMMAND = command.SET_TASK;
    vLog.log("Setting task to", i : string, hstring=vstring);
    this.SEND(newMsg, iM);
    this.receiveMessage();

    vLog.log("Setting ID to", vId : string, hstring=vstring);
    newMsg = new messaging.msg(vId);
    newMsg.COMMAND = command.SET_ID;
    this.SEND(newMsg, iM);
    this.receiveMessage();
    return vp;
  }

  proc runDONOTUSE() {
    while true {
      this.receiveMessage();
    }
  }

}
