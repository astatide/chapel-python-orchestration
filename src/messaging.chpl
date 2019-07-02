// This is just a source file for our messages.
// It's essentially a self executing record.

use genes;
use ZMQ;
use ygglog;
use Spawn;
use Time;

extern proc chpl_nodeName(): c_string;
config const awaitResponse = false;
config const yieldWhileWait = false;
config const heartBeat: int = 100;


record statusRecord {
  var OK: int = 0;
  var ERROR: int = 1;
  var IGNORED: int = 2;
  var SEALED: int = 3;
}

record commandRecord {
  var RETURN_STATUS: int = 0;
  var SET_TASK: int = 1;
  var RECEIVE_AND_PROCESS_DELTA: int = 2;
  var RECEIVE_SCORE: int = 3;
  var SET_ID: int = 4;
  var SHUTDOWN: int = 5;
  var MOVE: int = 6;
}

var status: statusRecord;
var command: commandRecord;

record msg {
  // there are two types of records; commands and results.
  var STATUS: int;
  var COMMAND: int;
  var s: string;
  var i: int;
  var r: real;
  var exists: int = 0;
  // this.complete() means we complete!

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

class msgHandlerMultiple {

  //var channelsOpened: [filesOpened] channel(true,iokind.dynamic,true);

  var size: int = 1;
  // this is for single receivers/senders.
  var mId: int = 1;

  var fin: [0..size] channel(false,iokind.dynamic,true);
  var fout: [0..size] channel(true,iokind.dynamic,true);
  var STATUS: int;
  var context: Context;

  var sendPorts: [0..size] string;
  var recvPorts: [0..size] string;

  var sendSocket: [0..size] Socket;
  var recvSocket: [0..size] Socket;

  // not in use yet.
  var blocking: bool = true;

  var msgQueue: [1..0] msg;

  proc init(n: int) {
    this.size = n;
  }

  proc initSendSocket(i: int) {
    this.sendSocket[i] = this.context.socket(ZMQ.REQ);
    this.sendSocket[i].bind("tcp://*:*");
    this.sendPorts[i] = this.sendSocket[i].getLastEndpoint().replace("0.0.0.0",chpl_nodeName():string);
  }

  proc initUnlinkedRecvSocket(i: int) {
    // so, we're going to set up and use a random port.
    this.recvSocket[i] = this.context.socket(ZMQ.REP);
    this.recvSocket[i].bind("tcp://*:*");
    this.recvPorts[i] = this.recvSocket[i].getLastEndpoint().replace("0.0.0.0",chpl_nodeName():string);
  }

  proc initPrevSendSocket(i: int, port: string) {
    this.sendSocket[i] = this.context.socket(ZMQ.REQ);
    this.sendSocket[i].connect(port);
    this.sendPorts[i] = port;
  }

  proc initRecvSocket(i: int, port: string) {
    // so, we're going to set up and use a random port.
    this.recvSocket[i] = this.context.socket(ZMQ.REP);
    this.recvSocket[i].connect(port);
    this.recvPorts[i] = port;
  }

  proc __receiveMessage__(i: int) {
    this.__PROCESS__(this.RECV(i), i);
  }

  proc __OK__(i: int) {
    //this.sendSocket[i].send(status.OK);
    this.recvSocket[i].send(status.OK);
  }

  // convenient handler function

  proc receiveMessage(i: int) { this.__receiveMessage__(i); }
  proc receiveMessage() { this.__receiveMessage__(this.mId); }

  proc OK(i: int) { this.__OK__(i); }
  proc OK() { this.__OK__(this.mId); }

  // these are stubs the children should implement, if necessary.
  // Each time a message is considered received by process, it _must_
  // end with a status okay.

  proc __PROCESS__(m: msg, i: int) { OK(i); this.PROCESS(m, i); }
  proc __PROCESS__(m: msg) { OK(this.mId); this.PROCESS(m); }

  proc PROCESS(m: msg, i: int) { }
  proc PROCESS(m: msg) { }

  // these are functions for sending.  Essentially, all listen functions
  // must receive a status, or they are blocked.

  proc __SEND__(m: msg, i: int) {
    // send the message!
    //fout[i].writeln(m);
    //fout[i].flush();
    this.sendSocket[i].send(m);
    if awaitResponse {
      this.RECV_STATUS(i);
    }
    return true;
  }

  proc SEND(m: msg, i: int) { return this.__SEND__(m, i); }
  proc SEND(m: msg) { return this.__SEND__(m, this.mId); }

  // these are functions for receiving.  Essentially, all listen functions
  // must receive a status, or they are blocked.

  proc __RECV__(i: int) {
    // receive the message!
    //fin[i].readln(m);
    var m: msg;
    m.exists = -1;
    m = this.recvSocket[i].recv(msg);
    while m.exists == -1 do chpl_task_yield();
    return m;
  }

  proc RECV(i: int) { return this.__RECV__(i); }
  proc RECV() { return this.__RECV__(this.mId); }

  proc valhalla(i: int, vId: string, mSize : string, vLog: ygglog.YggdrasilLogging, vstring: ygglog.yggHeader) {
    // ha ha, cause Valkyries are in Valhalla, get it?  Get it?
    // ... no?
    // set up a ZMQ client/server
    var iM: int = i; //+(maxValkyries*here.id);
    vLog.log("Initializing sockets", hstring=vstring);
    this.initSendSocket(iM);
    this.initUnlinkedRecvSocket(iM);

    vLog.log("Spawning Valkyrie", hstring=vstring);
    var vp: subprocess(kind=iokind.dynamic, locking=true);
    begin with (ref vp) vp = spawn(["./v.sh", this.sendPorts[iM], this.recvPorts[iM], mSize : string], stdout=FORWARD, stderr=FORWARD, stdin=FORWARD, locking=true);

    vLog.log("SPAWN COMMAND:", "./valkyrie", "--recvPort", this.sendPorts[iM], "--sendPort", this.recvPorts[iM], "--vSize", mSize : string, hstring=vstring);

    var newMsg = new messaging.msg(i);
    newMsg.COMMAND = messaging.command.SET_TASK;
    vLog.log("Setting task to", i : string, hstring=vstring);
    this.SEND(newMsg, iM);

    vLog.log("Setting ID to", vId : string, hstring=vstring);
    newMsg = new messaging.msg(vId);
    newMsg.COMMAND = messaging.command.SET_ID;
    this.SEND(newMsg, iM);
    return vp;
  }

  proc runDONOTUSE() {
    while true {
      this.receiveMessage();
    }
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

  proc init(n: int) {
    this.size = n;
  }

  proc initSendSocket(i: int) {
    this.socket[i] = this.context.socket(ZMQ.REQ);
    this.socket[i].bind("tcp://*:*");
    this.ports[i] = this.socket[i].getLastEndpoint().replace("0.0.0.0",chpl_nodeName():string);
  }

  proc initRecvSocket(i: int, port: string) {
    // so, we're going to set up and use a random port.
    this.socket[i] = this.context.socket(ZMQ.REP);
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
    m.exists = -1;
    m = this.socket[i].recv(msg);
    while m.exists == -1 do chpl_task_yield();
    return m;
  }

  proc RECV(i: int) { return this.__RECV__(i); }
  proc RECV() { return this.__RECV__(this.mId); }

  proc valhalla(i: int, vId: string, mSize : string, vLog: ygglog.YggdrasilLogging, vstring: ygglog.yggHeader) {
    // ha ha, cause Valkyries are in Valhalla, get it?  Get it?
    // ... no?
    // set up a ZMQ client/server
    var iM: int = i; //+(maxValkyries*here.id);
    vLog.log("Initializing sockets", hstring=vstring);
    this.initSendSocket(iM);
    //this.initUnlinkedRecvSocket(iM);

    vLog.log("Spawning Valkyrie", hstring=vstring);
    var vp: subprocess(kind=iokind.dynamic, locking=true);
    vp = spawn(["./v.sh", this.ports[iM], this.ports[iM], mSize : string], stdout=FORWARD, stderr=FORWARD, stdin=FORWARD, locking=true);

    vLog.log("SPAWN COMMAND:", "./valkyrie", "--recvPort", this.ports[iM], "--sendPort", this.ports[iM], "--vSize", mSize : string, hstring=vstring);

    var newMsg = new messaging.msg(i);
    newMsg.COMMAND = messaging.command.SET_TASK;
    vLog.log("Setting task to", i : string, hstring=vstring);
    this.SEND(newMsg, iM);
    this.receiveMessage();

    vLog.log("Setting ID to", vId : string, hstring=vstring);
    newMsg = new messaging.msg(vId);
    newMsg.COMMAND = messaging.command.SET_ID;
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
