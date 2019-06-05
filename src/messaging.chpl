// This is just a source file for our messages.
// It's essentially a self executing record.

use genes;
use ZMQ;
use ygglog;
use Spawn;

extern proc chpl_nodeName(): c_string;
config const awaitResponse = false;


record statusRecord {
  var OK: int = 0;
  var ERROR: int = 1;
  var IGNORED: int = 2;
}

record commandRecord {
  var RETURN_STATUS: int = 0;
  var SET_TASK: int = 1;
  var RECEIVE_AND_PROCESS_DELTA: int = 2;
  var RECEIVE_SCORE: int = 3;
  var SET_ID: int = 4;
  var SHUTDOWN: int = 5;
}

var status: statusRecord;
var command: commandRecord;

record msg {
  // there are two types of records; commands and results.
  var TYPE: int = 0;
  var ISINT: int = 1;
  var ISSTR: int = 2;
  var ISREL: int = 3;
  var ISDEL: int = 4;
  var STATUS: int;
  var COMMAND: int;
  var s: string;
  var i: int;
  var r: real;

  // this.complete() means we complete!

  proc init() {}

  proc init(s: string) {
    this.complete();
    this.TYPE = this.ISSTR;
    this.s = s;
  }
  proc init(i: int) {
    this.complete();
    this.TYPE = this.ISINT;
    this.i = i;
  }
  proc init(r: real) {
    this.complete();
    this.TYPE = this.ISREL;
    this.r = r;
  }
  proc init(d: genes.deltaRecord) {
    this.complete();
    this.TYPE = this.ISDEL;
    this.s = d : string;
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
  //var nChannels: domain(int);
  //var sendPorts: [nChannels] string;
  //var recvPorts: [nChannels] string;

  //var sendSocket: [nChannels] Socket;
  //var recvSocket: [nChannels] Socket;
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
    this.sendSocket[i] = this.context.socket(ZMQ.PUSH);
    this.sendSocket[i].bind("tcp://*:*");
    this.sendPorts[i] = this.sendSocket[i].getLastEndpoint().replace("0.0.0.0",chpl_nodeName():string);
  }

  proc initPrevSendSocket(i: int, port: string) {
    this.sendSocket[i] = this.context.socket(ZMQ.PUSH);
    this.sendSocket[i].connect(port);
    this.sendPorts[i] = port;
  }

  proc initRecvSocket(i: int, port: string) {
    // so, we're going to set up and use a random port.

    this.recvSocket[i] = this.context.socket(ZMQ.PULL);
    this.recvSocket[i].connect(port);
    this.recvPorts[i] = port;
  }

  proc initUnlinkedRecvSocket(i: int) {
    // so, we're going to set up and use a random port.

    this.recvSocket[i] = this.context.socket(ZMQ.PULL);
    this.recvSocket[i].bind("tcp://*:*");
    this.recvPorts[i] = this.recvSocket[i].getLastEndpoint().replace("0.0.0.0",chpl_nodeName():string);

  }

  proc setChannels(fin, fout, i /* takes a channel as input */) {
    this.fin[i] = fin;
    this.fout[i] = fout;
  }

  proc setChannels(fin, fout /* takes a channel as input */) {
    var i: int;
    this.fin[i] = fin;
    this.fout[i] = fout;
  }

  proc receiveLoop(i: int) {
  }

  proc __receiveMessage__(i: int) {
    // listen for a message.
    //var m: msg;
    //fin[i].readln(m);
    // this is a wee loop.
    var m = this.recvSocket[i].recv(msg);
    this.__PROCESS__(m, i);
  }

  proc __OK__(i: int) {
    this.sendSocket[i].send(status.OK);
    //fout[i].writeln(status.OK);
    //fout[i].flush();
  }

  proc __SEND_STATUS__(s: int, i: int) {
    this.sendSocket[i].send(s);
    //fout[i].writeln(s);
    //fout[i].flush();
  }

  proc __RECV_STATUS__(i: int) {
    //var s: int;
    // ?  This is not working.
    //fin[i].readln(s);
    var s = this.recvSocket[i].recv(int);
    //return s;
    if s == status.OK {
      // yay
    }
  }

  // convenient handler function

  proc receiveMessage(i: int) { this.__receiveMessage__(i); }
  proc receiveMessage() { this.__receiveMessage__(this.mId); }

  proc OK(i: int) { this.__OK__(i); }
  proc OK() { this.__OK__(this.mId); }

  proc SEND_STATUS(s:int, i: int) { this.__SEND_STATUS__(s,i); }
  proc SEND_STATUS(s: int) { this.__SEND_STATUS__(s, this.mId); }

  proc RECV_STATUS(i: int) { this.__RECV_STATUS__(i); }
  proc RECV_STATUS() { this.__RECV_STATUS__(this.mId); }

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

  proc __RECV__(ref m: msg, i: int) {
    // receive the message!
    //fin[i].readln(m);
    m = this.recvSocket[i].recv(msg);
    if awaitResponse {
      OK(i);
    }
    return true;
  }

  proc RECV(ref m: msg, i: int) { return this.__RECV__(m, i); }
  proc RECV(ref m: msg) { return this.__RECV__(m, this.mId); }

  proc valhalla(i: int, vId: string, mSize : string vLog: ygglog.YggdrasilLogging, vstring: ygglog.yggHeader) {
    // ha ha, cause Valkyries are in Valhalla, get it?  Get it?
    // ... no?
    // set up a ZMQ client/server
    var iM: int = i; //+(maxValkyries*here.id);
    vLog.log("Initializing sockets", hstring=vstring);
    this.initSendSocket(iM);
    this.initUnlinkedRecvSocket(iM);

    vLog.log("Spawning Valkyrie", hstring=vstring);
    var vp = spawn(["./v.sh", this.sendPorts[iM], this.recvPorts[iM], mSize : string], stdout=FORWARD, stderr=FORWARD, stdin=FORWARD, locking=false);

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

}
