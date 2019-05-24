// This is just a source file for our messages.
// It's essentially a self executing record.

use genes;

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
}

var status: statusRecord;
var command: commandRecord;

record msg {
  // there are two types of records; commands and results.
  var STATUS: int;
  var COMMAND: int;
}

class msgHandler {

  //var channelsOpened: [filesOpened] channel(true,iokind.dynamic,true);

  var size: int = 1;
  // this is for single receivers/senders.
  var id: int = 0;

  var fin: [0..size] channel(false,iokind.dynamic,true);
  var fout: [0..size] channel(true,iokind.dynamic,true);
  var STATUS: int;


  proc initblah(n: int) {
    // makes sure that our array is good.
    this.size = n;
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

  proc __receiveMessage__(i: int) {
    // listen for a message.
    var m: msg;
    fin[i].readln(m);
    this.__PROCESS__(m, i);
  }

  proc __OK__(i: int) {
    fout[i].writeln(status.OK);
    fout[i].flush();
  }

  proc __SEND_STATUS__(s: int, i: int) {
    fout[i].writeln(s);
    fout[i].flush();
  }

  proc __RECV_STATUS__(i: int) {
    var s: int;
    // ?  This is not working.
    fin[i].readln(s);
    //return s;
    if s == status.OK {
      // yay
    }
  }

  // convenient handler function

  proc receiveMessage(i: int) { this.__receiveMessage__(i); }
  proc receiveMessage() { this.__receiveMessage__(this.id); }

  proc OK(i: int) { this.__OK__(i); }
  proc OK() { this.__OK__(this.id); }

  proc SEND_STATUS(s:int, i: int) { this.__SEND_STATUS__(s,i); }
  proc SEND_STATUS(s: int) { this.__SEND_STATUS__(s, this.id); }

  proc RECV_STATUS(i: int) { this.__RECV_STATUS__(i); }
  proc RECV_STATUS() { this.__RECV_STATUS__(this.id); }

  // these are stubs the children should implement, if necessary.
  // Each time a message is considered received by process, it _must_
  // end with a status okay.

  proc __PROCESS__(m: msg, i: int) { OK(i); this.PROCESS(m, i); }
  proc __PROCESS__(m: msg) { OK(this.id); this.PROCESS(m); }

  proc PROCESS(m: msg, i: int) { }
  proc PROCESS(m: msg) { }

  // these are functions for sending.  Essentially, all listen functions
  // must receive a status, or they are blocked.

  proc __SEND__(m: msg, i: int) {
    // send the message!
    fout[i].writeln(m);
    fout[i].flush();
    this.RECV_STATUS(i);
  }

  proc __SEND__(d: genes.deltaRecord, i: int) {
    // send the message!
    fout[i].writeln(d);
    fout[i].flush();
    this.RECV_STATUS(i);
  }

  proc __SEND__(j: int, i: int) {
    // send the message!
    fout[i].writeln(j);
    fout[i].flush();
    this.RECV_STATUS(i);
  }

  proc __SEND__(j: real, i: int) {
    // send the message!
    fout[i].writeln(j);
    fout[i].flush();
    this.RECV_STATUS(i);
  }

  proc SEND(m: msg, i: int) { this.__SEND__(m, i); }
  proc SEND(m: msg) { this.__SEND__(m, this.id); }

  proc SEND(d: genes.deltaRecord, i: int) { this.__SEND__(d, i); }
  proc SEND(d: genes.deltaRecord) { this.__SEND__(d, this.id); }

  proc SEND(j: int, i: int) { this.__SEND__(j, i); }
  proc SEND(j: int) { this.__SEND__(j, this.id); }

  proc SEND(j: real, i: int) { this.__SEND__(j, i); }
  proc SEND(j: real) { this.__SEND__(j, this.id); }

  // these are functions for receiving.  Essentially, all listen functions
  // must receive a status, or they are blocked.

  proc __RECV__(ref m: msg, i: int) {
    // receive the message!
    fin[i].readln(m);
    OK(i);
  }

  proc __RECV__(ref d: genes.deltaRecord, i: int) {
    // receive the message!
    fin[i].readln(d);
    OK(i);
  }

  proc __RECV__(ref j: int, i: int) {
    // receive the message!
    fin[i].readln(j);
    OK(i);
  }

  proc __RECV__(ref j: real, i: int) {
    // receive the message!
    fin[i].readln(j);
    OK(i);
  }

  proc RECV(ref m: msg, i: int) { this.__RECV__(m, i); }
  proc RECV(ref m: msg) { this.__RECV__(m, this.id); }

  proc RECV(ref d: genes.deltaRecord, i: int) { this.__RECV__(d, i); }
  proc RECV(ref d: genes.deltaRecord) { this.__RECV__(d, this.id); }

  proc RECV(ref j: int, i: int) { this.__RECV__(j, i); }
  proc RECV(ref j: int) { this.__RECV__(j, this.id); }

  proc RECV(ref j: real, i: int) { this.__RECV__(j, i); }
  proc RECV(ref j: real) { this.__RECV__(j, this.id); }
}
