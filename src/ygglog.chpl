
use spinlock;
use IO;
use Time;

class YggdrasilLogging {
  // This is a class to let us handle all input and output.
  var currentDebugLevel: int;
  var DEVEL = -1;
  var DEBUG = 0;
  var WARNING = 1;
  var RUNTIME = 2;
  var maxCharacters = 160;
  var headerLiner = '_';
  var headerStarter = '/';
  var indent = 5;
  var l = new shared spinlock.SpinLock();
  var tId: int;
  //var wc: channel;
  var filesOpened: domain(string);
  var channelsOpened: [filesOpened] channel(true,iokind.dynamic,true);
  var channelDebugHeader: [filesOpened] string;
  var fileHandles: [filesOpened] file;
  var lastDebugHeader = '';
  var time = Time.getCurrentTime();
  //var l: [filesOpened] spinlock.SpinLock;

  proc init() {
    //this.filesOpened = new domain(string);
    //this.channelsOpened = [filesOpened] channel(true,iokind.dynamic,true);
    //this.filesOpened.add('stdout');
    //this.channelsOpened['stdout'] = stdout;
  }

  proc formatHeader(mstring: string, mtype: string) {
    // Formats a header for us to print out to stdout.
    var header = ' '.join(this.headerStarter*5, mtype, this.headerStarter);
    var nToEnd = this.maxCharacters - header.size;
    header = header + (this.headerStarter*nToEnd);
    return header;
  }

  proc printToConsole(msg, debugLevel: string, hstring: string) {
    // check whether we're going to stdout or not.
    if this.lastDebugHeader == '' {
      if !this.filesOpened.contains('stdout') {
        this.filesOpened.add('stdout');
        this.channelsOpened['stdout'] = open('EVOCAP.log', iomode.cw).writer();
      }
    }
    var wc = stdout;
    var useStdout: bool = true;
    var s = hstring.split('----');
    //var s2 = hstring.split('--');
    var vstring = hstring;
    var lf: file;
    var lastDebugHeader: string;
    l.lock();
    var id: string;
    if s[1] == 'EVOCAP' {
      id = s[2];
      // First, check to see whether we've created the file.
      if this.filesOpened.contains(id) {
        if propagator.unitTestMode {
          // if we're in debug mode, we close the channels.
          // Otherwise, we leave them open.  It's for exception handling.
          var fileSize = this.fileHandles[id].length();
          this.channelsOpened[id] = this.fileHandles[id].writer(start=fileSize);
        }
        wc = this.channelsOpened[id];
        if propagator.stdoutOnly {
          wc = stdout;
        }
      } else {
        lf = open('logs/V-' + s[3] + '.log' : string, iomode.cw);
        this.filesOpened.add(id);
        this.channelsOpened[id] = lf.writer();
        this.fileHandles[id] = lf;
        wc = this.channelsOpened[id];
        if propagator.stdoutOnly {
          wc = stdout;
        }
        // First Valkyrie!
        wc.writeln('VALKYRIE TASK: ' + s[3] : string + ' ID: ' + s[2] : string);
        wc.writeln('');
      }
      vstring = s[4];
      useStdout = false;
      //lastDebugHeader = this.channelDebugHeader[s[2]];
    } else {
      id = 'stdout';
    }
    vstring = '%010.2dr'.format(Time.getCurrentTime() - this.time) + ' -- ' + vstring;
    var tm: int;
    if debugLevel != this.channelDebugHeader[id] {
      wc.writeln(this.formatHeader(vstring, debugLevel));
      if useStdout {
        this.channelsOpened[id].writeln(this.formatHeader(vstring, debugLevel));
      }
      this.channelDebugHeader[id] = debugLevel;
    }
    if vstring != '' {
        wc.write(' '*(this.indent+1), vstring, ' : ');
        if useStdout {
          this.channelsOpened[id].write(' '*(this.indent+1), vstring, ' : ');
        }
        tm = (' '*(this.indent+1) + vstring + ' : ').size;
    } else {
        wc.write(' '*(this.indent+1), 'YGGDSL : ');
        if useStdout {
          this.channelsOpened[id].write(' '*(this.indent+1), 'YGGDSL : ');
        }
        tm = (' '*(this.indent+1) + 'YGGDSL : ').size;
    }
    for im in msg {
      for m in im.split(maxsplit = -1) {
        if tm + m.size > this.maxCharacters {
          wc.writeln('');
          wc.write(' '*this.indent*3);
          if useStdout {
            this.channelsOpened[id].writeln('');
            this.channelsOpened[id].write(' '*this.indent*3);

          }
          tm = this.indent*3;
        }
        tm += m.size+1;
        wc.write(m : string, ' ');
        if useStdout {
          this.channelsOpened[id].write(m : string, ' ');
        }
      }
    }
    wc.writeln('');
    if useStdout {
      this.channelsOpened[id].writeln('');
    }
    if id != 'stdout' {
      //writeln(wc.type : string);
      if propagator.unitTestMode {
        // If we're in debug mode, sync the file every time.
        // This ensures that if/when we fail out, our logs are complete.
        if !propagator.stdoutOnly {
          // We can also just bail on the logging and only use stdout.
          wc.close();
        }
        this.fileHandles[id].fsync();
      }
        //wc.commit();
        //wc.close();
        //lf.close();
    }
    l.unlock();
  }

  proc noSpecialPrinting(msg, debugLevel: string, hstring: string) {
    // check whether we're going to stdout or not.
    if this.lastDebugHeader == '' {
      if !this.filesOpened.contains('stdout') {
        this.filesOpened.add('stdout');
        this.channelsOpened['stdout'] = open('EVOCAP.log', iomode.cw).writer();
      }
    }
    var wc = stdout;
    var useStdout: bool = true;
    var s = hstring.split('----');
    var vstring: string;
    var lf: file;
    var lastDebugHeader: string;
    l.lock();
    var id: string;
    if s[1] == 'EVOCAP' {
      id = s[2];
      // First, check to see whether we've created the file.
      if this.filesOpened.contains(id) {
        wc = this.channelsOpened[id];
      } else {
        lf = open('logs/V-' + s[3] + '.log' : string, iomode.cw);
        this.filesOpened.add(id);
        this.channelsOpened[id] = lf.writer();
        wc = this.channelsOpened[id];
        wc.writeln('VALKYRIE TASK: ' + s[3] : string + ' ID: ' + s[2] : string);
        wc.writeln('');
      }
      vstring = s[4];
      useStdout = false;
    } else {
      id = 'stdout';
    }
    var tm: int;
    if debugLevel != this.channelDebugHeader[id] {
      wc.writeln(this.formatHeader(vstring, debugLevel));
      if useStdout {
        this.channelsOpened[id].writeln(this.formatHeader(vstring, debugLevel));
      }
      this.channelDebugHeader[id] = debugLevel;
    }
    // We're not splitting; just printing.
    for m in msg {
    //  wc.writeln('');
      wc.write(' '*(this.indent+1));
      if useStdout {
        //this.channelsOpened[id].writeln('');
        this.channelsOpened[id].write(' '*(this.indent+1));
      }
      wc.writeln(m : string, ' ');
      if useStdout {
        this.channelsOpened[id].writeln(m : string, ' ');
      }
    }
    l.unlock();
  }


  proc genericMessage(msg, mtype: int, debugLevel: string, gt: bool) {
    if gt {
      if this.currentDebugLevel <= mtype {
        this.printToConsole(msg, debugLevel, hstring='');
      }
    } else {
      if this.currentDebugLevel == mtype {
        this.printToConsole(msg, debugLevel, hstring='');
      }
    }
  }

  proc genericMessage(msg, mtype: int, debugLevel: string, hstring: string, gt: bool) {
    if gt {
      if this.currentDebugLevel <= mtype {
        this.printToConsole(msg, debugLevel, hstring);
      }
    } else {
      if this.currentDebugLevel == mtype {
        this.printToConsole(msg, debugLevel, hstring);
      }
    }
  }

  proc debug(msg...?n) {
    this.genericMessage(msg, this.DEBUG, 'DEBUG', gt=true);
  }

  proc debug(msg...?n, hstring: string = '') {
    this.genericMessage(msg, this.DEBUG, 'DEBUG', hstring, gt=true);
  }

  proc devel(msg...?n) {
    this.genericMessage(msg, this.DEVEL, 'DEVEL', gt=false);
  }

  proc devel(msg...?n, hstring: string = '') {
    this.genericMessage(msg, this.DEVEL, 'DEVEL', hstring, gt=false);
  }

  proc warning(msg...?n) {
    this.genericMessage(msg, this.WARNING, 'WARNING', gt=true);
  }

  proc warning(msg...?n, hstring: string = '') {
    this.genericMessage(msg, this.WARNING, 'WARNING', hstring, gt=true);
  }

  proc log(msg...?n) {
    this.genericMessage(msg, this.RUNTIME, 'RUNTIME', gt=true);
  }

  proc log(msg...?n, hstring: string = '') {
    this.genericMessage(msg, this.RUNTIME, 'RUNTIME', hstring, gt=true);
  }

  proc critical(msg...?n, hstring: string = '') {
    this.genericMessage(msg, this.currentDebugLevel, 'CRITICAL FAILURE', hstring, gt=true);
  }

  proc critical(msg...?n) {
    this.genericMessage(msg, this.currentDebugLevel, 'CRITICAL FAILURE', hstring='', gt=true);
  }

  proc header(msg...?n) {
    this.noSpecialPrinting(msg, 'RUNTIME', hstring='');
  }
  proc header(msg...?n, hstring: string = '') {
    this.noSpecialPrinting(msg, 'RUNTIME', hstring=hstring);
  }

}
