
use spinlock;

class YggdrasilLogging {
  // This is a class to let us handle all input and output.
  var currentDebugLevel: int;
  var lastDebugHeader = '';
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

  proc formatHeader(mstring: string, mtype: string) {
    // Formats a header for us to print out to stdout.
    var header = ' '.join(this.headerStarter*5, mtype, this.headerStarter);
    var nToEnd = this.maxCharacters - header.size;
    header = header + (this.headerStarter*nToEnd);
    return header;
  }

  proc printToConsole(msg, debugLevel: string, hstring: string) {
    l.lock();
    if debugLevel != this.lastDebugHeader {
      writeln(this.formatHeader(hstring, debugLevel));
      this.lastDebugHeader = debugLevel;
    }
    if hstring != '' {
        write(' '*(this.indent+1), hstring, ' : ');
    } else {
        write(' '*(this.indent+1), 'YGGDSL : ');
    }
    var tm = this.indent;
    for im in msg {
      for m in im.split(maxsplit = -1) {
        if tm + m.size > this.maxCharacters {
          writeln('');
          write(' '*this.indent*3);
          tm = this.indent*3;
        }
        tm += m.size+1;
        write(m : string, ' ');
      }
    }
    writeln('');
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

}
