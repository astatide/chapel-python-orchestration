
class SpinLock {
  var l: atomic bool;

  inline proc lock() {
    while l.testAndSet(memory_order_acquire) do chpl_task_yield();
  }

  inline proc unlock() {
    l.clear(memory_order_release);
  }
}

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
  var l = new shared SpinLock();
  var tId: int;

  proc formatHeader(tId: int, mtype: string) {
    // Formats a header for us to print out to stdout.
    var header = ' '.join(this.headerStarter*5, mtype, ':', 'tId ', tId : string, ' ');
    var nToEnd = this.maxCharacters - header.size;
    header = header + (this.headerStarter*nToEnd);
    return header;
  }

  proc printToConsole(msg, debugLevel: string, tId: int) {
    l.lock();
    if debugLevel != this.lastDebugHeader {
      writeln(this.formatHeader(tId, debugLevel));
      this.lastDebugHeader = debugLevel;
    }
    if tId >= 0 {
        write(' '*(this.indent+1), 'TASK ', tId: string, ' : ');
    } else {
        write(' '*(this.indent+1), 'YGGDSL : ');
    }
    var tm = this.indent;
    //for m in msg.split(maxsplit = -1) {
    for im in msg {
      for m in im.split(maxsplit = -1) {
        if tm + m.size > this.maxCharacters {
          writeln('');
          write(' '*this.indent*3);
          tm = this.indent*3;
        }
        tm += m.size+1;
        write(m, ' ');
      }
    }
    writeln('');
    l.unlock();
  }

  proc debug(msg...?n) {
    var nmsg: [1..n-1] string;
    if this.currentDebugLevel <= this.DEBUG {
      if msg[msg.size].type == int {
        tId = msg[n];
        //msg = msg[1..(msg.size-1)];
        //msg.domreturnremove(msg.size);
        for param m in 1..n-1 {
          nmsg[m] = msg[m];
        }
        this.printToConsole(nmsg, 'DEBUG', tId);
      } else {
        tId = -1;
        this.printToConsole(msg, 'DEBUG', tId);
      }
    }
  }

  proc warning(msg...?n, tId: int = 0) {
    if this.currentDebugLevel <= this.WARNING {
      this.printToConsole(msg, 'WARNING', tId);
    }
  }

  proc print(msg...?n, tId: int = 0) {
    if this.currentDebugLevel <= this.RUNTIME {
      this.printToConsole(msg, 'RUNTIME', tId);
    }
  }

  proc log(msg...?n, tId: int = 0) {
    if this.currentDebugLevel <= this.RUNTIME {
      this.printToConsole(msg, 'RUNTIME', tId);
    }
  }
}
