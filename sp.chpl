

use Spawn;
use Time;

config const useValhalla: bool = false;
config const maxTasks: int = 24;


class vSpawner {

  var areSpawned: single bool;
  var numSpawned: atomic int;

  proc run() {
    if useValhalla {
      this.runValhalla();
    } else {
      this.runSansValhalla();
    }
  }

  proc runValhalla() {

    coforall L in Locales {
      on L do {
        coforall i in 1..maxTasks {
          var mH = new messaging.msgHandler(1);
          var vLog = new shared ygglog.YggdrasilLogging();
          vLog.currentDebugLevel = debug;
          var vLock = new shared spinlock.SpinLock();
          vLock.t = 'Valkyrie';
          var v = new valkyrie();
          v.currentTask = i;
          v.currentLocale = L : string;
          v.yh += 'run';
          for iL in v.logo {
            vLog.header(iL, hstring=v.header);
          }
          // also, spin up the tasks.
          //this.lock.wl(v.header);
          var t: real = Time.getCurrentTime();
          var vp = mH.valhalla(1, v.id, mSize : string, vLog, vstring=v.header);
          //var vp = spawn(["./v.sh", this.sendPorts[iM], this.recvPorts[iM], mSize : string], stdout=FORWARD, stderr=FORWARD, stdin=FORWARD, locking=true);
          writeln("Hello from task %i on ".format(i) + here.id : string + "; done in %r time!".format(Time.getCurrentTime() - t));
        }
      }
    }
  }

  proc runSansValhalla() {

    coforall L in Locales {
      on L do {
        coforall i in 1..maxTasks {
          var t: real = Time.getCurrentTime();
          var vp = spawn(["./v.sh", this.sendPorts[iM], this.recvPorts[iM], 33483 : string], stdout=FORWARD, stderr=FORWARD, stdin=FORWARD, locking=false);
          writeln("Hello from task %i on ".format(i) + here.id : string + "; done in %r time!".format(Time.getCurrentTime() - t));
        }
      }
    }
  }
}

var vs = new owned vSpawner();
vs.run();
