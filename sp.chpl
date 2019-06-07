

use Spawn;
use Time;
use messaging;
use ygglog;


config const useValhalla: bool = false;
config const useCSpawn: bool = true;
config const maxTasks: int = 24;

extern proc cSpawn() : c_int;


require "sp.c";

class vSpawner {

  var areSpawned: single bool;
  var numSpawned: atomic int;

  proc run() {
    if useCSpawn {
      this.runCSpawn();
    } else if useValhalla {
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
          vLog.currentDebugLevel = 0;
          // also, spin up the tasks.
          //this.lock.wl(v.header);
          var yh = new ygglog.yggHeader();
          var t: real = Time.getCurrentTime();
          var vp = mH.valhalla(1, i : string, 33483 : string, vLog, vstring=yh);
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
          //var vp = spawn(["./v.sh"], stdout=FORWARD, stderr=FORWARD, stdin=FORWARD, locking=false);
          var vp = spawn(["bash", "v.sh"], stdout=FORWARD, stderr=FORWARD, stdin=FORWARD, locking=false);
          vp.wait();
          writeln("Hello from task %i on ".format(i) + here.id : string + "; done in %r time!".format(Time.getCurrentTime() - t));
        }
      }
    }
  }
  proc runCSpawn() {

    coforall L in Locales {
      on L do {
        coforall i in 1..maxTasks {
          var t: real = Time.getCurrentTime();
          cSpawn();
          //var vp = spawn(["./v.sh"], stdout=FORWARD, stderr=FORWARD, stdin=FORWARD, locking=false);
          //vp.wait();
          writeln("Hello from task %i on ".format(i) + here.id : string + "; done in %r time!".format(Time.getCurrentTime() - t));
        }
      }
    }
  }
}

var vs = new owned vSpawner();
vs.run();
