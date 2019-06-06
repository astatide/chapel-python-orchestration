

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

config const mSize = 20;
config const maxPerGeneration = 10;
config const mutationRate = 0.03;
config const maxValkyries = 1;
config const startingSeeds = 4;
config const createEdgeOnMove = true;
config const edgeDistance = 10;
config const debug = -1;
config const generations = 100;
config const unitTestMode = false;
config const stdoutOnly = false;
// The locks are noisy, but we do need to debug them sometimes.
// This shuts them up unless you really want them to sing.  Their song is
// a terrible noise; an unending screech which ends the world.
// (okay, they're just super verbose)
config var lockLog = false;
config var flushToLog = false;

config var nChromosomes = 6;
config var chromosomeSize = 36;
config var nDuplicates = 4;

class A {
  proc blah {}
}


class vSpawner {

  proc run() {
    coforall L in Locales {
      on L do {
        coforall i in 1..maxValkyries {
          // spin up the Valkyries!
          //var yggLocalCopy = this.ygg.clone();
          var mH = new messaging.msgHandler(1);
          var vLog = new shared ygglog.YggdrasilLogging();
          vLog.currentDebugLevel = debug;
          var vLock = new shared spinlock.SpinLock();
          vLock.t = 'Valkyrie';
          //var yggLocalCopy = this.ygg.clone();
          // ?? This doesn't seem to actually be working.
          //yggLocalCopy.log = vLog;
          //yggLocalCopy.lock.log = vLog;
          var v = new valkyrie();
          v.currentTask = i;
          v.currentLocale = L : string;
          v.yh += 'run';
          for iL in v.logo {
            vLog.header(iL, hstring=v.header);
          }
          // also, spin up the tasks.
          //this.lock.wl(v.header);
          var vp = mH.valhalla(1, v.id, mSize : string, vLog, vstring=v.header);
          if this.numSpawned.fetchAdd(1) < ((Locales.size*maxValkyries)-1) {
            // we want to wait so that we spin up all processes.
            this.areSpawned;
          } else {
            this.areSpawned = true;
          }
          writeln("Hello from " + here.id : string + "; done!");
        }
      }
    }
  }
}

var vs = new owned vSpawner();
vs.run();
