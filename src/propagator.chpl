// Do I need all this?  Probably not but I'll sort that issue later.

use rng;
use genes;
use network;
use uuid;
use Math;

config const mSize = 20;
config var maxPerGeneration = 400;
config var mutationRate = 0.03;
config var maxValkyries = 1;
config var startingSeeds = 10;
config var createEdgeOnMove = true;
config var edgeDistance = 2;

// As we have our tree of life, so too do we have winged badasses who choose
// who lives and who dies.
// (this is a record for a worker class)
record valkyrie {
  var matrixValues: [0..mSize-1] real;
  var takeAnyPath: bool = false;
  var moved: bool = false;
  var canMove: bool = false;
  //var matrixValues: [0..20] real;
  //var matrixValues: int;
  var currentNode: string;
  // Is there anything else I need?  Who knows!
  proc moveToRoot() {
    // Zero out the matrix, return the root id.
    this.matrixValues = 0;
    this.currentNode = 'root';
  }

  // Assume we're given a delta object so that we may express it.
  proc move(delta: deltaRecord, id: string) {
    delta.express(this.matrixValues);
    this.currentNode = id;
  }
}

class SpinLock {
  var l: atomic bool;

  inline proc lock() {
    while l.testAndSet(memory_order_acquire) do chpl_task_yield();
  }

  inline proc unlock() {
    l.clear(memory_order_release);
  }
}

class Propagator {
  // this is going to actually hold all the logic for running EvoCap.
  var generation: int;
  //var maxValkyries: int;

  // now a few globals.
  //var processedArray: [1..maxPerGeneration] atomic bool;
  var nodesToProcess: domain(string);
  var lock = new shared SpinLock();
  var processedArray: [nodesToProcess] atomic bool;
  //var nodesToProcess: [1..maxPerGeneration] string;
  var inCurrentGeneration: atomic int;
  var nextGeneration: domain(string);

  var ygg: shared network.GeneNetwork();

  proc init() {
    this.ygg = new shared network.GeneNetwork();
    this.ygg.initializeNetwork(n_seeds=startingSeeds);
    //this.processedArray.write(true);
    var ids = this.ygg.ids;
    ids.remove('root');
    //this.inCurrentGeneration.add(1);
    for i in ids {
      //this.nodesToProcess[this.inCurrentGeneration.read()+1] = i;
      this.processedArray[i].write(false);
      this.inCurrentGeneration.add(1);
    }
    //writeln(this.ygg.nodes);
    writeln(this.inCurrentGeneration);
  }

  proc run() {
    // start up the main procedure by creating some valkyries.
    //var v: new valkyrieRecord;
    // declaring this variable seems to just kill it.  Entirely.  WHY.
    //for i in 1..maxValkyries {
    //  for j in this.ygg.ids {
    //    writeln(j);
    //  }
    //forall i in 1..maxValkyries with ( var v: valkyrie ) {
    var nnodes: atomic int;
    coforall i in 1..maxValkyries {
      //writeln(v.matrixValues);
      var v = new valkyrie;
      v.moveToRoot();
      for gen in 1..998 {
        var calculatedDistance: bool = false;
        var currToProc: string;
        var currMin: real = Math.INFINITY : real;
        var pathDomain: domain(string);
        var pathSet: [pathDomain] network.pathHistory;
        //var pathSet: network.pathHistory;
        pathDomain.clear();
        // first, calculate the distance from the current node to all others.
        if !calculatedDistance {
          var toProcess: domain(string);
          for id in this.nodesToProcess {
            if !this.processedArray[id].read() {
              // If we have yet to process it, sort it out.
              toProcess.add(id);
            }
          }
          pathSet += this.ygg.calculatePathArray(v.currentNode, toProcess);
          calculatedDistance = true;
        }
        while this.inCurrentGeneration.read()!= 0 {
          currMin =  Math.INFINITY;
          for id in this.nodesToProcess {
            //writeln(pathSet);
            if pathSet[id].distance() < currMin {
              currMin = pathSet[id].distance();
              currToProc = id;
            }
          }
          // try it now!
          // look, I know this will break it.
          //writeln(this.processedArray[currToProc].testAndSet());
          if !this.processedArray[currToProc].testAndSet() {
            this.lock.lock();
            this.nodesToProcess.remove(currToProc);
            this.lock.unlock();
            pathDomain.remove(currToProc);
            writeln('TASK ', i, ', SEED # ', this.ygg.nodes[currToProc].debugOrderOfCreation, ' : ', v.matrixValues);
            this.ygg.move(v, currToProc, createEdgeOnMove, edgeDistance);
            this.inCurrentGeneration.sub(1);
            this.lock.lock();
            var nextNode = this.ygg.nextNode(currToProc);
            this.nextGeneration.add(nextNode);
            this.lock.unlock();

            if this.inCurrentGeneration.read() == 0 {
              break;
            }
          }
          pathDomain.remove(currToProc);
          currMin =  Math.INFINITY;
          currToProc = '';
        }
        // now, switch over the list.
        this.lock.lock();
        //pathDomain.clear();
        if this.nextGeneration.size != 0 {
          this.nodesToProcess = this.nextGeneration;
          this.nextGeneration.clear();
          this.inCurrentGeneration.write(this.nodesToProcess.size);
          this.processedArray.write(false);
        } // otherwise, we assume we already did it.  Because we did.
        this.lock.unlock();



      }
      //writeln(nnodes.read());
    }
  }

}
