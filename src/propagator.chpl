// Do I need all this?  Probably not but I'll sort that issue later.

use rng;
use genes;
use network;
use uuid;
use Math;
use VisualDebug;
use ygglog;
use spinlock;

config const mSize = 20;
config var maxPerGeneration = 400;
config var mutationRate = 0.03;
config var maxValkyries = 1;
config var startingSeeds = 10;
config var createEdgeOnMove = true;
config var edgeDistance = 2;
config var debug = -1;

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

  var currentTask: int;
  var nMoves: int;
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

  proc header {
    return ' '.join('V', this.currentTask : string, 'M', this.nMoves : string);
  }
}

class Propagator {
  // this is going to actually hold all the logic for running EvoCap.
  var generation: int;
  //var maxValkyries: int;

  // now a few globals.
  //var processedArray: [1..maxPerGeneration] atomic bool;
  var nodesToProcess: domain(string);
  var processedArray: [nodesToProcess] atomic bool;
  //var nodesToProcess: [1..maxPerGeneration] string;
  var inCurrentGeneration: atomic int;
  var nextGeneration: domain(string);

  var ygg: shared network.GeneNetwork();
  var log: shared ygglog.YggdrasilLogging();
  var lock: shared spinlock.SpinLock;

  proc init() {
    this.ygg = new shared network.GeneNetwork();
    this.log = new shared ygglog.YggdrasilLogging();
    this.log.currentDebugLevel = debug;
    this.ygg.log = this.log;
    this.ygg.lock.log = this.log;
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
    this.log.debug('INITIALIZED', this.inCurrentGeneration.read() : string, 'seeds.', hstring='Ragnarok');
    this.lock = new shared spinlock.SpinLock();
    this.lock.t = 'Ragnarok';
    this.lock.log = this.log;
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
    startVdebug("E2");
    //this.log.debug('STARTING YGGDRASIL');
    var nnodes: atomic int;
    coforall i in 1..maxValkyries {
      //writeln(v.matrixValues);
      var v = new valkyrie();
      v.currentTask = i;
      v.moveToRoot();
      for gen in 1..100 {
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
          pathSet += this.ygg.calculatePathArray(v.currentNode, toProcess, v.header);
          calculatedDistance = true;
        }
        while this.inCurrentGeneration.read()!= 0 {
          currMin =  Math.INFINITY;
            this.lock.lock();
          for id in this.nodesToProcess {
            //writeln(pathSet);
            if pathDomain.member(id) {
              if pathSet[id].distance() < currMin {
                currMin = pathSet[id].distance();
                currToProc = id;
              }
            }
          }
          this.lock.unlock();
          // try it now!
          // look, I know this will break it.
          //writeln(this.processedArray[currToProc].testAndSet());
          // If we can't get anything, that means we're just waiting for things to have finished processing.
          if currToProc != '' {
            if !this.processedArray[currToProc].testAndSet() {
              this.lock.lock(v.header);
              this.nodesToProcess.remove(currToProc);
              this.lock.unlock(v.header);
              //pathDomain.remove(currToProc);
              //writeln('TASK ', i, ', SEED # ', this.ygg.nodes[currToProc].debugOrderOfCreation, ' : ', v.matrixValues);
              //this.log.log(' '.join('TASK', i : string, 'SEED #', currToProc : string, ':', v.matrixValues : string), i);
              //this.log.tId = i;
              //this.log.debug('SEED #', currToProc : string, hstring=' '.join('TASK', i : string));
              this.log.debug('SEED #', currToProc : string, hstring=v.header);
              //this.log.devel('Hey, so, this is like, a test, you know what I mean?  I want a lot of things here.  Lots and lots of big things.  Things that will definitely test out the logging infrastructure.  Look, I know that you are tired.  I know that you are scared.  Hell, I am, too.  We are all scared.  We are all tired.  But we have to keep fighting.  We have to keep testing this.  It really is the only way to debug this.  So buck up.  Chin up.  Pull your little kitten arms up.');
              //this.log.debug(v.matrixValues : string, i);

              //this.log.debug('STARTING TO MOVE');
              this.ygg.move(v, currToProc, createEdgeOnMove, edgeDistance);
              //this.inCurrentGeneration.sub(1);
              this.lock.lock(v.header);
              this.log.debug('Attempting to create another node', hstring=v.header);
              var nextNode = this.ygg.nextNode(currToProc, hstring=v.header);
              this.log.debug('Node added; attempting to increase count for nextGeneration', hstring=v.header);
              this.nextGeneration.add(nextNode);
              this.inCurrentGeneration.sub(1);
              this.lock.unlock(v.header);
              //if this.inCurrentGeneration.read() == 0 {
              //  break;
              //}
            }
          }
          //pathDomain.remove(currToProc);
          currMin =  Math.INFINITY;
          currToProc = '';
        }
        // now, switch over the list.
        this.lock.lock(v.header);
        //pathDomain.clear();
        if this.inCurrentGeneration.read() == 0 {
          this.log.debug('Switching generations', v.header);
          this.nodesToProcess = this.nextGeneration;
          this.nextGeneration.clear();
          this.inCurrentGeneration.write(this.nodesToProcess.size);
          this.processedArray.write(false);
        } else {
          this.log.debug('Already swapped', v.header);
        } // otherwise, we assume we already did it.  Because we did.
        this.lock.unlock(v.header);



      }
      //writeln(nnodes.read());
    }
    stopVdebug();
  }

}
