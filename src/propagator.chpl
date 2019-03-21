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
config var generations = 100;

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
  var valkyriesDone: sync int = maxValkyries;
  var moveOn: [1..generations] single bool;

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
    //this.valkyriesDone = maxValkyries;
    //this.valkyriesDone.reset();
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
      for gen in 1..generations {
        //this.moveOn = false;
        this.log.debug('Starting GEN', gen : string, hstring=v.header);
        var calculatedDistance: bool = false;
        var currToProc: string;
        var currMin: real = Math.INFINITY : real;
        var pathDomain: domain(string);
        var pathSet: [pathDomain] network.pathHistory;
        var distance: [pathDomain] real;
        var currNode: int;
        var dSorted: [0] real;
        var toProcess: domain(string);
        //var pathSet: network.pathHistory;
        this.log.debug('Determining who needs processing', hstring=v.header);
        // first, calculate the distance from the current node to all others.
        for id in this.nodesToProcess {
          if !this.processedArray[id].read() {
            // If we have yet to process it, sort it out.
            toProcess.add(id);
          }
        }
        this.log.debug('Beginning processing', hstring=v.header);
        //pathSet += this.ygg.calculatePathArray(v.currentNode, toProcess, v.header);
        // try it now!
        // look, I know this will break it.
        //writeln(this.processedArray[currToProc].testAndSet());
        // If we can't get anything, that means we're just waiting for things to have finished processing.
        // we just want the sorted bit.
        //this.log.debug(pathDomain.isEmpty() : string, hstring=v.header);
        while !toProcess.isEmpty() {
          // This will just return the closest one, and is really all we need.
          currToProc = this.ygg.returnNearestUnprocessed(v.currentNode, toProcess, v.header);
          this.log.debug('Attempting to unlock node', currToProc, hstring=v.header);
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
            //this.lock.lock(v.header);
            this.log.debug('Attempting to create another node', hstring=v.header);
            var nextNode = this.ygg.nextNode(currToProc, hstring=v.header);
            this.log.debug('Node added; attempting to increase count for nextGeneration', hstring=v.header);
            this.nextGeneration.add(nextNode);
            this.inCurrentGeneration.sub(1);
            //this.lock.unlock(v.header);
            //if this.inCurrentGeneration.read() == 0 {
            //  break;
            //}
          }
          this.log.debug('Removing node from list to process', currToProc, hstring=v.header);
          toProcess.remove(currToProc);
          currToProc = '';
        }
        //this.valkyriesDone.sub(1);
        var vd = this.valkyriesDone;
        if vd != 1 {
          this.log.debug('Waiting in gen', gen : string, v.header);
          this.valkyriesDone = vd - 1;
          this.moveOn[gen];
          this.log.debug('MOVING ON in gen', gen : string, v.header);
        } else {
          // time to move the fuck on.
          this.moveOn[gen] = true;
          this.valkyriesDone = maxValkyries;
          //this.lock.lock(v.header);
          // reset that shit, yo.
          this.log.debug('Switching generations', v.header);
          this.nodesToProcess = this.nextGeneration;
          this.nextGeneration.clear();
          this.inCurrentGeneration.write(this.nodesToProcess.size);
          this.processedArray.write(false);
          //this.lock.unlock(v.header);
          this.log.log('GEN', (gen + 1) : string, this.inCurrentGeneration.read() : string, "to process", hstring='NormalRuntime');
        }
        // now, switch over the list.
        //this.log.log('VALKYRIE DONE in gen ', gen : string, v.header);
        //var vd = this.valkyriesDone; // should block until we're done.
        // nice dream, if I say so myself.
        //while this.valkyriesDone.read() != 0 do {
        //  this.log.log('VALKYRIE Sleeping in gen ', gen : string, maxValkyries : string, this.valkyriesDone.read() : string, v.header);
        //  chpl_task_yield();
        //}
        //pathDomain.clear();
         // otherwise, we assume we already did it.  Because we did.
        //this.log.debug();
      }
      //writeln(nnodes.read());
    }
    stopVdebug();
  }

}
