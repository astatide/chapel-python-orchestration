// Do I need all this?  Probably not but I'll sort that issue later.

use rng;
use genes;
use network;
use uuid;
use Math;
use VisualDebug;
use ygglog;
use spinlock;
use Time;

config const mSize = 20;
config var maxPerGeneration = 400;
config var mutationRate = 0.03;
config var maxValkyries = 1;
config var startingSeeds = 10;
config var createEdgeOnMove = true;
config var edgeDistance = 10;
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
  var priorityNodes: domain(string);
  var nPriorityNodesProcessed: int;

  var currentTask: int;
  var nMoves: int;
  var nProcessed: int;
  var gen: int;
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
    return ' '.join('V', this.currentTask : string, 'M', this.nMoves : string, 'G', this.gen : string);
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
  var valkyriesDone: [1..generations] atomic int;
  var moveOn: [1..generations] single bool;
  var valkyriesProcessed: [1..maxValkyries] atomic int;
  var priorityValkyriesProcessed: [1..maxValkyries] atomic real;
  var generationTime: real;

  proc init() {
    this.ygg = new shared network.GeneNetwork();
    this.log = new shared ygglog.YggdrasilLogging();
    this.log.currentDebugLevel = debug;
    this.ygg.log = this.log;
    this.ygg.lock.log = this.log;
    this.ygg.initializeNetwork(n_seeds=startingSeeds);
    //this.processedArray.write(true);
    var ids = this.ygg.ids;
    //ids.remove('root');
    //this.inCurrentGeneration.add(1);
    for i in this.ygg.ids {
      //this.nodesToProcess[this.inCurrentGeneration.read()+1] = i;
      if i != 'root' {
        this.nodesToProcess.add(i);
        this.processedArray[i].write(false);
        this.inCurrentGeneration.add(1);
      }
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
        v.gen = gen;
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
        var path: network.pathHistory;
        toProcess.clear();
        this.lock.rl(v.header);
        if this.generationTime == 0 : real {
          this.generationTime = Time.getCurrentTime();
        }
        this.lock.url(v.header);
        this.log.debug('Beginning processing', hstring=v.header);
        this.log.debug(this.nodesToProcess : string, hstring=v.header);
        var prioritySize = v.priorityNodes.size;
        //pathSet += this.ygg.calculatePathArray(v.currentNode, toProcess, v.header);
        // try it now!
        // look, I know this will break it.
        //writeln(this.processedArray[currToProc].testAndSet());
        // If we can't get anything, that means we're just waiting for things to have finished processing.
        // we just want the sorted bit.
        //this.log.debug(pathDomain.isEmpty() : string, hstring=v.header);
        //while !toProcess.isEmpty() {
        while this.inCurrentGeneration.read() > 0 {
          // Yeah, this is WAY fucking faster.
          //writeln(toProcess);
          currToProc = '';
          toProcess.clear();
          //this.lock.rl(v.header);
          for id in this.nodesToProcess {
            //if !this.processedArray[id].read() {
              // If we have yet to process it, sort it out.
              if !this.processedArray[id].read() {
            //    writeln(id);
            // Make sure we use the priority nodes first.
                toProcess.add(id);
              }
            //}
          }
          if !(v.priorityNodes & toProcess).isEmpty() {
            // If this isn't an empty set, then prioritize the nodes here.
            // Note that this always means that if the nodes aren't in the current
            // generation, we ignore them.
            toProcess = toProcess & v.priorityNodes;
          }
          //this.lock.url(v.header);
          //writeln(toProcess);
          // This will just return the closest one, and is really all we need.
          if !toProcess.isEmpty() {
            (currToProc, path) = this.ygg.returnNearestUnprocessed(v.currentNode, toProcess, v.header);
            // Sometimes, it's not returning the correct one.
            this.log.debug(this.nodesToProcess : string, '//', toProcess : string, ':', v.currentNode : string, 'TO', currToProc : string, hstring=v.header);
            //if gen == 2 {
            //  this.lock.wl(v.header);
            //}
            this.log.debug('Attempting to unlock node', currToProc, 'nodes', this.nodesToProcess : string, hstring=v.header);
            // Sometimes
            //if !this.processedArray[currToProc].exchange(true) {
            if !this.processedArray[currToProc].testAndSet() {
              //this.lock.lock(v.header);
              //this.nodesToProcess.remove(currToProc);
              //this.lock.unlock(v.header);
              //pathDomain.remove(currToProc);
              //writeln('TASK ', i, ', SEED # ', this.ygg.nodes[currToProc].debugOrderOfCreation, ' : ', v.matrixValues);
              //this.log.log(' '.join('TASK', i : string, 'SEED #', currToProc : string, ':', v.matrixValues : string), i);
              //this.log.tId = i;
              //this.log.debug('SEED #', currToProc : string, hstring=' '.join('TASK', i : string));
              if v.priorityNodes.contains(currToProc) {
                v.priorityNodes.remove(currToProc);
                v.nPriorityNodesProcessed += 1;
              }
              this.log.debug('SEED #', currToProc : string, hstring=v.header);
              //this.log.devel('Hey, so, this is like, a test, you know what I mean?  I want a lot of things here.  Lots and lots of big things.  Things that will definitely test out the logging infrastructure.  Look, I know that you are tired.  I know that you are scared.  Hell, I am, too.  We are all scared.  We are all tired.  But we have to keep fighting.  We have to keep testing this.  It really is the only way to debug this.  So buck up.  Chin up.  Pull your little kitten arms up.');
              //this.log.debug(v.matrixValues : string, i);

              //this.log.debug('STARTING TO MOVE');
              this.ygg.move(v, currToProc, path, createEdgeOnMove=true, edgeDistance);
              //this.inCurrentGeneration.sub(1);
              //this.lock.lock(v.header);
              this.log.debug('Attempting to create another node', currToProc, hstring=v.header);
              var nextNode = this.ygg.nextNode(currToProc, hstring=v.header);
              this.log.debug('Node added; attempting to increase count for nextGeneration', hstring=v.header);
              v.nProcessed += 1;
              v.moved = true;
              this.lock.wl(v.header);
              // We only want to add to an empty domain here such that we only
              // prioritize nodes which are close to the current node.
              // Eventually, if we mutate, we'll add that in, too.
              v.priorityNodes.clear();
              v.priorityNodes.add(nextNode);
              this.nextGeneration.add(nextNode);
              this.lock.uwl(v.header);
              this.log.debug('Attempting to decrease count for inCurrentGeneration', hstring=v.header);
              this.inCurrentGeneration.sub(1);
              this.log.debug('inCurrentGeneration successfully reduced', hstring=v.header);
              //this.lock.unlock(v.header);
              //if this.inCurrentGeneration.read() == 0 {
              //  break;
              //}
            }
            // While it seems odd we might try this twice, this helps us keep
            // track of algorithm efficiency by determining whether we're processing
            // the nodes in our priority queue or not.
            if v.priorityNodes.contains(currToProc) {
              v.priorityNodes.remove(currToProc);
            }
          }
          //this.log.debug('Removing node from list to process', currToProc, hstring=v.header);
          //toProcess.remove(currToProc);
          // Just try clearing the whole damn domain.
          //toProcess.clear();
          //currToProc = '';
          this.log.debug('Remaining in generation:', this.inCurrentGeneration.read() : string, 'NODES:', this.nodesToProcess : string, 'priorityNodes:', v.priorityNodes : string, hstring=v.header);
        }
        // if we haven't moved, we should move our valkyrie to something in the current generation.  It makes searching substantially easier.
        if !v.moved {
          if currToProc != '' {
            this.ygg.move(v, currToProc, path, createEdgeOnMove=true, edgeDistance);
            // now, don't do anything, mind you.  Just move it up.
          }
        }
        //this.valkyriesDone.sub(1);
        //var vd = this.valkyriesDone[gen];
        if this.valkyriesDone[gen].fetchAdd(1) < (maxValkyries-1) {
          //this.valkyriesDone[gen] = vd - 1;
          //v.priorityNodes.clear();
          v.moved = false;
          this.log.debug('Waiting in gen', gen : string, v.header);
          this.valkyriesProcessed[i].write(v.nProcessed);
          this.priorityValkyriesProcessed[i].write(v.nPriorityNodesProcessed : real / prioritySize : real);
          this.moveOn[gen];
          //this.log.debug('MOVING ON in gen', gen : string, v.header);
          this.lock.rl(v.header);
          this.log.debug('MOVING ON in gen', gen : string, this.nodesToProcess : string, v.header);
          this.lock.url(v.header);
          //this.lock.wl(v.header);
          //this.valkyriesProcessed = ' '.join(this.valkyriesProcessed, 'V ', v.currentTask : string, ': ', v.nProcessed : string, ' ');
          v.nProcessed = 0;
          v.nPriorityNodesProcessed = 0;
          //toProcess.clear();
          //this.lock.uwl(v.header);
        } else {
          //this.valkyriesDone[gen] = maxValkyries;
          //v.priorityNodes.clear();
          v.moved = false;
          this.lock.wl(v.header);
          // reset that shit, yo.
          //this.valkyriesProcessed = 'V ', v.currentTask : string, ': ', v.nProcessed : string, ' ';
          //this.valkyriesProcessed = ' '.join(this.valkyriesProcessed, 'V ', v.currentTask : string, ': ', v.nProcessed : string, ' ');
          this.log.debug('Switching generations', v.header);
          this.nodesToProcess.clear();
          //this.processedArray: [nodesToProcess] atomic bool;
          //for node in this.nodesToProcess {
          //  this.nodesToProcess.remove(node);
            //this.processedArray[node].write(true);
          //}
          for node in this.nextGeneration {
            //this.nodesToProcess = this.nextGeneration;
            this.nodesToProcess.add(node);
            //this.nextGeneration.remove(node);
            this.processedArray[node].write(false);
          }
          this.nextGeneration.clear();
          //this.log.debug(this.processedArray.read() : string, hstring=v.header);
          this.inCurrentGeneration.write(this.nodesToProcess.size);
          //this.processedArray.write(false);
          this.valkyriesProcessed[i].write(v.nProcessed);
          this.priorityValkyriesProcessed[i].write(v.nPriorityNodesProcessed : real / prioritySize : real);
          var processedString: string;
          for y in 1..maxValkyries {
            processedString += ' '.join('// VP', y : string, '-', this.valkyriesProcessed[y].read() : string, '// ');
          }
          var avg = startingSeeds : real / maxValkyries : real ;
          var std: real;
          var eff: real;
          for y in 1..maxValkyries {
            std += (this.valkyriesProcessed[y].read() - avg)**2;
            eff += this.priorityValkyriesProcessed[y].read();
          }
          std /= maxValkyries;
          eff /= maxValkyries;
          std = (1 - (sqrt(std)/avg));
          processedString = ''.join(' // BALANCE:  ', std : string, ' // ', ' EFFICIENCY:  ', eff : string, ' // ');
          this.log.log('GEN', gen : string, 'processed in', (Time.getCurrentTime() - this.generationTime) : string, processedString : string, hstring='NormalRuntime');
          //this.log.log(this.nodesToProcess : string);
          this.generationTime = 0 : real;
          this.lock.uwl(v.header);
          this.valkyriesProcessed.write(0);
          this.priorityValkyriesProcessed.write(0);
          v.nPriorityNodesProcessed = 0;
          v.nProcessed = 0;
          // time to move the fuck on.
          this.moveOn[gen] = true;
          //toProcess.clear();
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
