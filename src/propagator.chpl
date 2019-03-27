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
use IO.FormattedIO;

var UUID = new owned uuid.UUID();
UUID.UUID4();

config const mSize = 20;
config var maxPerGeneration = 400;
config var mutationRate = 0.03;
config var maxValkyries = 1;
config var startingSeeds = 10;
config var createEdgeOnMove = true;
config var edgeDistance = 10;
config var debug = -1;
config var generations = 100;
config var unitTestMode = false;

// As we have our tree of life, so too do we have winged badasses who choose
// who lives and who dies.
// (this is a record for a worker class)
record valkyrie {
  var matrixValues: [0..mSize-1] real;
  var takeAnyPath: bool = false;
  var moved: bool = false;
  var canMove: bool = false;
  var currentNode: string;
  // Is there anything else I need?  Who knows!
  var priorityNodes: domain(string);
  var nPriorityNodesProcessed: int;

  var currentTask: int;
  var nMoves: int;
  var nProcessed: int;
  var gen: int;

  var id = UUID.UUID4();

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

  proc sendToFile {
    return 'EVOCAP----' + this.id + '----' + this.currentTask : string + '----';
  }

  proc header {
    return ' '.join(this.sendToFile, 'V', '%05i'.format(this.currentTask) : string, 'M', '%05i'.format(this.nMoves), 'G', '%05i'.format(this.gen));
  }
}

class Propagator {
  // this is going to actually hold all the logic for running EvoCap.
  var generation: int;

  // now a few globals.
  var nodesToProcess: domain(string);
  var processedArray: [nodesToProcess] atomic bool;
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
  var authors: domain(string) = ['Audrey Pratt', 'Benjamin Robbins'];
  var version: real = 0.1;
  // I guess this is protected or important in Chapel, in some way?
  //var release: string; // alpha

  proc logo() {
    return '';
  }

  proc header() {
    var lorder: domain(int);
    var logo: [lorder] string;
    var order: domain(int);
    var about: [order] string;
    logo[0] = '▄██   ▄      ▄██████▄     ▄██████▄  ████████▄     ▄████████    ▄████████    ▄████████  ▄█   ▄█       ';
    logo[1] = '███   ██▄   ███    ███   ███    ███ ███   ▀███   ███    ███   ███    ███   ███    ███ ███  ███       ';
    logo[2] = '███▄▄▄███   ███    █▀    ███    █▀  ███    ███   ███    ███   ███    ███   ███    █▀  ███▌ ███       ';
    logo[3] = '▀▀▀▀▀▀███  ▄███         ▄███        ███    ███  ▄███▄▄▄▄██▀   ███    ███   ███        ███▌ ███       ';
    logo[4] = '▄██   ███ ▀▀███ ████▄  ▀▀███ ████▄  ███    ███ ▀▀███▀▀▀▀▀   ▀███████████ ▀███████████ ███▌ ███       ';
    logo[5] = '███   ███   ███    ███   ███    ███ ███    ███ ▀███████████   ███    ███          ███ ███  ███       ';
    logo[6] = '███   ███   ███    ███   ███    ███ ███   ▄███   ███    ███   ███    ███    ▄█    ███ ███  ███▌    ▄ ';
    logo[7] = ' ▀█████▀    ████████▀    ████████▀  ████████▀    ███    ███   ███    █▀   ▄████████▀  █▀   █████▄▄██ ';
    logo[8] = '                                                 ███    ███                                ▀         ';
    // Taken from: http://patorjk.com/software/taag/#p=display&f=Delta%20Corps%20Priest%201&t=YGGDRASIL
    //this.release = 'A';
    about[0] = 'An implementation of EvoCap';
    about[1] = 'By: ';
    var z: int = 1;
    for i in this.authors {
      if z != this.authors.size {
          about[1] += i + ', ';
      } else {
          about[1] += i;
      }
      z += 1;
    }
    about[2] = 'Version: %.2dr%s'.format(this.version, 'A');
    about[3] = 'Copyright Cray, probably; DO NOT DISTRIBUTE';
    for i in 0..8 {
        this.log.header(logo[i]);
    }
    for i in 0..3 {
      this.log.header(about[i]);
    }
  }

  proc init() {
    // We initialize the network, creating the GeneNetwork object, logging
    // infrastructure
    // We could actually create different loggers, if we wanted; the classes
    // and infrastructure support that.  Might be faster, dunno.
    // Typically, we probably won't have that much output, though, so.
    this.ygg = new shared network.GeneNetwork();
    this.log = new shared ygglog.YggdrasilLogging();
    this.log.currentDebugLevel = debug;
    this.ygg.log = this.log;
    this.ygg.lock.log = this.log;
    this.ygg.initializeNetwork(n_seeds=startingSeeds);
    var ids = this.ygg.ids;
    for i in this.ygg.ids {
      if i != 'root' {
        this.nodesToProcess.add(i);
        this.processedArray[i].write(false);
        this.inCurrentGeneration.add(1);
      }
    }
    this.log.debug('INITIALIZED', this.inCurrentGeneration.read() : string, 'seeds.', hstring='Ragnarok');
    this.lock = new shared spinlock.SpinLock();
    this.lock.t = 'Ragnarok';
    this.lock.log = this.log;
  }

  proc run() {
    // Print out the header, yo.
    this.header();
    // start up the main procedure by creating some valkyries.
    var nnodes: atomic int;
    coforall i in 1..maxValkyries {
      var v = new valkyrie();
      v.currentTask = i;
      v.moveToRoot();
      for gen in 1..generations {
        v.gen = gen;
        this.log.log('Starting GEN', '%{######}'.format(gen), hstring=v.header);
        var currToProc: string;
        var toProcess: domain(string);
        var path: network.pathHistory;
        // Likely not necessary.
        toProcess.clear();
        this.lock.wl(v.header);
        if this.generationTime == 0 : real {
          this.generationTime = Time.getCurrentTime();
        }
        this.lock.uwl(v.header);
        this.log.debug('Beginning processing', hstring=v.header);
        this.log.debug(this.nodesToProcess : string, hstring=v.header);
        var prioritySize = v.priorityNodes.size;
        while this.inCurrentGeneration.read() > 0 {
          // We clear this out because it is faster to just re-enumerate the
          // nodes that need processing, rather than explicitly calculating
          // the path towards every node.  Particularly as that results in tasks
          // performing a lot of unnecessary computations once a lot of nodes
          // have been processed.
          currToProc = '';
          toProcess.clear();
          for id in this.nodesToProcess {
              // As this is an atomic variable, we don't need to lock.
              if !this.processedArray[id].read() {
                toProcess.add(id);
              }
          }
          if !(v.priorityNodes & toProcess).isEmpty() {
            // If this isn't an empty set, then prioritize the nodes here.
            // Note that this always means that if the nodes aren't in the current
            // generation, we ignore them.
            // This is probably _not_ the way we'll want to do this, ultimately, but.
            this.log.debug('Using the joint of toProcess and priorityNodes:', (v.priorityNodes & toProcess) : string, hstring=v.header);
            this.log.debug('Current toProcess:', toProcess : string, 'current priorityNodes:', v.priorityNodes : string, hstring=v.header);
            toProcess = toProcess & v.priorityNodes;
          }
          // Assuming we have some things to process, do it!
          if !toProcess.isEmpty() {
            (currToProc, path) = this.ygg.returnNearestUnprocessed(v.currentNode, toProcess, v.header);
            // Still some weird edge cases; this just helps me sort out what things are doing.
            this.log.debug(this.nodesToProcess : string, '//', toProcess : string, ':', v.currentNode : string, 'TO', currToProc : string, hstring=v.header);
            this.log.debug('Attempting to unlock node:', currToProc, 'nodesToProcess', this.nodesToProcess : string, hstring=v.header);
            if !this.processedArray[currToProc].testAndSet() {
              // If this node is one of the ones in our priority queue, remove it
              // as we clearly processing it now.
              if v.priorityNodes.contains(currToProc) {
                v.priorityNodes.remove(currToProc);
                v.nPriorityNodesProcessed += 1;
              }
              this.log.debug('Processing seed ID', currToProc : string, hstring=v.header);
              this.ygg.move(v, currToProc, path, createEdgeOnMove=true, edgeDistance);
              this.log.debug('Attempting to move ID', currToProc, 'into the next generation.', hstring=v.header);
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
            }
            // While it seems odd we might try this twice, this helps us keep
            // track of algorithm efficiency by determining whether we're processing
            // the nodes in our priority queue or not.
            if v.priorityNodes.contains(currToProc) {
              v.priorityNodes.remove(currToProc);
            }
          }
          this.log.debug('Remaining in generation:', this.inCurrentGeneration.read() : string, 'NODES:', this.nodesToProcess : string, 'priorityNodes:', v.priorityNodes : string, 'toProcess:', toProcess : string, hstring=v.header);
          for z in this.nodesToProcess {
            this.log.debug('Has the node been processed?  Node: ', z: string, '-', this.processedArray[z].read() : string, hstring=v.header);
          }
        }
        // if we haven't moved, we should move our valkyrie to something in the current generation.  It makes searching substantially easier.
        if !v.moved {
          if currToProc != '' {
            this.ygg.move(v, currToProc, path, createEdgeOnMove=true, edgeDistance);
            // Get rid of the priority nodes; we've moved, after all.
            v.priorityNodes.clear();
            // We just need to make the current priorityNodes the intersection
            // of the current node's edges and what we're processing in the next
            // generation.
            this.lock.rl(v.header);
            v.priorityNodes.add((this.ygg.edges[currToProc] & this.nextGeneration));
            this.lock.url(v.header);
            // We're not doing any processing; just moving.
          }
        }
        if this.valkyriesDone[gen].fetchAdd(1) < (maxValkyries-1) {
          // Reset a lot of the variables for the Valkyrie while we're idle.
          // Then wait until all the other Valkyries have finished.
          // In addition, add to some global variables so that we can compute
          // some statistics of how well we're running.
          // Then wait on the sync variable.
          v.moved = false;
          this.log.debug('Waiting in gen', gen : string, v.header);
          this.valkyriesProcessed[i].write(v.nProcessed);
          this.priorityValkyriesProcessed[i].write(v.nPriorityNodesProcessed : real / prioritySize : real);
          v.nProcessed = 0;
          v.nPriorityNodesProcessed = 0;
          // moveOn is an array of sync variables.  We're blocked from reading
          // until that's set to true.
          this.moveOn[gen];
          this.lock.rl(v.header);
          this.log.debug('MOVING ON in gen', gen : string, this.nodesToProcess : string, v.header);
          this.lock.url(v.header);
        } else {
          // Same stuff here, but as this is the last Valkyrie, we also
          // do global cleanup to ensure the global arrays are ready.
          v.moved = false;
          this.lock.wl(v.header);
          this.log.debug('Switching generations', v.header);
          // Clear out the current nodesToProcess domain, and swap it for the
          // ones we've set to process for the next generation.
          this.nodesToProcess.clear();
          for node in this.nextGeneration {
            this.nodesToProcess.add(node);
            this.processedArray[node].write(false);
          }
          this.nextGeneration.clear();
          // Set the count variable.
          this.inCurrentGeneration.write(this.nodesToProcess.size);
          this.valkyriesProcessed[i].write(v.nProcessed);
          // Compute some rough stats.  Buggy.
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
          this.log.log('GEN', '%05i'.format(gen), 'processed in', '%05.2dr'.format(Time.getCurrentTime() - this.generationTime) : string, processedString : string, hstring='NormalRuntime');
          this.generationTime = 0 : real;
          this.lock.uwl(v.header);
          this.valkyriesProcessed.write(0);
          this.priorityValkyriesProcessed.write(0);
          v.nPriorityNodesProcessed = 0;
          v.nProcessed = 0;
          // time to move the fuck on.
          this.moveOn[gen] = true;
        }
      }
    }
  }
}
