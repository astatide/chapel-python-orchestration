// Do I need all this?  Probably not but I'll sort that issue later.

use rng;
use genes;
use network;
use uuid;
use Math;
//use VisualDebug;
use ygglog;
use spinlock;
use Time;
use IO.FormattedIO;
use chromosomes;
use gjallarbru;
use Spawn;
use messaging;
use HashedDist;

record scoreComparator {
  proc keyPart(x: (string, real), i: int) {
    if i > 2 then
      return (-1, ('NULL', 0));
    return (0, (x[1], x[2]));
  }
}


//var UUIDP = new owned uuid.UUID();
//UUIDP.UUID4();


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
//// The locks are noisy, but we do need to debug them sometimes.
// This shuts them up unless you really want them to sing.  Their song is
// a terrible noise; an unending screech which ends the world.
// (okay, they're just super verbose)
//config var flushToLog = false;

config var nChromosomes = 6;
config var chromosomeSize = 36;
config var nDuplicates = 4;

// Empty record serves as comparator
record Comparator { }

// compare method defines how 2 elements are compared
proc Comparator.compare(a, b) {
  return abs(a) - abs(b);
}

var absComparator: Comparator;
var cLock = new shared spinlock.SpinLock();
cLock.t = 'Chromosomes';
cLock.log = new shared ygglog.YggdrasilLogging();
var chromosomeDomain: domain(string) dmapped Hashed(idxType=string, mapper = new network.mapperByLocale());
var chromes: [chromosomeDomain] chromosomes.Chromosome;

// As we have our tree of life, so too do we have winged badasses who choose
// who lives and who dies.
// (this is a record for a worker class)
class valkyrie : msgHandler {
  // it's okay for now.
  var matrixValues: [0..mSize] c_double;
  var log: shared ygglog.YggdrasilLogging();
  var takeAnyPath: bool = false;
  var moved: bool = false;
  var canMove: bool = false;
  var currentNode: string;
  // Is there anything else I need?  Who knows!
  var priorityNodes: domain(string);
  var nPriorityNodesProcessed: int;
  var moveSuccessful: int = 0;
  var moveFailed: int = 1;

  var currentTask: int;
  var nMoves: int;
  var nProcessed: int;
  var gen: int;
  var currentLocale: string;

  var UUIDP = new owned uuid.UUID();
  var id = UUIDP.UUID4();

  var yh = new ygglog.yggHeader();

  var __score__ : real;


  proc moveToRoot() {
    // Zero out the matrix, return the root id.
    this.matrixValues = 0;
    this.currentNode = 'root';
  }

  proc init() {
    // basically, the inheritance isn't working as I would have expected.
    // see https://github.com/chapel-lang/chapel/issues/8232
    super.init(1);
    this.complete();
    this.log = new shared ygglog.YggdrasilLogging();
    this.log.currentDebugLevel = debug;
  }

  proc init(n: int) {
    // basically, the inheritance isn't working as I would have expected.
    // see https://github.com/chapel-lang/chapel/issues/8232
    super.init(n);
    this.complete();
    this.log = new shared ygglog.YggdrasilLogging();
    this.log.currentDebugLevel = debug;
  }

  override proc PROCESS(m: msg, i: int) {
    // overriden from the messaging class
    this.log.debug("MESSAGE RECEIVED:", m : string, hstring = this.header);
    select m.COMMAND {
      when messaging.command.RECEIVE_SCORE {
        m.open(this.__score__);
      }
    }
    this.log.debug("SCORE IS", this.__score__ : string, hstring = this.header);
    //stdout.flush();
  }

  inline proc score {
    var s = this.__score__;
    this.__score__ = 0;
    return s;
  }

  // Assume we're given a delta object so that we may express it.
  proc move(delta: deltaRecord, id: string) {
    delta.express(this.matrixValues);
    this.currentNode = id;
    if unitTestMode {
      if this.matrixValues[0] != this.currentNode : real {
        // in unit test mode, we set up our IDs and matrix values such that
        // every value of the matrix should be equal to ID.
        // in that event, return a failure code.
        return this.moveFailed;
      } else {
        return this.moveSuccessful;
      }
    }
    return this.moveSuccessful;
  }

  proc sendToFile {
    return 'EVOCAP----' + this.id + '----' + this.currentTask : string + '----';
  }

  proc header {
    //return ' '.join(this.sendToFile, 'V', '%05i'.format(this.currentTask) : string, 'M', '%05i'.format(this.nMoves), 'G', '%05i'.format(this.gen));
    this.yh.header = 'VALKYRIE';
    this.yh.id = this.id;
    this.yh.currentLocale = this.currentLocale;
    this.yh.currentTask = this.currentTask;
    return this.yh;
  }

  iter logo {
    var lorder: domain(int);
    var logo: [lorder] string;
    logo[0] = ' ▄█    █▄     ▄████████  ▄█          ▄█   ▄█▄ ▄██   ▄      ▄████████  ▄█     ▄████████ ';
    logo[1] = '███    ███   ███    ███ ███         ███ ▄███▀ ███   ██▄   ███    ███ ███    ███    ███ ';
    logo[2] = '███    ███   ███    ███ ███         ███▐██▀   ███▄▄▄███   ███    ███ ███▌   ███    █▀  ';
    logo[3] = '███    ███   ███    ███ ███        ▄█████▀    ▀▀▀▀▀▀███  ▄███▄▄▄▄██▀ ███▌  ▄███▄▄▄     ';
    logo[4] = '███    ███ ▀███████████ ███       ▀▀█████▄    ▄██   ███ ▀▀███▀▀▀▀▀   ███▌ ▀▀███▀▀▀     ';
    logo[5] = '███    ███   ███    ███ ███         ███▐██▄   ███   ███ ▀███████████ ███    ███    █▄  ';
    logo[6] = '███    ███   ███    ███ ███▌    ▄   ███ ▀███▄ ███   ███   ███    ███ ███    ███    ███ ';
    logo[7] = '▀██████▀    ███    █▀  █████▄▄██   ███   ▀█▀  ▀█████▀    ███    ███ █▀     ██████████ ';
    logo[8] = '                       ▀           ▀                     ███    ███                   ';
    logo[9] = 'VALKYRIE %s on locale %i, running task %i'.format(this.id, here.id, this.currentTask);
    for i in 0..9 {
      yield logo[i];
    }
  }
}

class Propagator {
  // this is going to actually hold all the logic for running EvoCap.
  var generation: int;

  // now a few globals.
  var nodesToProcess: domain(string);
  var processedArray: [nodesToProcess] atomic bool;
  // actually, this should probably be its own thing.
  //var scoreDomain: domain(string);
  var scoreArray: [0..4,1..maxPerGeneration] real = -1; //Math.INFINITY;
  var idArray: [0..4,1..maxPerGeneration] string;
  var inCurrentGeneration: atomic int;
  var nextGeneration: domain(string);

  var ygg: shared network.GeneNetwork();
  var yh = new ygglog.yggHeader();
  var log: shared ygglog.YggdrasilLogging();
  var lock: shared spinlock.SpinLock;
  var valkyriesDone: [1..generations] atomic int;
  var moveOn: [1..generations] single bool;
  var readyForChromosomes: [1..generations] single bool;
  var areSpawned: single bool;
  var numSpawned: atomic int;
  var valkyriesProcessed: [1..maxValkyries*Locales.size] atomic int;
  var priorityValkyriesProcessed: [1..maxValkyries*Locales.size] atomic real;
  var finishedChromoProp: atomic int;
  var generationTime: real;
  var authors: domain(string) = ['Audrey Pratt', 'Benjamin Robbins'];
  var version: real = 0.1;
  // I guess this is protected or important in Chapel, in some way?
  var shutdown: bool = false;

  // Because the chromosomes are an abstraction of the gene network, and are
  // in many respects related more to the movement rather than graph problems,
  // the propagator is responsible for it.


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
        if z == this.authors.size-1 {
          if this.authors.size > 2 {
            about[1] += i + ', and ';
          } else {
            about[1] += i + ' and ';
          }
        } else {
          about[1] += i + ', ';
        }
      } else {
          about[1] += i;
      }
      z += 1;
    }
    about[2] = 'Version: %.2dr%s'.format(this.version, 'A');
    about[3] = 'Copyright Cray (2019), probably; DO NOT DISTRIBUTE';
    for i in 0..8 {
        this.log.header(logo[i]);
    }
    for i in 0..3 {
      this.log.header(about[i]);
    }
  }

  proc initRun() {
    // We initialize the network, creating the GeneNetwork object, logging
    // infrastructure
    // We could actually create different loggers, if we wanted; the classes
    // and infrastructure support that.  Might be faster, dunno.
    // Typically, we probably won't have that much output, though, so.
    this.ygg = new shared network.GeneNetwork();
    this.yh = new ygglog.yggHeader();
    this.yh += 'Ragnarok';
    this.log = new shared ygglog.YggdrasilLogging();
    this.log.currentDebugLevel = debug;
    this.ygg.log = this.log;
    this.ygg.lock.log = this.log;
    this.lock = new shared spinlock.SpinLock();
    this.lock.t = 'Ragnarok';
    this.lock.log = this.log;
    //this.ygg.initializeRoot();
    // basically, re-add the root node to make sure its connections are up to date
  }

  proc initChromosomes(ref nG: shared network.networkGenerator) {
    forall deme in 0..4 with (ref nG) {
      forall i in 1..nChromosomes with (ref nG) {
        // Here, we're going to be given the instructions for generating chromosomes.
        // No reason this can't be parallel, so let's do it.
        //this.log.debug('Spawning chromosome', this.yh);
        var nc = new chromosomes.Chromosome();
        nc.id = nG.generateChromosomeID;
        //this.log.debug('Chromosome ID', nc.id, 'spawned.  Preparing genes.', this.yh);
        nc.prep(startingSeeds, chromosomeSize-startingSeeds);
        nc.currentDeme = deme;
        this.log.debug('Genes prepped in Chromosome ID; converting into nodes', nc.id, this.yh);
        nc.log = this.log;
        var n: int = 1;
        nc.generateNodes(nG, this.yh);
        //this.lock.wl();
        cLock.wl();
        chromosomeDomain.add(nc.id);
        chromes[nc.id] = nc;
        cLock.uwl();
        //this.lock.uwl();
      }
    }
  }

  proc advanceChromosomes(ref nG: shared network.networkGenerator) {
    //vLog.debug('Advancing the chromosomes.', vheader);
    var newCD: domain(string);
    var newC: [newCD] chromosomes.Chromosome;
    //this.lock.rl();
    cLock.rl();
    for chrome in chromosomeDomain {
      if !chromes[chrome].isProcessed.testAndSet() {
        var nc = chromes[chrome];
        for i in 1..nDuplicates {
          var cc = nc.clone();
          cc.id = nG.generateChromosomeID;
          cc.log = this.log;
          cc.advanceNodes(nG, this.yh);
          newCD.add(cc.id);
          newC[cc.id] = cc;
        }
      }
    }
    //this.lock.url();
    cLock.url();
    //this.lock.wl();
    cLock.wl();
    network.globalLock.rl();
    for chrome in newCD {
      chromosomeDomain.add(chrome);
      chromes[chrome] = newC[chrome];
      for node in newC[chrome].geneIDs {
        if node != 'root' {
          this.nextGeneration.add(node);
          writeln(globalNodes[node].chromosome, ' : ', chrome);
          writeln(node);
          //writeln(globalNodes[node]);
          //assert(globalNodes[node].chromosome == chrome);
        }
      }
    }
    network.globalLock.url();
    cLock.uwl();
    //this.lock.uwl();
  }

  proc exitRoutine() throws {
    // command the logger to shut down, then exit.
    this.lock.wl(this.yh);
    // NEVER LET GO.
    for i in 1..maxValkyries {
      // tell the Valkyries to quit their shit.
      var m = new messaging.msg(0);
      m.COMMAND = messaging.command.SHUTDOWN;
      //SEND(m, i+(maxValkyries*here.id));
    }
    this.log.critical('SHUTDOWN INITIATED');
    this.log.exitRoutine();
    this.lock.uwl(this.yh);
    throw new owned Error();
  }

  proc setShutdown() {
    this.shutdown = true;
  }

  proc run() {
    // Print out the header, yo.
    this.header();
    this.yh.header = 'NormalRuntime';
    this.yh += 'run';
    this.log.debug("Setting up locales and valkyries", this.yh);
    // I think the other valkyries should be able to do their thing.
    // should probably do this for each task but hey whatever.
    // We're catching a signal interrupt, which is slightly mangled for some reason.
    // start up the main procedure by creating some valkyries.
    coforall L in Locales {
      on L do {
        if true {
          var yH = new ygglog.yggHeader();
          yH += 'Ragnarok';
          this.log.debug("Spawn local network and networkGenerator", yH);
          var yggLocalCopy = new shared network.GeneNetwork();
          // CHANGE ME
          yggLocalCopy.log = this.log;
          yggLocalCopy.lock.log = this.log;
          var nG = new shared network.networkGenerator();
          // we're gonna want a list of network IDs we can use.
          this.log.debug("Local networks spawned; creating chromosomes", yH);
          this.initChromosomes(nG);
          this.log.debug("Adding new nodes to unprocessed list", yH);
          nG.addUnprocessed();
          this.log.debug("Setting the current generation count", yH);
          // now, make sure we know we have to process all of these.
          this.inCurrentGeneration.add(nG.currentId.read()-1);
          //this.log.debug("About to add existing nodes to the processing list", yH);
          //var ids = this.ygg.ids;
          //ref YNC = yggNodeCopy;
          //var yggNodeCopy: network.GeneNetwork;
          //begin with (ref yggNodeCopy) this.ygg.clone(yggNodeCopy);
          //this.ygg.clone(yggLocalCopy);

          forall i in 1..maxValkyries with (ref nG, ref yggLocalCopy) {
            // spin up the Valkyries!
            var vLog = new shared ygglog.YggdrasilLogging();
            vLog.currentDebugLevel = debug;
            var vLock = new shared spinlock.SpinLock();
            vLock.t = 'Valkyrie';
            var v = new shared valkyrie(1);
            v.currentTask = i;
            v.currentLocale = L : string;
            v.yh += 'run';
            for iL in v.logo {
              vLog.header(iL, hstring=v.header);
            }
            vLog.log('Initiating spawning sequence', hstring=v.header);
            var vp = v.valhalla(1, v.id, mSize : string, vLog, vstring=v.header);
            var nSpawned = this.numSpawned.fetchAdd(1);
            if nSpawned < ((Locales.size*maxValkyries)-1) {
              // we want to wait so that we spin up all processes.
              vLog.log('Clone complete; awaiting arrival of other valkyries.  Ready:', nSpawned : string, hstring=v.header);
              this.areSpawned;
            } else {
              this.areSpawned = true;
            }
            v.moveToRoot();

            for gen in 1..generations {

              v.gen = gen;
              vLog.log('Starting GEN', '%{######}'.format(gen), hstring=v.header);
              var currToProc: string;
              var toProcess: domain(string);
              var path: network.pathHistory;
              var removeFromSet: domain(string);
              toProcess.clear();
              vLog.debug('Beginning processing', hstring=v.header);
              //vLog.debug(this.nodesToProcess : string, hstring=v.header);
              var prioritySize = v.priorityNodes.size;
              vLog.debug('Assessing nodes that must be handled', hstring=v.header);
              currToProc = '';
              //toProcess.clear();

              for id in nG.all {
                toProcess.add(id);
                vLog.debug('Adding node ID: ', id : string, hstring=v.header);
              }
              //vLog.debug('What is up, fellow nodes? NODES: ', toProcess : string, hstring=v.header);

              if toProcess.isEmpty() {
                // This checks atomics, so it's gonna be slow.
                // In an ideal world, we rarely call it.
                //network.globalLock.rl();
                for id in network.globalUnprocessed {
                  if !network.globalIsProcessed[id].read() {
                    toProcess.add(id);
                  }
                }
                //network.globalLock.url();
              }
              while this.inCurrentGeneration.read() > 0 {
                // We clear this out because it is faster to just re-enumerate the
                // nodes that need processing, rather than explicitly calculating
                // the path towards every node.  Particularly as that results in tasks
                // performing a lot of unnecessary computations once a lot of nodes
                // have been processed.
                // Assuming we have some things to process, do it!
                currToProc = '';
                vLog.debug('toProcess created', hstring=v.header);
                if !toProcess.isEmpty() {
                  // We can remove nodes from the domain processedArray is built on, which means we need to catch and process.
                  var existsInDomainAndCanProcess: bool = false;
                  // This function now does the atomic test.
                  vLog.debug('Returning nearest unprocessed', hstring=v.header);
                  (currToProc, path, removeFromSet) = yggLocalCopy.returnNearestUnprocessed(v.currentNode, toProcess, v.header, network.globalIsProcessed);
                  for i in removeFromSet {
                    if toProcess.contains(i) {
                      toProcess.remove(i);
                    }
                  }
                  vLog.debug('Unprocessed found.  ID:', currToProc : string, hstring=v.header);
                  if currToProc != '' {
                    // If this node is one of the ones in our priority queue, remove it
                    // as we clearly processing it now.
                    if v.priorityNodes.contains(currToProc) {
                      v.priorityNodes.remove(currToProc);
                      v.nPriorityNodesProcessed += 1;
                    }
                    v.moved = true;
                    vLog.debug('Removing from local networkGenerator, if possible.', hstring=v.header);
                    nG.removeUnprocessed(currToProc);
                    toProcess.remove(currToProc);
                    // Actually, reduce the count BEFORE we do this.
                    // Otherwise we could have threads stealing focus that should
                    // actually be idle.
                    vLog.debug('Attempting to decrease count for inCurrentGeneration', hstring=v.header);
                    this.inCurrentGeneration.sub(1);
                    vLog.debug('inCurrentGeneration successfully reduced', hstring=v.header);
                    //writeln('What are our demes? ', network.globalNodes[currToProc].demeDomain : string);
                    //network.globalLock.rl();
                    ref actualNode = network.globalNodes[currToProc];
                    //network.globalLock.url();
                    for deme in actualNode.returnDemes() {
                      vLog.debug('Starting work for ID:', currToProc: string, 'on deme #', deme : string, hstring=v.header);
                      vLog.debug('Processing seed ID', currToProc : string, hstring=v.header);
                      vLog.debug('PATH:', path : string, hstring=v.header);
                      var d = yggLocalCopy.deltaFromPath(path, path.key(0), hstring=v.header);
                      d.to = currToProc;
                      var newMsg = new messaging.msg(d);
                      newMsg.i = deme;
                      newMsg.COMMAND = messaging.command.RECEIVE_AND_PROCESS_DELTA;
                      vLog.debug("Attempting to run Python on seed ID", currToProc : string, hstring=v.header);
                      vLog.debug("Sending the following msg:", newMsg : string, hstring=v.header);
                      v.SEND(newMsg);
                      vLog.debug("Message & delta sent; awaiting instructions", hstring=v.header);
                      var m = v.RECV();
                      var score = m.r;
                      vLog.debug('SCORE FOR', currToProc : string, 'IS', score : string, hstring=v.header);
                      // we should _not_ need to readlock these domains, as the global domains cannot be and ARE not resized during this loop.
                      //network.globalLock.rl();
                      network.globalNodes[currToProc].setDemeScore(deme, score);
                      //network.globalLock.url();
                      // add to the chromosome.
                      vLog.debug('Adding to chromosome score.', hstring=v.header);
                      //network.globalLock.rl();
                      var nc = network.globalNodes[currToProc].chromosome;
                      //network.globalLock.url();
                      cLock.rl();
                      var inChromeID = chromes[nc].returnNodeNumber(currToProc);
                      cLock.url();
                      //if inChromeID == -1 {

                      //}
                      vLog.debug('NodeNumber:', inChromeID : string, "Node ID:", currToProc : string, "Chromosome ID:", nc : string, "Deme:", deme : string, hstring=v.header);
                      cLock.rl();
                      vLog.debug('DemeDomain in chromosome:', chromes[nc].geneIDs : string, hstring=v.header);
                      chromes[nc].scores[inChromeID] = score;
                      cLock.url();

                      //network.globalLock.url();
                    }
                  } else {
                    // actually, if that's the case, we can't do shit.  So break and yield.
                    break;
                  }
                  // While it seems odd we might try this twice, this helps us keep
                  // track of algorithm efficiency by determining whether we're processing
                  // the nodes in our priority queue or not.
                  //if v.priorityNodes.contains(currToProc) {
                  //  v.priorityNodes.remove(currToProc);
                  //}
                } else {
                  // Rest now, my child. Rest, and know your work is done.
                  vLog.debug('And now, I rest.  Remaining in generation:', this.inCurrentGeneration.read() : string, 'priorityNodes:', v.priorityNodes : string, hstring=v.header);
                  while this.inCurrentGeneration.read() != 0 do chpl_task_yield();
                  vLog.debug('Waking up!', hstring=v.header);
                }
                vLog.debug('Remaining in generation:', this.inCurrentGeneration.read() : string, 'priorityNodes:', v.priorityNodes : string, hstring=v.header);
                if this.shutdown {
                  this.exitRoutine();
                }
              }
              // if we haven't moved, we should move our valkyrie to something in the current generation.  It makes searching substantially easier.
              // but we should reinclude that logic _later_.  As it's busted.
              if !v.moved {
                // do something about it, why don't you.
              }
              if this.valkyriesDone[gen].fetchAdd(1) < ((Locales.size*maxValkyries)-1) {
                // Reset a lot of the variables for the Valkyrie while we're idle.
                // Then wait until all the other Valkyries have finished.
                // In addition, add to some global variables so that we can compute
                // some statistics of how well we're running.
                // Then wait on the sync variable.
                v.moved = false;
                vLog.debug('Waiting in gen', gen : string, v.header);
                this.valkyriesProcessed[i+(here.id*maxValkyries)].write(v.nProcessed);
                this.priorityValkyriesProcessed[i+(here.id*maxValkyries)].write(v.nPriorityNodesProcessed : real / prioritySize : real);
                vLog.log('GEN:', gen : string, 'TOTAL MOVES:', v.nMoves : string, 'PROCESSED:', v.nProcessed : string, 'PRIORITY PROCESSED', v.nPriorityNodesProcessed : string, hstring=v.header);
                v.nProcessed = 0;
                v.nPriorityNodesProcessed = 0;

                // after this, we process the chromosomes and go.
                this.readyForChromosomes[gen];
                vLog.log('Grabbing chromosomes to process', hstring=v.header);
                // moveOn is an array of sync variables.  We're blocked from reading
                // until that's set to true.
                this.advanceChromosomes(nG);
                nG.addUnprocessed();
                vLog.debug("Setting the current generation count", v.header);
                // now, make sure we know we have to process all of these.
                //this.inCurrentGeneration.add(nG.currentId.read()-1);
                this.finishedChromoProp.add(1);
                this.moveOn[gen];
                this.lock.rl(v.header);
                vLog.debug('MOVING ON in gen', gen : string, this.nodesToProcess : string, v.header);
                this.lock.url(v.header);
              } else {
                // Same stuff here, but as this is the last Valkyrie, we also
                // do global cleanup to ensure the global arrays are ready.
                vLog.debug('Handling cleanup on gen', gen : string, v.header);
                v.moved = false;
                this.nextGeneration.clear();
                // we'll just throw this in here for now.
                // Only do the max!
                //var bestInGen: real = this.scoreArray[1];
                var (bestInGen, minLoc) = maxloc reduce zip(this.scoreArray, this.scoreArray.domain);
                var chromosomesToAdvance: domain(string);
                var c: [chromosomesToAdvance] chromosomes.Chromosome;
                vLog.debug('Determining which chromosomes to advance', v.header);
                for chrome in chromosomeDomain {
                  var deme = chromes[chrome].currentDeme;
                  //var (lowestScore, minLoc) = minloc reduce zip(this.scoreArray[deme], this.scoreArray.domain);
                  var lowestScore : real = Math.INFINITY;
                  var minLoc : int;
                  vLog.debug('Determining lowest score...', v.header);
                  for z in 1..maxPerGeneration {
                    if this.scoreArray[deme,z] < lowestScore {
                      lowestScore = this.scoreArray[deme,z];
                      minLoc = z;
                    }
                  }
                  vLog.debug('Finding the highest scoring node on this chromosome and seeing if it is good enough.', v.header);
                  var (bestScore, bestNode) = chromes[chrome].bestGeneInDeme[chromes[chrome].currentDeme];
                  if bestScore > lowestScore {
                    this.scoreArray[deme, minLoc] = bestScore;
                    this.idArray[deme, minLoc] = chrome;
                  }
                }
                for deme in 0..4 {
                  for z in 1..maxPerGeneration {
                    if this.idArray[deme,z] != '' {
                      vLog.debug('Advancing chromosome ID:', this.idArray[deme,z], v.header);
                      chromosomesToAdvance.add(this.idArray[deme,z]);
                    }
                  }
                }
                // clear the domain of our losers.
                vLog.debug('Clearing the domain of those who are not continuing.', v.header);
                //if true {
                var vheader = v.header;
                for chrome in chromosomeDomain {
                  if !chromosomesToAdvance.contains(chrome) {
                    chromosomeDomain.remove(chrome);
                  }
                }
                this.readyForChromosomes[gen] = true;
                this.advanceChromosomes(nG);
                //}
                this.scoreArray = -1;
                this.log.debug("Setting the current generation count", this.yh);
                // now, make sure we know we have to process all of these.
                //this.inCurrentGeneration.add(nG.currentId.read()-1);
                while this.finishedChromoProp.read() < ((Locales.size*maxValkyries)-1) do chpl_task_yield();
                this.finishedChromoProp.write(0);
                vLog.debug('Switching generations', v.header);
                nG.addUnprocessed();
                // Clear out the current nodesToProcess domain, and swap it for the
                // ones we've set to process for the next generation.
                this.nodesToProcess.clear();
                for node in this.nextGeneration {
                  this.nodesToProcess.add(node);
                  this.processedArray[node].write(false);
                }
                for node in this.nextGeneration {
                  this.inCurrentGeneration.add(1);
                  vLog.debug("Node ID:", node : string, hstring=v.header);
                  vLog.debug('Chromosome:', chromes[globalNodes[node].chromosome] : string, hstring=v.header);
                  var isInChromosome: bool = false;
                  for n in chromes[globalNodes[node].chromosome].geneIDs {
                    if !isInChromosome {
                      if n == node {
                        isInChromosome = true;
                      }
                    }
                  }
                  assert(isInChromosome);
                }
                this.nextGeneration.clear();
                // Set the count variable.
                //this.inCurrentGeneration.write(this.nodesToProcess.size);
                this.valkyriesProcessed[i+(here.id*maxValkyries)].write(v.nProcessed);
                // Compute some rough stats.  Buggy.
                this.priorityValkyriesProcessed[i+(here.id*maxValkyries)].write(v.nPriorityNodesProcessed : real / prioritySize : real);
                vLog.log('GEN:', gen : string, 'TOTAL MOVES:', v.nMoves : string, 'PROCESSED:', v.nProcessed : string, 'PRIORITY PROCESSED', v.nPriorityNodesProcessed : string, hstring=v.header);
                var processedString: string;
                // this is really an IDEAL average.
                var avg = startingSeeds : real / maxValkyries : real ;
                var std: real;
                var eff: real;
                for y in 1..maxValkyries*Locales.size {
                  var diff = this.valkyriesProcessed[y].read() - avg;
                  std += diff**2;
                  if this.valkyriesProcessed[y].read() != 0 {
                    eff += this.priorityValkyriesProcessed[y].read() : real;
                  }
                }
                std = abs(avg - sqrt(std/maxValkyries))/avg;
                eff /= maxValkyries;
                processedString = ''.join(' // BALANCE:  ', std : string, ' // ', ' EFFICIENCY:  ', eff : string, ' // ');
                this.log.log('GEN', '%05i'.format(gen), 'processed in', '%05.2dr'.format(Time.getCurrentTime() - this.generationTime) : string, 'BEST: %05.2dr'.format(bestInGen), processedString : string, hstring=this.yh);
                this.yh.printedHeader = true;
                this.generationTime = Time.getCurrentTime() : real;
                //this.lock.uwl(v.header);
                this.valkyriesProcessed.write(0);
                this.priorityValkyriesProcessed.write(0);
                v.nPriorityNodesProcessed = 0;
                v.nProcessed = 0;
                // time to move the fuck on.
                this.generation = gen + 1;
                this.moveOn[gen] = true;
              }
            }
          }
        } else {
          while this.generation < generations do chpl_task_yield();
          writeln("fin.");
      }
        // wait, you damn fool.
        //  var moveOn: [1..generations] single bool;
      }
    }
  }
}
