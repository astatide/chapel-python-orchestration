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
  var areSpawned: single bool;
  var numSpawned: atomic int;
  var valkyriesProcessed: [1..maxValkyries*Locales.size] atomic int;
  var priorityValkyriesProcessed: [1..maxValkyries*Locales.size] atomic real;
  var generationTime: real;
  var authors: domain(string) = ['Audrey Pratt', 'Benjamin Robbins'];
  var version: real = 0.1;
  // I guess this is protected or important in Chapel, in some way?
  //var release: string; // alpha
  var shutdown: bool = false;
  //var currentGe

  // Because the chromosomes are an abstraction of the gene network, and are
  // in many respects related more to the movement rather than graph problems,
  // the propagator is responsible for it.
  var chromosomeDomain: domain(string);
  var chromes: [chromosomeDomain] chromosomes.Chromosome;

  //var sendPorts: [0..maxValkyries] string;
  //var recvPorts: [0..maxValkyries] string;

  //var sendSocket: [0..maxValkyries] Socket;
  //var recvSocket: [0..maxValkyries] Socket;
  //var nChannels: domain(int);


  // this is from the msgHandler class.
  //proc init(n: int) {
    // basically, the inheritance isn't working as I would have expected.
    // see https://github.com/chapel-lang/chapel/issues/8232
    //super.init(maxValkyries*Locales.size);
    //this.size = maxValkyries;
  //}


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
    //this.ygg.initializeNetwork(n_seeds=startingSeeds);
    this.log.debug("Initialising chromosomes", this.yh);
    this.ygg.initializeRoot();
    this.initChromosomes();
    // basically, re-add the root node to make sure its connections are up to date
    this.ygg.add_node(this.ygg.rootNode, this.yh);
    this.log.debug("Initialising root node", this.yh);
    this.log.debug("About to add existing nodes to the processing list", this.yh);
    var ids = this.ygg.ids;
    this.log.debug(this.ygg.ids : string, this.yh);
    for i in this.ygg.ids {
      if i != 'root' {
        if i != this.ygg.testNodeId {
          // Adding special nodes is a pain.  I should probably set a processing flag.
          this.nodesToProcess.add(i);
          this.processedArray[i].write(false);
          this.inCurrentGeneration.add(1);
        }
      }
    }
    this.log.debug(this.nodesToProcess: string, this.yh);
    this.log.debug('INITIALIZED', this.inCurrentGeneration.read() : string, 'seeds.', this.yh);
  }

  proc initChromosomes() {
    forall deme in 0..4 {
      forall i in 1..nChromosomes {
        // Here, we're going to be given the instructions for generating chromosomes.
        // No reason this can't be parallel, so let's do it.
        this.log.debug('Spawning chromosome', this.yh);
        var nc = new chromosomes.Chromosome();
        this.log.debug('Chromosome ID', nc.id, 'spawned.  Preparing genes.', this.yh);
        nc.prep(startingSeeds, chromosomeSize-startingSeeds);
        nc.currentDeme = deme;
        this.log.debug('Genes prepped in Chromosome ID; converting into nodes', nc.id, this.yh);
        nc.log = this.log;
        var n: int = 1;
        nc.generateNodes(this.ygg);
        //for combo in nc.geneSets() {
        //  var c = combo : string;
          //this.log.debug('LOC IN GENE:', n : string, 'SET:', hstring=this.yh);
        //  n += 1;
        //}
        this.lock.wl();
        this.chromosomeDomain.add(nc.id);
        this.chromes[nc.id] = nc;
        this.lock.uwl();
      }
    }
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
    // I think the other valkyries should be able to do their thing.
    // should probably do this for each task but hey whatever.
    // We're catching a signal interrupt, which is slightly mangled for some reason.
    // start up the main procedure by creating some valkyries.
    coforall L in Locales {
      on L do {
        if true {
          //super.init(maxValkyries*Locales.size);
          // create a logger, just for us!
          //var vLog = new shared ygglog.YggdrasilLogging();
          //vLog.currentDebugLevel = debug;
          //var localeUpdated: [1..generations] atomic bool;
          //this.log.debug("Copying network onto locale", here.id : string, this.yh);
          // why the fuck is this suddenly a problem?
          //this.log.debug("Network copied onto locale ", here.id : string, this.yh);
          //var nodeHasCopy: single bool;
          //vLock.log = vLog;
          // they really do not like to do this.  Why?
          var yggLocalCopy = new shared network.GeneNetwork();
          //ref YNC = yggNodeCopy;
          //var yggNodeCopy: network.GeneNetwork;
          //begin with (ref yggNodeCopy) this.ygg.clone(yggNodeCopy);
          this.ygg.clone(yggLocalCopy);
          //var yggLocalCopy = new shared network.GeneNetwork();
          coforall i in 1..maxValkyries {
            // spin up the Valkyries!
            //var yggLocalCopy = this.ygg.clone();
            //var mH = new messaging.msgHandler(1);
            //var mH = new shared yggMsgHandler(1);
            var vLog = new shared ygglog.YggdrasilLogging();
            vLog.currentDebugLevel = debug;
            var vLock = new shared spinlock.SpinLock();
            vLock.t = 'Valkyrie';
            //var yggLocalCopy: network.GeneNetwork;
            // ?? This doesn't seem to actually be working.
            //yggLocalCopy.log = vLog;
            //yggLocalCopy.lock.log = vLog;
            var v = new shared valkyrie(1);
            v.currentTask = i;
            v.currentLocale = L : string;
            v.yh += 'run';
            for iL in v.logo {
              vLog.header(iL, hstring=v.header);
            }
            // also, spin up the tasks.
            //this.lock.wl(v.header);
            //var yggLocalCopy: network.GeneNetwork;
            // spin it off baby.
            //begin with (ref yggLocalCopy) yggLocalCopy = this.ygg.clone();
            //nodeHasCopy;
            //var yggLocalCopy = this.ygg;
            //yggLocalCopy = yggNodeCopy.clone();
            //yggLocalCopy = this.ygg.clone();
            //var ayh = new ygglog.yggHeader();
            vLog.log('Initiating spawning sequence', hstring=v.header);
            var vp = v.valhalla(1, v.id, mSize : string, vLog, vstring=v.header);
            vLog.log('Spawn function complete; awaiting node copy of network', hstring=v.header);
            vLog.log('Cloning network for task', i : string, hstring=v.header);
            //var yggLocalCopy = this.ygg.clone();
            //var yggLocalCopy = yggNodeCopy.clone();
            //writeln("Original");
            //writeln(this.ygg : string);
            //writeln("Node copy");
            //writeln(yggNodeCopy : string);
            //var yggLocalCopy = new shared network.GeneNetwork();
            //ref YLC = yggLocalCopy;
            //while !(yggNodeCopy.isCopyComplete) do chpl_task_yield();
            vLog.log('Network has the following IDs', yggLocalCopy.ids : string, hstring=v.header);
            //yggLocalCopy = yggNodeCopy.clone();
            //yggNodeCopy.clone(yggLocalCopy);
            vLog.log('Clone complete; awaiting arrival of other valkyries', hstring=v.header);
            if this.numSpawned.fetchAdd(1) < ((Locales.size*maxValkyries)-1) {
              // we want to wait so that we spin up all processes.
              this.areSpawned;
            } else {
              this.areSpawned = true;
            }
            //this.lock.uwl(v.header);
            v.moveToRoot();

            for gen in 1..generations {

              v.gen = gen;
              vLog.log('Starting GEN', '%{######}'.format(gen), hstring=v.header);
              var currToProc: string;
              var toProcess: domain(string);
              var path: network.pathHistory;
              toProcess.clear();
              //this.lock.wl(v.header);
              //if this.generationTime == 0 : real {
              //  this.generationTime = Time.getCurrentTime();
              //}
              //if !localeUpdated[gen].testAndSet() {
              //yggLocalCopy = this.ygg;
              //yggLocalCopy = this.ygg.clone();
              //this.ygg.update(yggLocalCopy);
              //}
              //this.lock.uwl(v.header);
              vLog.debug('Beginning processing', hstring=v.header);
              vLog.debug(this.nodesToProcess : string, hstring=v.header);
              var prioritySize = v.priorityNodes.size;
              while this.inCurrentGeneration.read() > 0 {
                // We clear this out because it is faster to just re-enumerate the
                // nodes that need processing, rather than explicitly calculating
                // the path towards every node.  Particularly as that results in tasks
                // performing a lot of unnecessary computations once a lot of nodes
                // have been processed.
                vLog.debug('Assessing nodes that must be handled', hstring=v.header);
                currToProc = '';
                toProcess.clear();
                for id in this.nodesToProcess {
                    toProcess.add(id);
                }
                if !(v.priorityNodes & toProcess).isEmpty() {
                  toProcess = toProcess & v.priorityNodes;
                }
                // Assuming we have some things to process, do it!
                vLog.debug('toProcess created', hstring=v.header);
                if !toProcess.isEmpty() {
                  // We can remove nodes from the domain processedArray is built on, which means we need to catch and process.
                  var existsInDomainAndCanProcess: bool = false;
                  // This function now does the atomic test.
                  vLog.debug('Returning nearest unprocessed', hstring=v.header);
                  (currToProc, path) = yggLocalCopy.returnNearestUnprocessed(v.currentNode, toProcess, v.header, this.processedArray);
                  vLog.debug('Unprocessed found.', hstring=v.header);
                  if currToProc != '' {
                    for deme in yggLocalCopy.nodes[currToProc].demeDomain {
                      // Actually, reduce the count BEFORE we do this.
                      // Otherwise we could have threads stealing focus that should
                      // actually be idle.
                      vLog.debug('Attempting to decrease count for inCurrentGeneration', hstring=v.header);
                      this.inCurrentGeneration.sub(1);
                      vLog.debug('inCurrentGeneration successfully reduced', hstring=v.header);
                      // If this node is one of the ones in our priority queue, remove it
                      // as we clearly processing it now.
                      if v.priorityNodes.contains(currToProc) {
                        v.priorityNodes.remove(currToProc);
                        v.nPriorityNodesProcessed += 1;
                      }
                      vLog.debug('Processing seed ID', currToProc : string, hstring=v.header);
                      var d = yggLocalCopy.deltaFromPath(path, path.key(0), hstring=v.header);
                      //var d = yggLocalCopy.move(v, currToProc, path, createEdgeOnMove=false, edgeDistance);
                      d.to = currToProc;
                      var newMsg = new messaging.msg(d);
                      newMsg.i = deme;
                      newMsg.COMMAND = messaging.command.RECEIVE_AND_PROCESS_DELTA;
                      vLog.debug("Attempting to run Python on seed ID", currToProc : string, hstring=v.header);
                      vLog.debug("Sending the following msg:", newMsg : string, hstring=v.header);
                      v.SEND(newMsg);
                      vLog.debug("Message & delta sent; awaiting instructions", hstring=v.header);
                      //v.RECV(newMsg, i);
                      /*
                      var vheader = v.header;
                      vheader += "ValkyriePython";
                      var l: string;
                      while vp.stdout.readline(l) {
                        if l == "VALKYRIE PROCESSED MSG\n" {
                          // Do nothing.  Don't read again, that's for sure.
                          // probably have a race condition here, so.
                          break;
                        } else {
                          this.log.log(l, hstring=vheader);
                        }
                      }*/
                      //var retMsg = mH.RECV();
                      //mH.receiveMessage();
                      //mH.receiveMessage();
                      //v.receiveMessage();
                      var m = v.RECV();
                      var score = m.r;

                      //var score: real;
                      //var scoreString: real;
                      //scoreString = retMsg.open(score);
                      vLog.debug('SCORE FOR', currToProc : string, 'IS', score : string, hstring=v.header);
                      //vLog.debug('SCORE FOR', currToProc : string, 'IS', retMsg.r : string, hstring=v.header);
                      //vLog.debug('MSG FOR', currToProc : string, 'IS', retMsg : string, hstring=v.header);
                      //vLog.debug('SCORE FOR', currToProc : string, 'IS', scoreString : string, hstring=v.header);

                      //this.lock.wl(v.header);
                      /*if false {
                        var (maxVal, maxLoc) = maxloc reduce zip(this.scoreArray, this.scoreArray.domain);
                        if score < maxVal {
                          this.scoreArray[maxLoc] = score;
                          this.idArray[maxLoc] = currToProc;
                        }
                      } else {*/
                      // set the score on the node.
                      // this is thread safe.
                      this.ygg.nodes[currToProc].scores[deme] = score;
                      /*
                      var sA = this.scoreArray[deme, 1..maxPerGeneration];
                      // Is that a problem?
                      //var (minVal, minLoc) = minloc reduce zip(sA, sA.domain);
                      var minVal : real = Math.INFINITY;
                      var minLoc : int;
                      for (v,l) in zip(sA, sA.domain) {
                        if v <= minVal {
                          minVal = v;
                          minLoc = l;
                        }
                      }
                      if score >= minVal {
                        this.scoreArray[deme,minLoc] = score;
                        this.idArray[deme, minLoc] = currToProc;
                      }
                      //}
                      this.lock.uwl(v.header);
                      */
                    }
                  }
                  // While it seems odd we might try this twice, this helps us keep
                  // track of algorithm efficiency by determining whether we're processing
                  // the nodes in our priority queue or not.
                  if v.priorityNodes.contains(currToProc) {
                    v.priorityNodes.remove(currToProc);
                  }
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
              if !v.moved {
                if currToProc != '' {
                  //yggLocalCopy.move(v, currToProc, path, createEdgeOnMove=false, edgeDistance);
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
                // moveOn is an array of sync variables.  We're blocked from reading
                // until that's set to true.
                this.moveOn[gen];
                this.lock.rl(v.header);
                vLog.debug('MOVING ON in gen', gen : string, this.nodesToProcess : string, v.header);
                this.lock.url(v.header);
              } else {
                // Same stuff here, but as this is the last Valkyrie, we also
                // do global cleanup to ensure the global arrays are ready.
                vLog.debug('Handling cleanup on gen', gen : string, v.header);
                v.moved = false;

                /*
                // first, get the best!
                var sA = this.scoreArray[deme, 1..maxPerGeneration];
                // Is that a problem?
                //var (minVal, minLoc) = minloc reduce zip(sA, sA.domain);
                var minVal : real = Math.INFINITY;
                var minLoc : int;
                for (v,l) in zip(sA, sA.domain) {
                  if v <= minVal {
                    minVal = v;
                    minLoc = l;
                  }
                }
                if score >= minVal {
                  this.scoreArray[deme,minLoc] = score;
                  this.idArray[deme, minLoc] = currToProc;
                }*/

                // we'll just throw this in here for now.
                // Only do the max!
                //var bestInGen: real = this.scoreArray[1];
                var (bestInGen, minLoc) = maxloc reduce zip(this.scoreArray, this.scoreArray.domain);
                var chromosomesToAdvance: domain(string);
                var c: [chromosomesToAdvance] chromosomes.Chromosome;
                vLog.debug('Determining which chromosomes to advance', v.header);
                for node in this.idArray {
                  // these are the best nodes, so work em!
                  if node != '' {
                    // this is in the event that all of our genes suck.
                    // do we really care about saving genes if the entire set sucks?
                    for nc in this.ygg.nodes[node].chromosomes {
                      if !chromosomesToAdvance.contains(nc) {
                        chromosomesToAdvance.add(nc);
                        //c[nc] = this.chromes[nc];
                      }
                    }
                  }
                }
                // clear the domain of our losers.
                vLog.debug('Clearing the domain of those who are not continuing.', v.header);
                if true {
                  var vheader = v.header;
                  for chrome in this.chromosomeDomain {
                    if !chromosomesToAdvance.contains(chrome) {
                      this.chromosomeDomain.remove(chrome);
                    }
                  }
                  vLog.debug('Advancing the chromosomes.', vheader);
                  forall chrome in chromosomesToAdvance {
                    var nc = this.chromes[chrome];
                    vLog.debug('Pushing forward chromosome ID', nc.id : string , vheader);
                    for node in nc.geneIDs {
                      this.lock.wl(vheader);
                      this.nextGeneration.add(node);
                      this.lock.uwl(vheader);
                    }
                    vLog.debug('Advancing chromosome ID', nc.id : string , vheader);
                    for i in 1..nDuplicates {
                      //var cc: chromosomes.Chromosome;
                      var cc = nc.clone();
                      vLog.debug('New chromosome ID', cc.id : string , vheader);
                      cc.advanceNodes(this.ygg);
                      for node in cc.geneIDs {
                        this.lock.wl(vheader);
                        this.nextGeneration.add(node);
                        this.lock.uwl(vheader);
                      }
                      this.lock.wl(vheader);
                      this.chromosomeDomain.add(cc.id);
                      this.chromes[cc.id] = cc;
                      this.lock.uwl(vheader);
                    }
                  }
                }
                /*
                for ij in 1..maxPerGeneration {
                  currToProc = this.idArray[ij];
                  //if this.scoreArray[ij] == Math.INFINITY {
                  //  break;
                  //}
                  if this.scoreArray[ij] == 0 {
                    break;
                  }
                  this.log.debug('Attempting to move ID', currToProc, 'into the next generation.', hstring=v.header);
                  var nextNode = this.ygg.nextNode(currToProc, hstring=v.header);
                  // They should really know about each other, I mean, come on.
                  assert(this.ygg.nodes[currToProc].nodeInEdges(nextNode, v.header));
                  assert(this.ygg.nodes[nextNode].nodeInEdges(currToProc, v.header));
                  var mergeTest: string;
                  this.log.debug('Node', nextNode : string, 'added', hstring=v.header);
                  v.nProcessed += 1;
                  v.moved = true;
                  // test it!
                  //if unitTestMode {
                  if currToProc != this.idArray[minLoc] {
                    this.log.debug('Attempting to merge ID', currToProc, 'with', this.idArray[minLoc], hstring=v.header);
                    mergeTest = this.ygg.mergeNodes(currToProc, this.idArray[minLoc], hstring=v.header);
                    this.log.debug('Node', mergeTest : string, 'added', hstring=v.header);
                    //}
                  }
                  //this.lock.wl(v.header);
                  // We're testing to see if we can do this.
                  // I want this in there ultimately, but it needs to
                  // not result in a race condition.
                  //this.nodesToProcess.remove(currToProc);
                  // We only want to add to an empty domain here such that we only
                  // prioritize nodes which are close to the current node.
                  // Eventually, if we mutate, we'll add that in, too.
                  //v.priorityNodes.clear();
                  v.priorityNodes.add(nextNode);
                  this.nextGeneration.add(nextNode);
                  //if unitTestMode {
                  if currToProc != this.idArray[minLoc] {
                    v.priorityNodes.add(mergeTest);
                    this.nextGeneration.add(mergeTest);
                  }
                  //}
                }*/
                //this.scoreArray = Math.INFINITY;
                this.scoreArray = -1;

                vLog.debug('Switching generations', v.header);
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
