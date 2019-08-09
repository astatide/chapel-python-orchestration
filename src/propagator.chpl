// Do I need all this?  Probably not but I'll sort that issue later.
use rng;
use genes;
use network;
//use uuid;
use Math;
//use VisualDebug;
use ygglog;
use spinlock;
use Time;
use IO.FormattedIO;
use chromosomes;
//use gjallarbru;
use Spawn;
use messaging;
use HashedDist;
use Time;
//use CommDiagnostics;
use AllLocalesBarriers;
use valkyrie;

//use ReplicatedDist;

record scoreComparator {
  proc keyPart(x: (string, real), i: int) {
    if i > 2 then
      return (-1, ('NULL', 0));
    return (0, (x[1], x[2]));
  }
}

//config const mSize = 20;
config const highFitnessToKeep = 10;
config const highNovelToKeep = 10;
config const archiveChance = 0.05;
config const maxPerGeneration = 1000;
config const mutationRate = 0.03;
config const maxValkyries = 1;
config const startingSeeds = 4;
//config const createEdgeOnMove = true;
//config const edgeDistance = 10;
config const debug = -1;
config const generations = 100;
config const unitTestMode = false;

config var nChromosomes = 6;
config var chromosomeSize = 36;
config var nDuplicates = 4;
config var nDemes = 4;

config var createEdgeOnMove: bool = true;
config var stepsForEdge: int = 15;

config const reportTasks: bool = false;

config var exportNetwork: bool = false;

config var topKNovel: int = 10;

//allLocalesBarrier.reset(maxValkyries);

// These are all the global things we need to access from everywhere.

// Chromosome stuff
var cLock = new shared spinlock.NetworkSpinLock();
var chromosomeDomain: domain(string) dmapped Hashed(idxType=string, mapper = new network.mapperByLocale());
var chromosomeArchiveDomain: domain(string) dmapped Hashed(idxType=string, mapper = new network.mapperByLocale());
var chromes: [chromosomeDomain] chromosomes.Chromosome;
var archive: [chromosomeArchiveDomain] chromosomes.Chromosome;
cLock.t = 'Chromosomes';
var chromosomesToAdvance: domain(string);
//cLock.log = new shared ygglog.YggdrasilLogging();

// valkyrie stuff.
var valkyriesDone: [1..generations] atomic int;
var moveOn: [1..generations] single bool;
// for chromosome prop later on
var readyForChromosomes: [1..generations] single bool;
var finishedChromoProp: atomic int;
var howManyValks = (((Locales.size)*maxValkyries)-1);

// node stuff
var nodesToProcess: domain(string);
var processedArray: [nodesToProcess] atomic bool;
var scoreArray: [0..4,1..highFitnessToKeep] real = -1; //Math.INFINITY;
var idArray: [0..4,1..highFitnessToKeep] string;
var novelArray: [0..4,1..highNovelToKeep] real = -1; //Math.INFINITY;
var novelIdArray: [0..4,1..highNovelToKeep] string;

// network stuff
var inCurrentGeneration: atomic int;
//var nextGeneration: domain(string);
var valkyriesProcessed: [1..maxValkyries*Locales.size] atomic int;
var priorityValkyriesProcessed: [1..maxValkyries*Locales.size] atomic real;

record CustomMapper {
  proc this(ind:int, targetLocs: [?D] locale) : D.idxType {
    return ind;
  }
}

class Propagator {
  // this is going to actually hold all the logic for running EvoCap.
  var generation: int;

  var ygg: shared network.networkMapper();
  var yh = new ygglog.yggHeader();
  var log: shared ygglog.YggdrasilLogging();
  var lock: shared spinlock.SpinLock;
  var generationTime: real;
  var authors: domain(string) = ['Audrey Pratt', 'Benjamin Robbins'];
  var version: real = 0.1;
  var shutdown: bool = false;
  var udevrandom = new owned rng.UDevRandomHandler();
  var newrng = udevrandom.returnRNG();

  proc init() {
    this.complete();
    // We initialize the network, creating the networkMapper object, logging
    // infrastructure
    this.yh = new ygglog.yggHeader();
    this.yh += 'Ragnarok';
    this.yh.sendTo = "RAGNAROK-" + here.id : string;
    this.yh.header = 'ragnarok';
    this.yh.id = this.yh.sendTo;
    this.yh.useFile = true;
    this.log = new shared ygglog.YggdrasilLogging();
    this.log.currentDebugLevel = debug;
    this.lock = new shared spinlock.SpinLock();
    this.lock.t = 'Ragnarok';
  }

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
        this.log.header(logo[i], this.yh);
    }
    for i in 0..3 {
      this.log.header(about[i], this.yh);
    }
  }

  proc exitRoutine() throws {
    // command the logger to shut down, then exit.
    this.lock.wl();
    // NEVER LET GO.
    for i in 1..maxValkyries {
      // tell the Valkyries to quit their shit.
      var m = new messaging.msg(0);
      //m.COMMAND = this.command.SHUTDOWN;
      //SEND(m, i+(maxValkyries*here.id));
    }
    this.log.critical('SHUTDOWN INITIATED');
    this.log.exitRoutine();
    this.lock.uwl();
    throw new owned Error();
  }

  proc setShutdown() {
    this.shutdown = true;
  }

  // Because the chromosomes are an abstraction of the gene network, and are
  // in many respects related more to the movement rather than graph problems,
  // the propagator is responsible for it.

  proc initChromosomes(ref nG: shared network.networkGenerator, ref nM: shared network.networkMapper, yH: ygglog.yggHeader) {

    on this.locale {
      forall deme in 0..4 with (ref nG, ref nM) {
        forall i in 1..nChromosomes with (ref nG, ref nM) {
          // Here, we're going to be given the instructions for generating chromosomes.
          // No reason this can't be parallel, so let's do it.
          var nc = new chromosomes.Chromosome();
          nc.id = nG.generateChromosomeID;
          nc.prep(startingSeeds, chromosomeSize-startingSeeds);
          nc.currentDeme = deme;
          var n: int = 1;
          nc.generateNodes(nG, nM, yH);
          cLock.wl();
          chromosomeDomain.add(nc.id);
          chromes[nc.id] = nc;
          cLock.uwl();
        }
      }
    }
  }

  proc returnRandomChromosome() {
    // make a copy of the current domain.  For... reasons.
    var localDomain: [1..(highFitnessToKeep+highNovelToKeep)] string;
    var j = 1;
    for i in chromosomeDomain {
      localDomain[j] = i;
      j += 1;
    }
    // Get random index.
    var randomSelection = (newrng.getNext() * (highFitnessToKeep+highNovelToKeep)) : int;
    return localDomain[randomSelection];
  }

  proc advanceChromosomes(ref nG: shared network.networkGenerator, ref nM: shared network.networkMapper, yH: ygglog.yggHeader, gen: int) {
    on this.locale {
      var newCD: domain(string);
      var newC: [newCD] chromosomes.Chromosome;
      cLock.rl();
      // how many should this task get?  Only get about that many.
      // you... you know, you really need at least one.
      //var maxProcessed: int = max(ceil((nDemes * maxPerGeneration / Locales.size) / maxValkyries-1): int, 1);
      //var processed: int = 0;
      var cacheCD: domain(string);
      var cacheC: [cacheCD] chromosomes.Chromosome;
      var n: int = max((maxPerGeneration/maxValkyries) : int, 1);
      for i in 1..(maxPerGeneration/maxValkyries) : int {
        //if !chromes[chrome].isProcessed.testAndSet() {
        var chrome = returnRandomChromosome();
        var nc: chromosomes.Chromosome;
        // cache the chromosome, as we'll be doing a trillion and a half pulls from it.
        // So it's worth, you know, sorting or whatever.
        if cacheCD.contains(chrome) {
          nc = cacheC[chrome].clone();
        } else {
          cacheCD.add(chrome);
          cacheC[chrome] = chromes[chrome].clone();
          nc = cacheC[chrome];
        }
        var oldId = chromes[chrome].id;
        this.log.log('Advancing chromosome ID:', oldId, hstring=yH);
        for i in 1..nDuplicates {
          var cc = nc.clone();
          cc.id = nG.generateChromosomeID;
          cc.advanceNodes(nG, nM, yH, gen);
          newCD.add(cc.id);
          newC[cc.id] = cc;
        }
        //}
      }
      cLock.url();
      cLock.wl();
      //network.globalLock.rl();
      for chrome in newCD {
        chromosomeDomain.add(chrome);
        chromes[chrome] = newC[chrome];
      }
      //network.globalLock.url();
      cLock.uwl();
    }
  }

  proc setBestChromosomes(yh: ygglog.yggHeader) {
    var (bestInGen, minLoc) = maxloc reduce zip(scoreArray, scoreArray.domain);
    //var chromosomesToAdvance: domain(string);
    var c: [chromosomesToAdvance] chromosomes.Chromosome;
    var udevrandom = new owned rng.UDevRandomHandler();
    var newrng = udevrandom.returnRNG();

    this.log.log('Determining which chromosomes to advance', yh);
    for chrome in chromosomeDomain {
      // we're also adding things to the archive, if necessary.
      //if newrng.getNext() < archiveChance {
      //  this.log.log('Adding chromosome ID', chrome, 'to the archive',yh);
      //  chromosomeArchiveDomain.add(chrome);
      //  archive[chrome] = chromes[chrome];
      //}
      // copy it back to us.
      var deme = chromes[chrome].currentDeme;
      var (bestScore, bestNode) = chromes[chrome].bestGeneInDeme[chromes[chrome].currentDeme];
      var (lowestScore, minLoc) = minloc reduce zip(scoreArray[deme,..], scoreArray.domain.dim(2));
      this.log.log('Finding the highest scoring node on chromosome ID', chrome, 'and seeing if it is good enough.', yh);
      if bestScore > lowestScore {
        scoreArray[deme, minLoc] = bestScore;
        idArray[deme, minLoc] = chrome;
      }
    }
    for deme in 0..4 {
      for z in 1..highFitnessToKeep {
        if idArray[deme,z] != '' {
          this.log.log('Adding the following chromosome ID to be advanced:', idArray[deme,z], yh);
          chromosomesToAdvance.add(idArray[deme,z]);
        }
      }
    }
    // clear the domain of our losers.
    this.log.log('Clearing the domain of those who are not continuing.', yh);
    var delChrome: domain(string);
    for chrome in chromosomeDomain {
      if !chromosomesToAdvance.contains(chrome) {
        delChrome.add(chrome);
      }
    }
    for chrome in delChrome {
      chromosomeDomain.remove(chrome);
    }
  }

  proc setNovelChromosomes(yh: ygglog.yggHeader) {
    var (bestInGen, minLoc) = maxloc reduce zip(novelArray, novelArray.domain);
    //var chromosomesToAdvance: domain(string);
    var c: [chromosomesToAdvance] chromosomes.Chromosome;
    var udevrandom = new owned rng.UDevRandomHandler();
    var newrng = udevrandom.returnRNG();

    for chrome in chromosomeDomain {
      if newrng.getNext() < archiveChance {
        this.log.log('Adding chromosome ID', chrome, 'to the archive',yh);
        chromosomeArchiveDomain.add(chrome);
        archive[chrome] = chromes[chrome];
      }
    }

    this.log.log('Determining which novel chromosomes to advance', yh);
    for c1 in chromosomeDomain {
      var deme = chromes[c1].currentDeme;
      for c2 in chromosomeDomain {
        // copy it back to us.
        chromes[c1].calculateNovelty(chromes[c2]);
      }
      // we need to check against the archive, too.
      for c2 in chromosomeArchiveDomain {
        // copy it back to us.
        chromes[c1].calculateNovelty(archive[c2]);
      }
      chromes[c1].finalizeNovelty();
      var (bestScore, bestNode) = chromes[c1].novelGeneInDeme[chromes[c1].currentDeme];
      var (lowestScore, minLoc) = minloc reduce zip(novelArray[deme,..], novelArray.domain.dim(2));
      this.log.log('Finding the most novel on chromosome ID', c1, 'and seeing if it is good enough.', yh);
      if bestScore > lowestScore {
        novelArray[deme, minLoc] = bestScore;
        novelIdArray[deme, minLoc] = c1;
      }
    }
    for deme in 0..4 {
      for z in 1..highNovelToKeep {
        if novelIdArray[deme,z] != '' {
          this.log.log('Adding the following chromosome ID to be advanced:', novelIdArray[deme,z], yh);
          chromosomesToAdvance.add(novelIdArray[deme,z]);
        }
      }
    }
  }

  proc startLoggingTasks() {
    if reportTasks {
      begin {
        while true {
          var T: Time.Timer;
          T.start();
          this.log.log('runningTasks: ', here.runningTasks() : string, this.yh);
          this.log.log('queuedTasks: ', here.queuedTasks() : string, this.yh);
          this.log.log('blockedTasks: ', here.blockedTasks() : string, this.yh);
          this.log.log('totalThreads: ', here.totalThreads() : string, this.yh);
          this.log.log('idleThreads: ', here.idleThreads() : string, this.yh);
          T.stop();
          sleep(10 - T.elapsed(TimeUnits.seconds));
        }
      }
    }
    begin {
      while true {
        var T: Time.Timer;
        T.start();
        this.log.log('CURRENT GENERATION COUNT:', inCurrentGeneration.read() : string, this.yh);
        T.stop();
        sleep(30 - T.elapsed(TimeUnits.seconds));
      }
    }
  }

  proc exportCurrentNetworkState(yh: ygglog.yggHeader) {
    if exportNetwork {
      this.log.log('Exporting network', hstring=yh);
      network.globalLock.rl();
      network.exportGlobalNetwork(0);
      network.globalLock.url();
      this.log.log('Export complete!', hstring=yh);
    }
  }

  proc run() {
    // Print out the header, yo.
    this.yh += 'run';
    this.header();

    this.log.log("Setting up logging features", this.yh);
    if this.locale == Locales[0] {
      this.startLoggingTasks();
    }

    this.log.log("Spawn local network and networkGenerator", this.yh);
    var ygg = new shared network.networkMapper();
    var nG = new shared network.networkGenerator();      // we're gonna want a list of network IDs we can use.

    if this.locale == Locales[0] {
      // reset the barrier
      allLocalesBarrier.reset(maxValkyries);
    }

    ygg.log = this.log;
    ygg.log.currentDebugLevel = debug;

    this.log.log("Local networks spawned; creating chromosomes", this.yh);
    initChromosomes(nG, ygg, this.yh);
    this.log.log("Adding new nodes to unprocessed list", this.yh);
    nG.addUnprocessed(ygg);
    this.log.log("Setting the current generation count", this.yh);

    begin inCurrentGeneration.add(nG.currentId.read()-1);

    coforall i in 1..maxValkyries with (ref nG, ref ygg) {
      {
        // spin up the Valkyries!
        var v = new shared valkyrie.valkyrieHandler(1);
        v.currentTask = i;
        v.currentLocale = here : string;
        v.setSendTo();
        v.yh += 'run';
        v.currentNode = nG.root;
        // share the logger.  It's fine.
        v.log = this.log;

        // we want to pipe some of the output to the ragnarok logs.
        var currentYggHeader: ygglog.yggHeader = v.header;

        for iL in v.logo {
          this.log.header(iL, hstring=currentYggHeader);
        }

        this.log.log('Initiating spawning sequence', hstring=currentYggHeader);
        var vp = v.valhalla(1, v.id, this.log, vstring=currentYggHeader);

        // wait until everyone is nice and spawned.
        allLocalesBarrier.barrier();

        // network export shit.
        if here == Locales[0] && i == 1 {
          this.exportCurrentNetworkState(currentYggHeader);
        }

        // timer stuff.
        var T: Time.Timer;

        for gen in 1..generations {
          v.gen = gen;
          this.log.log('Starting GEN', '%{######}'.format(gen), hstring=currentYggHeader);
          var currToProc: string;
          var toProcess: domain(string);
          var path: network.pathHistory;
          var removeFromSet: domain(string);
          this.log.log('Beginning processing; Assessing nodes that must be handled', hstring=v.header);

          for id in nG.currentGeneration {
            // don't add root.
            if id != 'root' {
              toProcess.add(id);
              this.log.log('Adding node ID: ', id : string, hstring=v.header);
            }
          }

          // this is only if we don't have any, for some reason.  This should
          // basically always be false.
          if toProcess.isEmpty() {
            // This checks atomics, so it's gonna be slow.
            // In an ideal world, we rarely call it.
            for id in network.globalUnprocessed {
              if !network.globalIsProcessed[id].read() {
                toProcess.add(id);
              }
            }
          }
          this.log.debug('toProcess created', hstring=v.header);
          while !toProcess.isEmpty() {

            // Assuming we have some things to process, do it!
            currToProc = '';
            // We can remove nodes from the domain processedArray is built on, which means we need to catch and process.
            // This function now does the atomic test.
            this.log.log('Returning nearest unprocessed', hstring=v.header);
            var steps: int;
            T.start();
            (currToProc, path, removeFromSet, steps) = ygg.returnNearestUnprocessed(v.currentNode, toProcess, v.header, network.globalIsProcessed);
            T.stop();
            this.log.log('Unprocessed found in:', T.elapsed() : string, 'ID:', currToProc : string, hstring=v.header);
            T.clear();
            for z in removeFromSet {
              if toProcess.contains(z) {
                this.log.debug('Removing ID:', z : string, hstring=v.header);
                toProcess.remove(z);
              }
            }
            if currToProc != '' {
              this.log.debug('Removing from local networkGenerator, if possible.', hstring=v.header);
              // AHA!
              nG.removeUnprocessed(currToProc);
              toProcess.remove(currToProc);
              //this.log.debug('Attempting to decrease count for inCurrentGeneration', hstring=v.header);
              begin inCurrentGeneration.sub(1);

              // this handles all the ZMQ and TF bits.
              var delta = ygg.deltaFromPath(path, v.currentNode, hstring=v.header);
              delta.from = v.currentNode;
              delta.to = currToProc;
              network.globalLock.rl();
              ref newNode = network.globalNodes[currToProc];
              if createEdgeOnMove {
                if steps > stepsForEdge {
                  ref oldNode = network.globalNodes[v.currentNode];
                  oldNode.join(newNode, delta, v.header);
                  newNode.join(oldNode, delta*-1, v.header);
                }
              }
              T.start();
              for (score, deme) in v.processNode(newNode, delta) {
                T.stop();
                // now, set the chromosome.
                cLock.rl();
                ref nc = chromes[newNode.chromosome];
                nc.l.wl();
                var inChromeID = nc.returnNodeNumber(currToProc);
                this.log.log("Node ID:", currToProc : string, "Score:", score : string, "Deme:", deme : string, "Time:", T.elapsed() : string, hstring=v.header);
                this.log.log('Node:', newNode : string, hstring=v.header);
                T.clear();
                //this.log.log('NodeNumber:', inChromeID : string, "Node ID:", currToProc : string, "Chromosome ID:", nc : string, "Deme:", deme : string, hstring=v.header);
                this.log.log('DemeDomain in chromosome:', nc.geneIDs : string, hstring=v.header);
                try {
                  assert(nc.scores.domain.contains(inChromeID));
                } catch {
                  this.log.log('ID NOT IN CHROMOSOME:', currToProc : string, hstring=v.header);
                  this.log.log('Node:', newNode : string, hstring=v.header);
                  this.log.log('Chromosome', nc : string, hstring=v.header);
                }
                nc.scores[inChromeID] = score;
                nc.l.uwl();
                cLock.url();
              }
              network.globalLock.url();
            } else {
              // actually, if that's the case, we can't do shit.  So break and yield.
              break;
            }
            //begin this.log.log('Remaining in generation:', inCurrentGeneration.read() : string, 'priorityNodes:', v.priorityNodes : string, hstring=v.header);
            if this.shutdown {
              this.exitRoutine();
            }
          }
          if !v.moved {
            // if we haven't moved, we should move our valkyrie to something in the current generation.  It makes searching substantially easier.
            // but we should reinclude that logic _later_.  As it's busted.
            // do something about it, why don't you.
          }

          valkyriesDone[gen].fetchAdd(1);
          //if valkyriesDone[gen].fetchAdd(1) < howManyValks {
          if this.locale == Locales[0] && i == 1 {
            // why are we waiting on locale0?  Sanity.
            this.log.log('Waiting until all locales are done', this.yh);
            begin {
              // We do eventually want to end this.
              while valkyriesDone[gen].read() < howManyValks+1 {
                var T: Time.Timer;
                T.start();
                this.log.log('Finished valkyries:', valkyriesDone[gen].read() : string, this.yh);
                T.stop();
                sleep(10 - T.elapsed(TimeUnits.seconds));
              }
            }
            while valkyriesDone[gen].read() < howManyValks+1 do chpl_task_yield();
            this.continueEndOfGeneration(v, nG, gen, ygg, this.yh);
          } else {
            this.waitEndOfGeneration(v, nG, gen, ygg, currentYggHeader);
          }
        }
      }
    }
  }

  proc continueEndOfGeneration(ref v: shared valkyrie.valkyrieHandler, ref nG: shared network.networkGenerator, gen: int, ref ygg: shared network.networkMapper, yh: ygglog.yggHeader) {
    // Same stuff here, but as this is the last Valkyrie, we also
    // do global cleanup to ensure the global arrays are ready.
    this.log.log('Handling cleanup on gen', gen : string, yh);
    v.moved = false;
    //nextGeneration.clear();
    // we'll just throw this in here for now.
    // Only do the max!
    this.setNovelChromosomes(yh);
    this.setBestChromosomes(yh);
    // export the network!
    this.exportCurrentNetworkState(yh);
    //nG.setCurrentGeneration();
    readyForChromosomes[gen] = true;
    advanceChromosomes(nG, ygg, yh, gen+1);
    scoreArray = -1;
    this.log.debug("Setting the current generation count", yh);
    // now, make sure we know we have to process all of these.
    while finishedChromoProp.read() < (howManyValks) do chpl_task_yield();
    finishedChromoProp.write(0);
    this.log.log('Switching generations', yh);
    nG.addUnprocessed(ygg);
    begin inCurrentGeneration.add(nG.currentId.read()-1);
    // Clear out the current nodesToProcess domain, and swap it for the
    // ones we've set to process for the next generation.
    on Locales[0] {
      nodesToProcess.clear();
      for node in network.globalUnprocessed {
        nodesToProcess.add(node);
        processedArray[node].write(false);
      }
      forall node in network.globalUnprocessed {
        inCurrentGeneration.add(1);
      }
    }
    //nextGeneration.clear();
    // Set the count variable.
    valkyriesProcessed[v.currentTask+(here.id*maxValkyries)].write(v.nProcessed);
    // Compute some rough stats.  Buggy.
    //priorityValkyriesProcessed[i+(here.id*maxValkyries)].write(v.nPriorityNodesProcessed : real / prioritySize : real);
    //this.log.log('GEN:', gen : string, 'TOTAL MOVES:', v.nMoves : string, 'PROCESSED:', v.nProcessed : string, 'PRIORITY PROCESSED', v.nPriorityNodesProcessed : string, hstring=yh);
    var processedString: string;
    // this is really an IDEAL average.
    var avg = startingSeeds : real / maxValkyries : real ;
    var std: real;
    var eff: real;
    for y in 1..maxValkyries*Locales.size {
      var diff = valkyriesProcessed[y].read() - avg;
      std += diff**2;
      if valkyriesProcessed[y].read() != 0 {
        eff += priorityValkyriesProcessed[y].read() : real;
      }
    }
    std = abs(avg - sqrt(std/maxValkyries))/avg;
    eff /= maxValkyries;
    processedString = ''.join(' // BALANCE:  ', std : string, ' // ', ' EFFICIENCY:  ', eff : string, ' // ');
    //this.log.log('GEN', '%05i'.format(gen), 'processed in', '%05.2dr'.format(Time.getCurrentTime() - this.generationTime) : string, 'BEST: %05.2dr'.format(bestInGen), processedString : string, hstring=this.yh);
    this.yh.printedHeader = true;
    this.generationTime = Time.getCurrentTime() : real;
    valkyriesProcessed.write(0);
    priorityValkyriesProcessed.write(0);
    v.nPriorityNodesProcessed = 0;
    v.nProcessed = 0;
    // time to move the fuck on.
    this.log.log('MOVING ON in gen', gen : string, yh);
    this.generation = gen + 1;
    moveOn[gen] = true;
  }

  proc waitEndOfGeneration(ref v: shared valkyrie.valkyrieHandler, ref nG: shared network.networkGenerator, gen: int, ref ygg: shared network.networkMapper, yh: ygglog.yggHeader) {
    // Reset a lot of the variables for the Valkyrie while we're idle.
    // Then wait until all the other Valkyries have finished.
    // In addition, add to some global variables so that we can compute
    // some statistics of how well we're running.
    // Then wait on the sync variable.
    if v.currentTask == 1 {
      //nG.setCurrentGeneration();
    }
    v.moved = false;
    this.log.log('Waiting in gen', gen : string, yh);
    valkyriesProcessed[v.currentTask+(here.id*maxValkyries)].write(v.nProcessed);
    //priorityValkyriesProcessed[v.currentTask+(here.id*maxValkyries)].write(v.nPriorityNodesProcessed : real / prioritySize : real);
    this.log.log('GEN:', gen : string, 'TOTAL MOVES:', v.nMoves : string, 'PROCESSED:', v.nProcessed : string, 'PRIORITY PROCESSED', v.nPriorityNodesProcessed : string, hstring=yh);
    v.nProcessed = 0;
    v.nPriorityNodesProcessed = 0;

    // after this, we process the chromosomes and go.
    //if nG.generation.fetchAdd(1) == gen {
    //  nG.setCurrentGeneration();
    //} else {
    //  nG.generation.sub(1);
    //}

    readyForChromosomes[gen];
    this.log.log('Grabbing chromosomes to process', hstring=yh);
    // moveOn is an array of sync variables.  We're blocked from reading
    // until that's set to true.
    advanceChromosomes(nG, ygg, yh, gen+1);
    this.log.debug("Setting the current generation count", yh);
    finishedChromoProp.add(1);
    moveOn[gen];
    if v.currentTask == 1 {
      nG.addUnprocessed(ygg);
      begin inCurrentGeneration.add(nG.currentId.read()-1);
    }
    this.lock.rl();
    this.log.log('MOVING ON in gen', gen : string, yh);
    this.lock.url();
  }
}
