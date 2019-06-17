// Audrey Pratt

use genes;
use rng;
use uuid;
use Random;
use Math;
use Sort;
use propagator;
use ygglog;
use spinlock;
use SysError;
use chromosomes;
use HashedDist;

class NodeNotInEdgesError : Error {
  proc init() { }
}

var globalLock = new shared spinlock.SpinLock();
globalLock.t = 'Ragnarok';
globalLock.log = new shared ygglog.YggdrasilLogging();

config const nodeBlockSize = 100000;

// here's some stuff to spread out the array.

record mapperByLocale {
  proc this(ind : string, targetLocs: [?D] locale) : D.idxType {
    // we just map according to the first 4 numbers of the ID.
    // the ID is just the locale, padded out to 4, followed by a UUID.
    return ind[1..4] : int;
  }
}

// okay, it should be spread out.  Just nodes everywhere.
var globalIDs: domain(string) dmapped Hashed(idxType=string, mapper = new mapperByLocale());
var globalNodes: [globalIDs] shared genes.GeneNode;
var globalUnprocessed: domain(string) dmapped Hashed(idxType=string, mapper = new mapperByLocale());
var globalIsProcessed: [globalUnprocessed] atomic bool;

class networkGenerator {

  var NUUID = new owned uuid.UUID();
  var idSet: [1..0] string;
  var IDs: domain(string);
  var processed: [IDs] bool = false;
  var nodes: [IDs] shared genes.GeneNode;
  var currentId: atomic int = 1;
  var firstUnprocessed: atomic int = 1;
  var l = new shared spinlock.SpinLock();
  var isUpdating: atomic bool = false;
  var irng = new owned rng.UDevRandomHandler();

  proc init() {
    this.complete();
    this.l.t = 'networkGenerator';
    this.l.log = new shared ygglog.YggdrasilLogging();
    this.generateEmptyNodes(nodeBlockSize);
    this.addToGlobal();
  }

  proc spawn() {
    this.l.wl();
    this.IDs.clear();
    // probably won't work, but hey.
    this.idSet.clear();
    this.generateEmptyNodes(nodeBlockSize);
    this.addToGlobal();
    this.currentId.write(1);
    this.isUpdating.write(false);
    this.l.uwl();
  }

  iter currentGeneration {
    for i in this.firstUnprocessed.read()..this.currentId.read() {
      this.l.rl();
      var id = this.idSet[i];
      if this.processed[id] {
        this.l.url();
        yield id;
      }
      this.l.url();
    }
  }

  proc removeUnprocessed(id : string) {
    this.l.wl();
    this.processed[id] = true;
    this.l.uwl();
  }

  proc setCurrentGeneration() {
    this.firstUnprocessed.write(this.currentId.read());
  }

  proc generateID {
    // returns a UUID, prepended by the locale.
    return '%04i'.format(here.id) + '-' + NUUID.UUID4();
  }

  proc generateEmptyNodes(n: int) {
    // this will pre-generate a large set of UUIDs for us.
    // this is designed for the global arrays.
    for i in 1..n {
      // generate new nodes.  And UUIDs.
      var node = new shared genes.GeneNode();
      node.id = this.generateID;
      IDs.add(node.id);
      nodes[node.id] = node;
      this.idSet.push_front(node.id);
    }
  }

  proc getNode() : string {
    // this will return an unused node.
    this.l.rl();
    var nId : int = 1;
    while nId < nodeBlockSize {
      var nId = this.currentId.fetchAdd(1);
      // check and see whether it exists.
      if this.idSet.domain.contains(nId) {
        var node = this.idSet[nId];
        if !globalNodes[node].initialized.testAndSet() {
          // we can use it!
          this.l.url();
          return node;
        }
      }
    }
    this.l.url();
    // if we're here, we need more nodes!
    // make sure the call doesn't fail by returning this function again.
    if !this.isUpdating.testAndSet() {
      // avoid a race condition where we clear the flag after nodes have been generated,
      // but tried to grab one before things were ready.  Not likely, but hey.
      if this.currentId.read() >= nodeBlockSize {
        this.spawn();
      }
    }
    return this.getNode();
  }

  proc addToGlobal() {
    var removeSet: domain(string);
    globalLock.wl();
    for node in this.IDs {
      if !globalIDs.contains(node) {
        globalIDs.add(node);
        globalNodes[node] = this.nodes[node];
      } else {
        removeSet.add(node);
      }
    }
    globalLock.uwl();
    for node in removeSet {
      this.IDs.remove();
    }
  }

  proc addUnprocessed() {
    globalLock.wl();
    for node in currentGeneration {
      globalUnprocessed.add(node);
      globalIsProcessed[node].write(false);
    }
    globalLock.uwl();
  }

  proc newSeed() {
    // Generates a new seed for use with deltas, etc.
    // we're returning a long.
    return this.irng.getrandbits(64);
  }
}


record pathHistory {
  // This is basically an ordered dictionary.
  // Chapel doesn't yet have something so this like,
  // so it's up to me (up to ME!) to implement it here.
  var n: domain(int);
  var node: [n] string;

  inline iter these() {
    // This works.  Grand!
    for i in 0..this.n.size-1 do {
      yield (i, this.node[i]);
    }
  }

  proc key(i : int) {
    return this.node[i];
  }

  proc compare(a, b) {
    return abs(a) - abs(b);
  }

  // There HAS to be a better way to do this.
  proc remove(a) {
    // search through and find the index to remove.
    var n_remove = 0;
    var new_n: domain(int);
    var new_node: [new_n] string;
    for i in 0..this.n.size-1 do {
      // Oddly enough I do think this is working, although it's awful.
      if this.node[i] == a {
        n_remove = i+1;
      } else {
        new_n.add(i-n_remove);
        new_node[i-n_remove] = this.node[i];
      }
      // Adjust the ordering, at that.
    }
    this.n = new_n;
    this.node = new_node;
  }

  proc distance() {
    return this.n.size;
  }

}

record pathSet {
  // This is so that we can re-orient sets of paths relative to a new point.
  // The idea here is to allow us to use a cache to improve search times.
  // As they can get quite long, really.
  var entryPoints: domain(string);
  var paths: [entryPoints] pathHistory;
}

class GeneNetwork {
  // Hash table for our nodes and edges.  Basically, it's a dictionary of lists;
  // kind of easy to think of it that way, for those of us coming from Python.
  var id: string;
  var ids: domain(string);
  var edges: [ids] domain(string);
  var nodes: [ids] shared genes.GeneNode;
  // this tells us who was responsible for doing the last update.
  // or actually, just whether we have the most up to date version of the node.
  var nodeVersion: [ids] int;
  var lock: shared spinlock.SpinLock;
  var irng = new owned rng.UDevRandomHandler();
  var NUUID = new owned uuid.UUID();

  var rootNode: shared genes.GeneNode;
  var log: shared ygglog.YggdrasilLogging;

  var testNodeId: string;

  var isCopyComplete: bool = false;

  proc add_node(ref node: shared genes.GeneNode) {
    this.__addNode__(node, hstring=new ygglog.yggHeader());
  }

  proc add_node(ref node: shared genes.GeneNode, hstring: ygglog.yggHeader) {
    this.__addNode__(node, hstring);
  }

  proc __addNode__(ref node: shared genes.GeneNode, hstring: ygglog.yggHeader) : void {
    //writeln(nodes);
    // We are working with the actual node objects, here.
    // Add to our domain!
    // We need to block until such time as we're ready;
    var vstring: ygglog.yggHeader;
    vstring = hstring + '__addNode__';
    this.lock.wl(vstring);
    this.log.debug('Adding node', node.id : string, 'to GeneNetwork', hstring=vstring);
    this.ids.add(node.id);
    this.nodeVersion[node.id] = node.revision;
    //this.nodes[node.id] = node;
    for edge in node.nodes {
      this.edges[node.id].add(edge);
    }
    //this.newIDs.add(node.id);
    this.lock.uwl(vstring);
  }

  proc init() {
    this.lock = new shared spinlock.SpinLock();
    this.lock.t = 'GeneNetwork';
    this.rootNode = new shared genes.GeneNode(id='root', ctype='root');
    this.complete();
    // create a new
    this.id = this.NUUID.UUID4();
  }

  proc initializeRoot() {
    this.rootNode.log = this.log;
    this.rootNode.l.log = this.log;
    this.add_node(this.rootNode, new ygglog.yggHeader() + 'initializeNetwork');
  }

  proc initializeNetwork(n_seeds=10: int, gen_seeds=true: bool) {
    var seed: int;
    var id: string;
    var delta: genes.deltaRecord;
    // We send each gene a logger; it's useful for debugging, but likely not
    // production.
    this.rootNode.log = this.log;
    this.rootNode.l.log = this.log;
    for n in 1..n_seeds {
      if propagator.unitTestMode {
        seed = n*1000;
        id = (n*1000) : string;
      } else {
        seed = this.newSeed();
        // With a blank ID, the gene will assign itself a UUID.
        id = '';
      }
      var node = new shared genes.GeneNode(id=id, ctype='seed', parentSeedNode='', parent='root');
      node.log = this.log;
      node.l.log = this.log;
      //writeln(this.log : string, node.log : string, node.l.log : string);
      if propagator.unitTestMode {
        node.debugOrderOfCreation = n*1000;
      } else {
        node.debugOrderOfCreation = n;
      }
      delta = new genes.deltaRecord();
      delta.seeds.add(seed);
      delta.delta[seed] = 1;
      this.rootNode.join(node, delta, new ygglog.yggHeader() + 'initializeNetwork');
      this.add_node(node, new ygglog.yggHeader() + 'initializeNetwork');
    }
    if propagator.unitTestMode {
      seed = ((n_seeds+1)*1000)+10000;
      this.testNodeId = seed : string;
      var node = new shared genes.GeneNode(id=seed : string, ctype='seed', parentSeedNode='', parent='root');
      node.log = this.log;
      node.l.log = this.log;
      node.debugOrderOfCreation = seed;
      delta = new genes.deltaRecord();
      delta.seeds.add(seed);
      delta.delta[seed] = 1;
      this.log.debug('Adding testMergeNode, delta:', delta : string, hstring=new ygglog.yggHeader() + 'initializeNetwork');
      this.rootNode.join(node, delta, new ygglog.yggHeader() + 'initializeNetwork');
      this.add_node(node, new ygglog.yggHeader() + 'initializeNetwork');
    }
    this.add_node(this.rootNode, new ygglog.yggHeader() + 'initializeNetwork');
  }

  proc initializeSeedGenes(seeds: domain(int)) {
    // We're going to create a whole host of seed type nodes.
    // These have a special deltaRecord; I'm going to encode an INFINITY
    // as 'blow away the matrix', then ... or should I?
    for seed in seeds {
      var node = new shared genes.GeneNode(seed);
      this.add_node(node, new ygglog.yggHeader() + 'initializeSeedGenes');
    }
  }

  proc returnNearestUnprocessed(id_A: string, id_B: domain(string), hstring: ygglog.yggHeader, processedArray) {
    var vstring = hstring + 'returnNearestUnprocessed';
    return this.__calculatePath__(id_A, id_B, hstring=vstring, processedArray=processedArray, checkArray=true);
  }

  proc calculatePath(id_A: string, id_B: string, hstring: ygglog.yggHeader) {
    // This allows us to search for a specific one with the same logic.
    var vstring = hstring + 'calculatePath';
    var b_dom: domain(string);
    var b: string;
    var path: pathHistory;
    b_dom.add(id_B);
    var tmp: domain(string);
    var processedArray: [tmp] atomic bool;
    (b, path) = this.__calculatePath__(id_A, b_dom, hstring=vstring, processedArray=processedArray, checkArray=false);
    return path;
  }

  proc __calculatePath__(id_A: string, in id_B: domain(string), hstring: ygglog.yggHeader, processedArray, checkArray: bool) throws {
    // This is an implementation of djikstra's algorithm.
    //var nodes: domain(string);
    var visited: [this.ids] bool;
    var dist: [this.ids] real;
    var paths: [this.ids] pathHistory;
    var currentNode = id_A;
    var unvisited: domain(string);
    var unvisited_d: domain(real);
    var currMinDist = Math.INFINITY;
    var currMinNode = id_A;
    var currMinNodeIndex = 0;
    var i: int;
    var completed: [id_B] bool = false;
    var vstring: ygglog.yggHeader;
    var thisIsACopy: bool = true;
    vstring = hstring + '__calculatePath__';
    //nodes.add[id_A];
    dist[id_A] = 0;
    paths[id_A].n.add(0);
    paths[id_A].node[0] = id_A;
    // catch an empty id_B!
    if id_B.isEmpty() {
      //this.log.debug('id_B is empty; is this okay?', hstring=vstring);
      return (id_A, paths[id_A]);
    }
    while true {
      i += 1;
      // Seems sometimes this locks, but doesn't unlock.
      // Is this from thread switching, I wonder?
      // AH!  I think it was from thread switching at the OS level, maybe.
      // I should apparently speak with Elliot about this, if I'm curious.
      if !thisIsACopy {
        this.lock.rl(vstring);
      }
      // If we need to update, trigger it.
      if this.nodeVersion[currentNode] < genes.FINALIZED {
        if globalNodes[currentNode].revision > this.nodeVersion[currentNode] {
          this.add_node(globalNodes[currentNode]);
        }
      }
      // we now assume this is an incomplete network.
      for edge in this.edges[currentNode] do {
        if !this.ids.contains(edge) {
          // add the edge to our network if we haven't done so.
          this.add_node(globalNodes[edge]);
          visited[edge] = false;
          dist[edge] = Math.INFINITY;
        }
        if !visited[edge] {
          var d = min(dist[edge], dist[currentNode]+1);
          unvisited.add(edge);
          unvisited_d.add(d);
          dist[edge] = d;

          if d == dist[currentNode]+1 {
            paths[edge].n.clear();
            var z: int;
            for (j, e) in paths[currentNode] {
              paths[edge].n.add(j : int);
              paths[edge].node[j : int] = e;
              z += 1;
            }
            // We're doing this as a tuple to help sorting later.
            // That'll also help us calculate how many hops we have to make,
            // which will be convenient when we're trying to determine who
            // should do what.
            paths[edge].n.add(d: int);
            paths[edge].node[d: int] = edge;
          }
        }
      }
      if !thisIsACopy {
        this.lock.url(vstring);
      }
      visited[currentNode] = true;
      // Doing it like this means we never have a race condition.
      // Should help with load balancing and efficiency.
      // Oh man, and does it ever.  Basically, we don't leave this routine
      // untl we have one we KNOW can process, or there's nothing left to
      // process.
      if id_B.contains(currentNode) {
        if checkArray {
          // We should actually do the testAndSet here, although I sort of
          // dislike having the network access the array.  If false, we can use it!
          if !processedArray[currentNode].testAndSet() {
            break;
          } else {
            // This means we've actually already processed it, so
            // we'll pretend it's not a part of id_B by removing it.
            // This will help us in the event that we've been beaten to this node.
            id_B.remove(currentNode);
            if id_B.isEmpty() {
              // If we've removed everything, then we can't process anything.
              // Returning an empty string dodges the processing logic.
              return ('', paths[id_A]);
            }
          }
        } else {
          break;
        }
      }
      if unvisited.isEmpty() {
      } else {
        // get the current minimum from here.
        i = 0;
        currMinDist = Math.INFINITY;
        for node in unvisited {
          i += 1;
          if currMinDist > dist[node] {
            currMinDist = dist[node];
            currMinNode = node;
            currMinNodeIndex = i;
          }
        }
        currentNode = currMinNode;
        unvisited_d.remove(currMinDist);
        unvisited.remove(currMinNode);
      }
    }
    //this.log.debug('id_B:', id_B : string, 'currentNode:', currentNode, 'id_A:', id_A, hstring=vstring);
    return (currentNode, paths[currentNode]);

  }

  proc deltaFromPath(in path: network.pathHistory, id: string, hstring: ygglog.yggHeader) : genes.deltaRecord throws {
    // This is an attempt to automatically create a deltaRecord from
    // a path.  We pass in a copy as we want to remove the id from it.
    // Not sure how that'll affect performance, but worth keeping an eye on.
    var vstring = hstring + 'deltaFromPath';
    var d = new genes.deltaRecord();
    var edge: genes.GeneEdge;
    var currentNode = id;
    var pl: int;
    path.remove(id);
    for (i, pt) in path {
      //this.log.debug(i: string, pt: string, hstring=vstring);
      this.lock.rl(vstring);
      // LOCK THE NODES.
      globalNodes[currentNode].l.rl(vstring);
      if globalNodes[currentNode].nodes.contains(pt) {
        edge = globalNodes[currentNode].edges[pt : string];
        //this.log.debug('EDGE:', edge : string, hstring=vstring);
      } else {
        this.log.critical('EDGE', pt : string, 'NOT IN EDGE LIST FOR', currentNode, hstring=vstring);
        this.log.critical('EDGELIST for 1st:', globalNodes[pt : string].nodes : string, hstring=vstring);
        this.log.critical('EDGELIST for 2nd:', globalNodes[currentNode].nodes : string, hstring=vstring);
        this.log.critical('PATH WAS:', path : string, hstring=vstring);
        globalNodes[currentNode].l.url(vstring);
        this.lock.url(vstring);
        //throw new owned NodeNotInEdgesError();
      }
      globalNodes[currentNode].l.url(vstring);
      this.lock.url(vstring);
      if edge.edgeType == genes.DELTA {
        for (s, c) in edge.delta {
          d += (s, c);
        }
      } else if edge.edgeType == genes.PATH {
        for edgeId in edge.path {
          // trace back to root, convert to a delta, add it in.
          d += this.calculateHistory(edgeId, vstring);
        }
      }
      currentNode = pt;
      pl += 1;
    }
    for (s, c) in d {
      if c == 0 {
        // Get rid of the seed is the coefficient is 0.  We don't need that stuff.
        // YOU HEAR THAT?  NOT WANTED HERE.
        d.remove(s);
      }
    }
    return d;
  }

  proc calculateHistory(id: string, hstring: ygglog.yggHeader) : genes.deltaRecord {
    // Since all nodes carry their ancestor,
    // simply calculate the path back to the seed node.
    var vstring: ygglog.yggHeader;
    vstring = hstring + 'calculateHistory';
    // Actually, can we just do back to root?
    var path = this.calculatePath(id, 'root', hstring=vstring);
    this.log.debug('Path calculated:', path : string, hstring=vstring);
    var delta = this.deltaFromPath(path, id, hstring=vstring);

    return delta;
  }

  proc clone(ref networkCopy: shared GeneNetwork) {
    //var networkCopy = new shared GeneNetwork();
    networkCopy.log = this.log;
    //networkCopy.initializeRoot();
    forall i in this.ids {
      this.lock.wl();
      networkCopy.ids.add(i);
      this.lock.uwl();
      for edge in this.edges[i] {
        networkCopy.edges[i].add(edge);
      }
    }
    networkCopy.isCopyComplete = true;
    return networkCopy;
  }

  proc update(ref otherNetwork: shared GeneNetwork) {
    for node in this.newIDs {
      for edge in this.nodes[node].nodes {
        // these are all the edges it's connected to.
        otherNetwork.edges[node].add(edge);
        otherNetwork.edges[edge].add(node);
        otherNetwork.nodes[edge].nodes.add(node);
        otherNetwork.nodes[edge].edges[node] = this.nodes[node].edges[edge].reverse();
      }
    }
    this.newIDs.clear();
  }

}
