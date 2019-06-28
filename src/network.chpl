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

config const nodeBlockSize = 10000;

// here's some stuff to spread out the array.

record mapperByLocale {
  proc this(ind : string, targetLocs: [?D] locale) : D.idxType {
    // we just map according to the first 4 numbers of the ID.
    // the ID is just the locale, padded out to 4, followed by a UUID.
    //writeln('Just doing my thing and being a mapper: ', ind);
    //writeln(ind);
    if ind == 'root'{
      return 0;
    } else if ind != '' {
      return ind[1..4] : int;
    }
    // you should never be called, but hey.
    return 0;
  }
}

// okay, it should be spread out.  Just nodes everywhere.
var globalIDs: domain(string) dmapped Hashed(idxType=string, mapper = new mapperByLocale());
var globalNodes: [globalIDs] shared genes.GeneNode;
var globalUnprocessed: domain(string) dmapped Hashed(idxType=string, mapper = new mapperByLocale());
var globalIsProcessed: [globalIDs] atomic bool;

class networkGenerator {

  var NUUID = new owned uuid.UUID();
  var idSet: [1..0] string;
  var IDs: domain(string);
  var processed: [IDs] bool = false;
  var nodes: [IDs] shared genes.GeneNode;
  var initialized: [IDs] atomic bool;
  var currentId: atomic int = 1;
  var firstUnprocessed: atomic int = 1;
  var l = new shared spinlock.SpinLock();
  var isUpdating: atomic bool = false;
  var irng = new owned rng.UDevRandomHandler();
  var N: int = 0;
  var newNodeStartingPoint: int = 1;

  proc init() {
    this.complete();
    this.l.wl();
    this.initializeRoot();
    this.l.t = 'networkGenerator';
    this.l.log = new shared ygglog.YggdrasilLogging();
    this.generateEmptyNodes(nodeBlockSize);
    this.addToGlobal();
    this.l.uwl();
  }

  proc spawn() {
    this.l.wl();
    //writeln("Spawn begin. READ LOCK HANDLES: ", this.l.readHandles.read() : string);
    this.generateEmptyNodes(nodeBlockSize);
    this.addToGlobal();
    // right.  That's not going to work, because we haven't processed them.
    //writeln("Now, add them to the global unprocessed arrayß");
    //this.addUnprocessed();
    this.isUpdating.clear();
    //writeln("Spawn end. READ LOCK HANDLES: ", this.l.readHandles.read() : string);
    this.l.uwl();
  }

  proc initializeRoot() {
    // this is to init root.
    // We'll call this numerous times, but only one will win.
    // Not that it matters.
    globalLock.wl();
    if !globalIDs.contains('root') {
      var rootNode = new shared genes.GeneNode();
      rootNode.id = 'root';
      rootNode.revision = genes.SPAWNED;
      globalIDs.add('root');
      globalNodes['root'] = rootNode;
    }
    globalLock.uwl();
  }

  iter currentGeneration {
    var cId: int;
    cId = this.currentId.read();
    if cId > this.N {
      // do not go higher than the actual number of nodes we have.
      cId = this.N;
    }
    //writeln("Taking from our currentGeneration! What's our starting point? ", this.firstUnprocessed.read() : string);
    //var addToUnprocessed: int = this.firstUnprocessed.read();
    for i in this.firstUnprocessed.read()..cId {
      this.l.rl();
      var id = this.idSet[i];
      if !this.processed[id] {
        //this.l.url();
        //this.firstUnprocessed.write(addToUnprocessed);
        yield id;
      } else {
        //addToUnprocessed += 1;
      }
      this.l.url();
    }
    //this.firstUnprocessed.write(addToUnprocessed);
  }

  proc randomInGen {
    // return a random node ID in this generation.
    var RNG = new owned Random.PCGRandom.RandomStream(eltType = int);
    var cId: int;
    cId = this.currentId.read();
    if cId > this.N {
      // do not go higher than the actual number of nodes we have.
      cId = this.N;
    }
    var rint = RNG.getNext(min = this.firstUnprocessed.read(), max = cId);
    return this.idSet[rint];
  }

  iter all {
    var cId: int;
    cId = this.currentId.read();
    //writeln("What is our current ID?: ", this.currentId.read() : string);
    if cId > this.N {
      // do not go higher than the actual number of nodes we have.
      cId = this.N;
    }
    for i in 1..cId {
      //writeln("Current readHandles: ", this.l.readHandles.read() : string);
      this.l.rl();
      var id = this.idSet[i];
      //writeln(this.processed[id] : string);
      if !this.processed[id] {
        //this.l.url();
        yield id;
      }
      this.l.url();
    }
  }

  proc removeUnprocessed(id : string) {
    //writeln("readHandles on nG: ", this.l.readHandles.read() : string);
    this.l.wl();
    if this.IDs.contains(id) {
      this.processed[id] = true;
    }
    this.l.uwl();
    globalLock.wl();
    if globalIDs.contains(id) {
      globalUnprocessed.remove(id);
      //globalIsProcessed[id].write(true);
    }
    globalLock.uwl();
  }

  proc setCurrentGeneration() {
    this.firstUnprocessed.write(this.currentId.read());
  }

  proc generateID {
    // returns a UUID, prepended by the locale.
    return '%04i'.format(here.id) + '-GENE-' + NUUID.UUID4();
  }

  proc generateChromosomeID {
    // returns a UUID, prepended by the locale.
    return '%04i'.format(here.id) + '-GENE-' + NUUID.UUID4();
  }

  proc generateEmptyNodes(n: int) {
    // this will pre-generate a large set of UUIDs for us.
    // this is designed for the global arrays.
    for i in 1..n {
      // generate new nodes.  And UUIDs.
      var id = this.generateID;
      this.IDs.add(id);
      this.idSet.push_back(id);
      this.initialized[id].write(false);
    }
    this.N += n;
  }

  proc getNode() : string {
    // this will return an unused node.
    // block if we're updating.
    //while this.isUpdating.read() do chpl_task_yield();
    this.l.rl();
    var nId : int = 1;
    //writeln(nId : string);
    while nId < this.N {
      var nId = this.currentId.fetchAdd(1);
      if nId >= this.N {
        //this.currentId.sub(1);
        break;
      }
      var node = this.idSet[nId];
      if !this.initialized[node].testAndSet() {
        // we can use it!
        this.l.url();
        // before we return, we need to ensure we're not updating.
        // avoid a race condition.
        // I mean, maybe!  So stochastic!  Sometimes we get nodes and they don't exist.
        // in the global space!  So that's a thing.
        // BUT IT SHOULD.  BECAUSE IT DOES.  WHAT THE EFF YO.
        while this.isUpdating.read() do chpl_task_yield();
        return node;
      }
      //}
    }
    this.currentId.sub(1);
    //writeln("If you're seeing this, it's because we're out of nodes.  READ LOCK HANDLES: ", this.l.readHandles.read() : string);
    //this.l.url();
    // if we're here, we need more nodes!
    // make sure the call doesn't fail by returning this function again.
    this.l.url();
    if !this.isUpdating.testAndSet() {
      //writeln("If you're seeing this, it's because we successfully nabbed the lock.");
      // avoid a race condition where we clear the flag after nodes have been generated,
      // but tried to grab one before things were ready.  Not likely, but hey.
      //writeln("What's our current count up to? ", this.currentId.read() : string);
      //if this.currentId.read() >= this.N {
      //writeln("Running spawn function");
      //this.l.url();
      this.spawn();
      //}
    }
    // hold it, you whiny assholes.
    this.isUpdating.waitFor(false);
    var returnString = this.getNode();
    return returnString;
  }

  proc addToGlobal() {
    var removeSet: domain(string);
    globalLock.wl();
    for i in this.newNodeStartingPoint..this.N {
      var node = this.idSet[i];
      if !globalIDs.contains(node) {
        // we _might_ need to recreate the log.
        // and lock.
        // because this is _definitely_ a copy operation, but it's possible
        // that somehow the lock doesn't copy over properly.
        globalIDs.add(node);
        globalNodes[node] = new shared genes.GeneNode();
        globalNodes[node].id = node;
        globalNodes[node].l = new shared spinlock.SpinLock();
        globalNodes[node].l.t = ' '.join('GENE', node);
        globalNodes[node].l.log = globalNodes[node].log;
      } else {
        removeSet.add(node);
      }
    }
    globalLock.uwl();
    for node in removeSet {
      this.IDs.remove();
    }
    this.newNodeStartingPoint = this.N+1;
  }

  proc addUnprocessed() {
    globalLock.wl();
    for node in currentGeneration {
      if !globalUnprocessed.contains(node) {
        globalUnprocessed.add(node);
        globalIsProcessed[node].write(false);
      }
    }
    this.setCurrentGeneration();
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
  //var nodes: [ids] shared genes.GeneNode;
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

  proc add_node(id: string) {
    this.__addNode__(id, hstring=new ygglog.yggHeader());
  }

  proc add_node(id: string, hstring: ygglog.yggHeader) {
    this.__addNode__(id, hstring);
  }

  proc __addNode__(id: string, hstring: ygglog.yggHeader) : void {
    //writeln(nodes);
    // We are working with the actual node objects, here.
    // Add to our domain!
    // We need to block until such time as we're ready;
    var vstring: ygglog.yggHeader;
    vstring = hstring + '__addNode__';
    this.lock.wl(vstring);
    //this.log.debug('Adding node', id : string, 'to GeneNetwork ID:', this.id : string, hstring=vstring);
    if !this.ids.contains(id) {
      this.ids.add(id);
    }
    //globalLock.rl();
    //this.log.debug('Attempting to grab node ID:', id : string, 'on locale', this.locale : string, hstring=vstring);
    ref node = globalNodes[id];
    //this.log.debug('Node grabbed!  Node ID:', id : string, 'on locale', this.locale : string, hstring=vstring);
    //globalLock.url();
    this.nodeVersion[id] = node.returnRevision();
    //this.nodes[node.id] = node;
    for edge in node.returnEdgeIDs() {
      if !this.edges[id].contains(edge) {
        this.edges[id].add(edge);
      }
    }
    //this.newIDs.add(node.id);
    this.lock.uwl(vstring);
  }

  proc generateID {
    // returns a UUID, prepended by the locale.
    return '%04i'.format(here.id) + '-NETM-' + this.NUUID.UUID4();
  }

  proc init() {
    this.lock = new shared spinlock.SpinLock();
    this.lock.t = 'GeneNetwork';
    this.rootNode = new shared genes.GeneNode(id='root', ctype='root');
    this.complete();
    // create a new
    this.id = this.generateID;
  }

  proc initializeRoot() {
    this.rootNode.log = this.log;
    this.rootNode.l.log = this.log;
    //this.add_node(this.rootNode, new ygglog.yggHeader() + 'initializeNetwork');
  }

  proc returnNearestUnprocessed(id_A: string, id_B: domain(string), hstring: ygglog.yggHeader, ref processedArray) {
    var vstring = hstring + 'returnNearestUnprocessed';
    return this.__calculatePath__(id_A, id_B, hstring=vstring, processedArray=processedArray, checkArray=true);
  }

  proc calculatePath(id_A: string, id_B: string, hstring: ygglog.yggHeader) {
    // This allows us to search for a specific one with the same logic.
    var vstring = hstring + 'calculatePath';
    var b_dom: domain(string);
    var b: string;
    var removeFromSet: domain(string);
    var path: pathHistory;
    b_dom.add(id_B);
    var tmp: domain(string);
    var processedArray: [tmp] atomic bool;
    (b, path, removeFromSet) = this.__calculatePath__(id_A, b_dom, hstring=vstring, processedArray=processedArray, checkArray=false);
    return path;
  }

  proc __calculatePath__(id_A: string, in id_B: domain(string), hstring: ygglog.yggHeader, ref processedArray, checkArray: bool) throws {
    // This is an implementation of djikstra's algorithm.
    var nodes: domain(string);
    var visited: [nodes] bool = false;
    var dist: [nodes] real = Math.INFINITY;
    var paths: [nodes] pathHistory;
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
    var removeFromSet: domain(string);
    vstring = hstring + '__calculatePath__';
    //nodes.add[id_A];
    if !this.ids.contains(id_A) {
      this.add_node(id_A);
    }
    nodes.add(id_A);
    dist[id_A] = 0;
    paths[id_A].n.add(0);
    paths[id_A].node[0] = id_A;
    // catch an empty id_B!
    if id_B.isEmpty() {
      //this.log.debug('id_B is empty; is this okay?', hstring=vstring);
      return (id_A, paths[id_A], removeFromSet);
    }
    while !id_B.isEmpty() {
      i += 1;
      // Seems sometimes this locks, but doesn't unlock.
      // Is this from thread switching, I wonder?
      // AH!  I think it was from thread switching at the OS level, maybe.
      // I should apparently speak with Elliot about this, if I'm curious.
      // If we need to update, trigger it.
      this.lock.rl(vstring);
      //globalLock.rl();
      ref actualNode = globalNodes[currentNode];
      //globalLock.url();
      if !this.ids.contains(currentNode) {
        nodes.add(currentNode);
        this.lock.url(vstring);
        this.add_node(currentNode);
        this.lock.rl(vstring);
      } else if this.nodeVersion[currentNode] < genes.FINALIZED {
        if actualNode.returnRevision() > this.nodeVersion[currentNode] {
          this.lock.url(vstring);
          this.add_node(currentNode);
          this.lock.rl(vstring);
        }
      }
      // we now assume this is an incomplete network.
      for edge in this.edges[currentNode] do {
        // why are we a big, errortastical bitch?
        //if !this.ids.contains(edge) {
        if !nodes.contains(edge) {
          nodes.add(edge);
          visited[edge] = false;
          dist[edge] = Math.INFINITY;
        }
        if !visited[edge] {
          var d = min(dist[edge], dist[currentNode]+1);
          unvisited.add(edge);
          //unvisited_d.add(d);
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
      this.lock.url(vstring);
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
          //globalLock.rl();
          if !processedArray[currentNode].testAndSet() {
            //break;
            //globalLock.url();
            return (currentNode, paths[currentNode], removeFromSet);
          } else {
            // This means we've actually already processed it, so
            // we'll pretend it's not a part of id_B by removing it.
            // This will help us in the event that we've been beaten to this node.
            //globalLock.url();
            removeFromSet.add(currentNode);
            id_B.remove(currentNode);
            if id_B.isEmpty() {
              // If we've removed everything, then we can't process anything.
              // Returning an empty string dodges the processing logic.
              return ('', paths[id_A], removeFromSet);
            }
          }
        } else {
          return (currentNode, paths[currentNode], removeFromSet);
        }
      }
      if id_B.isEmpty() {
        // If we've removed everything, then we can't process anything.
        // Returning an empty string dodges the processing logic.
        return ('', paths[id_A], removeFromSet);
      }
      if unvisited.isEmpty() {
        // no paths
        return ('', paths[id_A], removeFromSet);
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
        //unvisited_d.remove(currMinDist);
        unvisited.remove(currMinNode);
      }
    }
    //this.log.debug('id_B:', id_B : string, 'currentNode:', currentNode, 'id_A:', id_A, hstring=vstring);
    //writeln("What are our paths?: ", paths : string);
    if !id_B.isEmpty() {
      return (currentNode, paths[currentNode], removeFromSet);
    } else {
      return ('', paths[id_A], removeFromSet);
    }

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
          d += (this.calculateHistory(edgeId[1], vstring) * edgeId[2]);
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

}
