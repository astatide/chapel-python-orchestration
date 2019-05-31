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

class NodeNotInEdgesError : Error {
  proc init() { }
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
  var ids: domain(string);
  var edges: [ids] domain(string);
  // We're going to have to be careful about object ownership.  Might not matter
  // too much at the moment, but.

  // Yep, we don't have to be careful because this is Chapel and holy fuck.
  // aaahahahaha, suck it languages built not for HPC!
  var nodes: [ids] shared genes.GeneNode;
  var lock: shared spinlock.SpinLock;

  var irng = new owned rng.UDevRandomHandler();

  var rootNode: shared genes.GeneNode;
  var log: shared ygglog.YggdrasilLogging;

  var testNodeId: string;

  proc add_node(ref node: shared genes.GeneNode) {
    this.__addNode__(node, hstring='');
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
    this.nodes[node.id] = node;
    for edge in node.nodes {
      this.edges[node.id].add(edge);
    }
    this.lock.uwl(vstring);
  }

  proc newSeed() {
    // Generates a new seed for use with deltas, etc.
    // we're returning a long.
    return this.irng.getrandbits(64);
  }

  proc init() {
    this.lock = new shared spinlock.SpinLock();
    this.lock.t = 'GeneNetwork';
    this.rootNode = new shared genes.GeneNode(id='root', ctype='root');
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

  proc newSeedGene(deme: int) {
    var seed = this.newSeed();
    var node = new shared genes.GeneNode(id='', ctype='seed', parentSeedNode='', parent='root');
    var delta = new genes.deltaRecord();
    delta += (seed, 1.0);
    node.demeDomain.add(deme);
    node.log = this.log;
    node.l.log = this.log;
    this.rootNode.join(node, delta, new ygglog.yggHeader() + 'newSeedGene');
    this.add_node(node, new ygglog.yggHeader() + 'newSeedGene');
    return node.id;
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
    var nodes: domain(string);
    var visited: [nodes] bool;
    var dist: [nodes] real;
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
    vstring = hstring + '__calculatePath__';
    nodes.add[id_A];
    dist[id_A] = 0;
    paths[id_A].n.add(0);
    paths[id_A].node[0] = id_A;
    // catch an empty id_B!
    if id_B.isEmpty() {
      this.log.debug('id_B is empty; is this okay?', hstring=vstring);
      return (id_A, paths[id_A]);
    }
    while true {
      i += 1;
      // Seems sometimes this locks, but doesn't unlock.
      // Is this from thread switching, I wonder?
      // AH!  I think it was from thread switching at the OS level, maybe.
      // I should apparently speak with Elliot about this, if I'm curious.
      this.lock.rl(vstring);
      this.log.debug('Attempting to pass through node', currentNode, 'does it exist?', this.ids.contains(currentNode) : string, vstring);
      if this.ids.contains(currentNode) {
        //assert(this.ids.contains(currentNode));
      } else {
        this.log.critical('NODE', currentNode : string, 'NOT IN LIST.  WHAT.  Existing nodes:', this.ids : string, hstring=vstring);
      }
      for edge in this.edges[currentNode] do {
        if !nodes.contains(edge) {
            nodes.add(edge);
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
          // dislike having the network access the array.
          //if processedArray[currentNode].read() {
          if !processedArray[currentNode].testAndSet() {
            // If it's false, we can use it!
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

  proc move(ref v: propagator.valkyrie, id: string, createEdgeOnMove: bool, edgeDistance: int) throws {
    // This is the pathless one.  We just need the path, then we're good.
    var path = this.calculatePath(v.currentNode, id, hstring=v.header);
    return this.__move__(v, id, path, createEdgeOnMove, edgeDistance);
  }

  proc move(ref v: propagator.valkyrie, id: string, path: pathHistory, createEdgeOnMove: bool, edgeDistance: int) throws {
    return this.__move__(v, id, path, createEdgeOnMove, edgeDistance);
  }

  proc __move__(ref v: propagator.valkyrie, id: string, path: pathHistory, createEdgeOnMove: bool, edgeDistance: int) throws {
    // This is a overloaded move function if we already have a path.
    // The other move function is for if we DON'T have a path.
    var vstring = v.header + '__move__';
    this.log.debug('Attempting to move from', path.key(0), 'to', id : string, hstring=vstring);
    this.log.debug('PATH', path : string, hstring=vstring);
    // Now we just process the path into a delta, and confirm that it is valid.
    // this is node we're moving FROM, not to, by the way.
    var d = this.deltaFromPath(path, path.key(0), hstring=vstring);
    var pl = path.distance();
    v.nMoves += 1;
    if createEdgeOnMove {
      if pl > edgeDistance {
        // If our edge distance is particularly long, create a shortcut.
        // This can greatly improve
        this.lock.wl(vstring);
        this.nodes[v.currentNode].join(this.nodes[id], d, vstring);
        this.edges[id].add(v.currentNode);
        this.edges[v.currentNode].add(id);
        this.lock.uwl(vstring);
      }
    }
    if propagator.unitTestMode {
      this.log.debug('Delta to move to is:', d : string, hstring=vstring);
    }
    // this is actually the command that moves us.
    // rather than actually handling the move, let's just send the delta back.
    /*
    var success = v.move(d, id);
    // for now, hardcode the errors.
    if success == 0 {
      this.log.debug('move successful', hstring=vstring);
    } else if success == 1 {
      this.log.critical('CRITICAL FAILURE: Valkyrie did not move correctly!', hstring=vstring);
      this.log.critical('Matrix should be:', id : string, 'but is:', v.matrixValues : string, hstring=vstring);
    }
    */
    return d;
  }

  proc deltaFromPath(in path: network.pathHistory, id: string, hstring: ygglog.yggHeader) throws {
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
      this.log.debug(i: string, pt: string, hstring=vstring);
      this.lock.rl(vstring);
      // LOCK THE NODES.
      this.nodes[currentNode].l.rl(vstring);
      if this.nodes[currentNode].nodes.contains(pt) {
        edge = this.nodes[currentNode].edges[pt : string];
        this.log.debug('EDGE:', edge : string, hstring=vstring);
      } else {
        this.log.critical('EDGE', pt : string, 'NOT IN EDGE LIST FOR', currentNode, hstring=vstring);
        this.log.critical('EDGELIST for 1st:', this.nodes[pt : string].nodes : string, hstring=vstring);
        this.log.critical('EDGELIST for 2nd:', this.nodes[currentNode].nodes : string, hstring=vstring);
        this.log.critical('PATH WAS:', path : string, hstring=vstring);
        this.nodes[currentNode].l.url(vstring);
        this.lock.url(vstring);
        //throw new owned NodeNotInEdgesError();
      }
      this.nodes[currentNode].l.url(vstring);
      this.lock.url(vstring);
      for (s, c) in edge.delta {
        d += (s, c);
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

  proc calculateHistory(id: string, hstring: ygglog.yggHeader) {
    // Since all nodes carry their ancestor,
    // simply calculate the path back to the seed node.
    var vstring: ygglog.yggHeader;
    vstring = hstring + 'calculateHistory';
    //var path = this.calculatePath(id, this.nodes[id].parentSeedNode, hstring=vstring);
    //var delta = this.deltaFromPath(path, this.nodes[id].parentSeedNode, hstring=vstring);
    // Actually, can we just do back to root?
    var path = this.calculatePath(id, 'root', hstring=vstring);
    this.log.debug('Path calculated:', path : string, hstring=vstring);
    var delta = this.deltaFromPath(path, id, hstring=vstring);

    return delta;
  }

  // Here's a function for merging two nodes.

  proc mergeNodes(id_A: string, id_B: string) {
    return this.__mergeNodes__(id_A, id_B, hstring='');
  }

  proc mergeNodes(id_A: string, id_B: string, hstring: ygglog.yggHeader) {
    return this.__mergeNodes__(id_A, id_B, hstring=hstring);
  }

  proc __mergeNodes__(id_A: string, id_B: string, hstring: ygglog.yggHeader) {
    var vstring: ygglog.yggHeader;
    vstring = hstring + '__mergeNodes__';
    var deltaA = this.calculateHistory(id_A, vstring);
    var deltaB = this.calculateHistory(id_B, vstring);
    this.log.debug('deltaA:', deltaA : string, 'deltaB:', deltaB : string, hstring=vstring);
    var nId: string;
    if propagator.unitTestMode {
      // Remember, this is a MERGE function.  Dumbass.
      nId = ((id_A : real + id_B : real)/2) : string;
    }
    var node = new shared genes.GeneNode(id=nId, ctype='merge', parentSeedNode=this.nodes[id_A].parentSeedNode, parent=id_A);
    // Remember that we're sending in logging capabilities for debug purposes.
    node.log = this.log;
    node.l.log = this.log;
    // Why is it in the reverse, you ask?  Because the calculateHistory method
    // returns the information necessary to go BACK to the seed node from the id given.
    // So this delta allows us to go from node A to the new node.
    var delta = ((deltaA*-1) + deltaB)/2;
    this.log.debug('Delta to move from', id_A : string, 'to', node.id : string, '-', (delta*-1) : string, hstring=vstring);
    node.join(this.nodes[id_A], delta, vstring);
    // Now, reverse the delta to join B.  I love the records in Chapel.
    node.join(this.nodes[id_B], delta*-1, vstring);
    this.add_node(node, vstring);
    // Now, don't forget to connect it to the existing nodes.
    this.lock.wl(vstring);
    this.edges[id_A].add(node.id);
    this.edges[id_B].add(node.id);
    this.lock.uwl(vstring);
    // Return the id, as that's all we need.
    return node.id;
  }

  proc mergeNodeList(ref idList : [] string, deme: int, hstring: ygglog.yggHeader) {
    return this.__mergeNodeList__(idList, deme, hstring=ygglog.yggHeader);
  }

  proc mergeNodeList(ref idList : [] string, deme: int) {
    var yh = new ygglog.yggHeader();
    return this.__mergeNodeList__(idList, deme, hstring=yh);
  }

  proc __mergeNodeList__(ref idList : [] string, deme: int, hstring: ygglog.yggHeader) {
    // we're getting a list of nodes, so we need to calculate
    var vstring: ygglog.yggHeader;
    vstring = hstring + '__mergeNodeList__';
    var deltaDomain: domain(string);
    var deltaList: [deltaDomain] genes.deltaRecord;
    for id in idList {
      deltaDomain.add(id);
      deltaList[id] = this.calculateHistory(id, vstring);
    }

    //this.log.debug('deltaA:', deltaA : string, 'deltaB:', deltaB : string, hstring=vstring);
    // spawn the node; we're making arbitrary decisions.
    var node = new shared genes.GeneNode(id='', ctype='merge', parentSeedNode=this.nodes[idList[1]].parentSeedNode, parent=idList[1]);
    // Remember that we're sending in logging capabilities for debug purposes.
    node.log = this.log;
    node.l.log = this.log;
    node.demeDomain.add(deme);
    // Why is it in the reverse, you ask?  Because the calculateHistory method
    // returns the information necessary to go BACK to the seed node from the id given.
    // So this delta allows us to go from node A to the new node.
    for i in idList {
      var delta: genes.deltaRecord;
      for j in idList {
        if i != j {
          delta += deltaList[j];
        } else {
          delta += deltaList[i]*-1;
        }
      }
      delta /= idList.size;
      node.join(this.nodes[i], delta, vstring);
      // Now, don't forget to connect it to the existing nodes.
      this.lock.wl(vstring);
      this.edges[i].add(node.id);
      this.lock.uwl(vstring);
    }
    // this function locks, so.
    this.add_node(node, vstring);
    // Return the id, as that's all we need.
    return node.id;
  }

  proc nextNode(id: string, deme: int) {
    return this.__nextNode__(id, deme, hstring='');
  }
  proc nextNode(id: string) {
    return this.__nextNode__(id, 0, hstring='');
  }

  proc nextNode(id: string, deme: int, hstring: ygglog.yggHeader) {
    return this.__nextNode__(id, deme, hstring);
  }
  proc nextNode(id: string, hstring: ygglog.yggHeader) {
    return this.__nextNode__(id, 0, hstring);
  }

  proc __nextNode__(id: string, deme: int, hstring: ygglog.yggHeader) throws {
    var vstring: ygglog.yggHeader;
    vstring = hstring + '__nextNode__';
    this.log.debug('Adding a seed on to ID', id : string, hstring);
    // MIGHT NOT NEED TO BE A THING
    var seed: int;
    var nId: string;
    //var node = new shared genes.GeneNode(id='');
    if propagator.unitTestMode {
      //nId = (this.nodes[id].debugOrderOfCreation+1) : string;
      nId = (id : real + 1) : string;
      seed = 1;
    } else {
      nId = '';
      seed = this.newSeed();
    }
    // DEBUG ME
    var node = this.nodes[id].new_node(seed=seed, coefficient=1, id=nId, hstring=vstring);
    if propagator.unitTestMode {
      node.debugOrderOfCreation = this.nodes[id].debugOrderOfCreation+1;
    }
    // Again, send the logger to both the lock and the node.
    node.log = this.log;
    node.l.log = this.log;
    node.demeDomain.add(deme);
    // Add to current node!  I can't believe you forgot this.
    this.lock.wl(hstring);
    this.edges[id].add(node.id);
    this.lock.uwl(hstring);
    if this.ids.contains(node.id) {
      writeln('WHAT THE FUCK');
    }
    this.add_node(node, vstring);
    // Make sure it happened properly.
    assert(node.nodeInEdges(id, vstring));
    assert(this.nodes[id].nodeInEdges(node.id, vstring));
    if !this.nodes[node.id].nodeInEdges(id, vstring) {
      writeln(id, ' ', node.id);
      writeln(this.nodes[node.id]);
      writeln(node);
    }
    assert(this.nodes[node.id].nodeInEdges(id, vstring));
    this.log.debug('Successfully added', (seed+1) : string, 'to ID', id : string, 'to create ID', node.id : string, hstring=hstring);
    return node.id;
  }

}
