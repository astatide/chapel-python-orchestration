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

  proc key(i) {
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

  proc add_node(in node: shared genes.GeneNode) {
    this.__addNode__(node, hstring='');
  }

  proc add_node(in node: shared genes.GeneNode, hstring: string) {
    this.__addNode__(node, hstring);
  }

  proc __addNode__(in node: shared genes.GeneNode, hstring: string) : void {
    //writeln(nodes);
    // We are working with the actual node objects, here.
    // Add to our domain!
    // We need to block until such time as we're ready;
    var vstring: string;
    if hstring != '' {
      vstring = ' '.join(hstring, '__addNode__');
    }
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
      this.rootNode.join(node, delta, 'initializeNetwork');
      this.add_node(node, 'initializeNetwork');
    }
    this.add_node(this.rootNode, 'initializeNetwork');
  }

  proc initializeSeedGenes(seeds: domain(int)) {
    // We're going to create a whole host of seed type nodes.
    // These have a special deltaRecord; I'm going to encode an INFINITY
    // as 'blow away the matrix', then ... or should I?
    for seed in seeds {
      var node = new shared genes.GeneNode(seed);
      this.add_nodes(node);
    }
  }

  proc calculatePath(id_A: string, id_B: string, hstring: string) {
    // This is an implementation of djikstra's algorithm.
    var nodes: domain(string);
    var visited: [nodes] bool;
    var dist: [nodes] real;
    //var paths: [nodes] domain((real, string));
    var paths: [nodes] pathHistory;
    var currentNode = id_A;
    var unvisited: domain(string);
    var unvisited_d: domain(real);
    var currMinDist = Math.INFINITY;
    var currMinNode = id_A;
    var currMinNodeIndex = 0;
    var i: int;
    var vstring: string;
    if hstring != '' {
      vstring = ' '.join(hstring, 'calculatePath');
    }
    //writeln(this.edges);
    // Build up the potential node list.
    //this.lock.lock();
    //for id in this.ids do {
    //  nodes.add(id);
    //  visited[id] = false;
    //  dist[id] = Math.INFINITY;
      // paths is sorted down there.
    //}
    //this.lock.unlock();
    // Am I removing 'root' from here?  I suspect yes.
    nodes.add[id_A];
    dist[id_A] = 0;
    //paths[id_A].add((0.0, id_A));
    //paths[id_A].add(0);
    paths[id_A].node[0] = id_A;
    //writeln(this.ids);
    while true {
      //writeln(paths, ' : ', unvisited);
      //this.lock.lock();
      this.lock.rl(vstring);
      for edge in this.edges[currentNode] do {
        if !nodes.member(edge) {
            nodes.add(edge);
            visited[edge] = false;
            dist[edge] = Math.INFINITY;
        }
        //writeln(nodes.member(edge), ' ', edge);
        if !visited[edge] {
          var d = min(dist[edge], dist[currentNode]+1);
          //writeln(d);
          unvisited.add(edge);
          unvisited_d.add(d);
          dist[edge] = d;

          if d == dist[currentNode]+1 {
            paths[edge].n.clear();
            //i = 0;
            for (j, e) in paths[currentNode] {
              //writeln(e);
              paths[edge].n.add(j : int);
              paths[edge].node[j : int] = e;
            }
            //paths[edge] = paths[currentNode] + edge;
            // We're doing this as a tuple to help sorting later.
            // That'll also help us calculate how many hops we have to make,
            // which will be convenient when we're trying to determine who
            // should do what.
            paths[edge].n.add(d: int);
            paths[edge].node[d: int] = edge;
          }
        }
      }
      //this.lock.unlock();
      this.lock.url(vstring);
      visited[currentNode] = true;

      if nodes.member(id_B) {
        if visited[id_B] {
          break;
        }
      }
      if unvisited.isEmpty() {
        break;
      } else {
        // get the current minimum from here.
        //var next_node_id = unvisited_d.find(unvisited_d.low)[1];
        i = 0;
        currMinDist = Math.INFINITY;
        for node in unvisited {
          i += 1;
          //writeln(currentNode, ' : ', currMinDist, ', ', node, ' : ', dist[node]);
          if currMinDist > dist[node] {
            currMinDist = dist[node];
            currMinNode = node;
            currMinNodeIndex = i;
          }
        }
        //writeln(currMinNode);
        currentNode = currMinNode;
        unvisited_d.remove(currMinDist);
        unvisited.remove(currMinNode);
      }
    }
    //writeln(nodes, paths);
    return paths[id_B];

  }

  proc calculatePathArray(id_A: string, id_B: domain(string), hstring: string) {
    // This is an implementation of djikstra's algorithm.
    var nodes: domain(string);
    var visited: [nodes] bool;
    var dist: [nodes] real;
    //var paths: [nodes] domain((real, string));
    var paths: [nodes] pathHistory;
    var currentNode = id_A;
    var unvisited: domain(string);
    var unvisited_d: domain(real);
    var currMinDist = Math.INFINITY;
    var currMinNode = id_A;
    var currMinNodeIndex = 0;
    var i: int;
    var completed: [id_B] bool = false;
    var vstring: string;
    if hstring != '' {
      vstring = ' '.join(hstring, 'calculatePath');
    }
    //writeln(this.edges);
    // Build up the potential node list
    //this.lock.lock();
    //for id in this.ids do {
    //  nodes.add(id);
    //  visited[id] = false;
    //  dist[id] = Math.INFINITY;
      // paths is sorted down there.
    //}
    //this.lock.unlock();
    // Am I removing 'root' from here?  I suspect yes.
    nodes.add[id_A];
    dist[id_A] = 0;
    //paths[id_A].add((0.0, id_A));
    //paths[id_A].add(0);
    paths[id_A].node[0] = id_A;
    //writeln(this.ids);
    while true {
      i += 1;
      //writeln(paths, ' : ', unvisited);
      //this.lock.lock();
      this.lock.rl(vstring);
      this.log.debug('Attempting to pass through', currentNode, 'edges', this.ids.contains(currentNode) : string, vstring);
      for edge in this.edges[currentNode] do {
      this.log.debug(currentNode, edge, vstring);
        if !nodes.member(edge) {
            nodes.add(edge);
            visited[edge] = false;
            dist[edge] = Math.INFINITY;
        }
        if !visited[edge] {
          var d = min(dist[edge], dist[currentNode]+1);
          //writeln(d);
          unvisited.add(edge);
          unvisited_d.add(d);
          dist[edge] = d;

          if d == dist[currentNode]+1 {
            paths[edge].n.clear();
            //i = 0;
            for (j, e) in paths[currentNode] {
              //writeln(e);
              paths[edge].n.add(j : int);
              paths[edge].node[j : int] = e;
            }
            //paths[edge] = paths[currentNode] + edge;
            // We're doing this as a tuple to help sorting later.
            // That'll also help us calculate how many hops we have to make,
            // which will be convenient when we're trying to determine who
            // should do what.
            paths[edge].n.add(d: int);
            paths[edge].node[d: int] = edge;
          }
        }
      }
      //this.lock.unlock();
      this.lock.url(vstring);
      visited[currentNode] = true;
      if id_B.member(currentNode) {
        completed[currentNode] = true;
      }

      if + reduce completed == id_B.size {
        break;
      }
      if unvisited.isEmpty() {
        break;
      } else {
        // get the current minimum from here.
        //var next_node_id = unvisited_d.find(unvisited_d.low)[1];
        i = 0;
        currMinDist = Math.INFINITY;
        for node in unvisited {
          i += 1;
          //writeln(currentNode, ' : ', currMinDist, ', ', node, ' : ', dist[node]);
          if currMinDist > dist[node] {
            currMinDist = dist[node];
            currMinNode = node;
            currMinNodeIndex = i;
          }
        }
        //writeln(currMinNode);
        currentNode = currMinNode;
        unvisited_d.remove(currMinDist);
        unvisited.remove(currMinNode);
      }
    }
    //writeln(nodes, paths);
    return paths;

  }

  proc returnNearestUnprocessed(id_A: string, id_B: domain(string), hstring: string) {
    // This is an implementation of djikstra's algorithm.
    var nodes: domain(string);
    var visited: [nodes] bool;
    var dist: [nodes] real;
    //var paths: [nodes] domain((real, string));
    var paths: [nodes] pathHistory;
    var currentNode = id_A;
    var unvisited: domain(string);
    var unvisited_d: domain(real);
    var currMinDist = Math.INFINITY;
    var currMinNode = id_A;
    var currMinNodeIndex = 0;
    var i: int;
    var completed: [id_B] bool = false;
    var vstring: string;
    if hstring != '' {
      vstring = ' '.join(hstring, 'returnNearestUnprocessed');
    }
    //writeln(this.edges);
    // Build up the potential node list
    //this.lock.lock();
    //for id in this.ids do {
    //  nodes.add(id);
    //  visited[id] = false;
    //  dist[id] = Math.INFINITY;
      // paths is sorted down there.
    //}
    //this.lock.unlock();
    // Am I removing 'root' from here?  I suspect yes.
    nodes.add[id_A];
    dist[id_A] = 0;
    //paths[id_A].add((0.0, id_A));
    //paths[id_A].add(0);
    paths[id_A].n.add(0);
    paths[id_A].node[0] = id_A;
    //writeln(this.ids);
    // catch an empty id_B!
    if id_B.isEmpty() {
      return (id_A, paths[id_A]);
    }
    while true {
      i += 1;
      //writeln(paths, ' : ', unvisited);
      //this.lock.lock();
      this.lock.rl(vstring);
      this.log.debug('Attempting to pass through', currentNode, 'edges', this.ids.contains(currentNode) : string, vstring);
      for edge in this.edges[currentNode] do {
        //this.log.debug(currentNode, edge, vstring);
        if !nodes.contains(edge) {
            nodes.add(edge);
            visited[edge] = false;
            dist[edge] = Math.INFINITY;
        }
        if !visited[edge] {
          var d = min(dist[edge], dist[currentNode]+1);
          //writeln(d);
          unvisited.add(edge);
          unvisited_d.add(d);
          dist[edge] = d;

          if d == dist[currentNode]+1 {
            paths[edge].n.clear();
            //i = 0;
            //paths[edge].n.add(0);
            //paths[edge].node[0] = currentNode;
            var z: int;
            for (j, e) in paths[currentNode] {
              //writeln(e);
              paths[edge].n.add(j : int);
              paths[edge].node[j : int] = e;
              z += 1;
            }
            //paths[edge] = paths[currentNode] + edge;
            // We're doing this as a tuple to help sorting later.
            // That'll also help us calculate how many hops we have to make,
            // which will be convenient when we're trying to determine who
            // should do what.
            paths[edge].n.add(d: int);
            paths[edge].node[d: int] = edge;
          }
        }
      }
      //this.lock.unlock();
      this.lock.url(vstring);
      visited[currentNode] = true;
      if id_B.contains(currentNode) {
        //writeln('LIAR!');
        break;
        completed[currentNode] = true;
      }
      if unvisited.isEmpty() {
        //break;
      } else {
        // get the current minimum from here.
        //var next_node_id = unvisited_d.find(unvisited_d.low)[1];
        i = 0;
        currMinDist = Math.INFINITY;
        for node in unvisited {
          i += 1;
          //writeln(currentNode, ' : ', currMinDist, ', ', node, ' : ', dist[node]);
          if currMinDist > dist[node] {
            currMinDist = dist[node];
            currMinNode = node;
            currMinNodeIndex = i;
          }
        }
        //writeln(currMinNode);
        currentNode = currMinNode;
        unvisited_d.remove(currMinDist);
        unvisited.remove(currMinNode);
        //this.log.log('CONTINUE', currentNode : string, this.edges[currentNode] : string, this.nodes[currentNode].nodes : string, 'id_B:', id_B : string, vstring);
      }
    }
    //writeln(nodes, paths);
    //writeln(currentNode, paths[currentNode], id_B, ' ', id_A);
    this.log.debug('id_B:', id_B : string, 'currentNode:', currentNode, 'id_A:', id_A, hstring=vstring);
    return (currentNode, paths[currentNode]);

  }

  proc move(ref v: propagator.valkyrie, id: string, createEdgeOnMove: bool, edgeDistance: int) {
    // Bit clonky, but for now.
    var hstring = ' '.join(v.header, 'move');
    this.log.debug('attempting to move', hstring=hstring);
    var (d, pl) = this.moveToNode(v.currentNode, id, hstring=v.header);
    v.nMoves += 1;
    if createEdgeOnMove {
      if pl > edgeDistance {
        // If our edge distance is particularly long, create a shortcut.
        this.lock.wl(hstring);
        this.nodes[v.currentNode].join(this.nodes[id], d, ' '.join(v.header, 'move'));
        this.edges[id].add(v.currentNode);
        this.edges[v.currentNode].add(id);
        this.lock.uwl(hstring);
      }
    }
    v.move(d, id);
    this.log.debug('move successful', hstring=hstring);
  }

  proc move(ref v: propagator.valkyrie, id: string, path: pathHistory, createEdgeOnMove: bool, edgeDistance: int) throws {
    // Bit clonky, but for now.
    var vstring = ' '.join(v.header, 'move');
    this.log.debug('attempting to move', hstring=vstring);
    //this.lock.(vstring);
    //this.lock.unlock(vstring);
    // Cool, we have a path.  Now we need to get all the edges and
    // aggregate the coefficients.
    var d = new genes.deltaRecord();
    var pl: int;
    // Get rid of the current node.
    var currentNode = id;
    //path.remove(id);
    this.log.debug('PATH', path : string, hstring=vstring);
    for (i, pt) in path {
      this.lock.rl(vstring);
      var edge: genes.GeneEdge;
      if currentNode != id {
        if this.nodes[currentNode].nodes.contains(pt : string) {
          edge = this.nodes[currentNode].edges[pt : string];
        } else {
          // Error throwing doesn't work, so request a lock and don't let it go.
          // this is because Chapel can't throw errors from a non inlined iterator.
          // Note that it DOES work when working in serial mode.
          this.lock.url(vstring);
          // FOR NOW, just lock it up.
          this.lock.wl(vstring);
          this.log.critical('CRITICAL FAILURE: Node', pt : string, 'not in edge list for node', currentNode : string, hstring=vstring);
          this.log.critical(path : string, hstring=vstring);
          //this.log.critical(this.nodes[currentNode].edges : string, this.nodes[pt : string].edges : string);
          this.log.critical(currentNode : string, '-', this.nodes[currentNode].nodes : string, ',', pt: string, '-', this.nodes[pt : string].nodes : string, hstring=vstring);
          throw new owned NodeNotInEdgesError();
        }
        for (s, c) in edge.delta {
          //delta.seeds.add(seed);
          //delta.delta[seed] += (c*-1) : real;
          d += (s, c);
        }
      }
      this.lock.url(vstring);
      currentNode = pt;
      pl += 1;
    }
    for (s, c) in d {
      if c == 0 {
        // Get rid of the seed is the coefficient is 0.  We don't need that stuff.
        d.remove(s);
      }
    }
    v.nMoves += 1;
    if createEdgeOnMove {
      if pl > edgeDistance {
        // If our edge distance is particularly long, create a shortcut.
        this.lock.wl(vstring);
        this.nodes[v.currentNode].join(this.nodes[id], d, ' '.join(v.header, 'move'));
        this.edges[id].add(v.currentNode);
        this.edges[v.currentNode].add(id);
        this.lock.uwl(vstring);
      }
    }
    v.move(d, id);
    this.log.debug('move successful', hstring=vstring);
  }

  proc moveToNode(id_A: string, id_B: string) {
    return this.__moveToNode__(id_A, id_B, hstring='');
  }

  proc moveToNode(id_A: string, id_B: string, hstring: string) {
    return this.__moveToNode__(id_A, id_B, hstring);
  }

  proc __moveToNode__(id_A: string, id_B: string, hstring: string) {
    // We do have a lock here.
    var vstring = ' '.join(hstring, '__moveToNode__');
    var path = this.calculatePath(id_A, id_B, hstring);
    // Cool, we have a path.  Now we need to get all the edges and
    // aggregate the coefficients.
    var delta = new genes.deltaRecord();
    var pathLength: int;
    // Get rid of the current node.
    var currentNode = id_A;
    path.remove(id_A);
    for (i, pt) in path {
      this.lock.rl(vstring);
      var edge = this.nodes[currentNode].edges[pt : string];
      this.lock.url(vstring);
      for (s, c) in edge.delta {
        delta += (s, c);
      }
      currentNode = pt;
      pathLength += 1;
    }
    for (s, c) in delta {
      if c == 0 {
        // Get rid of the seed if the coefficient is 0.  We don't need that stuff.
        // GET OUTTA HERE!  YOU'RE NOT WANTED!
        delta.remove(s);
      }
    }
    return (delta, pathLength);
  }

  proc calculateHistory(id: string, hstring: string) {
    // Since all nodes carry their ancestor,
    // simply calculate the path back to the seed node.
    var vstring: string;
    if hstring != '' {
      vstring = ' '.join(hstring, 'calculateHistory');
    }
    var path = this.calculatePath(id, this.nodes[id].parentSeedNode, hstring);
    // Cool, we have a path.  Now we need to get all the edges and
    // aggregate the coefficients.
    var delta = new genes.deltaRecord();
    var currentNode = id;
    path.remove(id);
    // path is essentially an ordered dictionary, so this works in that
    // path order is preserved.
    for (i, pt) in path {
      // get the node itself.
      this.lock.rl(vstring);
      var edge = this.nodes[currentNode].edges[pt : string];
      this.lock.url(vstring);
      for (seed, c) in zip(edge.delta.seeds, edge.delta.delta) {
        // If it doesn't exist...
        // actually, do I still need to explicitly add it?  Don't remember.
        delta.seeds.add(seed);
        delta.delta[seed] += (c*-1) : real;
      }
      currentNode = pt;
    }
    for (seed, c) in zip(delta.seeds, delta.delta) {
      if c == 0 {
        // Get rid of the seed is the coefficient is 0.
        // LIKE I SAID, NO ONE WANTS YOU HERE.
        delta.seeds.remove(seed);
      }
    }
    return delta;
  }

  // Here's a function for merging two nodes.

  proc mergeNodes(id_A: string, id_B: string) {
    return this.__mergeNodes__(id_A, id_B, hstring='');
  }

  proc mergeNodes(id_A: string, id_B: string, hstring: string) {
    return this.__mergeNodes__(id_A, id_B, hstring='');
  }

  proc __mergeNodes__(id_A: string, id_B: string, hstring: string) {
    var vstring: string;
    if hstring != '' {
        vstring = ' '.join(hstring, '__mergeNodes__');
    }
    var deltaA = this.calculateHistory(id_A, vstring);
    var deltaB = this.calculateHistory(id_B, vstring);
    var node = new shared genes.GeneNode(ctype='merge', parentSeedNode=this.nodes[id_A].parentSeedNode, parent=id_A);
    // Remember that we're sending in logging capabilities for debug purposes.
    node.log = this.log;
    node.l.log = this.log;
    // Why is it in the reverse, you ask?  Because the calculateHistory method
    // returns the information necessary to go BACK to the seed node from the id given.
    // So this delta allows us to go from node A to the new node.
    var delta = ((deltaA*-1) + deltaB)/2;
    node.join(this.nodes[id_A], delta*-1, vstring);
    // Now, reverse the delta to join B.  I love the records in Chapel.
    node.join(this.nodes[id_B], delta, vstring);
    this.add_node(node, vstring);
    // Return the id, as that's all we need.
    return node.id;
  }

  proc nextNode(id: string) {
    return this.__nextNode__(id, hstring='');
  }

  proc nextNode(id: string, hstring: string) {
    return this.__nextNode__(id, hstring);
  }

  proc __nextNode__(id: string, hstring: string) {
    var vstring: string;
    if hstring != '' {
      vstring = ' '.join(hstring, '__nextNode__');
    }
    this.log.debug('Adding a seed on to ID', id : string, hstring);
    // MIGHT NOT NEED TO BE A THING
    var seed: int;
    var nId: string;
    var node: shared genes.GeneNode;
    this.lock.wl(hstring);
    if propagator.unitTestMode {
      nId = (this.nodes[id].debugOrderOfCreation+1) : string;
      seed = 1;
    } else {
      nId = '';
      seed = this.newSeed();
    }
    node = this.nodes[id].new_node(seed=seed, coefficient=1, id=nId, hstring=vstring);
    if propagator.unitTestMode {
      node.debugOrderOfCreation = this.nodes[id].debugOrderOfCreation+1;
    }
    // Again, send the logger to both the lock and the node.
    node.log = this.log;
    node.l.log = this.log;
    // Add to current node!  I can't believe you forgot this.
    this.edges[id].add(node.id);
    this.lock.uwl(hstring);
    this.add_node(node, vstring);
    this.log.debug('Successfully added', (seed+1) : string, 'to ID', id : string, 'to create ID', node.id : string, hstring=hstring);
    return node.id;
  }

  // Set of testing functions.

  proc testAllTests() {

  }

  proc __test_create_network__() {
    var seed: int;
    //var node: unmanaged genes.GeneNode;
    var delta: genes.deltaRecord;
    const alpha = ['A', 'B', 'C'];
    this.rootNode.ctype = 'root';
    for n in 1..3 {
      seed = this.newSeed();
      var node = new shared genes.GeneNode(id=alpha[n], ctype='seed', parentSeedNode='', parent='root');
      delta = new genes.deltaRecord();
      delta.seeds.add(seed);
      delta.delta[seed] = 1;
      this.rootNode.join(node, delta);
      this.add_node(node);
    }
    this.add_node(this.rootNode);
  }

  proc testInitializeNetwork() {

  }

  proc testCalculatePath() {
    this.__test_create_network__();
    var i = 'A';
    //var node: unmanaged genes.GeneNode;
    for j in 1..7 {
      var node = this.nodes[i].new_node(this.newSeed(), 1, j : string);
      this.add_node(node);
      this.edges[i].add(node.id);
      i = j : string;
    }
  }

  proc testMergeNodes() {
    this.__test_create_network__();
    var i = 'A';
    //var node: unmanaged genes.GeneNode;
    for j in 1..7 {
      var node = this.nodes[i].new_node(this.newSeed(), 1, j : string);
      this.add_node(node);
      this.edges[i].add(node.id);
      i = j : string;
    }
    //this.mergeNodes('A', '7');
    this.moveToNode('A', '7');
  }

}
