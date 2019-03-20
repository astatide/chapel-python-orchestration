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

record pathHistory {
  var n: domain(int);
  var node: [n] string;

  iter these() {
    // This works.  Grand!
    //if this.n.size > 0 {
    for i in 0..this.n.size-1 do {
      yield (i, this.node[i]);
    }
    //}
    //yield node;
  }

  proc key(i) {
    return this.node[i];
  }

  proc compare(a, b) {
    return abs(a) - abs(b);
  }

  // There HAS to be a better way to do this.
  proc remove(a) {
    //if this.node.member(a) {
    // search through and find the index to remove.
    var n_remove = 0;
    var new_n: domain(int);
    var new_node: [new_n] string;
    for i in 0..this.n.size-1 do {
      //if this.node[i] == a {
      //  this.n.remove(i);
      //  n_remove = i+1;
      //}
      // Oddly enough I do think this is working, although it's awful.
      //writeln(this.node[i], ' : ', a, ' ', i-n_remove);
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
    //}
    //this.n.add('0');
  }

  proc distance() {
    return this.n.size;
  }

}

//var reverseHistoryTuple: ReverseComparator(Comparator);

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
  var nodes$: sync bool = false;
  var lock: shared spinlock.SpinLock;

  var irng = new owned rng.UDevRandomHandler();

  //var rootNode = new shared genes.GeneNode(id='  root');
  var rootNode: shared genes.GeneNode;
  var log: shared ygglog.YggdrasilLogging;

  // Kind of wondering whether this is the appropriate place to handle locales?
  // Despite the name, this is simply an array which stores where each locale
  // currently is.

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
    //var n = nodes$;
    //nodes$.writeEF(true);
    var vstring: string;
    if hstring != '' {
      vstring = ' '.join(hstring, '__addNode__');
    }
    this.lock.wl(vstring);
    this.ids.add(node.id);
    this.nodes[node.id] = node;
    for edge in node.nodes {
      this.edges[node.id].add(edge);
      if !this.ids.member(edge) {
        this.ids.add(edge);
      }
      if !this.edges[edge].member(node.id) {
        this.edges[edge].add(node.id);
      }
    }
    this.lock.uwl(vstring);
  }

  proc newSeed() {
    // Generates a new seed for use with deltas, etc.
    // we're returning a long.
    return this.irng.getrandbits(64);
    //return 0;
  }

  proc init() {
    this.lock = new shared spinlock.SpinLock();
    this.lock.t = 'GeneNetwork';
    this.rootNode = new shared genes.GeneNode(id='root', ctype='root');
  }

  proc initializeNetwork(n_seeds=10: int, gen_seeds=true: bool) {
    var seed: int;
    //var node: unmanaged genes.GeneNode;
    var delta: genes.deltaRecord;
    //const alpha = ['A', 'B', 'C'];
    //this.rootNode = new shared genes.GeneNode(id='root', ctype='root');
    // Initialize the lock
    //this.lock = new shared spinlock.SpinLock();
    this.rootNode.log = this.log;
    this.rootNode.l.log = this.log;
    for n in 1..n_seeds {
      //seed = this.newSeed();
      seed = n*1000;
      var node = new shared genes.GeneNode(id=(n*1000) : string, ctype='seed', parentSeedNode='', parent='root');
      node.log = this.log;
      node.l.log = this.log;
      //writeln(this.log : string, node.log : string, node.l.log : string);
      node.debugOrderOfCreation = n*1000;
      delta = new genes.deltaRecord();
      delta.seeds.add(seed);
      delta.delta[seed] = 1;
      this.rootNode.join(node, delta, 'initializeNetwork');
      this.add_node(node, 'initializeNetwork');
    }
    this.add_node(this.rootNode, 'initializeNetwork');
    // send in the logging infrastructure.
    //this.lock.log = this.log;
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

  proc moveToNode(id_A: string, id_B: string) {
    return this.__moveToNode__(id_A, id_B, hstring='');
  }

  proc moveToNode(id_A: string, id_B: string, hstring: string) {
    return this.__moveToNode__(id_A, id_B, hstring);
  }

  proc __moveToNode__(id_A: string, id_B: string, hstring: string) {
    // We do have a lock here.
    var vstring = ' '.join(hstring, '__moveToNode__');
    //this.lock.(vstring);
    var path = this.calculatePath(id_A, id_B, hstring);
    //this.lock.unlock(vstring);
    // Cool, we have a path.  Now we need to get all the edges and
    // aggregate the coefficients.
    var delta = new genes.deltaRecord();
    var pathLength: int;
    // Get rid of the current node.
    var currentNode = id_A;
    path.remove(id_A);
    for (i, pt) in path {
      //this.lock.lock(vstring);
      var edge = this.nodes[currentNode].edges[pt : string];
      //this.lock.unlock(vstring);
      for (s, c) in edge.delta {
        //delta.seeds.add(seed);
        //delta.delta[seed] += (c*-1) : real;
        delta += (s, c);
      }
      currentNode = pt;
      pathLength += 1;
    }
    for (s, c) in delta {
      if c == 0 {
        // Get rid of the seed is the coefficient is 0.  We don't need that stuff.
        delta.remove(s);
      }
    }
    return (delta, pathLength);
  }

  proc calculateHistory(id: string, hstring: string) {
    // Since all nodes carry their ancestor,
    // simply calculate the path back to the seed node.
    //this.lock.lock();
    var vstring: string;
    if hstring != '' {
      vstring = ' '.join(hstring, 'calculateHistory');
    }
    var path = this.calculatePath(id, this.nodes[id].parentSeedNode, hstring);
    //this.lock.unlock();
    // Cool, we have a path.  Now we need to get all the edges and
    // aggregate the coefficients.
    var delta = new genes.deltaRecord();
    // Get rid of the current node.
    //path.remove(path[path.size]);
    //sort(path, comparator=reverseHistoryTuple);
    var currentNode = id;
    //writeln(path);
    path.remove(id);
  //  writeln(path);
    // This does need to be sorted in order to get the actual edges.
    // OH YEAH IT'S DONE NOW AWWWWW YIS.
    for (i, pt) in path {
      // get the node itself.
      //writeln(pt);
      //writeln(pt[2]);
      //writeln(this.nodes[currentNode].edges);
      //writeln(this.nodes[currentNode].edges[pt].delta);
      //writeln(this.nodes[currentNode].edges[pt[2]]);
      //this.lock.lock();
      var edge = this.nodes[currentNode].edges[pt : string];
      //this.lock.unlock();
      //for (seed, c) in edge.delta {
      //writeln(edge);
      for (seed, c) in zip(edge.delta.seeds, edge.delta.delta) {
        // If it doesn't exist...
        //if delta.seeds.member(seed) == true {
        delta.seeds.add(seed);
        //}
        delta.delta[seed] += (c*-1) : real;
      }
      currentNode = pt;
    }
    for (seed, c) in zip(delta.seeds, delta.delta) {
      if c == 0 {
        // Get rid of the seed is the coefficient is 0.
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
    //var ndeltaA = new genes.deltaRecord;
    //var ndeltaB = new genes.deltaRecord;
    var node = new shared genes.GeneNode(ctype='merge', parentSeedNode=this.nodes[id_A].parentSeedNode, parent=id_A);
    // s, c = seed, coefficient
    //writeln(deltaA);
    // Why is it in the reverse, you ask?  Because the calculateHistory method
    // returns the information necessary to go BACK to the seed node from the id given.
    node.log = this.log;
    node.l.log = this.log;
    var delta = ((deltaA*-1) + deltaB)/2;
    node.join(this.nodes[id_A], delta*-1, vstring);
    node.join(this.nodes[id_B], delta, vstring);
    this.add_node(node, vstring);
    //var delta = (deltaA + deltaB);
    //delta /= 2;
    //delta = delta * 2;
    //writeln(delta/2);
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
    var seed = this.nodes[id].debugOrderOfCreation;
    var node = this.nodes[id].new_node(1, 1, (seed+1) : string, vstring);
    node.debugOrderOfCreation = seed+1;
    node.generation = this.nodes[id].generation + 1;
    node.log = this.log;
    node.l.log = this.log;
    this.add_node(node, vstring);
    this.log.debug('Successfully added', seed : string, 'to ID', id : string, 'to create ID', node.id : string, hstring=hstring);
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
