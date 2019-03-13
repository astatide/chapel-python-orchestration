// Audrey Pratt

use genes;
use rng;
use uuid;
use Random;
use Math;
use Sort;

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
  var nodes: [ids] genes.GeneNode;

  var irng = new owned rng.UDevRandomHandler();

  var rootNode = new unmanaged genes.GeneNode(id='root');

  proc add_nodes(nodes: domain(genes.GeneNode)) {
    //writeln(nodes);
    for node in nodes {
      // We are working with the actual node objects, here.
      // Add to our domain!
      this.ids.add(node.id);
      this.nodes[node.id] = node;
      for edge in node.nodes {
        this.edges[node.id].add(edge);
      }
    }
  }

  proc add_node(node: unmanaged) : void {
    //writeln(nodes);
    // We are working with the actual node objects, here.
    // Add to our domain!
    this.ids.add(node.id);
    this.nodes[node.id] = node;
    for edge in node.nodes {
      this.edges[node.id].add(edge);
    }
  }

  proc newSeed() {
    // Generates a new seed for use with deltas, etc.
    // we're returning a long.
    return this.irng.getrandbits(64);
  }

  proc initializeNetwork(n_seeds=10: int, gen_seeds=true: bool) {
    var seed: int;
    //var node: unmanaged genes.GeneNode;
    var delta: genes.deltaRecord;
    this.rootNode.ctype = 'root';
    this.add_node(this.rootNode);
    if gen_seeds {
      for n in 1..n_seeds {
        seed = this.newSeed();
        var node = new unmanaged genes.GeneNode(ctype='seed', parentSeedNode='', parent='root');
        delta = new genes.deltaRecord();
        //node.ctype = 'seed';
        //node.parentSeedNode = node.id;
        //node.parent = this.rootNode.id;
        delta.seeds.add(seed);
        delta.delta[seed] = 1;
        node.join(this.rootNode, delta);
        this.add_node(node);
      }
    }
  }

  proc initializeSeedGenes(seeds: domain(int)) {
    // We're going to create a whole host of seed type nodes.
    // These have a special deltaRecord; I'm going to encode an INFINITY
    // as 'blow away the matrix', then ... or should I?
    for seed in seeds {
      var node = new unmanaged genes.GeneNode(seed);
      this.add_nodes(node);
    }
  }

  proc calculatePath(id_A: string, id_B: string) {
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
    // Build up the potential node list.
    for id in this.ids do {
      nodes.add(id);
      visited[id] = false;
      dist[id] = Math.INFINITY;
      // paths is sorted down there.
    }
    dist[id_A] = 0;
    //paths[id_A].add((0.0, id_A));
    //paths[id_A].add(0);
    paths[id_A].node[0] = id_A;
    while true {
      //writeln(paths, ' : ', unvisited);
      for edge in this.edges[currentNode] do {
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
      visited[currentNode] = true;

      if visited[id_B] {
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
    return paths[id_B];

  }

  proc calculateHistory(id: string) {
    // Since all nodes carry their ancestor,
    // simply calculate the path back to the seed node.
    var path = this.calculatePath(id, this.nodes[id].parentSeedNode);
    // Cool, we have a path.  Now we need to get all the edges and
    // aggregate the coefficients.
    var delta = new genes.deltaRecord;
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
      var edge = this.nodes[currentNode].edges[pt : string];
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
    var deltaA = this.calculateHistory(id_A);
    var deltaB = this.calculateHistory(id_B);
    var ndeltaA = new genes.deltaRecord;
    var ndeltaB = new genes.deltaRecord;
    var node = new unmanaged genes.GeneNode(ctype='merge', parentSeedNode=this.nodes[id_A].parentSeedNode, parent=id_A);
    // s, c = seed, coefficient
    //writeln(deltaA);
    // Why is it in the reverse, you ask?  Because the calculateHistory method
    // returns the information necessary to go BACK to the seed node from the id given.
    for (s, c) in deltaA {
      ndeltaA.add(s, (c/2));
      ndeltaB.add(s, (-1*c/2));
    }
    //writeln(deltaB);
    for (s, c) in deltaB {
      //writeln((c*(-1/2)), ' : ', (c/2), ' : ', (-1*c/2));
      ndeltaA.add(s, (-1*c/2));
      ndeltaB.add(s, (c/2));
    }
    // blah blah, now, set up the new delta...
    // we need a parent seed node; we could just pick it randomly, but hey.
    node.join(this.nodes[id_A], ndeltaA);
    node.join(this.nodes[id_B], ndeltaB);
    this.add_node(node);
    writeln(ndeltaA);
    writeln(ndeltaB);
    writeln(node);
    writeln(' ya ya ya ');
    var delta = (deltaA + deltaB);
    //delta /= 2;
    delta = delta * 2;
    writeln(delta);
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
    this.add_node(this.rootNode);
    for n in 1..3 {
      seed = this.newSeed();
      var node = new unmanaged genes.GeneNode(id=alpha[n], ctype='seed', parentSeedNode='', parent='root');
      delta = new genes.deltaRecord();
      delta.seeds.add(seed);
      delta.delta[seed] = 1;
      node.join(this.rootNode, delta);
      this.add_node(node);
    }
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
    writeln('BLAH!');
    writeln(this.ids, ' : ', this.nodes);
    writeln(this.calculatePath('A', '7'));
    writeln(this.calculateHistory('7'));
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
    this.mergeNodes('A', '7');
  }

}
