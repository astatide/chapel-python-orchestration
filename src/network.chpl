// Audrey Pratt

use genes;
use rng;
use uuid;
use Random;
use Math;

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

  proc add_node(node: unmanaged ) {
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
      var node = new shared genes.GeneNode(seed);
      this.add_nodes(node);
    }
  }

  proc calculatePath(id_A: string, id_B: string) {
    // This is an implementation of djikstra's algorithm.
    var nodes: domain(string);
    var visited: [nodes] bool;
    var dist: [nodes] real;
    var paths: [nodes] domain((real, string));
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
    paths[id_A].add((0.0, id_A));
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
            paths[edge].clear();
            for e in paths[currentNode] {
              paths[edge].add(e);
            }
            //paths[edge] = paths[currentNode] + edge;
            // We're doing this as a tuple to help sorting later.
            // That'll also help us calculate how many hops we have to make,
            // which will be convenient when we're trying to determine who
            // should do what.
            paths[edge].add((d, edge));
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
    writeln(nodes, paths);
    return paths[id_B];

  }

}
