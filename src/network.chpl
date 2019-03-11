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
  var nodes: [0] genes.GeneNode;

  proc add_nodes(nodes: domain(genes.GeneNode)) {
    //writeln(nodes);
    for node in nodes {
      // We are working with the actual node objects, here.
      //e.clear();
      // Add to our domain!
      this.ids.add(node.id);
      for edge in node.nodes {
        //i += 1;
        //e.add(i);
        //e.add(edge);
        //writeln(node.id, edge);
        this.edges[node.id].add(edge);
      }
      //this.edges[node.id] = e;
    }
  }

  proc calculatePath(id_A: string, id_B: string) {
    // This is an implementation of djikstra's algorithm.
    var nodes: domain(string);
    var visited: [nodes] bool;
    var dist: [nodes] int;
    var paths: [nodes] string;
    var currentNode = id_A;
    var unvisited = [0] string;
    var unvisited_d = [0] int;
    // Build up the potential node list.
    for id in this.ids do {
      visited[id] = false;
      dist[id] = Math.INFINITY;
      paths[id] = [0] string;
    }
    dist[id_A] = 0;
    paths[id_A].push_front(id_A);
    while true {
      for edge in this.edges[currentNode] do {
        if ~visited[edge] {
          var d = min(dist[edge], dist[currentNode]+1);
          unvisited.push_front(edge);
          unvisited_d.push_front(d);

          if d == dist[currentNode]+1 {
            paths[edge] = paths[currentNode] + [edge];
          }
        }
      }
      visited[currentNode] = true;
      if ~unvisited.isEmpty() {
        // get the current minimum from here.
        var next_node_id = unvisited_d.find(unvisited_d.low)[1];
        currentNode = unvisited[next_node_id];
        delete unvisited_d[next_node_id];
        delete unvisited[next_node_id];
      }
      if visited[id_B] {
        break;
      }
      if unvisited.isEmpty() {
        break;
      }
    }
    return paths[id_B];

  }

}
