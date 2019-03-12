// by Audrey Pratt
// I'm sure some sort of license is appropriate here, but right now
// this is all internal Cray stuff.

use rng;
use uuid;
use Random;

// This pulls from its own RNG.  Guarantees a bit more entropy.
var UUID = new owned uuid.UUID();
UUID.UUID4();
//writeln(UUID.UUID4());


record noiseFunctions {

}

record deltaRecord {
  var seeds: domain(int);
  var delta: [seeds] real;

  iter these() {
    //yield (seeds, delta);
    yield (0, 0);
  }

  proc key(a) {
    return this.delta[a];
  }
}

class GeneEdge {
  // This is mostly just a handler class which stores the information necessary
  // to change the in memory array from one state to another.
  // It is keyed to the UUID of a node.
  // The values are the coefficients we should use apply to the seed.
  //var seeds : domain(int);
  //var delta : [seeds] int;
  var delta : deltaRecord;
  // These are values for the noise function that we'll be using.
  var mu: real;
  var sigma: real;
  // I just want this to be a pointer to a noise function, or something.
  // I don't want to store the function in the class if I can help it.
  // Actually, I guess it doesn't matter that much, really.  I can sort it out later.
  var direction: (string, string);
  var noise_function: int;

  proc init(delta) {
    this.delta = delta;
  }

  proc seedInDelta(seed: int) {
    for rseed in this.delta.seeds do {
      if rseed == seed {
        return true;
      }
    }
    // If it's not in there, it's not in there.
    return false;
  }

  proc expressDelta(matrix: [real]) {
    for seed in this.delta.seeds do {
      matrix += this.gaussian_noise(seed)*delta.delta[seed];
    }
  }

  // Sort of assuming this is _actually_ gaussian.
  proc gaussian_noise(matrix: [real], shape: int, seed: int) {
    // I'm assuming I can get the shape from the matrix, but still.
    // Just pulling from the global is fine, really; we don't need a
    // random stream for all of these.
    var noise: [1..shape] real;
    fillRandom(noise, seed);
    return noise;
  }
}

class GeneNode {
  // This is a node.  It contains the Chapel implementation of a hash table
  // (akin to a python dict); we're going to store the gene edges in it.
  var nodes: domain(string);
  var edges: [nodes] GeneEdge;
  var generation: int;
  var ctype: string;
  var parent: string;
  // we need a node ID.  I like the ability of being able to specify them.
  // but we should generate them by default.
  var id: string;

  // Here, we're gonna track our parent at history 0
  // should make it easier to return histories.
  var parentSeedNode: string;

  proc init(id='', ctype='', parent='', parentSeedNode='') {
    this.ctype = ctype;
    this.parent = parent;
    // Here, we make an ID if we don't already have one.
    if id == '' {
      this.id = UUID.UUID4();
    } else {
      this.id = id;
    }
    if parentSeedNode == '' {
      this.parentSeedNode = this.id;
    } else {
      this.parentSeedNode = parentSeedNode;
    }
  }
  // some validation functions
  proc node_in_edges(id: string) {
    // Well, okay, that turned out to be easy but whatever.
    return this.nodes.member(id);
  }

  // Now, the functions to handle the nodes!
  //   proc join(node: GeneNode, delta: [?dom]) {
  proc join(node: GeneNode, delta: deltaRecord) {
    // did I call that function correctly?
    //writeln(node, delta);
    this.edges[node.id] = new unmanaged GeneEdge(delta);
    // Now, reverse the delta.
    var r_delta = new deltaRecord();
    for seed in delta.seeds do {
      r_delta.delta[seed] = delta.delta[seed] * -1;
    }
    node.edges[this.id] = new unmanaged GeneEdge(r_delta);
    //writeln(node.edges[this.id]);
  }

  proc return_edge(id: string) {
    if this.node_in_edges(id) {
      return this.edges[id];
    } else {
      // Return an empty deltaRecord
      return deltaRecord;
    }
  }
  proc new_node(seed: int, coefficient: real, id='': string) {
    // This function is a generic call for whenever we make a modification
    // Mutations, adding a new seed, whatever.  We just create a new node
    // and join them properly.
    var node = new unmanaged GeneNode(id=id);
    var delta = new deltaRecord();

    node.parentSeedNode = this.parentSeedNode;
    node.parent = this.id;

    delta.delta[seed] = coefficient;
    this.join(node, delta);
    return node;
  }
}
