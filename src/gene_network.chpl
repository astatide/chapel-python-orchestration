// by Audrey Pratt
// I'm sure some sort of license is appropriate here, but right now
// this is all internal Cray stuff.

use rng;
use uuid;

// This pulls from its own RNG.  Guarantees a bit more entropy.
var UUID = new owned uuid.UUID();
UUID.UUID4();
//writeln(UUID.UUID4());


record noiseFunctions {

}

record deltaRecord {
  var seeds: domain(int);
  var delta: [seeds] int;
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

  proc seedInDelta() {

  }
  proc expressDelta() {

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

  proc init(id='') {
    // Here, we make an ID if we don't already have one.
    if id == '' {
      this.id = UUID.UUID4();
    } else {
      this.id = id;
    }
  }
  // Now, the functions to handle the nodes!
  //   proc join(node: GeneNode, delta: [?dom]) {
  proc join(node: GeneNode, delta: deltaRecord) {
    // did I call that function correctly?
    writeln(node, delta);
    this.edges[node.id] = new unmanaged GeneEdge(delta);
    // Now, reverse the delta.
    var r_delta = new deltaRecord();
    for seed in delta.seeds do {
      r_delta.delta[seed] = delta.delta[seed] * -1;
    }
    node.edges[this.id] = new owned GeneEdge(r_.delta);
    writeln(node.edges[this.id]);
  }
}
