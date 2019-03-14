// by Audrey Pratt
// I'm sure some sort of license is appropriate here, but right now
// this is all internal Cray stuff.

use rng;
use uuid;
use Random;

// This pulls from its own RNG.  Guarantees a bit more entropy.
var UUID = new owned uuid.UUID();
UUID.UUID4();
var udevrandom = new owned rng.UDevRandomHandler();
var newrng = udevrandom.returnRNG();
//writeln(UUID.UUID4());


record noiseFunctions {
  // This is for holding our functions.
  var which_function: int;
  var udevrandom = new owned rng.UDevRandomHandler();
  var newrng = udevrandom.returnRNG();

  proc noise(seed: int, c: real, ref matrix: [0] real) {
    return this.add_uniform_noise(seed, c, matrix);
  }

  proc add_uniform_noise(seed: int, c: real, ref matrix: [0] real) {
    // This a function that takes in our matrix and adds the appropriate
    // amount of noise.
    // Make a new array with the same domain as the input matrix.
    var m: [matrix.domain] real;
    this.newrng.fillRandom(m, seed=seed);
    matrix += (m*c);
  }
}

record deltaRecord {
  var seeds: domain(int);
  var delta: [seeds] real;

  iter these() {
    //yield (seeds, delta);
    for (s, c) in zip(this.seeds, this.delta) do {
      // We most definitely do not care about order for this.
      yield (s, c);
    }
  }

  proc this(a) ref {
    return this.delta[a];
  }

  proc add(s, c) {
    // try to add it.
    if !this.seeds.member(s) {
      this.seeds.add(s);
    }
    this.delta[s] += c;
    if this.delta[s] == 0 {
      // Pick up your shit, Todd.
      this.seeds.remove(s);
    }
  }

  proc remove(s) {
    this.seeds.remove(s);
  }

  proc express(ref matrix) {
    var m: [matrix.domain] real;
    for (s, c) in zip(this.seeds, this.delta) do {
      //matrix += noiseFunctions.noise(s, c, matrix);
      writeln(s, ' ', c);
      writeln(s.type : string);
      // What the hell is going on here?
      // error: unresolved access of 'int(64)' by '(int(64))'
      // Seriously?  What?
      //newrng.seed(int);
      // I'm sure something better exists, but for now.
      udevrandom.returnSpecificRNG(s).fillRandom(arr=m);
      matrix += (m*c);
      //matrix += c;
    }
  }

}

proc +(a: deltaRecord, b: deltaRecord) {
  var d = new deltaRecord;
  for (s, c) in a {
    d.add(s, c);
  }
  for (s, c) in b {
    d.add(s, c);
  }
  return d;
}

proc +=(ref a: deltaRecord, b: deltaRecord) {
  for (s, c) in b {
    a.add(s, c);
  }
}

proc +=(ref a: deltaRecord, b: (int, real)) {
  a.add(b[1], b[2]);
}

proc /=(ref a: deltaRecord, b: real) {
  for (s, c) in a {
    a[s] = a[s] / b;
  }
}

proc *=(ref a: deltaRecord, b: real) {
  for (s, c) in a {
    a[s] = a[s] * b;
  }
}

// Probably worth noting these are borked as hell.
proc /(a: deltaRecord, b: real) {
  var d = new deltaRecord;
  for (s, c) in a {
    //d.seeds.add(s);
    //d.delta[s] = (c/b);
    d.add(s, (c/b));
  }
  return d;
}

proc *(a: deltaRecord, b: real) {
  var d = new deltaRecord;
  for (s, c) in a {
    //d.seeds.add(s);
    //d.delta[s] = (c*b);
    d.add(s, (c*b));
  }
  return d;
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

  proc init() {

  }

  proc init(delta) {
    this.delta = delta;
  }

  proc init(delta, direction) {
    this.delta = delta;
    this.direction = direction;
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

  proc expressDelta(ref matrix: [real]) {
    for (s,c) in this.delta do {
      //matrix += this.gaussian_noise(seed)*delta.delta[seed];
      matrix += noiseFunctions.noise(s, c, matrix);
    }
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
    var d = (this.id, node.id);
    this.edges[node.id] = new unmanaged GeneEdge(delta, d);
    // Now, reverse the delta.  Which we can do by multiplying it by
    // -1.
    d = (node.id, this.id);
    node.edges[this.id] = new unmanaged GeneEdge(delta*-1, d);
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
