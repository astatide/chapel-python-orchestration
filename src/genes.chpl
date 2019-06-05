// by Audrey Pratt
// I'm sure some sort of license is appropriate here, but right now
// this is all internal Cray stuff.

use rng;
use uuid;
use Random;
use spinlock;
use ygglog;

// This pulls from its own RNG.  Guarantees a bit more entropy.
var GUUID = new owned uuid.UUID();
GUUID.UUID4();
var udevrandom = new shared rng.UDevRandomHandler();
var newrng = udevrandom.returnRNG();
//writeln(UUID.UUID4());

record cType {
  var ROOTNODE = -1;
  // Used for normal, 'add a seed' types.
  var SEED = 0;
  var MERGE = 1;
}

record noiseFunctions {
  // This is for holding our functions.
  var which_function: int;
  var udevrandom = new owned rng.UDevRandomHandler();
  var newrng = udevrandom.returnRNG();

  proc noise(seed: int, c: real, ref matrix: [0] real) {
    return this.add_uniform_noise(seed, c, matrix);
    //return this.constant(seed, c, matrix);
  }

  proc add_uniform_noise(seed: int, c: real, ref matrix: [0] real) {
    // This a function that takes in our matrix and adds the appropriate
    // amount of noise.
    // Make a new array with the same domain as the input matrix.
    var m: [matrix.domain] real;
    this.newrng.fillRandom(m, seed=seed);
    matrix += (m*c*.1);
  }

  proc constant(seed: int, c: real, ref matrix: [0] real) {
    var m: [matrix.domain] real;
    m = seed;
    matrix += (m*c);
  }
}

record deltaRecord {
  var seeds: domain(int);
  var delta: [seeds] real;
  var to: string = "root";
  var from: string = "root";
  //var udevrandom = new owned rng.UDevRandomHandler();
  //var newrng = udevrandom.returnRNG();

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
    if !this.seeds.contains(s) {
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

  proc prune() {
    // remove it.
    for (s, c) in zip(this.seeds, this.delta) do {
      if c == 0.0 {
        this.seeds.remove(s);
      }
    }
  }

  proc express(ref matrix) {
    var m: [0..matrix.size] c_double;
    for (s, c) in zip(this.seeds, this.delta) do {
      // This is the debug mode.
      //m = s;
      //matrix += (m*c);
      //var m: [matrix.domain] real;
      Random.fillRandom(m, seed=s);
      matrix += (m*c*0.01);
    }
  }


  proc writeThis(f) {
    var style = defaultIOStyle();
    style.string_format = 2;
    f._set_style(style);
    f <~> new ioLiteral("{SIZE=");
    f <~> this.seeds.size;
    f <~> new ioLiteral(",");
    f <~> new ioLiteral("TO=");
    f <~> this.to;
    //f.write(this.to, style=style);
    f <~> new ioLiteral(",");
    f <~> new ioLiteral("FROM=");
    f <~> this.from;
    //f.write(this.from, style=style);
    f <~> new ioLiteral(",");
    var first = true;
    for (s, c) in zip(this.seeds, this.delta) {
      if first {
        first = false;
      } else {
        f <~> new ioLiteral(",");
      }
      f <~> new ioLiteral("(");
      f <~> s;
      f <~> new ioLiteral(",");
      f <~> c;
      f <~> new ioLiteral(")");
    }
    f <~> new ioLiteral("}");  }

  proc readThis(f) {
    //this.seeds.clear();
    var size: int;
    var first = true;
    var s: int;
    var c: real;
    var to: string;
    var from: string;
    var style = defaultIOStyle();
    style.string_format = 2;
    f._set_style(style);
    //var style = new iostyle(string_format=2);
    f <~> new ioLiteral("{SIZE=");
    f <~> size;
    f <~> new ioLiteral(",");
    f <~> new ioLiteral("TO=");
    // hello problem my old friend
    f <~> to;
    //f.readln(to, style=style);
    f <~> new ioLiteral(",");
    f <~> new ioLiteral("FROM=");
    f <~> from;
    f <~> new ioLiteral(",");
    this.to = to;
    this.from = from;
    for i in 1..size {
      if first {
        first = false;
      } else {
        f <~> new ioLiteral(",");
      }
      f <~> new ioLiteral("(");
      f <~> s;
      f <~> new ioLiteral(",");
      f <~> c;
      f <~> new ioLiteral(")");
      this.seeds.add(s);
      this.delta[s] = c;
    }
    f <~> new ioLiteral("}");
  }
}

proc +(a: deltaRecord, b: deltaRecord) {
  var d = new deltaRecord();
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
  var d = new deltaRecord();
  for (s, c) in a {
    //d.seeds.add(s);
    //d.delta[s] = (c/b);
    d.add(s, (c/b));
  }
  return d;
}

proc *(a: deltaRecord, b: real) {
  var d = new deltaRecord();
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

  proc reverse() {
    var rEdge = new shared GeneEdge();
    rEdge.delta = (this.delta * -1);
    rEdge.direction = (this.direction[2], this.direction[1]);
    rEdge.mu = this.mu;
    rEdge.sigma = this.sigma;
    rEdge.noise_function = this.noise_function;
    return rEdge;
  }

}

class GeneNode {
  // This is a node.  It contains the Chapel implementation of a hash table
  // (akin to a python dict); we're going to store the gene edges in it.
  var nodes: domain(string);
  var edges: [nodes] shared GeneEdge;
  var generation: int;
  var ctype: string;
  var parent: string;
  // we need a node ID.  I like the ability of being able to specify them.
  // but we should generate them by default.
  var id: string;
  var debugOrderOfCreation: int;

  // Here, we're gonna track our parent at history 0
  // should make it easier to return histories.
  var parentSeedNode: string;

  var l: shared spinlock.SpinLock;
  var log: shared ygglog.YggdrasilLogging;

  // We want to hold scores on the nodes.
  var demeDomain: domain(int);
  var chromosomes: domain(string);
  var scores: [demeDomain] real;

  proc init(id='', ctype='', parent='', parentSeedNode='') {
    this.ctype = ctype;
    this.parent = parent;
    // Here, we make an ID if we don't already have one.
    if id == '' {
      this.id = GUUID.UUID4();
    } else {
      this.id = id;
    }
    if parentSeedNode == '' {
      this.parentSeedNode = this.id;
    } else {
      this.parentSeedNode = parentSeedNode;
    }
    this.l = new shared spinlock.SpinLock();
    this.l.t = ' '.join('GENE', this.id);
  }

  proc writeThis(wc) {
    wc.writeln(this.id);
    wc.writeln('nodes: ', this.nodes : string);
    wc.writeln('edges: ', this.edges : string);
  }

  // some validation functions
  proc nodeInEdges(id: string, hstring: ygglog.yggHeader) {
    // Well, okay, that turned out to be easy but whatever.
    this.l.rl(hstring);
    var ie = this.nodes.contains(id);
    this.l.url(hstring);
    return ie;
  }

  // Now, the functions to handle the nodes!
  proc join(node: shared GeneNode, delta: deltaRecord) {
    this.__join__(node, delta, hstring='');
  }
  proc join(node: shared GeneNode, delta: deltaRecord, hstring: ygglog.yggHeader) {
    this.__join__(node, delta, hstring);
  }

  proc __join__(node: shared GeneNode, delta: deltaRecord, hstring: ygglog.yggHeader) {
    // did I call that function correctly?
    //writeln(node, delta);
    var vstring: ygglog.yggHeader;
    vstring = hstring + '__join__';
    this.l.wl(vstring);
    node.l.wl(vstring);
    var d = (this.id, node.id);
    node.nodes.add(this.id);
    this.nodes.add(node.id);
    this.edges[node.id] = new shared GeneEdge(delta, d);
    // Now, reverse the delta.  Which we can do by multiplying it by
    // -1.
    d = (node.id, this.id);
    node.edges[this.id] = new shared GeneEdge(delta*-1, d);
    node.l.uwl(vstring);
    this.l.uwl(vstring);
  }

  proc return_edge(id:string) {
    return this.__returnEdge__(id, hstring='');
  }

  proc return_edge(id:string, hstring: ygglog.yggHeader) {
    return this.__returnEdge__(id, hstring);
  }

  proc __returnEdge__(id: string, hstring: ygglog.yggHeader) {
    var vstring: ygglog.yggHeader;
    vstring = hstring + '__returnEdge__';
    this.l.rl(vstring);
    var d: deltaRecord;
    if this.node_in_edges(id) {
      d = this.edges[id];
    }
    this.l.url(vstring);
    return d;
  }

  proc new_node(seed: int, coefficient: real, id='': string) {
    return this.__newNode__(seed, coefficient, id, hstring='');
  }

  proc new_node(seed: int, coefficient: real, id='': string, hstring: ygglog.yggHeader) {
    return this.__newNode__(seed, coefficient, id, hstring);
  }

  proc __newNode__(seed: int, coefficient: real, id='': string, hstring: ygglog.yggHeader) {
    // This function is a generic call for whenever we make a modification
    // Mutations, adding a new seed, whatever.  We just create a new node
    // and join them properly.
    var node = new shared GeneNode(id=id);
    var delta = new deltaRecord();

    node.parentSeedNode = this.parentSeedNode;
    node.parent = this.id;
    node.log = this.log;
    node.l.log = this.log;
    // Not sure whether I always want to do this, but hey.
    node.generation = this.generation + 1;

    delta.delta[seed] = coefficient;
    var vstring: ygglog.yggHeader;
    vstring = hstring + '__newNode__';
    this.join(node, delta, vstring);
    return node;
  }
}
