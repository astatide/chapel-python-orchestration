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

const NEW = 0;
const SPAWNED = 1;
const FINALIZED = 3;

const DELTA = 0;
const PATH = 1;

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

  iter returnRandomSeed() {
    // THIS IS NOT EFFICIENT
    // @TODO: fix it, yo.
    var udevrandom = new owned rng.UDevRandomHandler();
    var newrng = udevrandom.returnRNG();
    var randomSelection = (newrng.getNext() * this.seeds.size) : int;
    var i: int = 0;
    for s in this.seeds {
      if i == randomSelection {
        return (s, this.delta[s]);
      }
      i += 1;
    }
  }

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

proc =(ref a: deltaRecord, b: deltaRecord) {
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
  var edgeType: int;
  var path: [1..0] (string, real);
  //var pathCoefficient: int = 1;
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

  proc init(idList: [] (string, real)) {
    // this means we're adding in a path.
    this.edgeType = PATH;
    //this.path: [1..idList.size] (string, real);
    //this.pathCoefficient = c;
    this.complete();
    for i in 1..idList.size {
      this.path.push_back(idList[i]);
    }
  }

  proc init(delta) {
    this.delta = delta;
    this.edgeType = DELTA;
  }

  proc init(delta, direction) {
    this.delta = delta;
    this.edgeType = DELTA;
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

  proc clone() {
    var rEdge = new shared GeneEdge();
    rEdge.delta = this.delta;
    rEdge.direction = (this.direction[1], this.direction[2]);
    rEdge.mu = this.mu;
    rEdge.sigma = this.sigma;
    rEdge.noise_function = this.noise_function;
    return rEdge;
  }

}

class GeneNode {
  // This is a node.  It contains the Chapel implementation of a hash table
  // (akin to a python dict); we're going to store the gene edges in it.
  var id: string = '0000-GENE-00000000-0000-0000-0000-000000000000';
  var nodes: domain(string);
  var edges: [nodes] shared GeneEdge;
  var generation: int;
  var ctype: string;
  var parent: string;
  // we need a node ID.  I like the ability of being able to specify them.
  // but we should generate them by default.
  var debugOrderOfCreation: int;

  // Here, we're gonna track our parent at history 0
  // should make it easier to return histories.
  var parentSeedNode: string;

  var l: shared spinlock.SpinLock;
  var log: shared ygglog.YggdrasilLogging;

  // We want to hold scores on the nodes.
  var demeDomain: domain(int);
  var chromosomes: domain(string);
  var chromosome: string;
  var combinationID: string;
  var scores: [demeDomain] real;

  var processedBy: string;
  var processedOrder: int;

  var initialized: atomic bool = false;

  var revision: int = NEW;

  proc init() {
    // Blank initializer so that we can check on it later.
    // do set the lock, though.
    this.l = new shared spinlock.SpinLock();
    this.l.t = ' '.join('GENE', this.id);
    this.complete();
    this.l.log = this.log;
  }

  proc init(id, ctype='', parent='', parentSeedNode='') {
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

  //proc writeThis(wc) {
  //  wc.writeln(this.id);
  //  wc.writeln('nodes: ', this.nodes : string);
  //  wc.writeln('edges: ', this.edges : string);
  //}

  // some validation functions
  proc nodeInEdges(id: string, hstring: ygglog.yggHeader) {
    // Well, okay, that turned out to be easy but whatever.
    this.l.rl(hstring);
    var ie = this.nodes.contains(id);
    this.l.url(hstring);
    return ie;
  }

  proc joinPaths(ref node: shared GeneNode, idList: [] string, deme: int) {
    // we're going to make path edge connections between all the wee nodes.
    // This is because we might not ever actually need them.  So just store
    // the information necessary to reconstruct a delta, if necessary.
    var vstring: ygglog.yggHeader;
    vstring += '__join__';
    var z: int = 1;
    var shortList: [1..0] (string, real);
    var revList: [1..0] (string, real);
    for i in idList {
      if i != node.id {
        //idList.remove(z);
        shortList.push_back((i, 1/(idList.size-1:real)));
        revList.push_back((i, -1/(idList.size-1:real)));
      }
      z += 1;
    }
    on this.locale {
      var e = new shared GeneEdge(shortList);
      e.direction = (this.id, node.id);
      this.l.wl(vstring);
      this.nodes.add(node.id);
      this.edges[node.id] = e;
      this.l.uwl(vstring);
    }
    on node.locale {
      var re = new shared GeneEdge(revList);
      re.direction = (node.id, this.id);
      node.l.wl(vstring);
      node.nodes.add(this.id);
      node.edges[this.id] = re;
      node.l.uwl(vstring);
    }
  }

  // Now, the functions to handle the nodes!
  proc join(ref node: shared GeneNode, delta: deltaRecord) {
    this.__join__(node, delta, hstring=new ygglog.yggHeader());
  }
  proc join(ref node: shared GeneNode, delta: deltaRecord, hstring: ygglog.yggHeader) {
    this.__join__(node, delta, hstring);
  }

  proc __join__(ref node: shared GeneNode, delta: deltaRecord, hstring: ygglog.yggHeader) {
    // did I call that function correctly?
    //writeln(node, delta);
    on this.locale {
      var vstring: ygglog.yggHeader;
      vstring = hstring + '__join__';
      this.l.wl();
      var d = (this.id, node.id);
      this.nodes.add(node.id);
      this.edges[node.id] = new shared GeneEdge(delta, d);
      this.l.uwl();
    }
    on node.locale {
      node.l.wl();
      var d = (node.id, this.id);
      node.nodes.add(this.id);
      node.edges[this.id] = new shared GeneEdge(delta*-1, d);
      node.l.uwl();
    }
  }

  proc returnEdge(id:string) {
    return this.__returnEdge__(id, hstring='');
  }

  proc returnEdge(id:string, hstring: ygglog.yggHeader) {
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

  iter returnEdgeIDs() {
    //on this.locale {
      this.l.rl();
      for edge in this.nodes {
        yield edge;
      }
      this.l.url();
    //}
  }

  iter returnDemes() {
    // do it on our locale.  For _us_.
    // I prefer to think of it as our locale.
    //on this.locale {
      this.l.rl();
      for deme in this.demeDomain {
        yield deme;
      }
      this.l.url();
    //}
  }

  proc setDeme(deme: int) {
    on this.locale {
      this.l.wl();
      this.demeDomain.add(deme);
      this.l.uwl();
    }
  }

  proc setValkyrie(id: string, n: int) {
    on this.locale {
      //this.l.wl();
      this.processedBy = id;
      this.processedOrder = n;
      //this.l.uwl();
    }
  }

  proc setGeneration(gen: int) {
    on this.locale {
      //this.l.wl();
      this.generation = gen;
      //this.l.uwl();
    }
  }

  proc setDemeScore(deme: int, score: real) {
    on this.locale {
      // probably don't need to lock it as we're not gonna do concurrent access.
      //this.l.wl();
      if !this.demeDomain.contains(deme) {
        this.demeDomain.add(deme);
      }
      this.scores[deme] = score;
      //this.l.uwl();
    }
  }

  proc returnRevision() {
    //on this.locale {
    var rev: int;
    this.l.rl();
    rev = this.revision;
    this.l.url();
    return rev;
    //}
  }

  proc addSeed(seed: int, cId: string, deme: int, ref node: shared GeneNode, mutantDelta: genes.deltaRecord = new genes.deltaRecord()) {
    on this.locale {
      this.ctype = 'seed';
      this.parentSeedNode = 'root';
      this.parent = node.id;

      var delta = new genes.deltaRecord();

      // It's reverse because we're creating the connection from the new node backwards;
      // ergo, you must _undo_ the change.
      delta += (seed, -1.0);
      delta += mutantDelta*-1;
      //node.l.wl();
      //node.demeDomain.add(deme);
      this.setDeme(deme);
      //this.chromosomes.add(cId);
      //this.chromosome = cId;
      //node.l.uwl();
      //node.join(this, delta, new ygglog.yggHeader() + 'newSeedGene');
      this.join(node, delta, new ygglog.yggHeader() + 'newSeedGene');
    }
  }

  proc newCombinationNode(idList, seedList, deme, oldId, ref gN, mutantDelta: genes.deltaRecord = new genes.deltaRecord()) {
    //node.newCombinationNode(idList, seedList, this.geneIDs[n], network.globalNodes);
    // so, for each seed in the seedlist, we add it to the oldId link node.
    on this.locale {
      var delta = new genes.deltaRecord();
      for s in seedList {
        delta += (s, -1.0);
      }
      delta += mutantDelta;
      delta /= idList.size;
      this.setDeme(deme);
      network.globalLock.rl();
      ref oldNode = gN[oldId];
      network.globalLock.url();
      this.parent = oldId;
      this.join(oldNode, delta, new ygglog.yggHeader() + "newCombinationNode");
      // now, we add 1/N of that to each other one.
      // except... that's pretty tough.  We'll calculate this stuff on the fly.
      // We _know_ that we have these connections, so.
      // also, this is broken.  So don't make these edges, yet.
      //for id in idList {
      //  network.globalLock.rl();
      //  ref oldNode = gN[id];
      //  network.globalLock.url();
      //  this.joinPaths(oldNode, idList, deme);
      //}
    }
  }

  proc clone() {
    var node = new shared GeneNode();
    node.id = this.id;
    node.debugOrderOfCreation = this.debugOrderOfCreation;
    node.ctype = this.ctype;
    node.parent = this.parent;
    node.generation = this.generation;
    node.parentSeedNode = this.parentSeedNode;
    for n in this.nodes {
      node.nodes.add(n);
      node.edges[n] = this.edges[n].clone();
    }
    for deme in this.demeDomain {
      node.demeDomain.add(deme);
      node.scores[deme] = this.scores[deme];
    }
    for c in this.chromosomes {
      node.chromosomes.add(c);
    }
    return node;
  }
}
