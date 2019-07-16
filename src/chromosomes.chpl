// by Audrey Pratt

use uuid;
use Random;
use spinlock;
use genes;
use Sort;
use network;

// Global UUID, with an entropy lock, guarantees unique IDs for chromosomes.
// Keep in mind, it's possible that since we're using two UUID generators
// (one for genes, the other here) that the UUID generators may produce the same
// UUID.  Individual generators are locked to ensure that a given UUID stream
// is unique, but there's nothing stopping two UUID generators from pulling
// from entropy at the same time (this kept happening before I locked it).
// However, as the chromosome and gene domain are unique, we don't need uniqueness
// between the sets, and so a gene and chromosome sharing the same UUID is fine.
//var CUUID = new owned uuid.UUID();
//CUUID.UUID4();

//var AssocDom:domain(int);
//type AssocArrayType = [AssocDom] int;


/*

  A chromosome is essentially a set of gene (nodes) that has a known order.

*/

record array {
  var n: domain(int);
  var j: [0..0] int;

  //proc dsiLow {
  //  return j.domain.low;
  //}

  iter these_die() {
    //for i in 0..this.n.size-1 {
    //  if this.n.contains(i) {
    //    yield (i, this.j[i]);
    //  }
    //}
    var z: int;
    //sort(this.j);
    for i in 0..this.j.domain.size {
      //writeln(i);
      if this.j.domain.contains(i) {
        yield (z, this.j[i]);
      }
      z += 1;
    }
  }

}

record seedSet {
  var s: domain(int);
  var sA: [s] int;

  iter these() {
    for i in this.s {
      yield this.sA[i];
    }
  }

  proc this(a: int) {
    return this.sA[a];
  }

  proc add(a: int) {
    this.s.add(a);
    this.sA[a] += 1;
  }

}

record geneCombo {
  var geneNumbers: domain(int);
  var actualGenes: [geneNumbers] domain(int);
  var nSize: int;

  proc init(n: int) {
    this.nSize = n;
  }

  proc prep() {
    var l = this.GeneOrderListFix(nSize);
    var n: int = 1;
    for i in 1..nSize {
      var m = l[nSize,i].j;
      sort(m);
      for z in m {
        geneNumbers.add(n);
        actualGenes[n] = DNA(z);
        n += 1;
      }
    }
  }

  proc clone() {
    var gCC = new geneCombo(this.nSize);
    for i in this.geneNumbers {
      gCC.geneNumbers.add(i);
      for z in this.actualGenes[i] {
        gCC.actualGenes[i].add(z);
      }
    }
    return gCC;
  }

  proc DNA(a: int) {
    var q: int;
    // Can't be longer than that!
    var code: [1..64] int;
    var indexSet: domain(int);
    // by taking the modulo with 2, we generate unique combinations
    // (that are made non-unique when combined with other combinations)
    var modulo: int = 2;
    q = a;
    var i = 1;
    if q/modulo > 0 {
      while q/modulo > 0 {
        code[i] = q % modulo;
        q = q/modulo;
        i += 1;
      }
    }
    code[i] = q % modulo;
    for j in 1..i {
      if code[j] > 0 {
      indexSet.add(j);
      }
    }
    return indexSet;
  }
  proc GeneOrderListFix(k: int) {
    // This is a mapping function that, given an index, returns a set in
    // a sensible manner.
    var d1: domain(int);
    var d2: domain(int);
    var d3: domain(int);
    var l: [0..k,0..k] array;
    var n: int;

    for i in 0..k {
      for j in 0..k {
        l[i,j].j.domain.clear();
      }
      l[i,0].j.push_back(0);
    }
    for i in 1..k {
      for j in 1..i {
        for s in l[i - 1, j - 1].j {
          l[i,j].j.push_back((s * 2) + 1);
        }
        for s in l[i - 1, j].j {
          l[i,j].j.push_back(s * 2);
        }
      }
    }
    return l;
  }
}


record Chromosome {
  // Honestly, the chromosome just needs to be a record, with maybe a few methods.
  // who knows, really!
  var id: string;

  // These are going to allow us to set up the actual chromosome list.
  var nRootGenes: int = propagator.startingSeeds;
  var nFunctionGenes: int = propagator.chromosomeSize - propagator.startingSeeds;
  var totalGenes: int = propagator.chromosomeSize;
  var currentGenes: int;
  var l: shared spinlock.SpinLock;
  var log: shared ygglog.YggdrasilLogging;
  var lowestIsBest: bool=false;
  var currentDeme: int = 0;
  var isProcessed: atomic bool = false;

  // these are just the genes
  var geneNumbers: domain(int);
  var geneIDSet: domain(string);
  var geneIDs: [geneNumbers] string;
  var geneSeeds: [geneNumbers] int;
  var scores: [geneNumbers] real;

  // shadow genes.  These are the nodes that represented the previous state;
  // this is particularly useful for merge operations, as we have _already_
  // done a lot of the work to merge genes.
  var shadowGenes: [geneNumbers] string;
  //var shadowSeeds: [geneNumbers] int;
  var combinations = new geneCombo(propagator.startingSeeds);

  //var geneNumbers: domain(int);
  var actualGenes: [geneNumbers] domain(int);

  proc init() {
    //this.complete();
    //var CUUID = new owned uuid.UUID();
    //this.id = '%04i'.format(here.id) + "-CHRO-" + CUUID.UUID4();
    this.l = new shared spinlock.SpinLock();
    this.l.t = ' '.join('CHROMOSOME', this.id);
    this.log = new shared ygglog.YggdrasilLogging();
    this.complete();
    this.l.log = this.log;
  }

  proc init(id : string) {
    //this.complete();
    //var CUUID = new owned uuid.UUID();
    this.id = id;
    this.l = new shared spinlock.SpinLock();
    this.l.t = ' '.join('CHROMOSOME', id);
    this.log = new shared ygglog.YggdrasilLogging();
    this.complete();
    this.l.log = this.log;
  }

  proc prep(nRootGenes: int, nFunctionGenes: int) {
    this.combinations.prep();
    // Add in the root node.
    this.geneNumbers.add(0);
    this.geneIDs[0] = 'root';
    for i in this.combinations.geneNumbers {
      this.geneNumbers.add(i);
      for z in this.combinations.actualGenes[i] {
        this.actualGenes[i].add(z);
      }
    }
    //writeln("Genes!");
    //for i in this.geneNumbers {
    //  writeln(i : string, " ", this.actualGenes[i] : string);
    //}
    this.nRootGenes = nRootGenes;
    this.nFunctionGenes = nFunctionGenes;
    //this.totalGenes = this.nRootGenes + this.nFunctionGenes;
    this.totalGenes = this.combinations.geneNumbers.size;
  }

  proc add(i: int, geneId: string) {
    this.l.wl();
    if !this.geneNumbers.contains(i) {
      this.geneNumbers.add(i);
    }
    this.geneIDs[i] = geneId;
    this.l.uwl();
  }

  proc add(geneId: string) {
    this.l.wl();
    this.currentGenes += 1;
    this.geneNumbers.add(this.currentGenes);
    this.geneIDs[this.currentGenes] = geneId;
    this.l.uwl();
  }

  proc this(a: int) {
    return this.geneIDs[a];
  }

  iter these() {
    for i in 1..totalGenes {
      yield this.geneIDs[i];
    }
  }

  iter geneSets() {
    for i in 1..totalGenes {
      yield this.actualGenes[i];
    }
  }

  proc __generateNodes__(ref nG: shared network.networkGenerator, initial=false, hstring: ygglog.yggHeader) {
    // chromosomes should build nodes according to their desires.
    // first, prep all the initial nodes.
    var vstring = hstring + '__generateNodes__';
    if initial {
      //this.log.debug("Initializing nodes.", hstring=vstring);
    }
    for n in 1..totalGenes {
      if !this.geneNumbers.contains(n) {
        this.geneNumbers.add(n);
      }
      var id = nG.getNode();
      network.globalLock.rl();
      ref node = network.globalNodes[id];
      network.globalLock.url();
      node.revision = genes.SPAWNED;
      if n > 0 && n <= this.nRootGenes {
        // prep the root seeds.
        // handy function to return a node id.
        //this.log.debug('Getting seeds and ID', hstring=vstring);
        var seed = nG.newSeed();
        //this.log.debug('ID: %s, SEED: %i'.format(id, seed), hstring=vstring);
        if initial {
          //this.log.debug("INITIAL GO; adding seed to root.", hstring=vstring);
          network.globalLock.rl();
          ref rootNode = network.globalNodes[nG.root];
          network.globalLock.url();
          node.addSeed(seed = seed, cId = this.id, deme = this.currentDeme, node = rootNode);
        } else {
          //this.log.debug("NOT INITIAL; advancing old node.", hstring=vstring);
          network.globalLock.rl();
          ref oldNode = network.globalNodes[this.geneIDs[n]];
          network.globalLock.url();
          node.addSeed(seed = seed, cId = this.id, deme = this.currentDeme, node = oldNode);
          oldNode.revision = genes.FINALIZED;
        }
        //this.log.debug('Node successfully advanced', hstring=vstring);
        this.geneIDs[n] = id;
        this.add(n, id);
        this.geneSeeds[n] = seed;
        node.chromosomes.add(this.id);
        node.chromosome = this.id;
        node.combinationID = n : string;
        this.geneIDSet.add(id);
      }
      if n > this.nRootGenes {
        // now we use the combo.  We should pack it into a list and send it.
        // is there a better way?  I'm sure.
        //this.log.debug('COMBINATION GENES', hstring=vstring);
        var c = this.actualGenes[n];
        //var idList: [1..c.size] string;
        var idList: [1..c.size] string;
        var seedList: [1..c.size] int;
        var i: int = 1;
        var combLabel: string = '(';
        for oldId in c {
          idList[i] = this.geneIDs[oldId];
          seedList[i] = this.geneSeeds[oldId];
          i += 1;
          combLabel += oldId : string + ',';
        }
        combLabel += ')';
        //this.log.debug('Getting node!', hstring=vstring);
        if initial {
          //this.log.debug('Calling combination node', hstring=vstring);
          node.newCombinationNode(idList, seedList, this.currentDeme, nG.root, network.globalNodes);
        } else {
          //this.log.debug('Calling combination node', hstring=vstring);
          node.newCombinationNode(idList, seedList, this.currentDeme, this.geneIDs[n], network.globalNodes);
          // finalize it.
          network.globalNodes[this.geneIDs[n]].l.wl();
          network.globalNodes[this.geneIDs[n]].revision = genes.FINALIZED;
          network.globalNodes[this.geneIDs[n]].l.uwl();
        }
        //this.log.debug('Combination complete; setting node', hstring=vstring);
        this.geneIDs[n] = id;
        this.add(n, id);
        node.chromosomes.add(this.id);
        node.chromosome = this.id;
        node.combinationID = n : string;
        this.geneIDSet.add(id);
      }
    }
  }

  proc generateNodes(ref nG: shared network.networkGenerator, hstring: ygglog.yggHeader) {
    this.__generateNodes__(nG, initial=true, hstring=hstring);
  }

  proc advanceNodes(ref nG: shared network.networkGenerator, hstring: ygglog.yggHeader) {
    this.__generateNodes__(nG, initial=false, hstring=hstring);
  }

  proc clone () {
    /*
    var nRootGenes: int = propagator.startingSeeds;
    var nFunctionGenes: int = propagator.chromosomeSize - propagator.startingSeeds;
    var totalGenes: int = propagator.chromosomeSize;
    var currentGenes: int;
    var l: shared spinlock.SpinLock;
    var log: shared ygglog.YggdrasilLogging;
    var lowestIsBest: bool=false;
    var currentDeme: int = 0;
    var isProcessed: atomic bool = false;

    // these are just the genes
    var geneNumbers: domain(int);
    var geneIDSet: domain(string);
    var geneIDs: [geneNumbers] string;
    var geneSeeds: [geneNumbers] int;
    var scores: [geneNumbers] real;
    */
    // This is the copy operator, right?
    var b = new Chromosome();
    b.log = this.log;
    b.nRootGenes = this.nRootGenes;
    b.nFunctionGenes = this.nFunctionGenes;
    b.totalGenes = this.totalGenes;
    b.currentGenes = this.currentGenes;
    b.combinations = this.combinations.clone();
    b.lowestIsBest = this.lowestIsBest;
    b.currentDeme = this.currentDeme;
    // Unsure if this is a pointer or a copy, but.
    //for i in this.combinations.geneNumbers {
    //  b.geneNumbers.add(i);
    //  for z in this.combinations.actualGenes[i] {
    //    b.actualGenes[i].add(z);
    //  }
    //}
    for i in this.geneNumbers {
      b.geneNumbers.add(i);
      b.geneIDs[i] = this.geneIDs[i];
      b.scores[i] = this.scores[i];
      b.geneSeeds[i] = this.geneSeeds[i];
      for z in this.actualGenes[i] {
        b.actualGenes[i].add(z);
      }
    }
    return b;
  }

  /*
  Generates a unique set of combinations.  Given an index, will return the gene
  IDs that correspond to this particular combination.

  Generally, don't call this.
  */

  proc bestGene(ygg: network.GeneNetwork) {
    var bestNode: string;
    var bestScore: real = 0;
    if true {
      for i in 1..totalGenes {
        var score = this.geneIDs[i].scores[this.currentDeme];
        if score > bestScore {
          bestScore = score;
          bestNode = this.geneIDs[i];
        }
      }
    }
    return (bestNode, bestScore);
  }

  proc bestGeneInDeme(deme=0) {
    var bestNode: string;
    // Are scores high or low?  I guess it depends on our metric.  Blah.
    var bestScore: real = 0;
    if this.lowestIsBest {
      bestScore = Math.INFINITY : real;
      for i in 1..totalGenes {
        // Actually, I should make this a function on the gene.
        // That way we can handle the locking appropriately.
        //if this.geneIDs[i].demeDomain.contains(deme) {
          if this.scores[i] < bestScore {
            bestScore = this.scores[i];
            bestNode = this.geneIDs[i];
          }
        //}
      }
    } else {
      for i in 1..totalGenes {
        //if this.geneIDs[i].demeDomain.contains(deme) {
          if this.scores[i] < bestScore {
            bestScore = this.scores[i];
            bestNode = this.geneIDs[i];
          }
        //}
      }
    }
    return (bestScore, bestNode);
  }

  proc returnNodeNumber(node: string) {
    for i in this.geneNumbers {
      //assert(this.geneIDs.contains(i));
      if this.geneIDs[i] == node {
        return i;
      }
    }
    return -1;
  }

}

proc +=(ref a: Chromosome, b: string) {
  a.currentGenes += 1;
  a.geneNumbers.add(a.currentGenes);
  a.geneIDs[a.currentGenes] = b;
}
