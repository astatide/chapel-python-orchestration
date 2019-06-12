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
var CUUID = new owned uuid.UUID();
CUUID.UUID4();

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

  // these are just the genes
  var geneNumbers: domain(int);
  var geneIDs: [geneNumbers] string;

  // shadow genes.  These are the nodes that represented the previous state;
  // this is particularly useful for merge operations, as we have _already_
  // done a lot of the work to merge genes.
  var shadowGenes: [geneNumbers] string;
  var combinations = new geneCombo(propagator.startingSeeds);

  //var geneNumbers: domain(int);
  var actualGenes: [geneNumbers] domain(int);

  proc init() {
    this.id = CUUID.UUID4();
    this.l = new shared spinlock.SpinLock();
    this.l.t = ' '.join('CHROMOSOME', this.id);
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
    writeln("Genes!");
    for i in this.geneNumbers {
      writeln(i : string, " ", this.actualGenes[i] : string);
    }
    this.nRootGenes = nRootGenes;
    this.nFunctionGenes = nFunctionGenes;
    //this.totalGenes = this.nRootGenes + this.nFunctionGenes;
    this.totalGenes = this.combinations.geneNumbers.size;
  }

  proc add(i: int, geneId: string) {
    this.l.wl();
    if !this.geneNumbers.contains(i) {
      this.geneNumbers.add(this.currentGenes);
    }
    this.geneNumbers[this.currentGenes] = geneId;
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

  // This is an initialization function; we generate a set of instructions and
  // nodes such that we can generate the appropriate nodes.
  // yields: cType, gene position, and what genes to operate on, if necessary.
  iter generateGeneInstructions() {
  }

  proc generateNodes(ref ygg: shared network.GeneNetwork) {
    // chromosomes should build nodes according to their desires.
    // first, prep all the initial nodes.
    //var n: int;
    //for c in this.geneSets() {
    writeln("generateNodes ", totalGenes : string, " ", this.nRootGenes);
    for n in 1..totalGenes {
      writeln(n);
      var c = this.actualGenes[n];
      //select n {
        //if n == 0 do {} // yeah, go home!  No one likes you!
        if n > 0 && n <= this.nRootGenes {
          // prep the root seeds.
          //this.geneNumbers.add(n);
          writeln("NEW SEED GENE");
          this.geneIDs[n] = ygg.newSeedGene(this.id, this.currentDeme);
        }
        if n > this.nRootGenes {
          // now we use the combo.  We should pack it into a list and send it.
          // is there a better way?  I'm sure.
          var idList: [1..c.size] string;
          var i: int = 1;
          for id in c {
            idList[i] = this.geneIDs[id];
            i += 1;
          }
          //this.geneNumbers.add(n);
          writeln("MERGE GENE!");
          this.geneIDs[n] = ygg.mergeNodeList(this.id, idList, this.currentDeme);
        }
      //}
      //n += 1;
    }
    //writeln(this.geneNumbers)
  }

  proc advanceNodes(ref ygg: shared network.GeneNetwork) {
    // chromosomes should build nodes according to their desires.
    // first, prep all the initial nodes.
    //var n: int;
    //for c in this.geneSets() {
    //writeln(this.combinations : string);
    //for d in 1..totalGenes {
    //for c in this.geneSets() {
      //var c = this.combinations.actualGenes[d];
    for n in 1..totalGenes {
      var c = this.actualGenes[n];
      //select n {
        //if n == 0 do {} // yeah, go home!  No one likes you! (this is always root)
        if n > 0 && n <= this.nRootGenes {
          // Further the nodes!
          //this.geneNumbers.add(n);
          this.geneIDs[n] = ygg.nextNode(cId=this.id, id=this.geneIDs[n], deme=this.currentDeme);
        }
        if n > this.nRootGenes {
          // now we use the combo.  We should pack it into a list and send it.
          // is there a better way?  I'm sure.
          var idList: [1..c.size] string;
          var i: int = 1;
          for id in c {
            idList[i] = this.geneIDs[id];
            i += 1;
          }
          // it's fine for now.
          //this.geneNumbers.add(n);
          this.geneIDs[n] = ygg.mergeNodeList(this.id, idList, this.currentDeme);
        }
      //}
      //n += 1;
    }
  }

  /*
  var nRootGenes: int = propagator.startingSeeds;
  var nFunctionGenes: int = propagator.chromosomeSize - propagator.startingSeeds;
  var totalGenes: int = propagator.chromosomeSize;
  var currentGenes: int;
  var l: shared spinlock.SpinLock;
  var log: shared ygglog.YggdrasilLogging;
  var lowestIsBest: bool=false;
  var currentDeme: int = 0;

  // these are just the genes
  var geneNumbers: domain(int);
  var geneIDs: [geneNumbers] string;

  // shadow genes.  These are the nodes that represented the previous state;
  // this is particularly useful for merge operations, as we have _already_
  // done a lot of the work to merge genes.
  var shadowGenes: [geneNumbers] string;
  var combinations = new geneCombo(propagator.startingSeeds);
  */

  proc clone () {
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
    for i in this.geneNumbers {
      b.geneNumbers.add(i);
      b.geneIDs[i] = this.geneIDs[i];
      for z in this.actualGenes[i] {
        b.actualGenes[i].add(z);
      }
    }
    //b.id = CUUID.UUID4();
    //b.l = new shared spinlock.SpinLock();
    //b.l.t = ' '.join('CHROMOSOME', b.id);
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
        if this.geneIDs[i].demeDomain.contains(deme) {
          if this.geneIDs[i].scores[deme] < bestScore {
            bestScore = this.geneIDs[i].scores[deme];
            bestNode = this.geneIDs[i];
          }
        }
      }
    } else {
      for i in 1..totalGenes {
        if this.geneIDs[i].demeDomain.contains(deme) {
          if this.geneIDs[i].scores[deme] > bestScore {
            bestScore = this.geneIDs[i].scores[deme];
            bestNode = this.geneIDs[i];
          }
        }
      }
    }
  }
}

proc =(ref b: Chromosome, a: Chromosome) {
  // This is the copy operator, right?
  //var b = new Chromosome();
  b.nRootGenes = a.nRootGenes;
  b.nFunctionGenes = a.nFunctionGenes;
  b.totalGenes = a.totalGenes;
  b.currentGenes = a.currentGenes;
  // Unsure if this is a pointer or a copy, but.
  b.geneNumbers = a.geneNumbers;
  b.geneIDs = a.geneIDs;
  //return b;
}

proc +=(ref a: Chromosome, b: string) {
  a.currentGenes += 1;
  a.geneNumbers.add(a.currentGenes);
  a.geneIDs[a.currentGenes] = b;
}
