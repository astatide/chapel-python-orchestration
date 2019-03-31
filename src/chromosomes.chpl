// by Audrey Pratt

use uuid;
use Random;
use spinlock;
use genes;

// Global UUID, with an entropy lock, guarantees unique IDs for chromosomes.
// Keep in mind, it's possible that since we're using two UUID generators
// (one for genes, the other here) that the UUID generators may produce the same
// UUID.  Individual generators are locked to ensure that a given UUID stream
// is unique, but there's nothing stopping two UUID generators from pulling
// from entropy at the same time (this kept happening before I locked it).
// However, as the chromosome and gene domain are unique, we don't need uniqueness
// between the sets, and so a gene and chromosome sharing the same UUID is fine.
var UUID = new owned uuid.UUID();
UUID.UUID4();

/*

  A chromosome is essentially a set of gene (nodes) that has a known order.

*/

record Chromosome {
  // Honestly, the chromosome just needs to be a record, with maybe a few methods.
  // who knows, really!
  var id: string;

  // These are going to allow us to set up the actual chromosome list.
  var nRootGenes: int;
  var nFunctionGenes: int;
  var totalGenes: int;
  var currentGenes: int;
  var l: shared spinlock.SpinLock;
  var log: shared ygglog.YggdrasilLogging;

  var lowestIsBest: bool=true;

  // these are just the genes
  var geneNumbers: domain(int);
  var geneIDs: [geneNumbers] string;

  // shadow genes.  These are the nodes that represented the previous state;
  // this is particularly useful for merge operations, as we have _already_
  // done a lot of the work to merge genes.
  var shadowGenes: [geneNumbers] string;

  proc init(id='') {
    if id == '' {
      this.id = UUID.UUID4();
    } else {
      this.id = id;
    }
    this.l = new shared spinlock.SpinLock();
    this.l.t = ' '.join('CHROMOSOME', this.id);
  }

  proc prep(nRootGenes: int, nFunctionGenes: int) {
    this.nRootGenes = nRootGenes;
    this.nFunctionGenes = nFunctionGenes;
    this.totalGenes = this.nRootGenes + this.nFunctionGenes;
    // Add in the root node.
    this.geneNumbers.add(0);
    this.geneIDs[0] = 'root';

  }

  proc add(i: int, geneId: string) {
    if !this.geneNumbers.contains(i) {
      this.geneNumbers.add(this.currentGenes);
    }
    this.geneNumbers[this.currentGenes] = geneId;
  }

  proc add(geneId: string) {
    this.currentGenes += 1;
    this.geneNumbers.add(this.currentGenes);
    this.geneIDs[this.currentGenes] = geneId;
  }

  proc this(a: int) {
    return this.geneIDs[a];
  }

  iter these() {
    for i in 1..totalGenes {
      yield this.geneIDs[i];
    }
  }

  // This is an initialization function; we generate a set of instructions and
  // nodes such that we can generate the appropriate nodes.
  // yields: cType, gene position, and what genes to operate on, if necessary.
  iter generateGeneInstructions() {
    var alreadyDone: domain(string);
    for i in 1..nRootGenes {
      // last field not used here.
      yield (0, i, 0, 0);
    }
    // We want combinations of all seeds, including ungenerated ones, until
    // we have reached the total number of genes.
    // Since these are merge functions, we want n genes, where n^2 = nFunctionGenes.
    // Ergo, n = sqrt(nFunctionGenes); this is the number of combinations we can do.
    // We'll exclude ones that we've already done, however.
    var z: int = this.nRootGenes;
    //while z <= this.totalGenes {
    // ehhhhhh it'll get us there for the moment.  Won't quite produce the right
    // amount since I'm too lazy.
    for i in 1..sqrt(this.nFunctionGenes) : int {
      for j in i+1..sqrt(this.nFunctionGenes) : int {
        if i != j {
          z += 1;
          yield (1, z, i, j);
        }
      }
    }
  }

  proc bestGeneInDeme(deme='') {
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

proc =(ref a: Chromosome) {
  // This is the copy operator, right?
  var b = new Chromosome();
  b.nRootGenes = a.nRootGenes;
  b.nFunctionGenes = a.nFunctionGenes;
  b.totalGenes = a.totalGenes;
  b.currentGenes = a.currentGenes;
  // Unsure if this is a pointer or a copy, but.
  b.geneNumbers = a.geneNumbers;
  b.geneIDs = a.geneIDs;
  return b;
}

proc +=(ref a: Chromosome, b: string) {
  a.currentGenes += 1;
  a.geneNumbers.add(a.currentGenes);
  a.geneIDs[a.currentGenes] = b;
}
