// by Audrey Pratt

use uuid;
use Random;
use spinlock;
use genes;
use Sort;

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
      this.id = CUUID.UUID4();
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

  /*
  Generates a unique set of combinations.  Given an index, will return the gene
  IDs that correspond to this particular combination.

  Generally, don't call this.
  */

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
    //var l: [0..k,0..k,0..this.factorial(k)] int;
    //var l: [d1,d2,d3] int;
    var l: [0..k,0..k] array;
    var n: int;


    for i in 0..k {
      for j in 0..k {
        l[i,j].j.domain.clear();
      }
      //l[i,0].n.add(0);
      l[i,0].j.push_back(0);
    }

    for i in 1..k {
      for j in 1..i {
        //writeln(i,j);
        for s in l[i - 1, j - 1].j {
          //l[i,j].n.add(z);
          //writeln(z,s);
          //n = l[i,j].n.size;
          //n = l[i,j].j.domain.size;
          //writeln(n);
          //l[i,j].j[n] = (s * 2) + 1;
          l[i,j].j.push_back((s * 2) + 1);

        }
        //writeln(i,j);
        //writeln(l[i-1,j]);
        for s in l[i - 1, j].j {
          //l[i,j].n.add(z);
          //writeln(z,s, i, j);
          //n = l[i,j].n.size;
          //n = l[i,j].j.domain.size;
          //l[i,j].j[n] = s * 2;
          l[i,j].j.push_back(s * 2);

        }
        //writeln('who likes bread?');
        //writeln(i);
      }
    }
    writeln('DONE');
    return l;

    }

  // So the above does a lexicographical ordering, but is not what we want.
  proc DNA_gen(a: int) {
    var q: int;
    // Can't be longer than that!
    var code: [1..64] int;
    var indexSet: domain(int);


  }

  /* An ordering function which returns all combinations of order N */

  proc orderGenes(order: int) {
    var combSet: domain(int);
    var indices: [combSet] int;
    var q = (2**(order))-1;
    var seeds: int = 6;
    //combSet.add(1);
    //indices[1] = q;
    var orderSize = (this.factorial(seeds)/(this.factorial(order)*this.factorial(seeds-(order))));
    writeln(order, ' ', orderSize);
    for i in 0..orderSize-1 {
        combSet.add(i);
        indices[i] = q ^ 2**(order-1) ^ 2**(i+order-1);
        //writeln(indices[i]);
    }
    return indices;
  }

  proc getOrder(in a: int) {
    // This function calculates the number of bits that are equal to 1.
    // With the combinatorial algorithm we're using, this determines the
    // order of the function.
    var sum: int;
    while a != 0 {
      sum += 1;
      a &= (a - 1);
    }
    return sum;
  }

  // taken from the primers
  proc factorial(x: int) : int
  {
  if x < 0 then
    halt("factorial -- Sorry, this is not the gamma procedure!");

  return if x == 0 then 1 else x * factorial(x-1);
  }

  /*

  Generates the actual root seed set that a particular combination
  corresponds to.  Given an index, this returns which seeds, and what their
  corresponding combinations, should be.

  */

  iter DNAList() {
    var seeds: int = 6;
    for order in 1..seeds {
      var geneSet = this.orderGenes(order);
      for s in 0..geneSet.domain.size-1 {
        yield geneSet[s];
      }
    }
  }

  proc DNASetGenerator(a: int, first: bool) : seedSet {
    var coefSet = new seedSet();
    // we're doing a sort on this basically.
    // we have n choose k for each level.
    var order: int;
    var p: int = 0;
    var lp: int = 0;
    var seeds: int = 5;
    var diff: int;
    var iD: int;
    if first {
      while a > p {
        order += 1;
        //if seeds < order {
        //  break;
        //}
        lp = (this.factorial(seeds)/(this.factorial(order)*this.factorial(seeds-order)));
        diff = a - p - 1;
        p += lp;
      }
      if order-1 > seeds {
        // don't do this.
      }
      var orderFunctions = this.orderGenes(order);
      iD = orderFunctions[diff];
      // Now that we know our order...
      //var z = this.orderGenes(a);
    } else {
      iD = a;
    }
    //writeln(a, ', ', diff, ' ', order);
    //writeln(orderFunctions, ' : ', orderFunctions.domain);
    for i in this.DNA(iD) {
      if i > seeds {
        writeln(i, ' ', this.DNA(i), ' ', iD, ' ', this.DNA(iD), ' DNASetGenerator ', this.DNASetGenerator(i, false));
        //var j_coef = this.DNASetGenerator(i, false);
        var j_coef = this.DNASetGenerator(i, false);
        for j in j_coef.s {
          coefSet.add(j);
        }
      } else {
        coefSet.add(i);
      }
    }
  return coefSet;
  }

  // This is similar to the above, but is for pulling out what _type_
  // of node we are
  proc DNA_old(a: int) : 4*int {
    var test = false;
    var alreadyDone: domain(string);
    if a <= nRootGenes {
      return (0, a, a, a);
    } else {
      var functionOrder: int = 1;
      var y: int = 0;
      var z: int = this.nRootGenes;
      var shadowZ: int = this.nRootGenes;
      while z < this.totalGenes {
        for i in 1..shadowZ : int {
          if (i > this.nRootGenes || functionOrder == 1)
           {
            for j in i..shadowZ : int {
              if i != j {
                z += 1;
                assert(z > i);
                assert(z > j);
                if z == a {
                  if test {
                    return (1, z, this.DNA(i)[2], this.DNA(j)[2]);
                  } else {
                    return (1, z, i, j);
                  }
                }
              }
            }
          } else {
            for j in y+1..shadowZ : int {
              if i != j {
                z += 1;
                assert(z > i);
                assert(z > j);
                if z == a {
                  if test {
                    return (1, z, this.DNA(i)[2], this.DNA(j)[2]);
                  } else {
                    return (1, z, i, j);
                  }
                }
              }
            }
          }
        }
        y = shadowZ;
        shadowZ = z;
        functionOrder += 1;
      }
    }
    return (-1, 0, 0, 0);
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
