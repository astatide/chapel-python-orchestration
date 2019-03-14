// Do I need all this?  Probably not but I'll sort that issue later.

use rng;
use genes;
use network;
use uuid;

config const mSize = 20;
config var maxPerGeneration = 400;
config var mutationRate = 0.03;
config var maxValkyries = 1;
config var startingSeeds = 10;

// As we have our tree of life, so too do we have winged badasses who choose
// who lives and who dies.
// (this is a record for a worker class)
record valkyrie {
  var matrixValues: [0..mSize-1] real;
  //var matrixValues: [0..20] real;
  //var matrixValues: int;
  var currentNode: string;
  // Is there anything else I need?  Who knows!
  proc moveToRoot() {
    // Zero out the matrix, return the root id.
    this.matrixValues = 0;
    this.currentNode = 'root';
  }

  // Assume we're given a delta object so that we may express it.
  proc move(delta: deltaRecord, id: string) {
    delta.express(this.matrixValues);
    this.currentNode = id;
  }
}

class Propagator {
  // this is going to actually hold all the logic for running EvoCap.
  var generation: int;
  //var maxValkyries: int;

  // now a few globals.
  var processedArray: [1..maxPerGeneration] atomic bool;
  var nodesToProcess: [1..maxPerGeneration] string;
  var inCurrentGeneration: atomic int;

  var ygg: shared network.GeneNetwork();

  proc init() {
    this.ygg = new shared network.GeneNetwork();
    this.ygg.initializeNetwork(n_seeds=startingSeeds);
    this.processedArray.write(true);
    var ids = this.ygg.ids;
    ids.remove('root');
    //this.inCurrentGeneration.add(1);
    for i in ids {
      this.nodesToProcess[this.inCurrentGeneration.read()+1] = i;
      this.processedArray[this.inCurrentGeneration.read()+1].write(false);
      this.inCurrentGeneration.add(1);
    }
    //writeln(this.ygg.nodes);
    writeln(this.inCurrentGeneration);
  }

  proc run() {
    // start up the main procedure by creating some valkyries.
    //var v: new valkyrieRecord;
    // declaring this variable seems to just kill it.  Entirely.  WHY.
    //for i in 1..maxValkyries {
    //  for j in this.ygg.ids {
    //    writeln(j);
    //  }
    //forall i in 1..maxValkyries with ( var v: valkyrie ) {
    forall i in 1..maxValkyries {
      //writeln(v.matrixValues);
      var v = new valkyrie;
      v.moveToRoot();
      for (processed, id) in zip(this.processedArray, this.nodesToProcess) {
        // This blocks while it reads the value and sets it to true, then returns
        // said original value.  No race conditions!
        if !processed.testAndSet() {
          // Actually do something.
          writeln('I am task ', i, ' and I am going to move from node ', v.currentNode, ' to ', id);
          if id != 'root' {
            //writeln(this.ygg.calculatePath('root', id));
          }
          //writeln(this.ygg.edges);
          this.ygg.move(v, id);
          writeln('TASK ', i, ', SEED # ', this.ygg.nodes[id].debugOrderOfCreation, ' : ', v.matrixValues);
          //var d = this.ygg.moveToNode(v.currentNode, id);
          //v.move(v, id);
          //processed.write(true);
        }
      }
    }
  }

}
