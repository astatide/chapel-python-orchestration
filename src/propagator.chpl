// Do I need all this?  Probably not but I'll sort that issue later.

use rng;
use genes;
use network;
use uuid;

config var matrixSize = 20;
config var maxPerGeneration = 400;
config var mutationRate = 0.03;
config var maxValkyries = 6;
config var startingSeeds = 10;

// As we have our tree of life, so too do we have winged badasses who choose
// who lives and who dies.
// (this is a record for a worker class)
record valkyrie {
  var matrixValues: [0..matrixSize-1] real;
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

  var ygg: network.GeneNetwork();

  proc init() {
    this.ygg = new owned network.GeneNetwork();
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
    writeln(this.ygg.ids);
    forall i in 1..maxValkyries with ( var v: valkyrie ) {
      //var v: new valkyrie;
      v.moveToRoot();
      //writeln(v.matrixValues);
      for (processed, id) in zip(this.processedArray, this.nodesToProcess) {
        // This blocks while it reads the value and sets it to true, then returns
        // said original value.  No race conditions!
        if !processed.testAndSet() {
          // Actually do something.
          writeln('I am task ', i, ' and I am going to move node ', id );
          this.ygg.move(v, id);
          writeln(v.matrixValues);
          //processed.write(true);
        }
      }
    }
  }

}
