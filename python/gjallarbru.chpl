// This is the Chapel to C bit.

// We're gonna create a main then try and call the python C bits.

use propagator;
use Math;

extern type PyObject;
extern type npy_intp;
extern type PyThreadState;

extern proc returnNumpyArray(ref arr: c_float, ref dims: npy_intp) : PyObject;
extern proc weights(ref self: PyObject, ref args: PyObject): PyObject;
extern proc run();
extern proc pythonRun(arr: [] c_double, valkyrie: c_ulonglong, ref score : c_double, ref buffer : c_string) : real;
extern proc pythonInit(n: c_ulonglong): c_void_ptr;
extern proc pythonFinal();
extern proc newThread() : c_void_ptr;
extern proc PyThreadState_Swap(thread: c_void_ptr) : c_void_ptr;
extern proc PyGILState_Ensure(): c_void_ptr;
extern proc PyGILState_Release(lock: c_void_ptr);
extern proc garbageCollect(valkyrie: c_ulonglong);
//extern proc
//extern proc pythonRun();

require "gjallarbru.c";
//require "python/test.cpp";

class Gjallarbru {

  var threads: [1..propagator.maxValkyries] c_void_ptr;
  var runGC: atomic bool;
  var roundsProcessed: [1..propagator.maxValkyries] atomic int;
  var rounds: int = 0;
  var valkyriesDone: atomic int;
  var valkyriesUnblocked: atomic int;
  var moveOn: sync bool;
  var log: shared ygglog.YggdrasilLogging;
  //var ready: [1..propagator.maxValkyries] atomic bool;

  proc pInit() {
    // return the sub-interpreter
    this.log = new shared ygglog.YggdrasilLogging();
    this.log.currentDebugLevel = propagator.debug;
    this.threads = pythonInit(propagator.maxValkyries : c_ulonglong);
    //return threads;
  }

  proc gc() {
    garbageCollect(0 : c_ulonglong);
  }

  proc newInterpreter() {
    return newThread();
  }

  proc final() {
    pythonFinal();
  }

  proc createTestArray(l: int, d: int) {
    var length: c_ulonglong = l : c_ulonglong;
    var nd: c_ulonglong = d : c_ulonglong;
    var arr: [0..(length**nd)] c_double = 0;
    return arr;
  }

  proc createDimsArray(l: int, d: int) {
    var length: c_ulonglong = l : c_ulonglong;
    var nd: c_ulonglong = d : c_ulonglong;
    var dims: [0..nd-1] c_ulonglong = length;
    return dims;
  }

  proc testRun() {
    // Does almost seem to work?
    pythonRun(createTestArray(20, 2), 2 : c_ulonglong, createDimsArray(20, 2));
  }

  //       gjallarbru.pythonRun(v.matrixValues, 1 : c_ulonglong, gjallarbru.createDimsArray(mSize, 1));

  proc lockAndRun(matrix, valkyrie, hstring) {
    // This is just some bullshit to make us thread safe, I guess.
    //var gil = PyGILState_Ensure();
    // We're sending in a pointer and then writing to it.  Seems to work more cleanly.
    // or that's the hope.  Who fucking knows anymore.
    //this.log.debug('Launching and running Python code.', hstring=hstring);
    // We're gonna pass in the file object.
    //var score: real;
    var score2: [0..1] real = Math.INFINITY;
    var buffer : [0..1023] c_string;
    var moveOn: [0..1] bool = false;
    //string lf = ('logs/V-' + hstring.currentTask + '.log');
    //begin {
    var score: c_double;
    pythonRun(matrix, valkyrie : c_ulonglong, score, buffer[0]);
    var a: string;
    for i in 0..1023 {
      a += buffer[i] : string;
    }
    //writeln("Blah blah blah, " + a : string);
    score2[0] = score;
    moveOn[0] = true;
    //while (!moveOn[0]) {
    //  if buffer[0].size > 0 {
    //    this.log.debug(buffer[0] : string, hstring=hstring);
    //  }
    //}
    score = score2[0];
    //writeln("The score is: ", score : string);

    if false {
      if this.roundsProcessed[valkyrie].read() == rounds {
        // basically, once this happens, set it to done, then sit your ass _down_
        if this.valkyriesDone.fetchAdd(1) < (propagator.maxValkyries-1) {
          // waiting.
          writeln("Waiting");
          //this.moveOn;
          this.valkyriesUnblocked.add(1);
          while this.valkyriesUnblocked.read() > 0 do chpl_task_yield();
          writeln("No longer waiting");
          //this.roundsProcessed[valkyrie].write(0);
          //this.valkyriesDone.sub(1);
          // Clear on out and go.
        } else {
          // Set it and go.
          writeln("Processing gc");
          //writeln(this.valkyriesDone.read() : string);
          //this.valkyriesDone.sub(1);
          //this.roundsProcessed[valkyrie].write(0);
          // If this is called when TF is active, haaaa.
          this.gc();
          writeln("gc done");
          for i in 1..propagator.maxValkyries {
            this.roundsProcessed[valkyrie].write(0);
          }
          this.valkyriesDone.write(0);
          //this.moveOn = true;
          // Does this work in order?
          //while this.valkyriesUnblocked.read() < (propagator.maxValkyries-1) do chpl_task_yield();
          //this.moveOn.reset();
          this.valkyriesUnblocked.write(0);
          // Now we wait for the others to clear out.
          //while this.valkyriesDone.fetchAdd(1) > 0 do chpl_task_yield();
          //this.moveOn = false;
          writeln("At last, the gc thread moves on");
        }
      } else {
        this.roundsProcessed[valkyrie].add(1);
      }
    }
    //if !runGC.testAndSet() {} {
      // Fact is, we can't really run this until... aaaaanyway.
      // If you run GC and TF at the same time, haaaahahaha.
    //  gc();
    //  runGC.write(false);
    //}
    //writeln("from lockandrun, what is the score?");
    //writeln(score : real : string);
    return score;
    //PyGILState_Release(gil);
  }
}

// This is really only if you're compiling it as an executable.  It's more of a library.
/*
proc main() {
  init();
  testRun();
  final();
}
*/
