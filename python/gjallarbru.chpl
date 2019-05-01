// This is the Chapel to C bit.

// We're gonna create a main then try and call the python C bits.

extern type PyObject;
extern type npy_intp;
extern type PyThreadState;

extern proc returnNumpyArray(ref arr: c_float, ref dims: npy_intp) : PyObject;
extern proc weights(ref self: PyObject, ref args: PyObject): PyObject;
extern proc run();
extern proc pythonRun(arr: [] c_double, nd: c_ulonglong, dims: [] c_ulonglong, thread: c_void_ptr, ref score: c_double) : c_double;
extern proc pythonInit(n: c_ulonglong): c_void_ptr;
extern proc pythonFinal();
extern proc newThread() : c_void_ptr;
extern proc PyThreadState_Swap(thread: c_void_ptr) : c_void_ptr;
extern proc PyGILState_Ensure(): c_void_ptr;
extern proc PyGILState_Release(lock: c_void_ptr);
//extern proc
//extern proc pythonRun();

require "python/gjallarbru.c";

class Gjallarbru {

  var threads: [1..propagator.maxValkyries] c_void_ptr;

  proc pInit() {
    // return the sub-interpreter
    this.threads = pythonInit(propagator.maxValkyries : c_ulonglong);
    //return threads;
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

  proc lockAndRun(pi, matrix, nd, dims ) {
    // This is just some bullshit to make us thread safe, I guess.
    //var gil = PyGILState_Ensure();
    // We're sending in a pointer and then writing to it.  Seems to work more cleanly.
    // or that's the hope.  Who fucking knows anymore.
    var score: c_double;
    var newscore = pythonRun(matrix, nd, dims, pi, score);
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
