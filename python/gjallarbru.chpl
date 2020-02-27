// This is the Chapel to C bit.

// We're gonna create a main then try and call the python C bits.
use Math;
use List;

extern type PyObject;
extern type npy_intp;
extern type PyThreadState;

extern proc returnNumpyArray(ref arr: c_float, ref dims: npy_intp) : PyObject;
extern proc weights(ref self: PyObject, ref args: PyObject): PyObject;
extern proc run();
extern proc pythonRun(arr: [] c_double, valkyrie: c_ulonglong, deme : c_ulonglong, score: c_ptr(c_double)) : c_double;
//extern proc pythonRunFunction(moduleName: c_string, funcName : c_string, retvalue : c_ptr(c_double), argv, argc) : c_double;
extern proc pythonRunFunction(moduleName: c_string, funcName : c_string, retvalue : c_ptr(c_double), argv: [] c_string, argc: c_int, id: c_double) : c_double;
extern proc pythonRunFunctionNoArguments(moduleName: c_string, funcName : c_string, retvalue : c_ptr(c_double), id: c_double) : c_double;
extern proc pythonRunSimple(funcName : c_string);
extern proc pythonInit(n: c_ulonglong): c_void_ptr;
extern proc pythonFinal();
extern proc newThread() : c_void_ptr;
extern proc PyThreadState_Swap(thread: c_void_ptr) : c_void_ptr;
extern proc PyGILState_Ensure(): c_void_ptr;
extern proc PyGILState_Release(lock: c_void_ptr);
extern proc garbageCollect(valkyrie: c_ulonglong);

require "gjallarbru.c";

class Gjallarbru {

  var threads: [1..10] c_void_ptr;
  var runGC: atomic bool;
  var roundsProcessed: [1..10] atomic int;
  var rounds: int = 0;
  var valkyriesDone: atomic int;
  var valkyriesUnblocked: atomic int;
  var moveOn: sync bool;
  //var log: shared ygglog.YggdrasilLogging;

  proc pInit() {
    // return the sub-interpreter
    //this.log = new shared ygglog.YggdrasilLogging();
    //this.log.currentDebugLevel = propagator.debug;
    this.threads = pythonInit(10 : c_ulonglong);
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

  proc testRun() {
    // Does almost seem to work?
    pythonRun(createTestArray(20, 2), 2 : c_ulonglong, createDimsArray(20, 2));
  }


  proc lockAndRun(matrix, valkyrie, deme, hstring) {
    // We're sending in a pointer and then writing to it.  Seems to work more cleanly.
    // or that's the hope.  Who fucking knows anymore.
    // We're gonna pass in the file object.
    //var score: real;
    //var score2: [0..1] real = Math.INFINITY;
    //var buffer : [0..1023] c_string;
    //var moveOn: [0..1] bool = false;

    var score: c_double;
    //var s2: c_double;
    pythonRun(matrix, valkyrie : c_ulonglong, deme: c_ulonglong, c_ptrTo(score));
    //writeln("S2 in Chapel: ", s2 : real : string);
    //assert(score == 5.0);
    writeln("score in Chapel: ", score : string);
    //score2[0] = score;
    //moveOn[0] = true;

    //score = score2[0];
    stdout.flush();
    return score : real;
  }

  proc runFunction(moduleName: string, funcName: string = 'calculateNovelty', arg: list(string), id: real = 1) {
    var retValue: c_double;
    var argc: c_int;
    writeln("How many arguments: ", arg.size);
    if arg.size == 0 {
      //pythonRunFunctionNoArguments(moduleName.c_str(), funcName.c_str(), c_ptrTo(retValue));
      pythonRunFunctionNoArguments(moduleName.c_str(), funcName.c_str(), c_ptrTo(retValue), id);
      stdout.flush();
    } else {
      var argv: [1..arg.size] c_string;
      for i in 1..arg.size {
        argv[i] = arg[i].c_str();
        argc += 1;
      }
      pythonRunFunction(moduleName.c_str(), funcName.c_str(), c_ptrTo(retValue), argv, argc, id);
      stdout.flush();
    }
    return retValue : real;
  }

  proc runString(funcName: string = 'calculateNovelty') {
    pythonRunSimple(funcName.c_str());
  }


}
