// This is the Chapel to C bit.

// We're gonna create a main then try and call the python C bits.

extern type PyObject;
extern type npy_intp;
extern type PyThreadState;

extern proc returnNumpyArray(ref arr: c_float, ref dims: npy_intp) : PyObject;
extern proc weights(ref self: PyObject, ref args: PyObject): PyObject;
extern proc run();
extern proc pythonRun(arr: [] c_double, nd: c_ulonglong, dims: [] c_ulonglong);
extern proc pythonInit();
extern proc pythonFinal();
extern proc Py_NewInterpreter() : c_void_ptr;
extern proc PyThreadState_Swap(thread: c_void_ptr) : c_void_ptr;
extern proc PyGILState_Ensure(): c_void_ptr;
extern proc PyGILState_Release(lock: c_void_ptr);
//extern proc
//extern proc pythonRun();

require "python/gjallarbru.c";

proc init() {
  // return the sub-interpreter
  pythonInit();
}

proc newInterpreter() {
  return Py_NewInterpreter();
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
  var dims: [0..nd] c_ulonglong = length;
  return dims;
}

proc testRun() {
  // Does almost seem to work?
  pythonRun(createTestArray(20, 2), 2 : c_ulonglong, createDimsArray(20, 2));
}

//       gjallarbru.pythonRun(v.matrixValues, 1 : c_ulonglong, gjallarbru.createDimsArray(mSize, 1));

proc lockAndRun(pi, matrix, nd, dims ) {
  // This is just some bullshit to make us thread safe, I guess.
  var gil = PyGILState_Ensure();
  PyThreadState_Swap(pi);
  pythonRun(matrix, nd, dims);
  PyGILState_Release(gil);
}

// This is really only if you're compiling it as an executable.  It's more of a library.
/*
proc main() {
  init();
  testRun();
  final();
}
*/
