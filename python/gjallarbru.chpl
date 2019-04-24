// This is the Chapel to C bit.

// We're gonna create a main then try and call the python C bits.

extern type PyObject;
extern type npy_intp;

extern proc returnNumpyArray(ref arr: c_float, ref dims: npy_intp) : PyObject;
extern proc weights(ref self: PyObject, ref args: PyObject): PyObject;
extern proc run();
extern proc pythonRun(arr: [] c_double, nd: c_ulonglong, dims: [] c_ulonglong);
extern proc pythonInit();
extern proc pythonFinal();
//extern proc pythonRun();

require "gjallarbru.c";

proc init() {
  pythonInit();
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

// This is really only if you're compiling it as an executable.  It's more of a library.
proc main() {
  init();
  testRun();
  final();
}
