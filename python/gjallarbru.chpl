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


/*
static PyObject *returnNumpyArray(float *arr, npy_intp *dims);
static PyObject *weights(PyObject *self, PyObject *args);
run
main
*/

proc main() {
  writeln('man fuck you');
  //var arr = [0.0,1.0,2.0,3.0,4.0,5.0,6.0,7.0,8.0,11.0, 456.0] : c_float;
  var length: c_ulonglong = 100;
  var arr: [0..length] c_double = 700;
  var dims: [0..1] c_ulonglong = length;
  var a = arr;
  var d = dims;
  pythonInit();
  pythonRun(a, 1, d);
  writeln("What der fuck");
  pythonFinal();
  //writeln(arr : string);
}
