// This is the Chapel to C bit.

// We're gonna create a main then try and call the python C bits.

extern type PyObject;
extern type npy_intp;

extern proc returnNumpyArray(ref arr: c_float, ref dims: npy_intp) : PyObject;
extern proc weights(ref self: PyObject, ref args: PyObject): PyObject;
extern proc run();
extern proc pythonRun(arr: [] c_float);
extern proc pythonInit();
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
  var arr = [0.0,1.0,2.0,3.0,4.0,5.0,6.0,7.0,8.0,11.0] : c_float;
  var a = arr;
  pythonInit();
  pythonRun(a);
  writeln(arr : string);
}
