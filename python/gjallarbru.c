#include <Python.h>
#include <numpy/arrayobject.h>
#include <math.h>

// Declare the functions that we'll need later.

static PyObject *returnNumpyArray(float *arr, npy_intp *dims);
static PyObject *weights(PyObject *self, PyObject *args);
//static PyObject returnNumpyArray();


// You basically _always_ need this.  It's the methods that we're going to
// expose to the python module.
static PyMethodDef methods[] = {
  { "weights", weights, METH_VARARGS, "Descriptions"},
  { NULL, NULL, 0, NULL }
};

// Remove me if you enjoy segfaults!
// https://docs.scipy.org/doc/numpy/user/c-info.ufunc-tutorial.html#example-numpy-ufunc-for-one-dtype

// Differences for python 2 and 3.
#if PY_VERSION_HEX >= 0x03000000
static struct PyModuleDef moduledef = {
    PyModuleDef_HEAD_INIT,
    "gjallarbru",
    NULL,
    -1,
    methods,
    NULL,
    NULL,
    NULL,
    NULL
};
PyMODINIT_FUNC PyInit_gjallarbru(void) {
  PyObject *m;
  m = PyModule_Create(&moduledef);
  // This is necessary for the numpy bits to work.
  import_array();
  if (!m) {
    return NULL;
  }
  return m;

}
#else
// This is a python 2 style.  We don't really want it.
PyMODINIT_FUNC initgjallarbru(void) {
  PyObject *m, *logit, *d;
  m = Py_InitModule("gjallarbru", methods);
  // This is necessary for the numpy bits to work.
  import_array();
  if (m == NULL) {
    return;
  }


}
#endif

// This is working code that creates an array in C, then wraps it up
// as a numpy array.
static PyObject *returnNumpyArray(float *arr, npy_intp *dims) {
  //  return *self;
  // FUCKING FINALLY.
  PyObject *pArray;

  // FINALLY FUCKING WORKS
  pArray = PyArray_SimpleNewFromData(1, dims, NPY_FLOAT, (void *)(arr));
  PyArrayObject *np_arr = (PyArrayObject*)(pArray);

  Py_XINCREF(np_arr);
  Py_XINCREF(pArray);

  return np_arr;
}

static PyObject *weights(PyObject *self, PyObject *args) {
  //  return *self;
  // FUCKING FINALLY.

  // Create the array.
  float *arr;
  // MALLOC to the rescue bitches
  // I think the garbage collector in python gets collection happy
  // or something.  I have no idea.
  arr = (float*)malloc(10*sizeof(float));
  npy_intp dims[1];
  dims[0] = 10;
  for (int i = 0; i < 10; i++ ) {
    arr[i] = i;
  }

  PyArrayObject *np_arr = returnNumpyArray(arr, dims);

  return np_arr;
  //Py_RETURN_NONE;


}


PyObject* runPythonFunction(char *function, PyObject *pModule) {
  PyObject *pArgs, *pValue, *pFunc;
  int i;

  pFunc = PyObject_GetAttrString(pModule, function);
  /* pFunc is a new reference */

  if (pFunc && PyCallable_Check(pFunc)) {
      pValue = PyObject_CallObject(pFunc, NULL);
      if (pValue != NULL) {
        //printf("Result of call: %ld\n", PyInt_AsLong(pValue));
        Py_DECREF(pValue);
      }
      else {
        Py_DECREF(pFunc);
        Py_DECREF(pModule);
        PyErr_Print();
        fprintf(stderr,"Call failed\n");
        return pValue;
      }
  }
  else {
    if (PyErr_Occurred())
      PyErr_Print();
    fprintf(stderr, "Cannot find function \"%s\"\n", function);
  }
  Py_XDECREF(pFunc);
  Py_DECREF(pModule);

  return pValue;
}

PyObject* loadPythonModule(char * module) {
  PyObject *pName, *pModule;

  // This should allow us to actually add to the stupid path.
  PyRun_SimpleString("import sys");
  PyRun_SimpleString("sys.path.append('/Users/apratt/work/yggdrasil/python/')");
  //PyRun_SimpleString("import numpy as np");
  // Hooray and success and all that.  Sucks to your asmar.
  // Only for Python3!
  //pName = PyString_FromString(module);
  /* Error checking of pName left out */

  // I think this returns a pointer.

  // works with char.  Derp derp.
  pModule = PyImport_ImportModule(module);

  // yeah duh of course it fucking does.

  if (pModule != NULL) {
    return pModule;
  }
  else {
    PyErr_Print();
    fprintf(stderr, "Failed to load \"%s\"\n", module);
    return pModule;
  }

  //return pModule;

}

void run() {
  PyObject *pName, *pModule, *pFunc, *elfucko;
  PyObject *pArgs, *pValue, *pTest, *test;

  int i;
  //pName = PyString_FromString("test");
  pModule = loadPythonModule("gjTest.gjTest");

  pFunc = PyObject_GetAttrString(pModule, "testRun");
  if (pFunc && PyCallable_Check(pFunc)) {
    pValue = PyObject_CallObject(pFunc, NULL);
  } else {
    PyErr_Print();
  }
  if (pValue) {
    elfucko = PyObject_CallMethod(pValue, "printm", NULL);
  } else {
    PyErr_Print();
    printf("get bent");
  }

}

int main(int argc, char *argv[])
{
  PyImport_AppendInittab("gjallarbru", &PyInit_gjallarbru);
  Py_Initialize();
  run();
  printf("stupid");
}
