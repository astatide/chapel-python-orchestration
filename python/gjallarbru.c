#include <Python.h>
#include <numpy/arrayobject.h>
#include <math.h>
#include <stdlib.h>

// Declare the functions that we'll need later.

static PyObject *returnNumpyArray(double *arr, unsigned long long *dims);
static PyObject *weights(PyObject *self, PyObject *args);
//static PyObject returnNumpyArray();

double *globalArray;
PyArrayObject *numpyArray;
unsigned long long globalND;
unsigned long long *globalDims;
PyInterpreterState * mainInterpreterState;
PyThreadState * mainThreadState;

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
static PyObject *returnNumpyArray(double *arr, unsigned long long *dims) {
  PyObject *pArray;

  pArray = PyArray_SimpleNewFromData(globalND, dims, NPY_FLOAT64, (void *)(arr));
  PyArrayObject *np_arr = (PyArrayObject*)(pArray);

  Py_XINCREF(np_arr);
  Py_XINCREF(pArray);

  return np_arr;
}

static PyObject *weights(PyObject *self, PyObject *args) {

  numpyArray = returnNumpyArray(globalArray, globalDims);
  Py_XINCREF(numpyArray);
  return numpyArray;

}

PyObject* loadPythonModule(char * module) {
  PyObject *pName, *pMod;

  // This should allow us to actually add to the stupid path.
  PyRun_SimpleString("import sys");
  PyRun_SimpleString("sys.path.append('/Users/apratt/work/yggdrasil/python/')");
  pMod = PyImport_ImportModule(module);
  if (pMod != NULL) {
    return pMod;
  }
  else {
    PyErr_Print();
    return pMod;
  }

}

double run(char * function) {
  PyObject *pFunc, *pArgs, *pValue;
  // Gotta be super careful about this call.
  // There's probably some error checking but heyo.
  double score;
  PyObject * pModule;
  pModule = loadPythonModule("gjTest.gjTest");
  pFunc = PyObject_GetAttrString(pModule, function);
  if (pFunc && PyCallable_Check(pFunc)) {
    pValue = PyObject_CallObject(pFunc, NULL);
  } else {
    PyErr_Print();
  }
  if (pValue != NULL) {
    PyObject* repr = PyObject_Repr(pValue);
    PyObject* str = PyUnicode_AsEncodedString(repr, "utf-8", "~E~");
    const char *bytes = PyBytes_AS_STRING(str);
    Py_DECREF(pValue);
    score = atof(bytes);
    if (PyErr_Occurred()) {
      PyErr_Print();
    }
  } else {
    PyErr_Print();
  }
  return score;

}

double pythonRun(double * arr, unsigned long long nd, unsigned long long * dims, PyThreadState *pi, double *score)
{
  // We're setting the pointer.  Keep in mind that this hinges on properly
  // passing in the array; Chapel needs to make sure it's compatible with
  // what C expects.

  PyGILState_STATE gstate;
  double newscore;
  globalArray = arr;
  globalND = nd;
  globalDims = dims;
  gstate = PyGILState_Ensure();
  //PyEval_RestoreThread(mainThreadState);
  printf("Lock it down blah blah blah");
  // yeah, this so is not working.
  //PyEval_AcquireLock();
  printf("Hey, did you lock?  Okay, cool.  Now; can you create a new thread?");
  // the answer to that question is "no".
  //PyThreadState *ts = PyThreadState_New(pi);
  printf("Cool.  Can you swap threads?");
  //PyThreadState_Swap(ts);
  // no.  We don't.
  printf("Did we swap states correctly?");
  //PyImport_AppendInittab("gjallarbru", &PyInit_gjallarbru);
  *score = run("run");
  //newscore = *score;
  //Py_XDECREF(numpyArray);
  //PyThreadState_Swap(mainThreadState);
  //PyEval_ReleaseLock();
  //PyThreadState_Clear(ts);
  //PyThreadState_Delete(ts);
  PyGILState_Release(gstate);
  //PyEval_ReleaseThread();
  return newscore;
}

PyThreadState* newThread() {
  // we're sort of faffing about here with the GIL.
  //PyGILState_STATE gstate;
  //gstate = PyGILState_Ensure();

  PyThreadState *thread = Py_NewInterpreter();
  //PyThreadState *thread = PyThreadState_New(mainInterpreterState);
  // Actually, do we need this?
  //PyThreadState *thread = PyThreadState_New(interp);
  // get it son.
  //PyThreadState_Swap(thread);
  //PyGILState_Release(gstate);
  return thread;
}

PyThreadState* pythonInit(unsigned long long maxValkyries) {
  // disable buffering for debugging.
  //PyGILState_STATE gstate;
  //gstate = PyGILState_Ensure();

  PyThreadState *threads[maxValkyries];

  setbuf(stdout, NULL);
  PyImport_AppendInittab("gjallarbru", &PyInit_gjallarbru);
  Py_Initialize();
  PyEval_InitThreads();
  //import_array();
  // For some reason, this seems necessary for... anything.
  // However, when we call it in the second task, we get nothing.
  // Because there's no active thread.
  // Why does Python insist on not sandboxing itself?  I don't want to share
  // memory between python interpreters.
  // ugh it's just blah.
  mainThreadState = PyThreadState_Get();
  mainInterpreterState = mainThreadState->interp;
  for (int i = 0; i < maxValkyries; i++ ) {
    threads[i] = newThread();
  }
  PyEval_SaveThread();
  PyThreadState_Swap(mainThreadState);
  //PyThreadState_Swap(mainThreadState);
  return threads;
  // Release the GIL, but swap in the main thread.
  // also fuck off.
  //PyGILState_Release(gstate);
}

void pythonFinal() {
  Py_Finalize();
}

/*
int main(int argc, char *argv[])
{

  PyImport_AppendInittab("gjallarbru", &PyInit_gjallarbru);
  Py_Initialize();
  run();
  //printf("stupid");

}
*/
