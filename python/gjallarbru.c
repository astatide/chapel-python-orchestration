#include <Python.h>
#include <numpy/arrayobject.h>
#include <math.h>

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
  //  return *self;
  // FUCKING FINALLY.
  PyObject *pArray;

  // FINALLY FUCKING WORKS
  //printf("Hey is this working?");
  // That should be the numnber of dimensions.
  pArray = PyArray_SimpleNewFromData(globalND, dims, NPY_FLOAT64, (void *)(arr));
  //printf("Piece of fuck");
  PyArrayObject *np_arr = (PyArrayObject*)(pArray);

  Py_XINCREF(np_arr);
  Py_XINCREF(pArray);

  return np_arr;
}

static PyObject *weights(PyObject *self, PyObject *args) {

  // We're doing this here so that it stays in scope.
  //npy_intp tDims[globalDims];
  //for (int i = 0; i < globalLength; i++ ) {
  //  tDims[i] = globalLength;
  //}
  //printf("Are we going?");
  numpyArray = returnNumpyArray(globalArray, globalDims);
  Py_XINCREF(numpyArray);
  return numpyArray;

}

PyObject* loadPythonModule(char * module) {
  PyObject *pName, *pMod;

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
  pMod = PyImport_ImportModule(module);

  // yeah duh of course it fucking does.

  if (pMod != NULL) {
    return pMod;
  }
  else {
    PyErr_Print();
    //printf(stderr, "Failed to load \"%s\"\n", module);
    return pMod;
  }

}

double run() {
  PyObject *pFunc, *pArgs, *pValue;

  // Gotta be super careful about this call.
  // There's probably some error checking but heyo.
  double score;
  PyObject * pModule;
  ////printf("We're in run, now");
  // This might need to be a per thread thing.
  // This call fucks out because it's a fucking asshole.
  // You are such an asshole.  Why are you like this?
  pModule = loadPythonModule("gjTest.gjTest");
  pFunc = PyObject_GetAttrString(pModule, "testRun");
  ////printf("We successfully grabbed the module");
  if (pFunc && PyCallable_Check(pFunc)) {
    // you pretend to work.  YOU LIE.
    pValue = PyObject_CallObject(pFunc, NULL);
    //printf("ARE YOU A LIAR?!");
  } else {
    PyErr_Print();
  }
  if (pValue == NULL) {
    // we should also kill it if the score is null.
    PyErr_Print();
    // we can check for null on the python side, and if it's NULL, we can
    // die with a critical failure.  WITH HONOR.
    //return NULL;
  }
  //printf("Are we able to get the score?  Seems the function ran");
  score = PyFloat_AsDouble(pValue);
  //printf("Score?");
  return score;

}

double pythonRun(double * arr, unsigned long long nd, unsigned long long * dims, PyThreadState *pi)
{
  // We're setting the pointer.  Keep in mind that this hinges on properly
  // passing in the array; Chapel needs to make sure it's compatible with
  // what C expects.

  //printf("Do we get this far");
  double score;
  PyGILState_STATE gstate;
  globalArray = arr;
  globalND = nd;
  globalDims = dims;
  //printf("Acquire that shit");
  //PyEval_AcquireThread(pi);
  gstate = PyGILState_Ensure();
  // DEBUG
  //PyThreadState_Swap(pi);
  PyImport_AppendInittab("gjallarbru", &PyInit_gjallarbru);
  //printf("GIL acquired!");
  //printf("Threads swapped!");
  // oh hi I'm a little bitch who doesn't run.
  score = run();
  //PyEval_SimpleString("import sys\n");
  //printf("Shit, it ran!");
  // Just cause.
  Py_XDECREF(numpyArray);
  ////printf("Hey; don't abort.  I told you not to");
  // Why does this just... die?
  //PyGILState_Release(pi);
  //PyThreadState_Swap(NULL);
  //PyEval_ReleaseThread(pi);
  PyGILState_Release(gstate);
  //printf("All that other shit is done");
  return score;
}

PyThreadState* newThread() {
  // we're sort of faffing about here with the GIL.
  //PyGILState_STATE a = PyGILState_Ensure();
  //PyEval_AcquireLock();
  PyGILState_STATE gstate;
  gstate = PyGILState_Ensure();
  //PyThreadState *interp = Py_NewInterpreter();
  PyThreadState *thread = PyThreadState_New(mainInterpreterState);
  //PyThreadState *thread = PyThreadState_New(interp);
  PyGILState_Release(gstate);
  //PyEval_ReleaseLock();
  //PyGILState_Release(a);
  return thread;
}

/*
void pythonInit() {
  // disable buffering for debugging.
  setbuf(stdout, NULL);
  PyImport_AppendInittab("gjallarbru", &PyInit_gjallarbru);
  Py_Initialize();
}
*/

void pythonInit() {
  // disable buffering for debugging.
  setbuf(stdout, NULL);
  PyImport_AppendInittab("gjallarbru", &PyInit_gjallarbru);
  Py_Initialize();
  PyEval_InitThreads();
  // This call might need to be loaded into each individual thread, perhaps.
  mainThreadState = PyThreadState_Get();
  //PyEval_ReleaseLock();
  //PyEval_AcquireLock();
  mainInterpreterState = mainThreadState->interp;
  PyEval_SaveThread();
  //PyEval_ReleaseLock();
  //PyEval_SaveThread();
  // load up the module.  Only do it once.
  // We actually don't give a shit about the GIL, so we just ignore it.
  // The python programs are essentially _read only_ programs.
  // They're not here to perform modifications to the algorithm.
  //return Py_NewInterpreter();
}

void pythonFinal() {
  // disable buffering for debugging.
  //setbuf(stdout, NULL);
  //PyImport_AppendInittab("gjallarbru", &PyInit_gjallarbru);
  //printf("Killing python");
  //Py_NewInterpreter();
  //Py_Finalize();
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
