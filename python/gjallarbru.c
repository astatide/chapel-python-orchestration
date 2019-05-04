#include <Python.h>
#include <numpy/arrayobject.h>
#include <math.h>
#include <stdlib.h>
#include <stdbool.h>

// Declare the functions that we'll need later.

static PyObject *returnNumpyArray(double *arr, unsigned long long *dims);
static PyObject *weights(PyObject *self, PyObject *args);
static PyObject *weights_multi(PyObject *self, PyObject *args);
//static PyObject returnNumpyArray();

double *globalArray;
PyArrayObject *numpyArray;
unsigned long long globalND;
unsigned long long *globalDims;
PyInterpreterState * mainInterpreterState;
// Main thread doesn't go away, but the other threads seem to?
PyThreadState * mainThreadState;
volatile PyThreadState ** threads;
PyObject ** returnArrayList;
PyObject * returnList;
unsigned long long *dimArray;
Py_ssize_t returnArrayListSize;

int functionRunOnce = false;

// You basically _always_ need this.  It's the methods that we're going to
// expose to the python module.
static PyMethodDef methods[] = {
  { "weights", weights, METH_VARARGS, "Descriptions"},
  { "weights_multi", weights_multi, METH_VARARGS, "Descriptions"},
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

  //Py_XINCREF(np_arr);
  //Py_XINCREF(pArray);

  return np_arr;
}

static PyObject *weights(PyObject *self, PyObject *args) {

  numpyArray = returnNumpyArray(globalArray, globalDims);
  //Py_XINCREF(numpyArray);
  return numpyArray;

}

static PyObject *returnManyNumpyArrays(double *arr, unsigned long long *dims, Py_ssize_t nd) {
  PyObject *pArray;

  pArray = PyArray_SimpleNewFromData(nd, dims, NPY_FLOAT64, (void *)(arr));
  //PyArrayObject *np_arr = (PyArrayObject*)(pArray);

  // These ones are sort of unbounded...
  //Py_XINCREF(np_arr);
  //Py_XINCREF(pArray);

  return (PyArrayObject*)(pArray);
}

static PyObject *weights_multi(PyObject *self, PyObject *args) {
  // The idea here is that we want to return a list of weights.
  // or return it from blah blah blah.
  // So args is going to contain some standard stuff or whatever.
  /*
  const char *field = 0; PyObject *value; int typeField; size_t size = 0;
  if (!PyArg_ParseTuple(args, "isO|n", &typeField, &field, &value, &size)) {
    return 0;
  }
  */

  // clean that shit up, yo.
  if (functionRunOnce) {
    for (int i = 0; i < returnArrayListSize; i++) {
      Py_XDECREF(returnArrayList[i]);
    }
    printf("Freeing memory from previous call");
    Py_XDECREF(returnList);
    free(returnArrayList);
  }

  printf("Freed!");
  functionRunOnce = true;
  PyObject * argList, * tupleValue;
  double * cArray = globalArray;
  Py_ssize_t n, m;
  long long cD;
  cD = 0;

  printf("\nSTARTING FUNCTION; what is cArray? %p\n", cArray);

  unsigned long long elements;
  unsigned long long tValue;

  Py_XINCREF(args);
  if (!PyArg_ParseTuple(args, "O!", &PyList_Type, &argList)) {
    // Eh, what der fuck?
  //  PyErr_Print();
    PyErr_SetString(PyExc_TypeError, "Argument must be a list.");
    return NULL;
  }
  Py_XDECREF(args);
  Py_XINCREF(argList);
  //PyArg_ParseTuple(args, "O!", &PyList_Type, &argList);

  PyObject * tempTuple;

  PyObject* repr;
  PyObject* str;
  char *bytes;

  n = PyList_Size(argList);
  returnArrayListSize = n;
  m = 0;
  // Get the size of the amount of memory we need to malloc
  for (int i = 0; i < n; i++) {
    tempTuple = PyList_GetItem(argList, i);
    Py_XINCREF(tempTuple);
    m += PyTuple_Size(tempTuple);
    Py_XDECREF(tempTuple);
  }
  // probably fucking up the mallocs
  dimArray = malloc(m * sizeof(unsigned long long));
  returnArrayList = malloc(n * sizeof(PyArrayObject*));
  double * dArray = dimArray;
  //unsigned long long dArray [ m ];
  //unsigned long long * dimArray = dArray;
  //PyObject * rAList [ n ];
  //PyObject ** returnArrayList = rAList;
  for (int i = 0; i < n; i++) {
    elements = 1;
    m = 0;
    printf("\nGet the tuple item\n");
    tempTuple = PyList_GetItem(argList, i);
    Py_XINCREF(tempTuple);
    printf("\nDid the getting work?\n");
    if (PyErr_Occurred()) {
      PyErr_Print();
    //  return NULL;
    }
    printf("\nIf no error was there, then yes.\n");
    // not all of these are tuples, unfortunately.
    // It seems sometimes, they're just ints.
    if (PyTuple_Check(tempTuple)) {
      m = PyTuple_Size(tempTuple);
    } else {
      m = 1;
    }
    for (int ti = 0; ti < m; ti++) {
      // Set the dimensional array value to be equal to the value in the tuple.
      if (PyTuple_Check(tempTuple)) {
        tupleValue = PyTuple_GetItem(tempTuple, ti);
        Py_XINCREF(tupleValue);
        tValue = PyLong_AsUnsignedLongLong(tupleValue);
      } else {
        tValue = PyLong_AsUnsignedLongLong(tempTuple);
      }
      dimArray[ti] = tValue;
      elements *= tValue;
      if (PyTuple_Check(tempTuple)) {
        Py_XDECREF(tupleValue);
      }
      printf("\ntValue %llu, elements %llu\n", tValue, elements);
      // okay, so now we're going through the tuples and blah blah blah.
    }
    printf("\nHow big is m? %llu\n", m);
    Py_XDECREF(tempTuple);
    returnArrayList[i] = returnManyNumpyArrays(cArray, dimArray, m);
    Py_XINCREF(returnArrayList[i]);
    if (PyErr_Occurred()) {
      PyErr_Print();
    }
    // shift the array pointer up by the appropriate number of elements.
    dimArray += m;
    cArray += elements;
    // how many elements have we used?
    cD += m;
    //Py_XINCREF(returnArrayList[i]);
  }
  free(dArray);
  printf("\nCreate a tuple, then populate\n");
  returnList = PyList_New(n);
  Py_XINCREF(returnList);
  if (PyErr_Occurred()) {
    PyErr_Print();
  //  return NULL;
  }
  for (int i = 0; i < n; i++) {
    printf("\nPopulating List element %i\n", i);
    PyList_SetItem(returnList, i, returnArrayList[i]);
    printf("\nElement %i populated\n", i);
    //if (PyErr_Occurred()) {
    //  PyErr_Print();
    //  return NULL;
    //}
  }
  //free(returnArrayList);
  printf("\nDecref on argList\n");
  Py_XDECREF(argList);
  printf("\nNow, we return!\n");
  return returnList;

}

PyObject* loadPythonModule(char * module) {
  PyObject *pName, *pMod;

  // This should allow us to actually add to the stupid path.
  //printf("can we import sys?");
  PyRun_SimpleString("import sys");
  //printf("Okay; can we add our path to the path?");
  PyRun_SimpleString("sys.path.append('/Users/apratt/work/yggdrasil/python/')");
  //printf("Good; what about the actual import module command?");
  pMod = PyImport_ImportModule(module);
  //printf("hey, that worked.  So what gives?");
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
  //printf("Wait, so is it just this?");
  pModule = loadPythonModule("gjTest.gjTest");
  Py_XINCREF(pModule);
  //printf("Okay, so that loaded...");
  pFunc = PyObject_GetAttrString(pModule, function);
  Py_XINCREF(pFunc);
  if (pFunc && PyCallable_Check(pFunc)) {
    // Oh, so we're a fancy lad, eh.
    pValue = PyObject_CallObject(pFunc, NULL);
    Py_XINCREF(pValue);
  } else {
    PyErr_Print();
  }
  if (pValue != NULL) {
    PyObject* repr = PyObject_Repr(pValue);
    Py_XINCREF(repr);
    PyObject* str = PyUnicode_AsEncodedString(repr, "utf-8", "~E~");
    Py_XINCREF(str);
    const char *bytes = PyBytes_AS_STRING(str);
    //Py_XDECREF(pValue);
    score = atof(bytes);
    Py_XDECREF(repr);
    Py_XDECREF(str);
    if (PyErr_Occurred()) {
      PyErr_Print();
    }
  } else {
    PyErr_Print();
  }
  Py_XDECREF(pValue);
  Py_XDECREF(pModule);
  Py_XDECREF(pFunc);
  return score;

}

double pythonRun(double * arr, unsigned long long nd, unsigned long long * dims, PyThreadState *pi)
{
  // We're setting the pointer.  Keep in mind that this hinges on properly
  // passing in the array; Chapel needs to make sure it's compatible with
  // what C expects.

  PyGILState_STATE gstate;
  double score;
  PyObject * pModule;
  globalArray = arr;
  globalND = nd;
  globalDims = dims;
  PyThreadState *old;
  //gstate = PyGILState_Ensure();
  //PyEval_RestoreThread(mainThreadState);
  //PyEval_AcquireLock();
  //PyThreadState_Swap(mainThreadState);
  //PyEval_AcquireThread(pi);
  //printf("Lock it down blah blah blah");
  //PyThreadState * interp = Py_NewInterpreter();
  // yeah, this so is not working.
  //printf("Hey, did you lock?  Okay, cool.  Now; can you create a new thread?");
  // the answer to that question is "no".
  //PyThreadState *ts = PyThreadState_New(interp->interp);
  PyThreadState *ts = PyThreadState_New(mainInterpreterState);
  //printf("Cool.  Can you swap threads?");
  //old = PyThreadState_Swap(ts);
  //PyEval_SaveThread();
  PyEval_AcquireThread(ts);
  //Py_InitializeEx(0);
  //PyRun_SimpleString("import importlib; importlib.reload()");
  //void* throw = import_array();
  // hmmmm.  Maybe?
  //printf("Can you add a module, or do you suck?");
  //PyRun_SimpleString("print(134342)");
  //PyRun_SimpleString("import sys");
  //pModule = loadPythonModule("gjTest.gjTest");
  // no.  We don't.
  //printf("Did we swap states correctly?");
  //PyImport_AppendInittab("gjallarbru", &PyInit_gjallarbru);
  score = run("run");
  for (int i = 0; i < returnArrayListSize; i++) {
    Py_XDECREF(returnArrayList[i]);
    //PyObject_GC_UnTrack(returnArrayList[i]);
  }
  functionRunOnce = false;
  Py_XDECREF(returnList);
  //PyObject_GC_UnTrack(returnList);
  free(returnArrayList);
  //free(dimArray);
  printf("\n We ran, but did we finish? \n");
  //dimArray = malloc(m * sizeof(unsigned long long));
  //returnArrayList = malloc(n * sizeof(PyArrayObject*));
  PyEval_ReleaseThread(ts);
  printf("\n Did we release the thread? \n");
  //newscore = *score;
  //Py_XDECREF(numpyArray);
  //PyThreadState_Swap(mainThreadState);
  //PyThreadState_Swap(mainThreadState);
  //PyEval_ReleaseLock();
  //PyThreadState_Clear(ts);
  //PyThreadState_Delete(ts);
  //PyGILState_Release(gstate);
  //PyEval_ReleaseThread();
  return score;
}

PyThreadState* newThread() {
  // we're sort of faffing about here with the GIL.
  //PyGILState_STATE gstate;
  //gstate = PyGILState_Ensure();

  PyThreadState *thread = Py_NewInterpreter()->interp;
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
  //PyThreadState *threads[maxValkyries];

  // malloc the damn thing.
  threads = malloc(sizeof(PyThreadState*)*maxValkyries);
  PyImport_AppendInittab("gjallarbru", &PyInit_gjallarbru);

  setbuf(stdout, NULL);
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
    //threads[i] = newThread();
    //threads[i] = Py_NewInterpreter();
    threads[i] = PyThreadState_New(mainInterpreterState);

  }
  //PyEval_ReleaseLock();
  PyEval_SaveThread();
  ////printf("swap the fucking threads asshole");
  //PyThreadState_Swap(mainThreadState);
  ////printf("Make a new goddamn thread");
  //PyThreadState *ts = PyThreadState_New(threads[0]->interp);
  ////printf("Now fucking swap it");
  //PyThreadState_Swap(ts);
  ////printf("Good for you you fucking asshole");
  //Py_BEGIN_ALLOW_THREADS
  //PyRun_SimpleString("print(134342)");
  // Since all that works, I suspect we're going out of scope, somehow.
  // Jesus.
  //PyThreadState_Swap(mainThreadState);
  return threads;
  // Release the GIL, but swap in the main thread.
  // also fuck off.
  //PyGILState_Release(gstate);
}

void pythonFinal() {
  // blah?

  //Py_END_ALLOW_THREADS
  Py_Finalize();
}

/*
int main(int argc, char *argv[])
{

  PyImport_AppendInittab("gjallarbru", &PyInit_gjallarbru);
  Py_Initialize();
  run();
  ////printf("stupid");

}
*/