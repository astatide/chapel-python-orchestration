#include <Python.h>
#include <numpy/arrayobject.h>
#include <math.h>
#include <stdlib.h>
#include <stdbool.h>

// Declare the functions that we'll need later.

static PyObject *returnNumpyArray(double *arr, unsigned long long *dims);
static PyObject *weights(PyObject *self, PyObject *args);
PyObject *weights_multi(PyObject *self, PyObject *args);
//static PyObject returnNumpyArray();

double *globalArray;
PyObject * returnList;
PyObject * prevReturnList;
//PyArrayObject *numpyArray;
//unsigned long long globalND;
//unsigned long long *globalDims;
PyInterpreterState * mainInterpreterState;
// Main thread doesn't go away, but the other threads seem to?
PyThreadState * mainThreadState;
volatile PyThreadState ** threads;
PyObject * pModule;
int moduleImportedOnce = false;
//PyObject ** returnArrayList;

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
/*
static PyObject *returnNumpyArray(double *arr, unsigned long long *dims) {
  PyObject *pArray;

  pArray = PyArray_SimpleNewFromData(globalND, dims, NPY_FLOAT64, (void *)(arr));
  PyArrayObject *np_arr = (PyArrayObject*)(pArray);

  //Py_XINCREF(np_arr);
  //Py_XINCREF(pArray);

  return np_arr;
}

static PyObject *weights(PyObject *self, PyObject *args) {
  PyArrayObject *numpyArray;

  numpyArray = returnNumpyArray(globalArray, globalDims);
  //Py_XINCREF(numpyArray);
  Py_XDECREF(args);
  return numpyArray;

}
*/
static PyObject *weights(PyObject *self, PyObject *args) {
  //PyArrayObject *numpyArray;

  //numpyArray = returnNumpyArray(globalArray, globalDims);
  //Py_XINCREF(numpyArray);
  Py_XDECREF(args);
  return NULL;
  //return numpyArray;

}

static PyObject *returnManyNumpyArrays(double *arr, unsigned long long *dims, Py_ssize_t nd) {
  PyArrayObject *pArray;

  pArray = (PyArrayObject *)PyArray_SimpleNewFromData(nd, dims, NPY_FLOAT64, (void *)(arr));
  //printf("\npArray pointer %p\n", pArray);
  //PyArrayObject *np_arr = (PyArrayObject*)(pArray);

  // These ones are sort of unbounded...
  //Py_XINCREF(np_arr);
  //Py_XINCREF(pArray);

  return pArray;
}

PyObject *weights_multi(PyObject *self, PyObject *args) {
  // This is definitely a leaky, leaky function.
  // The idea here is that we want to return a list of weights.
  // or return it from blah blah blah.
  // So args is going to contain some standard stuff or whatever.

  // clean that shit up, yo.

  if (functionRunOnce) {
    //Py_XINCREF(returnList);
    //return returnList;
    // clear up the memory.
    // This should cause it to fail, which we want.
    Py_XDECREF(returnList);
  }

  functionRunOnce = true;

  PyObject * argList, * tupleValue, * tempTuple;
  double * cArray = globalArray;
  unsigned long long *dimArray;
  Py_ssize_t n, m;
  long long cD;
  Py_ssize_t mTotal;

  unsigned long long elements;
  unsigned long long tValue;

  argList = NULL;
  tupleValue = NULL;
  tempTuple = NULL;
  returnList = NULL;

  cD = 0;
  mTotal = 0;

  // So, it might already be dead?  Unless we grab ownership?
  Py_XINCREF(args);
  if (!PyArg_ParseTuple(args, "O!", &PyList_Type, &argList)) {
    // Eh, what der fuck?
    PyErr_SetString(PyExc_TypeError, "Argument must be a list.");
    return NULL;
  }

  n = PyList_Size(argList);
  m = 0;
  // Get the size of the amount of memory we need to malloc
  for (int i = 0; i < n; i++) {
    tempTuple = PyList_GetItem(argList, i);
    if (PyTuple_Check(tempTuple)) {
      //m = PyTuple_Size(tempTuple);
      m += PyTuple_Size(tempTuple);
    } else {
      // Assume an int, if not a tuple.
      // AH.  That's where the fucking... jesus.
      // I forgot to do error checking here.  I mean _come on_.
      m += 1;
    }
  }

  // probably fucking up the mallocs
  dimArray = malloc(m * sizeof(unsigned long long));
  //unsigned long long * dArray = dimArray;
  returnList = PyList_New(n);
  Py_XINCREF(returnList);

  for (int i = 0; i < n; i++) {
    elements = 1;
    m = 0;
    tempTuple = PyList_GetItem(argList, i);
    if (PyErr_Occurred()) {
      PyErr_Print();
    }
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
        tValue = PyLong_AsUnsignedLongLong(tupleValue);
      } else {
        tValue = PyLong_AsUnsignedLongLong(tempTuple);
      }
      dimArray[ti] = (unsigned long long)tValue;
      elements *= tValue;
    }
    // Keep it in scope.
    PyList_SET_ITEM(returnList, i, (PyArrayObject *)PyArray_SimpleNewFromData(m, dimArray, NPY_FLOAT64, (void *)(cArray)));
    if (PyErr_Occurred()) {
      PyErr_Print();
    }
    // shift the array pointer up by the appropriate number of elements.
    dimArray += m;
    mTotal += m;
    cArray += elements;
    cD += m;
  }
  dimArray -= mTotal;
  free(dimArray);

  if (PyErr_Occurred()) {
    PyErr_Print();
  }

  //printf("\nDecref on args\n");
  // If I call this on the args in this case, they die.
  //Py_XDECREF(args);
  Py_XDECREF(argList);
  //printf("\nNow, we return!\n");
  //Py_XINCREF(returnList);
  //Py_XDECREF(returnList);
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
  if (PyErr_Occurred()) {
    PyErr_Print();
  }
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
  //printf("Wait, so is it just this?");
  // Multiple calls to this may only produce shallow copies; we want to make
  // sure we don't destroy it in between calls, maybe?
  //if (!moduleImportedOnce) {
    pModule = loadPythonModule("gjTest.gjTest");
    //Py_XINCREF(pModule);
  //}
  //Py_XINCREF(pModule);
  //printf("Okay, so that loaded...");
  pFunc = PyObject_GetAttrString(pModule, function);
  if (PyErr_Occurred()) {
    PyErr_Print();
  }
  //Py_XINCREF(pFunc);
  if (pFunc && PyCallable_Check(pFunc)) {
    // Oh, so we're a fancy lad, eh.
    // Here, garbage collection can occur.
    pValue = PyObject_CallObject(pFunc, NULL);
    //Py_XINCREF(pValue);
  } else {
    PyErr_Print();
  }
  if (pValue != NULL) {
    PyObject* repr = PyObject_Repr(pValue);
    //Py_XINCREF(repr);
    PyObject* str = PyUnicode_AsEncodedString(repr, "utf-8", "~E~");
    //Py_XINCREF(str);
    const char *bytes = PyBytes_AS_STRING(str);
    score = atof(bytes);
    //Py_XDECREF(pValue);
    Py_XDECREF(str);
    Py_XDECREF(repr);
    if (PyErr_Occurred()) {
      PyErr_Print();
    }
  } else {
    PyErr_Print();
  }
  Py_XDECREF(pFunc);
  Py_XDECREF(pValue);
  // this MIGHT be a borrowed reference
  Py_XDECREF(pModule);
  return score;

}

PyThreadState* newThread() {
  // we're sort of faffing about here with the GIL.
  PyThreadState *thread = Py_NewInterpreter();
  return thread;
}

double pythonRun(double * arr, unsigned long long valkyrie)
{
  // We're setting the pointer.  Keep in mind that this hinges on properly
  // passing in the array; Chapel needs to make sure it's compatible with
  // what C expects.

  double score;

  if (!moduleImportedOnce) {
    // get it, and store it.
    // This thread is only accessible to us, and there's no point in killing it
    // until we're actually done.

  }
  PyThreadState *ts;
  PyInterpreterState *inpt;
  //inpt = newThread();
  //ts = PyThreadState_New(mainInterpreterState);
  // We apparently do not need the GIL for this.
  //inpt = PyInterpreterState_New();
  //ts = PyThreadState_New(threads[valkyrie]->interp);
  ts = PyThreadState_New(mainInterpreterState);
  // Nor do we need it for this.
  //ts = PyThreadState_New(inpt);
  // This will grab the GIL.
  PyEval_AcquireThread(ts);
  //PyThreadState_Swap(ts);

  // So, while this is a global variable, it's hardly thread safe.
  // and each thread operates on it when they choose.  AcquireThread is blocking,
  // so doing it here _should_ avoid issues where the threads are changing the value.
  // I hope, anyway.  Or will it?  Goddammit.  I wish I could avoid this.
  // Hm, I could just pass the matrix in, I guess.

  // This little item could allow us to maybe send info in.
  PyObject* inptDict = PyThreadState_GetDict();
  //PyGILState_STATE gstate;
  //gstate = PyGILState_Ensure();

  globalArray = arr;
  functionRunOnce = false;
  score = run("run");
  functionRunOnce = false;
  Py_CLEAR(returnList);
  returnList = NULL;
  //PyThreadState_Clear(ts);
  PyEval_ReleaseThread(ts);
  PyThreadState_Delete(ts);
  //PyGILState_Release(gstate);
  moduleImportedOnce = true;

  return score;
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
    //threads[i] = PyThreadState_New(mainInterpreterState);
    threads[i] = newThread();

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
  returnList = NULL;
  prevReturnList = NULL;
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
