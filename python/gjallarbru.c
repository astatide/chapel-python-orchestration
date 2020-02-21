#include <Python.h>
#include <numpy/arrayobject.h>
#include <math.h>
#include <stdlib.h>
#include <stdbool.h>
#include <stdatomic.h>
#include <sys/mman.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <unistd.h>
// we crash on OSX if we don't init the time.
#include <time.h>

// Declare the functions that we'll need later.

static PyObject *returnNumpyArray(double *arr, unsigned long long *dims);
static PyObject *weights(PyObject *self, PyObject *args);
PyObject *weights_multi(PyObject *self, PyObject *args);
PyObject *valkyrieID(PyObject *self, PyObject *args);
PyObject *ref(PyObject *self, PyObject *args);
PyObject *demeID(PyObject *self, PyObject *args);
static PyObject * gjallarbru_write(PyObject *self, PyObject *args);
//static PyObject returnNumpyArray();

double *globalArray;
PyObject * returnList;
PyObject * prevReturnList;
PyInterpreterState * mainInterpreterState;
// Main thread doesn't go away, but the other threads seem to?
PyThreadState * mainThreadState;
volatile PyThreadState ** threads;
volatile PyThreadState ** interps;
int * threadsInitialized;
PyObject * pModule;
int moduleImportedOnce = false;
//PyObject ** returnArrayList;

int functionRunOnce = false;

int initializedAlready = false;

// Wait on forks, maybe?
atomic_int * valkyriesDone;

unsigned long long globalMaxValkyries;

// You basically _always_ need this.  It's the methods that we're going to
// expose to the python module.
static PyMethodDef methods[] = {
  { "weights", weights, METH_VARARGS, "Descriptions"},
  { "weights_multi", weights_multi, METH_VARARGS, "Descriptions"},
  { "valkyrieID", valkyrieID, METH_VARARGS, "Descriptions"},
  { "ref", ref, METH_VARARGS, "Descriptions"},
  { "demeID", demeID, METH_VARARGS, "Descriptions"},
  { "write", gjallarbru_write, METH_VARARGS, "Descriptions"},
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
  import_array();
  // This is necessary for the numpy bits to work.
  ////printf("\nIMPORTING NUMPY API\n");
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
  //import_array();
  if (m == NULL) {
    return;
  }


}
#endif
static PyObject * gjallarbru_write(PyObject *self, PyObject *args)
{
    const char *what;
    if (!PyArg_ParseTuple(args, "s", &what))
        return NULL;
    // this right here is the stdout.
    //printf("==%s==", what);
    return Py_BuildValue("");
}

PyObject *valkyrieID(PyObject *self, PyObject *args) {
  // this just returns the thread ID so we can use it
  // from within Python.

  PyObject* inptDict = PyThreadState_GetDict();
  PyObject* valkyrie = PyDict_GetItemString(inptDict, "valkyrieID");

  return valkyrie;

}

PyObject *ref(PyObject *self, PyObject *args) {
  // this just returns the thread ID so we can use it
  // from within Python.

  PyObject* inptDict = PyThreadState_GetDict();
  PyObject* sOne = PyDict_GetItemString(inptDict, "sOne");
  PyObject* sTwo = PyDict_GetItemString(inptDict, "sTwo");

  PyObject * rL;
  rL = NULL;
  rL = PyList_New(2);
  Py_XINCREF(rL);
  if (PyErr_Occurred()) {
    PyErr_Print();
  }
  PyList_SET_ITEM(rL, 0, sOne);
  if (PyErr_Occurred()) {
    PyErr_Print();
  }
  PyList_SET_ITEM(rL, 1, sTwo);
  if (PyErr_Occurred()) {
    PyErr_Print();
  }

  return rL;

}


PyObject *demeID(PyObject *self, PyObject *args) {
  // this just returns the thread ID so we can use it
  // from within Python.

  PyObject* inptDict = PyThreadState_GetDict();
  PyObject* deme = PyDict_GetItemString(inptDict, "demeID");

  return deme;

}

PyObject *runTF(PyObject *self, PyObject *args) {

  // We're going to go ahead and pull the appropriate thread state
  // out of the module.  Then link it up to the pointer above.
  PyObject* inptDict = PyThreadState_GetDict();
  // So let's add to this dictionary yo!
  unsigned long long valkyrie = PyLong_AsUnsignedLongLong(PyDict_GetItemString(inptDict, "valkyrieID"));

  //double * vArray = globalArray[valkyrie];

  PyObject * argList, * tupleValue, * tempTuple;
  //double * cArray = vArray;
  //unsigned long long *dimArray;
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
  // we need to parse these into... a numpy array.  And possibly the model string?
  Py_XINCREF(args);
  if (!PyArg_ParseTuple(args, "O!", &PyList_Type, &argList)) {
    // Eh, what der...?
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
      // AH.  That's where the... blagh
      // I forgot to do error checking here.  I mean _come on_.
      m += 1;
    }
  }
}


static PyObject *weights(PyObject *self, PyObject *args) {
  Py_XDECREF(args);
  return NULL;

}

static PyObject *returnManyNumpyArrays(double *arr, unsigned long long *dims, Py_ssize_t nd) {
  PyArrayObject *pArray;

  pArray = (PyArrayObject *)PyArray_SimpleNewFromData(nd, dims, NPY_FLOAT64, (void *)(arr));

  return pArray;
}

PyObject *weights_multi(PyObject *self, PyObject *args) {
  // This is definitely a leaky, leaky function.
  // The idea here is that we want to return a list of weights.
  // or return it from blah blah blah.
  // So args is going to contain some standard stuff or whatever.

  // clean that up, yo.

  if (functionRunOnce) {
    // This _should_ be better than it is.
    // But it isn't.
    return returnList;
  }

  // We're going to go ahead and pull the appropriate thread state
  // out of the module.  Then link it up to the pointer above.
  PyObject* inptDict = PyThreadState_GetDict();
  // So let's add to this dictionary yo!
  unsigned long long valkyrie = PyLong_AsUnsignedLongLong(PyDict_GetItemString(inptDict, "valkyrieID"));

  double * vArray = globalArray;

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
    // Eh, what der...?
    PyErr_SetString(PyExc_TypeError, "Argument must be a list.");
    return NULL;
  }

  n = PyList_Size(argList);
  m = 0;
  // Get the size of the amount of memory we need to malloc
  for (int i = 0; i < n; i++) {
    tempTuple = PyList_GetItem(argList, i);
    if (PyTuple_Check(tempTuple)) {
      m += PyTuple_Size(tempTuple);
    } else {
      // Assume an int, if not a tuple.
      // AH.  That's where the... blargh.
      // I forgot to do error checking here.  I mean _come on_.
      m += 1;
    }
  }
  // probably mucking up the mallocs
  dimArray = malloc(m * sizeof(unsigned long long));
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
    PyList_SET_ITEM(returnList, i, (PyArrayObject *)PyArray_SimpleNewFromData(m, dimArray, NPY_FLOAT64, (void *)(globalArray)));
    if (PyErr_Occurred()) {
      PyErr_Print();
    }
    // shift the array pointer up by the appropriate number of elements.
    dimArray += m;
    mTotal += m;
    globalArray += elements;
    cD += m;
  }
  dimArray -= mTotal;
  free(dimArray);

  if (PyErr_Occurred()) {
    PyErr_Print();
  }
  Py_XDECREF(argList);
  return returnList;

}

PyObject* loadPythonModule(char * module) {
  PyObject *pName, *pMod;

  // This should allow us to actually add to the stupid path.
  // blah blah, stupid hacks.
  PyRun_SimpleString("import sys");
  PyRun_SimpleString("sys.path.append('/lus/snx11254/apratt/yggdrasil/python/')");
  PyRun_SimpleString("sys.path.append('/Users/apratt/work/yggdrasil/python/')");
  //PyRun_SimpleString("sys.path.append('/opt/python/3.6.5.7')");
  //PyRun_SimpleString("sys.path.append('/opt/cray/llm/21.4.570-7.0.0.1_5.9__ge50c6aa.ari/lib64/python')");
  //PyRun_SimpleString("sys.path.append('/opt/python/3.6.5.7/lib/python3.6/site-packages')");
  pMod = PyImport_ImportModule(module);
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
  double score = 0.0;
  //PyObject* inptDict = PyThreadState_GetDict();
  //unsigned long long valkyrie = PyLong_AsUnsignedLongLong(PyDict_GetItemString(inptDict, "valkyrieID"));
  pModule = loadPythonModule("gjTest.gjTest");

  pFunc = PyObject_GetAttrString(pModule, function);
  if (PyErr_Occurred()) {
    PyErr_Print();
  }

  if (pFunc && PyCallable_Check(pFunc)) {
    pValue = PyObject_CallObject(pFunc, NULL);
  } else {
    PyErr_Print();
  }
  if (pValue != NULL) {
    score = PyFloat_AsDouble(pValue);
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
  printf("Score is %f\n", score);
  return score;

}

double runNew(char * function, char * argv[], int argc) {
  PyObject *pFunc, *pArgs, *pValue;
  int i,j;
  double score = 0.0;
  int isNumber = 1;
  pModule = loadPythonModule("gjTest.gjTest");

  pFunc = PyObject_GetAttrString(pModule, function);

  // okay, now we want to submit the arguments.
  pArgs = PyTuple_New(argc);
  for (i = 0; i < argc; i++) {
    // Check and see whether each char is a digit.  If they're all digits, that's a number.
    // I mean, we're assuming.  I can't think of any other scenario where you have all digits.
    for (j = 0; j < argc; j++) {
     if (!isdigit(argv[i][j])) {
       // oh HO.  You're not a number then!?  I see how it is.
       isNumber = 0;
     }
    }
    if (isNumber) {
      // base 10, bitches.
      pValue = PyLong_FromUnsignedLongLong(strtol(argv[i], argv[i+1], 10));
    } else {
      // OKAY FINE LEAVE IT AS A STRING THEN GOD.
      pValue = PyUnicode_FromString(argv[i]);
    }
    PyTuple_SetItem(pArgs, i, pValue);
  }

  if (PyErr_Occurred()) {
    PyErr_Print();
  }

  if (pFunc && PyCallable_Check(pFunc)) {
    pValue = PyObject_CallObject(pFunc, pArgs);
  } else {
    PyErr_Print();
  }
  if (pValue != NULL) {
    score = PyFloat_AsDouble(pValue);
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
  printf("Score is %f\n", score);
  return score;

}

PyThreadState* newThread() {
  // we're sort of faffing about here with the GIL.
  PyThreadState *thread = Py_NewInterpreter();
  return thread;
}

struct threadArgs {
  double * arr;
  unsigned long long valkyrie;
  double * score2;
  char * buffer;
};


double pythonRun(double * arr, unsigned long long valkyrie, unsigned long long deme, double * score)
//void *pythonRunInThread(void * arguments)

{
  // We're setting the pointer.  Keep in mind that this hinges on properly
  // passing in the array; Chapel needs to make sure it's compatible with
  // what C expects.

  double s2 = 0.0;
  int pid;
  int stat;

  pid = 0;

  PyObject* inptDict = PyThreadState_GetDict();
  // ... remember that Chapel doesn't start at 0.
  PyDict_SetItemString(inptDict, "valkyrieID", PyLong_FromUnsignedLongLong(valkyrie-1));
  PyDict_SetItemString(inptDict, "demeID", PyLong_FromUnsignedLongLong(deme));
  globalArray = arr;

  functionRunOnce = false;
  s2 = run("run");
  //printf("S2 is %f\n", s2);
  functionRunOnce = false;
  Py_CLEAR(returnList);
  returnList = NULL;
  moduleImportedOnce = true;
  *score = s2;
  printf("C SCORE IS %f\n", s2);
  fflush(stdout);
  return s2;
}

double pythonRunFunction(char * funcName, double * retValue, char * argv[], int argc)

{
  // We're setting the pointer.  Keep in mind that this hinges on properly
  // passing in the array; Chapel needs to make sure it's compatible with
  // what C expects.

  double rv = 0.0;
  int pid;
  int stat;

  pid = 0;

  PyThreadState *ts = PyThreadState_New(mainInterpreterState);
  PyEval_AcquireThread(ts);

  PyObject* inptDict = PyThreadState_GetDict();
  // ... remember that Chapel doesn't start at 0.
  //PyDict_SetItemString(inptDict, "x", PyUnicode_FromString(sOne));
  //PyDict_SetItemString(inptDict, "y", PyUnicode_FromString(sTwo));
  PyDict_SetItemString(inptDict, "valkyrieID", PyLong_FromUnsignedLongLong(0));
  rv = runNew(funcName, argv, argc);
  Py_CLEAR(returnList);
  returnList = NULL;
  *retValue = rv;
  //fflush(stdout);
  //PyThreadState_Clear(ts);
  PyEval_ReleaseThread(ts);
  //PyThreadState_Delete(ts);
  return rv;
}

void pythonRunSimple(char * funcName)

{
  // We're setting the pointer.  Keep in mind that this hinges on properly
  // passing in the array; Chapel needs to make sure it's compatible with
  // what C expects.

  PyThreadState *ts = PyThreadState_New(mainInterpreterState);
  PyEval_AcquireThread(ts);

  PyObject* inptDict = PyThreadState_GetDict();
  PyDict_SetItemString(inptDict, "valkyrieID", PyLong_FromUnsignedLongLong(0));
  PyRun_SimpleString(funcName);
  PyEval_ReleaseThread(ts);
}


PyThreadState* pythonInit(unsigned long long maxValkyries) {
  // disable buffering for debugging.
  // we're adding this to the list of builtins in order to export it.
  //threads = malloc(sizeof(PyThreadState*)*maxValkyries);
  setbuf(stdout, NULL);
  PyImport_AppendInittab("gjallarbru", &PyInit_gjallarbru);
  Py_Initialize();
  PyEval_InitThreads();
  //PyRun_SimpleString("import tensorflow as tf");
  mainThreadState = PyThreadState_Get();
  mainInterpreterState = mainThreadState->interp;
  //PyEval_ReleaseLock();
  PyRun_SimpleString("import sys");
  PyRun_SimpleString("print('fired up!')");
  PyEval_SaveThread();
  initializedAlready = true;
  return threads;
}

void garbageCollect(unsigned long long valkyrie) {
  // Spawn a new thread, then collect the garbage.
  // Nothing should be active right now!  Just this thread.
  PyThreadState *ts;

  if (!threadsInitialized[valkyrie]) {
    threads[valkyrie] = PyThreadState_New(mainInterpreterState);
    threadsInitialized[valkyrie] = true;
  }
  ts = threads[valkyrie];
  PyEval_AcquireThread(ts);
  PyEval_ReleaseThread(ts);
}

void pythonFinal() {
  Py_Finalize();
}
