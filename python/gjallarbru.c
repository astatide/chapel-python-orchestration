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


//#include "../tf/aegir.c"

// Declare the functions that we'll need later.

static PyObject *returnNumpyArray(double *arr, unsigned long long *dims);
static PyObject *weights(PyObject *self, PyObject *args);
PyObject *weights_multi(PyObject *self, PyObject *args);
PyObject *valkyrieID(PyObject *self, PyObject *args);
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

/*
PyObject *checkLock(PyObject *self, PyObject *args) {
  //
  if (atomic_fetch_add(valkyriesDone, 1) < globalMaxValkyries-1) {

  } else {
    atomic_store(valkyriesDone, 0);
  }

}
*/

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
}


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
  ////printf("\npArray pointer %p\n", pArray);
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
    return returnList;
    //Py_XDECREF(returnList);
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
    // Eh, what der fuck?
    PyErr_SetString(PyExc_TypeError, "Argument must be a list.");
    return NULL;
  }

  //printf("STOP 1: args processed");
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
  //printf("STOP 2: array size obtained");
  // probably fucking up the mallocs
  dimArray = malloc(m * sizeof(unsigned long long));
  //unsigned long long * dArray = dimArray;
  returnList = PyList_New(n);
  Py_XINCREF(returnList);
  //printf("STOP 3: returnList created");

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
  //printf("STOP 4: arrays built");
  dimArray -= mTotal;
  free(dimArray);

  if (PyErr_Occurred()) {
    PyErr_Print();
  }

  ////printf("\nDecref on args\n");
  // If I call this on the args in this case, they die.
  //Py_XDECREF(args);
  Py_XDECREF(argList);
  ////printf("\nNow, we return!\n");
  //Py_XINCREF(returnList);
  //Py_XDECREF(returnList);
  //printf("STOP 5: return");
  return returnList;

}

PyObject* loadPythonModule(char * module) {
  PyObject *pName, *pMod;

  // This should allow us to actually add to the stupid path.
  ////printf("can we import sys?");
  PyRun_SimpleString("import sys");
  ////printf("Okay; can we add our path to the path?");
  PyRun_SimpleString("sys.path.append('/Users/apratt/work/yggdrasil/python/')");
  ////printf("Good; what about the actual import module command?");
  pMod = PyImport_ImportModule(module);
  ////printf("hey, that worked.  So what gives?");
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
  // This is really just for debugging.
  PyObject* inptDict = PyThreadState_GetDict();
  unsigned long long valkyrie = PyLong_AsUnsignedLongLong(PyDict_GetItemString(inptDict, "valkyrieID"));

  ////printf("Wait, so is it just this?");
  // Multiple calls to this may only produce shallow copies; we want to make
  // sure we don't destroy it in between calls, maybe?
  //if (!moduleImportedOnce) {
    pModule = loadPythonModule("gjTest.gjTest");
    //Py_XINCREF(pModule);
  //}
  //Py_XINCREF(pModule);
  ////printf("Okay, so that loaded...");
  pFunc = PyObject_GetAttrString(pModule, function);
  if (PyErr_Occurred()) {
    PyErr_Print();
  }
  //Py_XINCREF(pFunc);
  if (pFunc && PyCallable_Check(pFunc)) {
    // Oh, so we're a fancy lad, eh.
    // Here, garbage collection can occur.
    pValue = PyObject_CallObject(pFunc, NULL);
    //printf("Valkyrie ID: %i We have left CallObject\n", valkyrie);
    //Py_XINCREF(pValue);
  } else {
    PyErr_Print();
  }
  if (pValue != NULL) {
    PyObject* repr = PyObject_Repr(pValue);
      //printf("Valkyrie ID: %i Converted to repr\n", valkyrie);
    //Py_XINCREF(repr);
    PyObject* str = PyUnicode_AsEncodedString(repr, "utf-8", "~E~");
    //printf("Valkyrie ID: %i Converted to str\n", valkyrie);
    //Py_XINCREF(str);
    const char *bytes = PyBytes_AS_STRING(str);
    //printf("Valkyrie ID: %i Converted to double\n", valkyrie);
    score = atof(bytes);
    //printf("Valkyrie ID: %i Called atof\n", valkyrie);
    ////printf("Hey, the score is %f", score);
    //Py_XDECREF(pValue);
    Py_XDECREF(str);
    Py_XDECREF(repr);
    //printf("Valkyrie ID: %i Decremented references\n", valkyrie);
    if (PyErr_Occurred()) {
      PyErr_Print();
    }
  } else {
    PyErr_Print();
  }
  //printf("Valkyrie ID: %i We have left formatted the score\n", valkyrie);
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

struct threadArgs {
  double * arr;
  unsigned long long valkyrie;
  double * score2;
  char * buffer;
};


double pythonRun(double * arr, unsigned long long valkyrie, double * score2, char * buffer)
//void *pythonRunInThread(void * arguments)

{
  // We're setting the pointer.  Keep in mind that this hinges on properly
  // passing in the array; Chapel needs to make sure it's compatible with
  // what C expects.

  /*
  struct threadArgs *args = arguments;
  unsigned long long valkyrie = args->valkyrie;
  double * arr = args->arr;
  double * score2 = args->score2;
  char * buffer = args->buffer;
  */

  // fork the fucking thing.
  double *score;
  int pid;
  int stat;
  //int pipefd[2];
  //pipe(pipefd);

  score = mmap(NULL, sizeof *score, PROT_READ | PROT_WRITE,
                MAP_SHARED | MAP_ANONYMOUS, -1, 0);

  //for (int i = 0; i < 1; i++) {
  //printf("Valkyrie ID: %i About to call fork\n", valkyrie-1);
  //pid = fork();
  pid = 0;

  switch(pid) {
    case 0:
      // sometimes, this is as far as we get.  WHY?
      // We successfully fork, but...
      //printf("Valkyrie ID: %i Child proc; importing gjallarbru\n", valkyrie-1);
      setbuf(stdout, NULL);
      //PyImport_AppendInittab("gjallarbru", &PyInit_gjallarbru);
      ////printf("Valkyrie ID: %i gj imported; initializing Python\n", valkyrie-1);
      //Py_Initialize();
      ////printf("Valkyrie ID: %i python initialized; getting thread dict\n", valkyrie-1);
      // maybe this is the asshole.
      //PyEval_InitThreads();
      // child process
      ////printf("I'm child %d, my pid is %d\n", i, getpid());
      //close(pipefd[0]);    // close reading end in the child

      //dup2(pipefd[1], 1);  // send stdout to the pipe
      //dup2(pipefd[1], 2);  // send stderr to the pipe
      //PyThreadState *ts;
      //PyInterpreterState *npi;
      //_PyGILState_check_enabled = 0;
      //inpt = newThread();
      //ts = PyThreadState_New(mainInterpreterState);
      // We apparently do not need the GIL for this.
      //inpt = PyInterpreterState_New();
      //ts = PyThreadState_New(threads[valkyrie]->interp);
      //if (!threadsInitialized[valkyrie-1]) {
        //threads[valkyrie-1] = PyThreadState_New(interps[valkyrie-1]->interp);
      //  threads[valkyrie-1] = PyThreadState_New(mainInterpreterState);
      //  threadsInitialized[valkyrie-1] = true;
      //}
      //ts = threads[valkyrie-1];
      // Nor do we need it for this.
      //ts = PyThreadState_New(inpt);
      // This will grab the GIL.
      //PyEval_AcquireThread(ts);
      // if we're using fork.
      //PyGILState_STATE gstate;
      //gstate = PyGILState_Ensure();
      //void * blah = import_array();
      //PyThreadState_Swap(ts);

      // So, while this is a global variable, it's hardly thread safe.
      // and each thread operates on it when they choose.  AcquireThread is blocking,
      // so doing it here _should_ avoid issues where the threads are changing the value.
      // I hope, anyway.  Or will it?  Goddammit.  I wish I could avoid this.
      // Hm, I could just pass the matrix in, I guess.

      // This little item could allow us to maybe send info in.
      PyObject* inptDict = PyThreadState_GetDict();
      //printf("Valkyrie ID: %i thread dict obtained; inserting valkyrie id\n", valkyrie-1);
      // So let's add to this dictionary yo!
      // ... remember that Chapel doesn't start at 0.
      PyDict_SetItemString(inptDict, "valkyrieID", PyLong_FromUnsignedLongLong(valkyrie-1));
      //PyDict_SetItemString(inptDict, "logname", PyUnicode_FromString(logname));
      globalArray = arr;

      //globalArray = arr;
      functionRunOnce = false;
      //printf("Valkyrie ID: %i id inserted; running run function\n", valkyrie-1);
      *score = run("run");
      // if we can get here, then we're getting in and out of python quickly enough...
      //printf("Valkyrie ID: %i Score from run is %f\n", valkyrie-1, *score);
      functionRunOnce = false;
      Py_CLEAR(returnList);
      returnList = NULL;
      //PyThreadState_Clear(ts);
      //PyEval_ReleaseThread(ts);
      //PyThreadState_Delete(ts);
      //PyGILState_Release(gstate);
      //printf("Valkyrie ID: %i Release the GIL, shut it down\n", valkyrie-1);
      moduleImportedOnce = true;
      //close(pipefd[1]);
      //Py_Finalize();
      //printf("Valkyrie ID: %i Exiting\n", valkyrie-1);
      //exit(0);
      break;

    case -1:
      //printf("Fork Error");
      break;

    default: {
      char buffer2[1024];
      //printf("Valkyrie ID: %i Parent proc\n", valkyrie-1);

      //close(pipefd[1]);  // close the write end of the pipe in the parent

      //while (read(pipefd[0], buffer2, 1024) != 0)
      //{
        ////printf("READING YALL");
        ////printf("FROM PYTHON: %s\n", buffer2);
      //  buffer = buffer2;
      //}
      //wait(0);
      waitpid(-1, &stat, WUNTRACED);

      // wait on the specific child?
      //waitpid(pid, &stat, 0);
      if (WIFEXITED(stat)) {
        // this happens if everything is good and such.  Yay!;
      } else {
        //printf("Valkyrie ID: %i SOMETHING HAPPENED\n", valkyrie-1);
        if (WIFSTOPPED(stat)) {
          // this happens if we just freeze up.  Bit hacky, but hey.
          // for now, just try again...
          //printf("Valkyrie ID: %i FORK FAILED; TRY AGAIN\n", valkyrie-1);
          pythonRun(arr,valkyrie,score2,buffer);

        }
      }
      //printf("Valkyrie ID: %i Returning to Chapel\n", valkyrie-1);
    }
  }
  //}

  //double score;

  if (!moduleImportedOnce) {
    // get it, and store it.
    // This thread is only accessible to us, and there's no point in killing it
    // until we're actually done.

  }
  ////printf("Score from child proc is %f", *score);
  *score2 = *score;
  return *score;
}

/*
double pythonRunpThread(double * arr, unsigned long long valkyrie, double * score2, char * buffer) {
  pthread_t thread_id;

  struct threadArgs args;
  args.arr = arr;
  args.valkyrie = valkyrie;
  args.score2 = score2;
  args.buffer = buffer;

  if(pthread_create(&thread_id, NULL, pythonRunInThread, (void *)&args)) {

    f//printf(stderr, "Error creating thread\n");
    return 1;

  }
  if(pthread_join(thread_id, NULL)) {

    f//printf(stderr, "Error joining thread\n");
    return 2;

  }
  return *args.score2;
}
*/

PyThreadState* pythonInit(unsigned long long maxValkyries) {
  // disable buffering for debugging.
  // we're adding this to the list of builtins in order to export it.
  setbuf(stdout, NULL);
  PyImport_AppendInittab("gjallarbru", &PyInit_gjallarbru);


  Py_Initialize();
  //PyEval_InitThreads();
  initializedAlready = true;
  //import_array();
  //import_array();
  // For some reason, this seems necessary for... anything.
  // However, when we call it in the second task, we get nothing.
  // Because there's no active thread.
  // Why does Python insist on not sandboxing itself?  I don't want to share
  // memory between python interpreters.
  // ugh it's just blah.
  //globalArray = malloc(maxValkyries * sizeof(double*));
  //mainThreadState = PyThreadState_Get();
  //mainInterpreterState = mainThreadState->interp;
  //for (int i = 0; i < maxValkyries; i++ ) {
    //threads[i] = newThread();
    //threads[i] = Py_NewInterpreter();
    //threads[i] = PyThreadState_New(mainInterpreterState);
    //threads[i] = newThread();
    //interps[i] = Py_NewInterpreter();
    //threadsInitialized[i] = false;

  //}
  //PyEval_ReleaseLock();
  //PyEval_SaveThread();
  //////printf("swap the fucking threads asshole");
  //PyThreadState_Swap(mainThreadState);
  //////printf("Make a new goddamn thread");
  //PyThreadState *ts = PyThreadState_New(threads[0]->interp);
  //////printf("Now fucking swap it");
  //PyThreadState_Swap(ts);
  //////printf("Good for you you fucking asshole");
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

void garbageCollect(unsigned long long valkyrie) {
  // Spawn a new thread, then collect the garbage.
  // Nothing should be active right now!  Just this thread.
  PyThreadState *ts;

  //ts = PyThreadState_New(mainInterpreterState);
  if (!threadsInitialized[valkyrie]) {
    threads[valkyrie] = PyThreadState_New(mainInterpreterState);
    threadsInitialized[valkyrie] = true;
  }
  ts = threads[valkyrie];
  PyEval_AcquireThread(ts);
  //PyRun_SimpleString("import gc; gc.collect()");
  PyEval_ReleaseThread(ts);
  //PyThreadState_Delete(ts);
}

void pythonFinal() {
  // blah?

  //Py_END_ALLOW_THREADS
  //free(threads);
  //free(threadsInitialized);
  //free(interps);
  Py_Finalize();
}

/*
int main(int argc, char *argv[])
{

  PyImport_AppendInittab("gjallarbru", &PyInit_gjallarbru);
  Py_Initialize();
  run();
  //////printf("stupid");

}
*/
