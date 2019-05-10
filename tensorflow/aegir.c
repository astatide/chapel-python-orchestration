// This is helper shim/module for TF.  AEgir is apparently a sea.
// Flow, C, sea.  Get it?  Haha ha ha ha aahhhhhhhh.
// Audrey P, Cray, Inc.  2019

// We're going to want to store a lot of structs and pointers and such.

#include <math.h>
#include <stdlib.h>
#include <stdbool.h>
#include <stdatomic.h>
//#include <TF.h> // ?  I dunno yet, don't care.  That'll matter later.
// Taken from https://github.com/tensorflow/tensorflow/blob/master/tensorflow/examples/label_image/main.cc
#include <Python.h>
#include "tensorflow/cc/ops/const_op.h"
#include "tensorflow/cc/ops/image_ops.h"
#include "tensorflow/cc/ops/standard_ops.h"
#include "tensorflow/core/framework/graph.pb.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/graph/default_device.h"
#include "tensorflow/core/graph/graph_def_builder.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/core/stringpiece.h"
#include "tensorflow/core/lib/core/threadpool.h"
#include "tensorflow/core/lib/io/path.h"
#include "tensorflow/core/lib/strings/str_util.h"
#include "tensorflow/core/lib/strings/stringprintf.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/platform/init_main.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/core/public/session.h"
#include "tensorflow/core/util/command_line_flags.h"
#include "tensorflow/core/platform/protobuf.h"


// now, create an array of sessions.  Fuck Jeff Sessions, that racist ass
// Keebler elf.  Man, what a dickwad.
TF_Session * globalSessions;

// So, we basically want to create the graph in Python, then
// export and run it here.  So we'll create a graph, add our nodes
// to it, then run it.

// A lot of this is taken from the TF link above.  Status may be a return type.
// It's also possible this won't work.  That happens, too.
Status loadGraph(TF_Session * session, TF_Buffer* graph_def) {
  // we're given a session, then we'll create and load a new graph object.
  TF_Graph *graph;
  Status setGraphStatus;

  session->reset(NewSession(SessionOptions));

  // here's where we'll want to actually do something with the graph.
  // I think the graph is stored as a protocol buffer, so if we can grab
  // that from Python, that would be great.

  // for instance, here's the line from the example:
  // ReadBinaryProto(tensorflow::Env::Default(), graph_file_name, &graph_def);
  // that loads up a binary protocol buffer from a file.  We want to use
  // an in memory object, if possible.  So if we can find the model wrapping in the
  // source...

  /*
  // Import the graph serialized in `graph_def` into `graph`.
  // Convenience function for when no results are needed.
  TF_CAPI_EXPORT extern void TF_GraphImportGraphDef(
      TF_Graph* graph, const TF_Buffer* graph_def,
      const TF_ImportGraphDefOptions* options, TF_Status* status);
  */

  // Ah!  It seems we can directly grab a graph_def object from Python!
  // from google.protobuf import text_format
  // graph_def = graph_or_graph_def.as_graph_def()
  // text_format.MessageToString(graph_def, float_format='')
  // we can convert it to text, which might be convenient for passing
  // it back (if we _could_ use the binary, we should.)
  // We should assume that we have already loaded it, somehow.

  setGraphStatus = *session->Create(graph);
  if (!setGraphStatus.ok()) {
    // we failed in this case.
    return setGraphStatus;
  }
  return OK();
}

TF_Buffer *convertTextToGraphDef(char * graph_def_text, size_t proto_len) {
  // We're going to call the function to convert blah.
  TF_Buffer * graph_def;
  // will this work?
  graph_def = TF_NewBufferFromString(graph_def_text, proto_len);
  return graph_def;
}

void setupTensorflowSessions(unsigned long long maxValkyries, char * graph_def_text, size_t proto_len ) {
  // We're going to set up our global sessions and send them on.

  TF_Graph *graph;
  TF_Status *status;
  TF_SessionOptions *opt;
  Status setGraphStatus;

  // allocate your memory foo
  globalSessions = malloc(sizeof(TF_Session*)*maxValkyries);

  graph = convertTextToGraphDef(graph_def_text, proto_len);
  for (int i = 0; i < maxValkyries; i++) {
    // create a new session, add our graph (that should work maybe?), store.
    globalSessions[i] = TF_NewSession(graph, opt, status);
    // if we need to, we can create one graph per session.  Might have to,
    // ultimately, given they will have different weights.
  }

}
