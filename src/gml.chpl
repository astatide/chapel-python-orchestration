// This is for creating a .gml file.

use IO;
use genes;

class gmlExporter {

  var fileName: string;

  proc writeHeader(f) {
    f.writeln('graph');
    f.writeln('[');
  }
  proc writeFooter(f) {
    f.writeln(']');
  }

  proc writeNode(f, node: genes.GeneNode) {
    f.writeln('  node [');
    //f.writeln('  [');
    f.writeln('    id "', node.id : string, '"');
    f.writeln('    label "', node.id : string, '"');
    f.writeln('    combination ', node.combinationID : string);
    f.writeln('    chromosome "', node.chromosome : string, '"');
    f.writeln('    deme ', node.demeDomain : string);
    f.writeln('  ]');
  }

  proc writeEdge(f, node: genes.GeneNode) {
    for edge in node.nodes {
      f.writeln('  edge [');
      //f.writeln('  [');
      f.writeln('    source "', node.id : string, '"');
      f.writeln('    target "', edge : string, '"');
      f.writeln('  ]');
    }
  }
}
