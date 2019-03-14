use uuid;
use rng;
use genes;
use network;
use propagator;

var newrng = new owned rng.UDevRandomHandler();
//var new_uuid = new owned uuid.UUID();

//var A: [0] int;

var A = new shared genes.GeneNode(id='A');
//writeln(blah.id);
var B = new shared genes.GeneNode(id='B');
//writeln(blah2.id);

//var coefficients: domain(int);
//var seeds: [coefficients] int;

//seeds[164523] = 1;

//var delta = new genes.deltaRecord();
//delta.delta[12543] = 1;

//A.join(B, delta);
//var C = A.new_node(324, 1, 'C');
//var D = C.new_node(324, 1, 'D');
//var E = D.new_node(324, 1, 'E');


// E, D, C, A

var ygg = new owned network.GeneNetwork();

//forall i in {A, B, C, D, E} do {
//  Net.add_node(i);
//}
//writeln(Net.edges);

//writeln(Net.calculatePath(A.id, E.id));
//ygg.initializeNetwork();
//writeln(ygg.ids);
//writeln(ygg.edges);
//writeln(ygg.nodes);
//writeln('\n');
//ygg.testCalculatePath();
ygg.testMergeNodes();
//writeln(ygg.locale);

var v = new valkyrie;
v.moveToRoot();
ygg.move(v, '7');
//var d = ygg.move('root', '3');
//writeln(d);
//v.move(d, '7');

//writeln(v.matrixValues);

//on Locales[1 % numLocales] {
  //writeln(ygg.locale);
  //writeln(ygg.nodes.locale);
  //writeln(ygg.nodes);
  //writeln(ygg.nodes.locale);
//}
