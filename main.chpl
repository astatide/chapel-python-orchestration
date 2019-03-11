use uuid;
use rng;
use genes;
use network;

var newrng = new owned rng.UDevRandomHandler();
//var new_uuid = new owned uuid.UUID();

//var A: [0] int;

var A = new owned genes.GeneNode(id='A');
//writeln(blah.id);
var B = new owned genes.GeneNode(id='B');
//writeln(blah2.id);

//var coefficients: domain(int);
//var seeds: [coefficients] int;

//seeds[164523] = 1;

var delta = new genes.deltaRecord();
delta.delta[12543] = 1;

A.join(B, delta);
var C = A.new_node(324, 1);

var Net = new owned network.GeneNetwork();

Net.add_nodes({A, B, C});
writeln(Net.edges);
