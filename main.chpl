use uuid;
use rng;
use gene_network;

var newrng = new owned rng.UDevRandomHandler();
//var new_uuid = new owned uuid.UUID();

//var A: [0] int;

var A = new owned gene_network.GeneNode();
//writeln(blah.id);
var B = new owned gene_network.GeneNode(id='0');
//writeln(blah2.id);

//var coefficients: domain(int);
//var seeds: [coefficients] int;

//seeds[164523] = 1;

var delta = new gene_network.deltaRecord();
delta.delta[12543] = 1;

A.join(B, delta);
