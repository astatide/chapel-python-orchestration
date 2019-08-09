use valkyrie;
use ygglog;
use genes;
use messaging;

var log = new shared ygglog.YggdrasilLogging();
var vstring = new ygglog.yggHeader();

var v = new shared valkyrie.valkyrieHandler(1);
v.currentTask = 1;
v.currentLocale = here : string;
v.setSendTo();
v.yh += 'run';
v.currentNode = 'root';

for iL in v.logo {
  writeln(iL);
}
var vp = v.valhalla(1, v.id, "1200", log, vstring=vstring);

var newNode = new shared genes.GeneNode();
newNode.demeDomain.add(0);
var delta = new genes.deltaRecord();
delta += (12345, 1.0);
delta += (54334, 2.0);
var score, deme = v.processNode(newNode, delta);