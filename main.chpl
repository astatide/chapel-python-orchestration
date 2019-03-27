use uuid;
use rng;
use genes;
use network;
use propagator;
use spinlock;

use VisualDebug;

startVdebug("E1");

var ragnarok = new owned propagator.Propagator();
ragnarok.run();

stopVdebug();

var newrng = new owned rng.UDevRandomHandler();
