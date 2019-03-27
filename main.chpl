use uuid;
use rng;
use genes;
use network;
use propagator;
use spinlock;

use VisualDebug;

startVdebug("E1");

// We want to capture kill signals.  @LouisJenkinsCS helped me with this.
extern proc signal(sigNum : c_int, handler : c_fn_ptr) : c_fn_ptr;

var ragnarok = new owned propagator.Propagator();

proc handler(x : int) : void {
    ragnarok.setShutdown();
}

signal(2, c_ptrTo(handler));

ragnarok.run();

stopVdebug();

var newrng = new owned rng.UDevRandomHandler();
