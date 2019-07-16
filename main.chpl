use uuid;
use rng;
use genes;
use network;
use propagator;
use spinlock;
use Spawn;

//use VisualDebug;

// We want to capture kill signals.  @LouisJenkinsCS helped me with this.
//extern proc signal(sigNum : c_int, handler : c_fn_ptr) : c_fn_ptr;


writeln("STARTING YGGDRASIL");
coforall L in Locales {
  on L do {
    // kill all valkyries
    var vp = spawn(["pkill", "-9", "valkyrie"], stdout=FORWARD, stderr=FORWARD, stdin=FORWARD, locking=true);  
  }
}
coforall L in Locales {
  if (propagator.useLocale0 || !(L == Locales[0])) {
    on L do {
      var ragnarok = new shared propagator.Propagator(propagator.maxValkyries);
      ragnarok.initRun();
      //proc handler(x : int) : void {
      //    ragnarok.setShutdown();
      //}
      // Capturing sigint.
      //signal(2, c_ptrTo(handler));
      ragnarok.run(L);
    }
  }
}

var newrng = new owned rng.UDevRandomHandler();
