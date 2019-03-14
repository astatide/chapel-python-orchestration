// Load up the random module.  We need randomness!
use Random;

// Chapel seems to just seed with the timer, so can we pull from the entropy source?
use IO;
// (turns out the answer is yes)

class UDevRandomHandler {
  // Swank!  This pulls in a bit stream from the entropy source woooooo.
  var Entropy = open('/dev/urandom', iomode.r);
  var EntropyStream = Entropy.reader();

  proc getrandbits(n: int) {
    // This is a function similar to the getrandbits in python; it just
    // returns a variable containing random bits.
    var x: int;
    this.EntropyStream.readbits(x, n);
    return x;
  }

  proc returnRNG() {
    var RandomNumberGenerator = makeRandomStream(real, seed=this.seed());
    return RandomNumberGenerator;
  }

  proc seed() {
    var x: int;
    return this.EntropyStream.readbits(x, 64);
  }

  proc returnSpecificRNG(seed: int) {
    var RandomNumberGenerator = makeRandomStream(real, seed=seed);
    return RandomNumberGenerator;
  }
}
