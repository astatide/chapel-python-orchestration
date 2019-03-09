// By Audrey Pratt, care of Cray

// Load up the random module.  We need randomness!
use Random;
use rng;

class UUID {
  // This is a generic class which will be able to output a UUID4 compliant ID
  // (I mean, that's the goal, anyway; who knows if I'm doing it right).
  // We want our random stuff, so.
  var entropySource = new owned UDevRandomHandler();
  var remainder: [0..#2] string;
  // We want an array to store stuff into.
  var uuid_int_rep: [1..16] uint(8);

  proc pull_random_data() {
    forall i in 1..16 do {
      // Just call the appropriate function on the entropy source.
      this.uuid_int_rep[i] = abs(this.entropySource.getrandbits(8)) : uint(8);
    }
  }
  proc convert_to_uuid4() {
    // Not entirely certain this is correct.
    this.uuid_int_rep[7] |= 64;
    this.uuid_int_rep[9] |= 2;
  }
  // This works!  Assumes... I think little endian?  Big stuff on the left.
  // Whatever, I don't have a CS degree, it's fine.  We'll sort it later.
  proc convert_to_hex(x: uint) {
    var result: uint;
    var i: uint;
    // Can I make this global?
    //var remainder: [0..#2] string;
    const hex = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'a', 'b', 'c', 'd', 'e', 'f'];
    // We have 8 bits of data; we need to split this into 4 bits and calculate
    // appropriately.
    // first, clear out the last four bits by ANDing them a number that is
    // 1 everywhere you don't want cleared, then doing a bit shift.
    this.remainder[0] = hex[(((x & 240) >> 4)+1): int];
    // now, clear out the first 4 bits.  No need to bit shift
    this.remainder[1] = hex[((x & 15)+1): int];
    return this.remainder;
  }

  proc UUID4() {
    // Currently I can't be bothered to support custom data.  Whatever.
    // I'm sure there's a more elegant solution for this, but this is just
    // proof of concept.
    var r_uuid: string;
    this.pull_random_data();
    this.convert_to_uuid4();
    r_uuid = '';
    for i in 1..16 do {
      for j in this.convert_to_hex(this.uuid_int_rep[i]) {
        r_uuid += j;
      }
    }
    r_uuid = r_uuid[1..8] + '-' + r_uuid[9..12] + '-' + r_uuid[13..16] +
             '-' + r_uuid[17..20] + '-' + r_uuid[21..32];
    return r_uuid;
  }
}
//var uuid = new owned UUID();
//uuid.UUID4();
//writeln(uuid.UUID4());
