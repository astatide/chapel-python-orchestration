use SysError;

//config var lockLog = false;

class TooManyLocksError : Error {
  proc init() { }
}

class SpinLock {
  //var l: chpl__processorAtomicType(bool);
  var l: atomic bool;
  //var n: chpl__processorAtomicType(int);
  var n: atomic int;
  var t: string;
  //var writeLock: chpl__processorAtomicType(int);
  //var readHandles: chpl__processorAtomicType(int);
  var writeLock: atomic int;
  var readHandles: atomic int;
  //// The locks are noisy, but we do need to debug them sometimes.
  // This shuts them up unless you really want them to sing.  Their song is
  // a terrible noise; an unending screech which ends the world.
  // (okay, they're just super verbose)
  var lockLog: bool;

  proc lock() throws {
    while l.read() || l.testAndSet(memory_order_acquire) do chpl_task_yield();
    this.n.add(1);
    if this.n.read() != 1 {
      throw new owned TooManyLocksError();
    }
  }

  proc unlock() throws {
    this.n.sub(1);
    if this.n.read() != 0 {
      throw new owned TooManyLocksError();
    }
    l.clear(memory_order_release);
  }

  proc rl() {
    // This checks to see whether the write lock is active, and if not,
    // allows reads.
    while this.writeLock.read() >= 1 do chpl_task_yield();
    this.readHandles.add(1);
  }

  proc url() {
    this.readHandles.sub(1);
  }

  proc wl() {
    // While we are actively reading, we do not write.
    this.writeLock.add(1);
    while this.readHandles.read() != 0 do chpl_task_yield();
    this.lock();
  }

  proc uwl() {
    this.writeLock.sub(1);
    this.unlock();
  }
}


class NetworkSpinLock : SpinLock {
      var l: atomic bool;
      var n: atomic int;
      var t: string;
      var writeLock: atomic int;
      var readHandles: atomic int;
      var lockLog: bool;
}