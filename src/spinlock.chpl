use ygglog;
use SysError;

//config var lockLog = false;

class TooManyLocksError : Error {
  proc init() { }
}

class SpinLock {
  //var l: chpl__processorAtomicType(bool);
  var l: atomic bool;
  var log: shared ygglog.YggdrasilLogging;
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

  proc lock(hstring: ygglog.yggHeader) throws {
    if this.lockLog {
      this.log.debug('locking', this.t, hstring);
    }
    while l.read() || l.testAndSet(memory_order_acquire) do chpl_task_yield();
    this.n.add(1);
    if this.n.read() != 1 {
      this.log.critical('CRITICAL FAILURE: During lock, spinlock has been acquired multiple times on', this.t, hstring);
      throw new owned TooManyLocksError();
    }
    if this.lockLog {
      this.log.debug(this.t, 'lock successful; locks - ', this.n.read() : string, hstring);
    }
  }

  proc unlock(hstring: ygglog.yggHeader) throws {
    this.n.sub(1);
    if this.n.read() != 0 {
      writeln("Too many locks on: ", this.t : string, " N: ", this.n.read());
      this.log.critical('CRITICAL FAILURE: During unlock, spinlock has been acquired multiple times on', this.t, hstring);
      throw new owned TooManyLocksError();
    }
    l.clear(memory_order_release);
    if this.lockLog {
      this.log.debug(this.t, 'unlocked; locks - ', this.n.read() : string, hstring);
    }
  }

  // Ha, we're still using this for the logs.
  proc lock() {
    while l.read() || l.testAndSet(memory_order_acquire) do chpl_task_yield();
  }

  proc unlock() {
    l.clear(memory_order_release);
  }

  proc rl(hstring: ygglog.yggHeader) {
    // This checks to see whether the write lock is active, and if not,
    // allows reads.
    while this.writeLock.read() >= 1 do chpl_task_yield();
    this.readHandles.add(1);
    if this.lockLog {
      this.log.debug('Locked RL on ', this.t : string, 'handles open -', this.readHandles.read() : string, hstring);
    }
  }

  proc url(hstring: ygglog.yggHeader) {
    if this.lockLog {
      this.log.debug('Releasing read lock on', this.t, hstring);
    }
    this.readHandles.sub(1);
    if this.lockLog {
      this.log.debug('Unlocked RL on ', this.t : string, 'handles open -', this.readHandles.read() : string, hstring);
    }
  }

  proc wl(hstring: ygglog.yggHeader) {
    // While we are actively reading, we do not write.
    if this.lockLog {
      this.log.debug('Requesting write lock on', this.t, 'current RL', this.readHandles.read() : string, hstring);
    }
    this.writeLock.add(1);
    while this.readHandles.read() != 0 do chpl_task_yield();
    if this.lockLog {
      this.log.debug('Write lock obtained on', this.t, 'current readHandles:', this.readHandles.read() : string, hstring);
    }
    this.lock(hstring);
  }

  proc uwl(hstring: ygglog.yggHeader) {
    this.writeLock.sub(1);
    if this.lockLog {
      this.log.debug('Releasing write lock on', this.t, 'current WL', this.writeLock.read() : string, hstring);
    }
    this.unlock(hstring);
  }

  proc wl() {
    this.wl(new ygglog.yggHeader());
  }

  proc uwl() {
    this.uwl(new ygglog.yggHeader());
  }

  proc rl() {
    this.rl(new ygglog.yggHeader());
  }

  proc url() {
    this.url(new ygglog.yggHeader());
  }
}

class NetworkSpinLock : SpinLock {
      var l: atomic bool;
      var log: shared ygglog.YggdrasilLogging;
      var n: atomic int;
      var t: string;
      var writeLock: atomic int;
      var readHandles: atomic int;
      var lockLog: bool;
}
