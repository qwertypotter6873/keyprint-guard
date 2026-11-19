/** One user submission at a time; changing context invalidates unfinished work. */
export class ActionGate {
  constructor() {
    this.generation = 0;
    this.pending = false;
  }

  invalidate() {
    this.generation += 1;
  }

  async run(action) {
    if (this.pending) return false;
    this.pending = true;
    const generation = this.generation;
    try {
      await action(() => this.generation === generation);
      return true;
    } finally {
      this.pending = false;
    }
  }
}
