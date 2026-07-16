/**
 * xnn-utils.js
 *
 * Functional (no classes) utilities for creating, stepping, and training an
 * XNN: a fully-recurrent, continuously-running integrate-and-fire network
 * with signed weights and per-neuron bias as an atypical neural parameter
 * (ATNP), trained via local, non-backprop credit assignment.
 *
 * MODEL SHAPE (plain JSON-serializable object -- JSON.stringify(model) /
 * JSON.parse(json) work directly, no custom (de)serialization needed):
 *
 *   {
 *     matrix: Connection[][],   // matrix[i][j] = the connection FROM neuron i TO neuron j
 *     vector: Neuron[],         // vector[i] = neuron i's own state
 *   }
 *
 *   Connection = { weight: number, factors: {} }
 *     weight is bounded to [-1, 1]. `factors` is an open bag for anything
 *     else a caller wants to attach to a specific edge (unused by the core
 *     mechanics here, but reserved -- e.g. a per-edge momentum term).
 *
 *   Neuron = { state: number, factors: {...} }
 *     state is bounded to [0, 1]. `factors` holds:
 *       bias           number, [0,1] -- trained ATNP field
 *       avgActivity    number -- running estimate of this neuron's firing
 *                      rate, used ONLY by homeostatic regulation
 *       targetActivity number -- this neuron's own homeostatic target
 *       trace          number -- eligibility trace, maintained automatically
 *                      by step(); train() consumes and (by default) resets it
 *       role           string|null -- optional free-form tag ('input',
 *                      'hidden', 'output', 'value', ...), for the caller's
 *                      own bookkeeping; not read by these functions
 *
 * CORE DYNAMICS (see accompanying paper, Sections 3-4 for full derivation):
 *   - Uniform Threshold Trigger: a neuron "fires" when its state reaches 1,
 *     pushes 1*weight along each outgoing connection, then resets to 0.
 *   - State Relaxation: any neuron that didn't fire has its resulting state
 *     squared (a decay that vanishes near 0 and 1, steepest around 0.5).
 *   - Homeostatic regulation runs every tick, independent of training,
 *     continuously nudging each neuron's bias toward its own target firing
 *     rate -- the standing defense against the saturation/lockstep collapse
 *     documented at length in the paper (Sections 5.6-5.10).
 *   - Eligibility trace: step() marks any neuron that fired as eligible
 *     (trace = 1) and, unless decayed, this persists across ticks. A flat
 *     (undecayed) trace was found to outperform decaying variants on the
 *     tasks tested (paper Section 9) -- decay defaults to 1.0 (no decay)
 *     for that reason, but is configurable.
 *
 * TRAINING MODES (see train() below): 'supervised' (a real, per-component
 * error vector against a known target -- e.g. delayed-recall) and 'td' (a
 * single scalar TD error applied only to the action actually taken, plus
 * value neurons -- e.g. tic-tac-toe self-play). These are genuinely
 * different credit-assignment shapes, not two names for the same thing --
 * see the paper's discussion of why conflating them would misrepresent
 * what either system actually does.
 *
 * In both modes, hidden-layer training uses Direct Feedback Alignment: a
 * FIXED, caller-supplied random projection turns the real error (vector or
 * scalar) into a per-hidden-neuron pseudo-error, applied through the exact
 * same local column-update mechanism as everything else. No backward pass,
 * no reading of the forward weights (no "weight transport"). This was the
 * single most important fix found in the whole investigation this module
 * is drawn from -- see paper Section 9 -- and is NOT optional if you want
 * hidden-to-hidden weights to receive any training signal at all.
 */

// ---------------------------------------------------------------------------
// Internal helpers (not exported -- implementation details)
// ---------------------------------------------------------------------------

function clamp(x, lo, hi) {
  return Math.max(lo, Math.min(hi, x));
}

// Deterministic PRNG (mulberry32) so `create` can be seeded for
// reproducibility -- plain Math.random() can't be seeded in JS.
function mulberry32(seed) {
  let a = seed >>> 0;
  return function () {
    a |= 0;
    a = (a + 0x6d2b79f5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

function randRange(rng, lo, hi) {
  return lo + rng() * (hi - lo);
}

function deepCopyModel(model) {
  return {
    matrix: model.matrix.map((row) =>
      row.map((conn) => ({ weight: conn.weight, factors: { ...conn.factors } }))
    ),
    vector: model.vector.map((n) => ({ state: n.state, factors: { ...n.factors } })),
  };
}

/**
 * The one local update rule everything else in this module is built from:
 * touch ONLY the incoming edges of a single target neuron (colIndex), only
 * from source neurons with a nonzero eligibility trace, and ONLY that
 * neuron's own bias. This is the "surgical" credit assignment found
 * necessary in the paper (Sections 5.9-5.10): updating a fired neuron's
 * entire outgoing row, or every neuron's bias uniformly, causes unrelated
 * neurons/outputs to drift together and saturate as a block.
 */
function applyColumnUpdate(model, colIndex, signal, cw, cb) {
  const n = model.vector.length;
  for (let i = 0; i < n; i++) {
    const trace = model.vector[i].factors.trace || 0;
    if (trace === 0) continue;
    const conn = model.matrix[i][colIndex];
    conn.weight = clamp(conn.weight + signal * cw * trace, -1, 1);
  }
  const target = model.vector[colIndex];
  target.factors.bias = clamp((target.factors.bias || 0) + signal * cb, 0, 1);
}

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

/**
 * Create a new, randomly-initialized XNN.
 *
 * @param {number} numNeurons - total neuron count. The connection matrix is
 *   fully connected (numNeurons x numNeurons, including self-loops).
 * @param {object} [options]
 * @param {number} [options.seed=1] - seed for the internal deterministic
 *   PRNG (ignored if options.rng is given).
 * @param {function} [options.rng] - a custom PRNG returning numbers in
 *   [0, 1); overrides options.seed if provided.
 * @param {[number,number]} [options.weightRange=[-1,1]] - initial connection
 *   weight range. Signed weights are required for representational
 *   capacity -- see paper Section 4.3 (positive-only weights are provably
 *   unable to represent inhibitory relationships).
 * @param {[number,number]} [options.biasRange=[0,1]] - initial per-neuron
 *   bias range.
 * @param {number|number[]} [options.targetActivity=0.25] - homeostatic
 *   target firing rate, either one value shared by all neurons or an array
 *   of length numNeurons for a per-neuron target.
 * @param {string[]} [options.roles=null] - optional array of length
 *   numNeurons tagging each neuron's role (e.g. 'input'/'hidden'/'output'),
 *   stored in that neuron's factors.role for the caller's own bookkeeping.
 * @returns {{matrix: object[][], vector: object[]}} a new model object,
 *   directly JSON-serializable.
 */
function create(numNeurons, options = {}) {
  const {
    seed = 1,
    rng = mulberry32(seed),
    weightRange = [-1, 1],
    biasRange = [0, 1],
    targetActivity = 0.25,
    roles = null,
  } = options;

  if (roles && roles.length !== numNeurons) {
    throw new Error("create: options.roles length must equal numNeurons");
  }

  const matrix = [];
  for (let i = 0; i < numNeurons; i++) {
    const row = [];
    for (let j = 0; j < numNeurons; j++) {
      row.push({ weight: randRange(rng, weightRange[0], weightRange[1]), factors: {} });
    }
    matrix.push(row);
  }

  const vector = [];
  for (let i = 0; i < numNeurons; i++) {
    const thisTarget = Array.isArray(targetActivity) ? targetActivity[i] : targetActivity;
    vector.push({
      state: 0,
      factors: {
        bias: randRange(rng, biasRange[0], biasRange[1]),
        avgActivity: thisTarget,
        targetActivity: thisTarget,
        trace: 0,
        role: roles ? roles[i] : null,
      },
    });
  }

  return { matrix, vector };
}

/**
 * Advance the model by one tick. Does not mutate the model passed in --
 * operates on, and returns, a copy.
 *
 * @param {object} model
 * @param {Object.<number,number>} [inputs={}] - a map of neuron index ->
 *   value to force-clamp AFTER this tick's dynamics are computed, overriding
 *   whatever the network itself computed for that neuron. This is the ONLY
 *   forced exception to ordinary evolution (the IO convention used
 *   throughout the paper) -- everything else, including every hidden
 *   neuron's state, evolves purely from the network's own dynamics.
 * @param {object} [options]
 * @param {number} [options.homeostaticLr=0.002] - how fast bias chases the
 *   activity target.
 * @param {number} [options.activityEmaAlpha=0.02] - smoothing rate of the
 *   running per-neuron activity estimate.
 * @param {number} [options.traceDecay=1.0] - eligibility trace decay factor
 *   applied each tick before marking newly-fired neurons; 1.0 means no
 *   decay (flat, whole-episode credit), which outperformed decaying
 *   variants in the tasks this module is drawn from (paper Section 9) --
 *   change this only if you have a specific reason to expect otherwise for
 *   your own task.
 * @returns {object} a new model object (the input is not mutated).
 */
function step(model, inputs = {}, options = {}) {
  const { homeostaticLr = 0.002, activityEmaAlpha = 0.02, traceDecay = 1.0 } = options;

  const next = deepCopyModel(model);
  const n = next.vector.length;

  // Which neurons fired, based on the INCOMING model's states.
  const fired = model.vector.map((neuron) => neuron.state >= 1);

  // Continuous homeostatic regulation -- independent of any training signal.
  for (let i = 0; i < n; i++) {
    const f = next.vector[i].factors;
    f.avgActivity = (1 - activityEmaAlpha) * f.avgActivity + activityEmaAlpha * (fired[i] ? 1 : 0);
    const target = f.targetActivity != null ? f.targetActivity : 0.25;
    f.bias = clamp(f.bias + homeostaticLr * (target - f.avgActivity), 0, 1);
  }

  // Incoming drive: sum of 1 * weight over all fired source neurons.
  const incoming = new Array(n).fill(0);
  for (let i = 0; i < n; i++) {
    if (!fired[i]) continue;
    const row = next.matrix[i];
    for (let j = 0; j < n; j++) {
      incoming[j] += row[j].weight;
    }
  }

  // Reset-on-fire, then relaxation (squaring) for neurons that didn't fire.
  for (let i = 0; i < n; i++) {
    const baseline = fired[i] ? 0 : model.vector[i].state;
    const raw = clamp(baseline + incoming[i] + next.vector[i].factors.bias, 0, 1);
    next.vector[i].state = raw > 0 ? raw * raw : 0;
  }

  // Clamp externally-supplied inputs, overriding the dynamics above.
  for (const idxStr of Object.keys(inputs)) {
    next.vector[Number(idxStr)].state = inputs[idxStr];
  }

  // Eligibility trace: decay first, then mark this tick's firers.
  for (let i = 0; i < n; i++) {
    const f = next.vector[i].factors;
    f.trace = (f.trace || 0) * traceDecay;
    if (fired[i]) f.trace = 1;
  }

  return next;
}

/**
 * Apply one local training update. Does not mutate the model passed in.
 *
 * Two supported modes, matching two genuinely different credit-assignment
 * shapes (see module docstring):
 *
 *   mode: 'supervised' -- a real, per-component error against a known
 *   target vector (e.g. a classification-style task with a definite
 *   correct answer).
 *     options.outputIndices   number[] - neuron indices forming the readout
 *     options.target          number[] - same length as outputIndices
 *     options.hiddenIndices    number[] [optional] - neurons to train via DFA
 *     options.feedbackMatrixB  number[][] [optional] - shape
 *       hiddenIndices.length x outputIndices.length; FIXED (never trained
 *       by this function) random projection matrix
 *
 *   mode: 'td' -- a single scalar TD error, applied only to the one action
 *   actually taken plus a set of value-estimate neurons (e.g. an
 *   actor-critic policy with no "correct" output vector to compare against).
 *     options.chosenIndex      number [optional] - the action's output neuron
 *     options.valueIndices     number[] [optional] - value-readout neurons
 *     options.tdError          number - reward + nextValue - previousValue,
 *       computed by the caller
 *     options.hiddenIndices    number[] [optional] - neurons to train via DFA
 *     options.feedbackVectorB  number[] [optional] - length
 *       hiddenIndices.length; FIXED random per-neuron projection (a vector,
 *       not a matrix, since the signal being projected is a single scalar)
 *
 * Shared options: cw (weight learning rate, default 0.05), cb (bias
 * learning rate, default 0.05), resetTrace (default true -- zero every
 * neuron's eligibility trace after applying this update, so the next
 * stretch of step() calls starts crediting from a clean slate).
 *
 * @param {object} model
 * @param {object} options
 * @returns {object} a new model object (the input is not mutated).
 */
function train(model, options = {}) {
  const { mode, cw = 0.05, cb = 0.05, resetTrace = true } = options;

  const next = deepCopyModel(model);

  if (mode === "supervised") {
    const { outputIndices, target, hiddenIndices = [], feedbackMatrixB = null } = options;
    if (!outputIndices || !target || outputIndices.length !== target.length) {
      throw new Error("train(mode='supervised'): outputIndices and target must be given and equal length");
    }

    const error = outputIndices.map((idx, k) => target[k] - next.vector[idx].state);

    // Output layer: the real, known error -- one component per neuron.
    outputIndices.forEach((idx, k) => {
      applyColumnUpdate(next, idx, error[k], cw, cb);
    });

    // Hidden layer: Direct Feedback Alignment. A fixed projection of the
    // real error into a per-neuron pseudo-error, no weight transport.
    if (feedbackMatrixB && hiddenIndices.length) {
      hiddenIndices.forEach((idx, h) => {
        const row = feedbackMatrixB[h];
        let pseudoError = 0;
        for (let k = 0; k < error.length; k++) pseudoError += row[k] * error[k];
        applyColumnUpdate(next, idx, pseudoError, cw, cb);
      });
    }
  } else if (mode === "td") {
    const {
      chosenIndex = null,
      valueIndices = [],
      tdError,
      hiddenIndices = [],
      feedbackVectorB = null,
    } = options;
    if (tdError == null) {
      throw new Error("train(mode='td'): options.tdError is required");
    }

    // Actor: only the action actually taken gets the (only) real signal --
    // every other output neuron is untouched by this transition.
    if (chosenIndex != null) {
      applyColumnUpdate(next, chosenIndex, tdError, cw, cb);
    }

    // Critic: value neurons share the same scalar TD error.
    valueIndices.forEach((idx) => applyColumnUpdate(next, idx, tdError, cw, cb));

    // Hidden layer: DFA variant for a scalar signal -- a fixed random
    // VECTOR (not matrix) projects tdError into a per-neuron pseudo-error.
    if (feedbackVectorB && hiddenIndices.length) {
      hiddenIndices.forEach((idx, h) => {
        applyColumnUpdate(next, idx, tdError * feedbackVectorB[h], cw, cb);
      });
    }
  } else {
    throw new Error(`train: unknown mode "${mode}" -- expected "supervised" or "td"`);
  }

  if (resetTrace) {
    next.vector.forEach((neuron) => {
      neuron.factors.trace = 0;
    });
  }

  return next;
}

module.exports = { create, step, train };