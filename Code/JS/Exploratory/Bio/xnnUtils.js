/**
 * xnn-utils.js (organoid variant)
 *
 * A from-scratch rebuild around bio-analogous local learning: spatially-
 * embedded, Dale's-Law-constrained, sparse-but-densely-stored topology;
 * pure spiking dynamics (leaky integrate-and-fire, rate-coded readout,
 * distance-dependent conduction delay); continuous spike-timing-dependent
 * plasticity (STDP) gated by a diffuse ambient neuromodulator standing in
 * for volume-transmitted reward signals (dopamine-like reward-prediction-
 * error); and slow structural plasticity (synapse formation/pruning) on
 * top of the fast weight dynamics.
 *
 * PARAMETER GROUNDING: default constants are set from real published
 * values wherever they exist, not invented and presented as if grounded --
 * each is cited inline. One tick is treated as approximately 1ms of real
 * time throughout, so STDP windows and conduction delay can be specified
 * in physiologically meaningful, mutually consistent units:
 *   - E/I ratio 80/20, inhibitory synapses ~2x excitatory strength
 *     (Braitenberg & Schuz 1998; canonical cortical value, widely used in
 *     spiking models; inhibitory/excitatory strength asymmetry is commonly
 *     modeled up to ~8x in balanced-network literature -- 2x used here as
 *     a conservative default)
 *   - STDP time constants tau+ = tau- = 15ms, window ~40ms (Bi & Poo 1998;
 *     Sjostrom & Gerstner 2010 review cites ~10-20ms as the standard range)
 *   - Conduction velocity ~1 m/s (unmyelinated axon range is ~0.5-10 m/s;
 *     organoid/cultured tissue is unmyelinated, so the slow end of the
 *     range is used, not the fast myelinated-tract value)
 *   - Target avalanche size distribution: power law with exponent -1.5
 *     (Beggs & Plenz 2003; replicated extensively since) -- see the
 *     accompanying analysis script, not asserted here, since this is a
 *     property to MEASURE from the network's own spontaneous activity,
 *     not something that can be hard-coded into a parameter.
 *   - Maturation to network-level synchrony genuinely takes weeks to
 *     months in real organoids/cultures (multiple MEA studies), not a
 *     handful of ticks -- structuralPlasticityFloor and maturityDecayRate
 *     are chosen so the model's own "developmental" window is long
 *     relative to a single task-learning run, not fast by construction.
 *
 * PUBLIC INTERFACE: create() and step() only. There is no separate train()
 * -- real synaptic plasticity is not a distinct mode a neuron enters, it is
 * continuous and intrinsic to ordinary operation, so local STDP and
 * neuromodulator-gated consolidation both live inside step(), always on.
 *
 * MODEL SHAPE (plain JSON-serializable object):
 *   {
 *     matrix: Connection[][],  // matrix[i][j] = the connection FROM i TO j
 *     vector: Neuron[],
 *     factors: { ...ambient, model-wide state, shared by every neuron }
 *   }
 *
 * "Dense in the abstract, sparse in implementation" is the explicit design
 * philosophy here: the matrix stays a full N x N array (as in the prior
 * LFA-oriented version of this module) with weight = 0 wherever a
 * connection does not exist, so nothing about the PUBLIC SHAPE assumes or
 * requires sparse storage -- but internally, `factors.exists` is tracked as
 * a first-class, independently-flippable fact, separate from weight, so a
 * future rewrite of the storage backend (e.g. adjacency lists, matching
 * real synapse counts that stay roughly constant regardless of network
 * size) could replace how matrix[i][j] is answered without changing what
 * it means to ask for it.
 *
 *   Connection = { weight: number, factors: {
 *     exists            boolean -- structural fact, set by the slow
 *                        structural-plasticity process below, NEVER by
 *                        ordinary consolidation. A connection with
 *                        exists=false always has weight=0 and cannot
 *                        transmit, but still accumulates eligibility --
 *                        sustained correlated activity is what can
 *                        eventually justify a real synapse forming there.
 *     eligibility       number, signed -- continuously-decaying STDP
 *                        trace; positive when this connection's pre/post
 *                        timing was potentiation-favorable, negative when
 *                        depression-favorable.
 *     lastConsolidated  tick of this connection's last non-negligible
 *                        weight change -- used to detect long-quiet
 *                        connections eligible for pruning.
 *     delay             ticks of propagation delay, fixed at creation from
 *                        distance / conduction velocity (minimum 1 -- no
 *                        connection is ever instantaneous).
 *   }}
 *
 *   Neuron = { state: [0,1], factors: {
 *     type              'excitatory' | 'inhibitory' (Dale's Law: fixed at
 *                        creation, constrains the SIGN of every one of
 *                        this neuron's existing outgoing weights forever)
 *     position           number[] -- fixed spatial coordinate, drives
 *                        distance-dependent connection probability AND
 *                        conduction delay
 *     restingPotential, leakRate   -- real exponential leak toward rest
 *                        when not firing, replacing the earlier squaring
 *                        relaxation (which had no biological grounding)
 *     threshold          fires when state >= threshold; this is now what
 *                        homeostasis adjusts (not an additive bias)
 *     avgActivity, targetActivity  -- drive homeostatic threshold movement
 *     lastSpikeTick      most recent firing tick, or null -- the whole
 *                        basis for STDP timing comparisons
 *     recentSpikeCount   exponentially-weighted recent spike rate -- THE
 *                        readout mechanism now; nothing reads raw `state`
 *                        for a decision, only this (rate coding)
 *     isSpontaneous, spontaneousRate  -- pacemaker neurons that can fire
 *                        stochastically, independent of input entirely
 *   }}
 *
 * Model-level factors = {
 *   ambientModulator          diffuse neuromodulator level, decays every
 *                             tick, gates how strongly eligibility
 *                             consolidates into real weight change.
 *                             Readable by any caller like everything else
 *                             in `factors`; the only way to influence it
 *                             from outside is the `feedback` argument to
 *                             step() (never direct mutation -- step never
 *                             mutates its input, same as everywhere else
 *                             in this project).
 *   ambientDecay              per-tick decay rate on ambientModulator
 *   age                       ticks processed since creation
 *   structuralPlasticityRate  how readily exists flags can flip right now;
 *                             starts high, decays toward a floor with age
 *                             -- volatile young connectivity stabilizing
 *                             with maturity
 *   pendingDrive              ring buffer of future-arriving spike drive,
 *                             one array of length N per delay slot -- the
 *                             mechanism behind conduction delay. Internal
 *                             bookkeeping; not meant to be hand-edited.
 *   rngState                  serializable PRNG state (a plain integer),
 *                             so the model can make its own random
 *                             decisions (spontaneous firing, structural
 *                             plasticity) every tick while remaining pure
 *                             and JSON-serializable -- no live closures
 *                             stored anywhere in the object.
 * }
 */

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

function clip(x, lo, hi) {
  return Math.max(lo, Math.min(hi, x));
}

// Pure, stateless PRNG step (mulberry32-derived): takes a state integer,
// returns { value, nextState }. Never a closure -- always just numbers --
// so the model can carry its own RNG state as an ordinary, serializable
// field and advance it one call at a time.
function rngStep(state) {
  let a = state | 0;
  a = (a + 0x6d2b79f5) | 0;
  let t = Math.imul(a ^ (a >>> 15), 1 | a);
  t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
  const value = ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  return { value, nextState: a };
}

function distance(a, b) {
  let sum = 0;
  for (let k = 0; k < a.length; k++) sum += (a[k] - b[k]) ** 2;
  return Math.sqrt(sum);
}

// ---------------------------------------------------------------------------
// create
// ---------------------------------------------------------------------------

/**
 * Create a new, spatially-embedded, Dale's-Law-respecting spiking network.
 *
 * @param {number} n - neuron count.
 * @param {object} [options]
 * @param {number} [options.seed=1]
 * @param {number} [options.dimensions=3] - spatial embedding dimensions
 *   (3 by default, matching real tissue rather than an arbitrary abstraction).
 * @param {number} [options.spaceSize=10] - neurons are placed uniformly in
 *   a [0, spaceSize]^dimensions cube.
 * @param {number} [options.excitatoryFraction=0.8] - fraction of neurons
 *   assigned excitatory type (roughly matching cortical E/I ratio).
 * @param {number} [options.connectionDecayLength=3] - distance scale over
 *   which connection probability falls off (exponential decay).
 * @param {number} [options.baseConnectionProb=0.5] - connection
 *   probability at distance 0.
 * @param {number} [options.maxWeight=1] - maximum connection weight magnitude.
 * @param {number} [options.spontaneousFraction=0] - fraction of neurons
 *   designated as spontaneously-active pacemakers.
 * @param {number} [options.spontaneousRate=0.01] - per-tick firing
 *   probability for spontaneous neurons, independent of input/state.
 * @param {number} [options.defaultThreshold=1]
 * @param {number} [options.targetActivity=0.2] - homeostatic target firing rate.
 * @returns {object} a new model, directly JSON-serializable.
 */
function create(n, options = {}) {
  const {
    seed = 1,
    dimensions = 3,
    spaceSize = 10,
    excitatoryFraction = 0.8, // Braitenberg & Schuz 1998; canonical cortical E/I ratio
    connectionDecayLength = 3,
    baseConnectionProb = 0.25, // EMPIRICALLY CALIBRATED, not a guess: the
    // earlier default of 0.5 produced deeply supercritical, runaway
    // spontaneous activity (avalanches spanning nearly the whole
    // simulation, no power-law signature); 0.15 was subcritical (avalanches
    // die out too fast, exponent -2.4 to -3.0 vs the real target of -1.5,
    // Beggs & Plenz 2003). 0.25 was the closest approach found to that
    // target during calibration (exponent ~-1.1; see analyze-avalanches.js)
    // -- still an approximation, not an exact match. See that script's
    // notes on why hitting -1.5 precisely likely needs either a finer
    // search or a genuine self-organizing mechanism, which this network
    // does not yet have.
    maxWeight = 1,
    inhibitoryWeightScale = 2, // inhibitory synapses modeled as several-fold stronger
    // than excitatory in balanced-network literature (up to ~8x); 2x used
    // here as a conservative default, not the extreme cited value.
    conductionVelocity = 1, // "distance units" per tick; ~1 m/s, the slow
    // (unmyelinated) end of the real 0.5-10 m/s range, matching organoid/
    // cultured tissue rather than myelinated adult white-matter tracts.
    maxDelay = 20,
    spontaneousFraction = 0,
    spontaneousRate = 0.01,
    defaultThreshold = 1,
    targetActivity = 0.2,
  } = options;

  let rngState = seed >>> 0;
  const nextRandom = () => {
    const r = rngStep(rngState);
    rngState = r.nextState;
    return r.value;
  };

  const vector = [];
  for (let i = 0; i < n; i++) {
    const position = Array.from({ length: dimensions }, () => nextRandom() * spaceSize);
    const type = nextRandom() < excitatoryFraction ? "excitatory" : "inhibitory";
    const isSpontaneous = nextRandom() < spontaneousFraction;
    vector.push({
      state: 0,
      factors: {
        type,
        position,
        restingPotential: 0,
        leakRate: 0.1,
        threshold: defaultThreshold,
        avgActivity: targetActivity,
        targetActivity,
        lastSpikeTick: null,
        recentSpikeCount: 0,
        isSpontaneous,
        spontaneousRate: isSpontaneous ? spontaneousRate : 0,
      },
    });
  }

  const matrix = [];
  for (let i = 0; i < n; i++) {
    const row = [];
    for (let j = 0; j < n; j++) {
      const d = distance(vector[i].factors.position, vector[j].factors.position);
      const prob = baseConnectionProb * Math.exp(-d / connectionDecayLength);
      const exists = nextRandom() < prob;
      const delay = Math.min(maxDelay, Math.max(1, Math.round(d / conductionVelocity)));
      let weight = 0;
      if (exists) {
        const isExcitatory = vector[i].factors.type === "excitatory";
        const magnitude = nextRandom() * maxWeight * (isExcitatory ? 1 : inhibitoryWeightScale);
        weight = isExcitatory ? magnitude : -magnitude;
      }
      row.push({ weight, factors: { exists, eligibility: 0, lastConsolidated: 0, delay } });
    }
    matrix.push(row);
  }

  const pendingDrive = Array.from({ length: maxDelay + 1 }, () => new Array(n).fill(0));

  return {
    matrix,
    vector,
    factors: {
      ambientModulator: 0,
      ambientDecay: 0.05,
      age: 0,
      structuralPlasticityRate: 1.0,
      pendingDrive,
      rngState,
    },
  };
}

// ---------------------------------------------------------------------------
// step
// ---------------------------------------------------------------------------

/**
 * Advance the model by one tick. Never mutates its input -- returns a new
 * model. This single function does everything: spiking dynamics,
 * homeostasis, continuous STDP eligibility accumulation, ambient
 * neuromodulator decay/delivery, modulator-gated consolidation into real
 * weight change, and (occasionally) slow structural plasticity.
 *
 * @param {object} model
 * @param {Object.<number,number>} [inputs={}] - neuron index -> forced
 *   state value, applied AFTER all dynamics this tick -- the one forced
 *   exception to ordinary evolution, exactly as in every prior version of
 *   this module.
 * @param {number} [feedback=0] - external reward/neuromodulator signal
 *   delivered this tick. This is the ENTIRE external-feedback interface --
 *   added into the (already-decaying) ambient modulator, never a direct
 *   mutation of model.factors.
 * @param {object} [options] - tunable constants; see inline defaults.
 * @returns {object} a new model.
 */
function step(model, inputs = {}, feedback = 0, options = {}) {
  const {
    homeostaticLr = 0.002,
    activityEmaAlpha = 0.02,
    thresholdBounds = [0.3, 1.0],
    stdpWindow = 40, // ms; Sjostrom & Gerstner 2010 review, ~10-20ms tau -> ~40ms window
    tauPlus = 15, // ms; Bi & Poo 1998 / standard STDP review value range 10-20ms
    tauMinus = 15,
    aPlus = 0.5,
    aMinus = 0.5,
    eligibilityDecay = 0.05,
    consolidationRate = 0.05,
    maxWeight = 1,
    spikeWindowDecay = 0.9,
    structuralPlasticityInterval = 50,
    growthThreshold = 3.0,
    growthProb = 0.05,
    pruneAge = 200,
    pruneProb = 0.05,
    maturityDecayRate = 0.999, // slow -- real organoid maturation takes
    // weeks to months (multiple MEA studies), not a handful of ticks; see
    // module doc comment.
    structuralPlasticityFloor = 0.05,
  } = options;

  const n = model.vector.length;
  const age = model.factors.age;
  const maxDelay = model.factors.pendingDrive.length - 1;

  let rngState = model.factors.rngState;
  const nextRandom = () => {
    const r = rngStep(rngState);
    rngState = r.nextState;
    return r.value;
  };

  // 1. Firing: threshold crossing, or a spontaneous roll for pacemakers.
  const fired = model.vector.map((neuron) => {
    if (neuron.state >= neuron.factors.threshold) return true;
    if (neuron.factors.isSpontaneous && nextRandom() < neuron.factors.spontaneousRate) return true;
    return false;
  });

  // 2. Incoming drive: ONLY what was scheduled to arrive THIS tick (slot 0
  // of the pending-drive ring buffer) -- every connection has a real,
  // distance-derived conduction delay (minimum 1 tick), so nothing is ever
  // instantaneous, matching real axonal propagation rather than the
  // zero-delay assumption every earlier version of this module made.
  const incoming = model.factors.pendingDrive[0];
  const pendingDrive = model.factors.pendingDrive.slice(1);
  pendingDrive.push(new Array(n).fill(0));

  for (let i = 0; i < n; i++) {
    if (!fired[i]) continue;
    const row = model.matrix[i];
    for (let j = 0; j < n; j++) {
      if (!row[j].factors.exists) continue;
      const slot = row[j].factors.delay - 1; // delay>=1, slot 0 of `pendingDrive` = next tick
      if (slot >= 0 && slot < maxDelay) pendingDrive[slot][j] += row[j].weight;
    }
  }

  // 3. Leaky integrate-and-fire state update (replaces the earlier,
  // biologically-ungrounded squaring relaxation): fired neurons reset to
  // resting potential; others leak a fraction of the way toward it before
  // incoming (delayed) drive is added.
  const newState = new Array(n);
  for (let i = 0; i < n; i++) {
    const f = model.vector[i].factors;
    const baseline = fired[i]
      ? f.restingPotential
      : model.vector[i].state + f.leakRate * (f.restingPotential - model.vector[i].state);
    newState[i] = clip(baseline + incoming[i], 0, 1);
  }

  // 4. Homeostasis adjusts THRESHOLD now, not an additive bias -- and spike
  // bookkeeping (lastSpikeTick for STDP, recentSpikeCount for readout).
  const newNeuronFactors = model.vector.map((neuron, i) => {
    const f = { ...neuron.factors };
    f.avgActivity = (1 - activityEmaAlpha) * f.avgActivity + activityEmaAlpha * (fired[i] ? 1 : 0);
    f.threshold = clip(
      f.threshold + homeostaticLr * (f.avgActivity - f.targetActivity),
      thresholdBounds[0],
      thresholdBounds[1]
    );
    if (fired[i]) f.lastSpikeTick = age;
    f.recentSpikeCount = f.recentSpikeCount * spikeWindowDecay + (fired[i] ? 1 : 0);
    return f;
  });

  // 5. Continuous STDP: decay every existing/eligible connection's trace,
  // then apply potentiation (post fired, check recent pre) and depression
  // (pre fired, check recent post) from spike-timing coincidence.
  const newMatrix = model.matrix.map((row) =>
    row.map((conn) => ({ weight: conn.weight, factors: { ...conn.factors } }))
  );

  for (let i = 0; i < n; i++) {
    for (let j = 0; j < n; j++) {
      const c = newMatrix[i][j].factors;
      if (c.exists || c.eligibility !== 0) c.eligibility *= 1 - eligibilityDecay;
    }
  }

  for (let j = 0; j < n; j++) {
    if (!fired[j]) continue;
    for (let i = 0; i < n; i++) {
      const preLast = model.vector[i].factors.lastSpikeTick;
      if (preLast != null && preLast !== age && age - preLast <= stdpWindow) {
        newMatrix[i][j].factors.eligibility += aPlus * Math.exp(-(age - preLast) / tauPlus);
      }
    }
  }
  for (let i = 0; i < n; i++) {
    if (!fired[i]) continue;
    for (let j = 0; j < n; j++) {
      const postLast = model.vector[j].factors.lastSpikeTick;
      if (postLast != null && postLast !== age && age - postLast <= stdpWindow) {
        newMatrix[i][j].factors.eligibility -= aMinus * Math.exp(-(age - postLast) / tauMinus);
      }
    }
  }

  // 6. Ambient neuromodulator: decay, then deliver this tick's feedback --
  // the entire external-feedback interface.
  const ambientModulator = model.factors.ambientModulator * (1 - model.factors.ambientDecay) + feedback;

  // 7. Consolidation: eligibility becomes real weight change, gated by the
  // CURRENT ambient modulator level -- three-factor plasticity. Dale's Law
  // is enforced here too (not just at creation): sign is clamped to match
  // the source neuron's type, every time.
  for (let i = 0; i < n; i++) {
    const isExcitatory = newNeuronFactors[i].type === "excitatory";
    for (let j = 0; j < n; j++) {
      const c = newMatrix[i][j];
      if (!c.factors.exists) continue;
      const delta = consolidationRate * ambientModulator * c.factors.eligibility;
      if (Math.abs(delta) < 1e-9) continue;
      c.weight = isExcitatory ? clip(c.weight + delta, 0, maxWeight) : clip(c.weight + delta, -maxWeight, 0);
      c.factors.lastConsolidated = age;
    }
  }

  // 8. Structural plasticity: rare, slow, scaled by the model's current
  // structuralPlasticityRate. Sustained high eligibility on a NONEXISTENT
  // connection can grow a new synapse; a long-quiet EXISTING connection
  // can be pruned. This is deliberately a different, slower process than
  // consolidation -- ordinary weight updates can never create or destroy
  // a connection on their own.
  if (age % structuralPlasticityInterval === 0) {
    const rate = model.factors.structuralPlasticityRate;
    for (let i = 0; i < n; i++) {
      const isExcitatory = newNeuronFactors[i].type === "excitatory";
      for (let j = 0; j < n; j++) {
        const c = newMatrix[i][j];
        if (!c.factors.exists) {
          if (Math.abs(c.factors.eligibility) >= growthThreshold && nextRandom() < rate * growthProb) {
            c.factors.exists = true;
            c.weight = isExcitatory ? 0.1 : -0.1;
            c.factors.lastConsolidated = age;
          }
        } else if (age - c.factors.lastConsolidated > pruneAge && nextRandom() < rate * pruneProb) {
          c.factors.exists = false;
          c.weight = 0;
        }
      }
    }
  }

  // 9. Clamp external sensory inputs -- the one forced exception, applied
  // last, overriding whatever the dynamics above computed.
  for (const idxStr of Object.keys(inputs)) {
    newState[Number(idxStr)] = inputs[idxStr];
  }

  const vector = newState.map((state, i) => ({ state, factors: newNeuronFactors[i] }));
  const structuralPlasticityRate = Math.max(
    structuralPlasticityFloor,
    model.factors.structuralPlasticityRate * maturityDecayRate
  );

  return {
    matrix: newMatrix,
    vector,
    factors: {
      ambientModulator,
      ambientDecay: model.factors.ambientDecay,
      age: age + 1,
      structuralPlasticityRate,
      pendingDrive,
      rngState,
    },
  };
}

module.exports = { create, step };