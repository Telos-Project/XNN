/**
 * xnn-utils.js (organoid variant)
 *
 * A bio-analogous local-learning substrate, built bottom-up from single-cell
 * dynamics: spatially-embedded, Dale's-Law-constrained, sparse-but-densely-
 * stored topology; spiking dynamics with a choice of two per-neuron models
 * (leaky integrate-and-fire with refractory period and spike-frequency
 * adaptation, or a full Izhikevich two-variable model capable of genuine
 * bursting); continuous spike-timing-dependent plasticity (STDP) split into
 * an always-on Hebbian pathway and an additional pathway gated by a diffuse
 * ambient neuromodulator (standing in for volume-transmitted, dopamine-like
 * reward-prediction-error); and slow structural plasticity (synapse
 * formation/pruning) on top of the fast weight dynamics.
 *
 * FACTORS-ONLY PHILOSOPHY: every quantity OXNN's own dynamics actually
 * depend on lives in `factors` -- `weight` (per connection) and `state`
 * (per neuron) are NOT used internally anywhere in this module. They are
 * deliberately KEPT in the data model, set to simple binary values (weight:
 * 1 if a connection exists else 0; state: 1 if a neuron fired this tick
 * else 0), specifically so other XNN variants built on more traditional
 * machine-learning conventions -- for which a connection's weight and a
 * neuron's activation are the load-bearing quantities -- remain
 * structurally compatible with an OXNN model. The REAL synaptic efficacy
 * lives in `factors.strength`; the REAL LIF membrane potential lives in
 * `factors.membranePotential` (Izhikevich-mode neurons use `factors.v`/`u`
 * instead, already factors-resident since that mode was introduced).
 * step() only ever touches top-level `weight`/`state` in two places: when a
 * connection's existence actually changes (creation, or structural growth/
 * pruning), and when writing this tick's binary fired-indicator for every
 * neuron -- never as part of ordinary consolidation, drive computation, or
 * firing determination, all of which read/write `factors` exclusively.
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
 *     not something that can be hard-coded into a parameter. NOTE: this
 *     was calibrated against the bare LIF neuron, before refractory period,
 *     adaptation, and Izhikevich mode existed -- not yet re-verified since.
 *   - Maturation to network-level synchrony genuinely takes weeks to
 *     months in real organoids/cultures (multiple MEA studies), not a
 *     handful of ticks -- structuralPlasticityFloor and maturityDecayRate
 *     are chosen so the model's own "developmental" window is long
 *     relative to a single task-learning run, not fast by construction.
 *   - ISI irregularity: real cortical neurons fire with coefficient of
 *     variation (CV) near 1 at high rates, close to a Poisson process
 *     (Softky & Koch 1993) -- and, critically, the paper's own central
 *     finding is that PLAIN leaky-integrate-fire with steady input FAILS
 *     to reproduce this (fires too regularly). History of measurements on
 *     this module: bare threshold-crossing LIF gave CV ~2.0 (too bursty,
 *     traced to avalanche-style cascading dynamics); refractory period +
 *     adaptation brought it to ~1.3; re-measured after the criticality
 *     recalibration below, the SAME LIF+adaptation network measured ~1.06
 *     with zero noise added -- a large, incidental improvement from fixing
 *     an unrelated parameter, worth remembering when calibrating one
 *     target can silently move another. An Izhikevich RS/FS network with
 *     homeostatic current (see homeostaticCurrent below) solved network-
 *     wide silence but measured CV ~0.45-0.53 -- too regular -- because
 *     homeostatic stabilization is definitionally variance-reducing.
 *     RESOLVED (substantially): adding genuine stochastic fluctuation to
 *     injected current, independent of the deterministic homeostatic bias
 *     (see noiseAmplitude below), closed most of this gap directly --
 *     CV ~0.93 at the best-found noise amplitude, WHILE fully preserving
 *     the activity fix (150/150 neurons stayed active across the entire
 *     tested noise range) -- confirming the literature's own explanation
 *     (fluctuating synaptic input, not smoother regulation) directly,
 *     rather than just plausibly. Not an exact match, and the relationship
 *     is non-monotonic (CV declines again well past the best-found
 *     amplitude) -- the right noise scale must be found per context, the
 *     same way izhikevichCurrentScale must be.
 *   - Firing-rate distribution shape and small-world topological structure
 *     (clustering, path length) are both real, checkable published targets
 *     that have not yet been measured against this substrate at all.
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
 * philosophy here: the matrix stays a full N x N array with weight = 0
 * wherever a connection does not exist, so nothing about the PUBLIC SHAPE
 * assumes or requires sparse storage -- but internally, `factors.exists` is
 * tracked as a first-class, independently-flippable fact, separate from
 * weight, so a future rewrite of the storage backend (e.g. adjacency lists,
 * matching real synapse counts that stay roughly constant regardless of
 * network size -- necessary for ever approaching organoid/brain scale, where
 * a dense N^2 array becomes intractable) could replace how matrix[i][j] is
 * answered without changing what it means to ask for it. This has NOT yet
 * been built; current scale is capped at a few hundred neurons.
 *
 *   Connection = { weight: [0,1] BINARY existence marker only, factors: {
 *     strength          number, signed -- the REAL synaptic efficacy (what
 *                        `weight` held directly before this refactor). All
 *                        internal dynamics (drive computation, consolidation,
 *                        Dale's Law enforcement) read/write this, never
 *                        top-level `weight`.
 *     exists            boolean -- structural fact, set by the slow
 *                        structural-plasticity process below, NEVER by
 *                        ordinary consolidation. A connection with
 *                        exists=false always has weight=0 and strength=0,
 *                        and cannot transmit, but still accumulates
 *                        eligibility -- sustained correlated activity is
 *                        what can eventually justify a real synapse
 *                        forming there.
 *     eligibility       number, signed -- continuously-decaying STDP
 *                        trace; positive when this connection's pre/post
 *                        timing was potentiation-favorable, negative when
 *                        depression-favorable. Tracked unconditionally,
 *                        regardless of whether consolidation is active.
 *     lastConsolidated  tick of this connection's last non-negligible
 *                        strength change.
 *     lastActive        tick this connection's |eligibility| last crossed
 *                        a meaningful threshold -- what pruning actually
 *                        keys on. Deliberately SEPARATE from
 *                        lastConsolidated: an earlier version tied pruning
 *                        to consolidation firing, which meant every
 *                        connection looked permanently "unconsolidated"
 *                        whenever ambient feedback was absent (see
 *                        hebbianRate/modulatoryRate below), causing
 *                        indiscriminate pruning unrelated to real activity.
 *     delay             ticks of propagation delay, fixed at creation from
 *                        distance / conduction velocity (minimum 1 -- no
 *                        connection is ever instantaneous).
 *   }}
 *
 *   Neuron = { state: [0,1] BINARY fired-indicator only (1 if fired this
 *              tick, else 0) -- NOT the real dynamical variable, factors: {
 *     type              'excitatory' | 'inhibitory' (Dale's Law: fixed at
 *                        creation, constrains the SIGN of every one of
 *                        this neuron's existing outgoing weights forever,
 *                        enforced at every consolidation step, not just
 *                        creation)
 *     position           number[] -- fixed spatial coordinate, drives
 *                        distance-dependent connection probability AND
 *                        conduction delay
 *     dynamicsModel     'lif' (default) | 'izhikevich' -- per-neuron choice.
 *                        Switching a neuron to izhikevich mode changes which
 *                        of the two field groups below are meaningful; the
 *                        unused group is simply ignored, not removed, so a
 *                        neuron's mode can be inspected or changed freely
 *                        without restructuring the object.
 *
 *     -- LIF-mode fields --
 *     membranePotential  the REAL dynamical variable (what top-level
 *                        `state` held directly before this refactor).
 *                        Firing determination, the leak, and the threshold
 *                        comparison all read/write this, never `state`.
 *     restingPotential, leakRate   -- real exponential leak toward rest
 *                        when not firing (replaces an earlier, biologically
 *                        ungrounded squaring relaxation).
 *     threshold          fires when membranePotential >= threshold; this is what
 *                        homeostasis adjusts for LIF neurons (not an
 *                        additive bias).
 *     refractoryPeriod   ticks after firing during which the neuron cannot
 *                        fire again, regardless of state.
 *     adaptation         slow recovery variable: builds with each spike,
 *                        decays otherwise, subtracted from drive --
 *                        spike-frequency adaptation, abstracted from the
 *                        real slow-potassium-current mechanism (not a
 *                        channel-level simulation). Single-cell testing
 *                        (isolated, plasticity disabled) confirmed this
 *                        reproduces two real classes -- Fast-Spiking
 *                        (minimal adaptation, sustained regular firing) and
 *                        Regular-Spiking (a decelerating transient settling
 *                        into a stable rate) -- but NOT genuine bursting: a
 *                        single linearly-decaying variable settles to a
 *                        fixed point, not a limit cycle, regardless of
 *                        parameters tested.
 *
 *     -- Izhikevich-mode fields (Izhikevich, 2003) --
 *     v, u               the two coupled state variables (membrane
 *                        potential and recovery). Fires when v>=30 (a fixed
 *                        biophysical constant -- NOT adjusted by
 *                        homeostasis, since doing so would blur the
 *                        class-defining a/b/c/d parameters below).
 *     izhikevichA/B/C/D  the four parameters that set firing class. Verified
 *                        (isolated single-cell, plasticity disabled) to
 *                        reproduce five real, distinct classes from
 *                        published parameter sets: Regular Spiking,
 *                        Fast Spiking, Intrinsically Bursting, Chattering,
 *                        and Low-Threshold Spiking -- INCLUDING genuine
 *                        repeating burst-then-silence cycles (confirmed via
 *                        exact-match repeating ISI patterns) that LIF+
 *                        adaptation could not reach. Current-dependent
 *                        transitions within a class (e.g. RS developing a
 *                        sharper initial burst at higher drive; chattering
 *                        becoming near-continuous at very high drive) were
 *                        also reproduced without any special-casing -- both
 *                        are real, independently documented phenomena.
 *     homeostaticCurrent slow, adjustable tonic bias current -- the
 *                        Izhikevich analogue of LIF's threshold homeostasis,
 *                        added to injected current rather than adjusting
 *                        the fixed v>=30 threshold. Verified to fully
 *                        resolve an under-activity problem in a network
 *                        context (46/150 -> 150/150 neurons active), but
 *                        this pushes CV further from the Softky-Koch target,
 *                        not closer -- see PARAMETER GROUNDING above.
 *
 *     -- Fields used by both modes --
 *     avgActivity, targetActivity  -- drive homeostatic adjustment (of
 *                        threshold for LIF, homeostaticCurrent for
 *                        Izhikevich).
 *     lastSpikeTick      most recent firing tick, or null -- the whole
 *                        basis for STDP timing comparisons, computed
 *                        identically regardless of which dynamics model
 *                        produced the spike (verified directly: STDP
 *                        eligibility and consolidation both work correctly
 *                        for connections originating from Izhikevich-mode
 *                        neurons, with no special-casing required).
 *     recentSpikeCount   exponentially-weighted recent spike rate -- THE
 *                        readout mechanism; nothing reads raw `state` for a
 *                        decision, only this (rate coding -- the simplest
 *                        of three real candidate codes, chosen for
 *                        tractability, not confirmed as correct; still the
 *                        least-tested design choice in this module).
 *     isSpontaneous, spontaneousRate  -- pacemaker neurons that can fire
 *                        stochastically, independent of input entirely.
 *                        NOTE: only implemented for LIF-mode neurons; an
 *                        Izhikevich-mode neuron's firing is currently
 *                        determined purely by its v/u dynamics, with no
 *                        equivalent stochastic-pacemaker option.
 *   }}
 *
 * Model-level factors = {
 *   ambientModulator          diffuse neuromodulator level, decays every
 *                             tick, scales the modulatory (reward-gated)
 *                             consolidation pathway -- see hebbianRate /
 *                             modulatoryRate below. Readable by any caller
 *                             like everything else in `factors`; the only
 *                             way to influence it from outside is the
 *                             `feedback` argument to step() (never direct
 *                             mutation -- step never mutates its input).
 *   ambientDecay              per-tick decay rate on ambientModulator.
 *   age                       ticks processed since creation.
 *   structuralPlasticityRate  how readily exists flags can flip right now;
 *                             starts high, decays toward a floor with age
 *                             -- volatile young connectivity stabilizing
 *                             with maturity.
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
 *
 * CONSOLIDATION (the weight-change rule): TWO plasticity pathways run
 * simultaneously, not one gated pathway --
 *
 *   delta = eligibility * (hebbianRate + modulatoryRate * ambientModulator)
 *
 * An earlier, single-rate version (delta = rate * ambientModulator *
 * eligibility) meant weight change was EXACTLY zero whenever no ambient
 * feedback was delivered, regardless of eligibility -- verified directly:
 * feeding the network strongly correlated input with feedback held at zero
 * throughout built genuine eligibility (0.86) while not one connection's
 * weight moved. This matters specifically because purely cortical tissue
 * (this module's target) has no intrinsic dopaminergic source at all --
 * three-factor, reward-gated plasticity is real but specific (corticostriatal
 * synapses), not something to assume as freely available everywhere.
 * hebbianRate is always-on, pattern-driven learning -- verified directly on
 * an isolated connection: weight climbed from 0.2 to saturation purely from
 * repeated correlated firing, with feedback held at exactly zero the whole
 * run (67 repetitions to saturate; adding periodic reward on top reduced
 * this to 4, confirming both pathways genuinely add together rather than
 * one overriding the other). Setting hebbianRate=0 recovers the old,
 * strictly-gated behavior if that's ever wanted.
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

// Box-Muller transform: two uniform draws -> one approximately-Gaussian
// sample. Real synaptic bombardment noise is commonly modeled as
// approximately Gaussian (diffusion approximation of many small Poisson
// inputs summing), so this is the natural choice over uniform noise.
function gaussianNoise(nextRandom) {
  const u1 = Math.max(nextRandom(), 1e-10); // avoid log(0)
  const u2 = nextRandom();
  return Math.sqrt(-2 * Math.log(u1)) * Math.cos(2 * Math.PI * u2);
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
    baseConnectionProb = 0.245, // RE-CALIBRATED (see conversation): the
    // original calibration (0.25, exponent ~-1.1) was measured on the bare
    // threshold-crossing neuron, BEFORE refractory period and spike-
    // frequency adaptation existed. Re-checking against the current, richer
    // neuron model found real drift: 0.25 now measures exponent ~-0.47
    // (supercritical -- refractory gating forces cascades to spread across
    // MORE neurons to sustain themselves, apparently pushing avalanches
    // larger, not smaller). The critical transition is also much NARROWER
    // now than before (0.24 subcritical at -2.05, 0.25 supercritical at
    // -0.47 -- almost the whole usable range within 0.01). 0.245 was the
    // closest re-calibrated approach found (exponent ~-1.59, target -1.5,
    // Beggs & Plenz 2003) -- tighter than the original approximation, but
    // still not exact, and still requires deliberate external tuning
    // rather than genuine self-organization. Re-verify again after any
    // further change to single-neuron dynamics (e.g. noise injection).
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
      state: 0, // BINARY fired-indicator only (1 if fired this tick, else 0) --
      // NOT the real dynamical variable. Kept in the data model, alongside
      // `weight` below, specifically so other XNN variants built on
      // traditional ML conventions remain structurally compatible; OXNN's
      // own step() never reads it for internal dynamics, only writes it
      // for external/compat consumers. See module doc comment.
      factors: {
        type,
        position,
        membranePotential: 0, // the REAL LIF dynamical variable (what `state`
        // used to hold directly). Izhikevich-mode neurons use v/u instead
        // (already factors-resident) and ignore this field entirely.
        restingPotential: 0,
        leakRate: 0.1,
        threshold: defaultThreshold,
        avgActivity: targetActivity,
        targetActivity,
        lastSpikeTick: null,
        recentSpikeCount: 0,
        isSpontaneous,
        spontaneousRate: isSpontaneous ? spontaneousRate : 0,
        refractoryPeriod: 2, // ticks after firing during which the neuron cannot fire again
        adaptation: 0, // slow recovery variable: builds with each spike, decays otherwise,
        // subtracted from drive -- spike-frequency adaptation (abstracted from the
        // real slow-potassium-current mechanism, not simulating channel kinetics)
        dynamicsModel: "lif", // 'lif' (default, unchanged) | 'izhikevich' -- an alternate,
        // optional per-neuron dynamics mode (Izhikevich, 2003): two coupled variables
        // (v, u) with a nonlinear v-u interaction, still an abstract reduction (not
        // channel-level simulation), but capable of genuine bursting limit cycles
        // that linear-subtraction adaptation on plain LIF cannot reach.
        v: -65, u: -13, // Izhikevich state, only meaningful when dynamicsModel='izhikevich'
        homeostaticCurrent: 0, // slow, adjustable tonic bias current -- the
        // Izhikevich analogue of LIF's threshold homeostasis. v>=30 is a fixed
        // biophysical constant (adjusting it would blur the RS/FS/IB/CH class
        // distinctions, which live in a/b/c/d, not the threshold), so this
        // adjusts overall excitability instead, the same PRINCIPLE as LIF's
        // mechanism applied through a different, class-appropriate lever.
        izhikevichA: 0.02, izhikevichB: 0.2, izhikevichC: -65, izhikevichD: 8, // RS defaults
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
      let strength = 0;
      if (exists) {
        const isExcitatory = vector[i].factors.type === "excitatory";
        const magnitude = nextRandom() * maxWeight * (isExcitatory ? 1 : inhibitoryWeightScale);
        strength = isExcitatory ? magnitude : -magnitude;
      }
      // `weight` is BINARY ONLY (1 if exists, 0 if not) -- not the real
      // synaptic efficacy. Kept in the data model, alongside neuron `state`
      // above, so other XNN variants built on traditional ML conventions
      // remain structurally compatible with OXNN models; OXNN's own step()
      // never reads it for internal dynamics (only factors.strength), and
      // only writes it when a connection's existence actually changes
      // (creation here, or structural growth/pruning in step()).
      row.push({
        weight: exists ? 1 : 0,
        factors: { exists, strength, eligibility: 0, lastConsolidated: 0, lastActive: 0, delay },
      });
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
    hebbianRate = 0.01, // always-on, pattern-driven consolidation -- what
    // lets the network learn from input structure alone with zero feedback
    modulatoryRate = 0.05, // additional reward-scaled consolidation on top
    // (matches the old single-rate default's magnitude when feedback is present)
    activityThreshold = 0.05, // |eligibility| above this counts as "genuinely active" for pruning purposes
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
    adaptationIncrement = 0.15, // added to adaptation on each spike
    adaptationDecay = 0.05, // per-tick decay of adaptation back toward 0 (slower than
    // ordinary state leak -- adaptation is meant to accumulate across several
    // spikes, not reset each tick)
    izhikevichCurrentScale = 1, // DEFAULT IS 1 -- i.e. no scaling, preserving
    // exact backward compatibility with every already-validated single-cell
    // test (which set connection weight directly AS the injected current,
    // e.g. weight=15 meaning I=15). A network with realistically bounded,
    // sparse synaptic weights (unlike an isolated test's single artificial
    // "current injection" connection) needs a MUCH larger scale passed
    // explicitly -- this is a context-specific parameter, not a new global
    // default, precisely to avoid silently invalidating prior results.
    izhikevichHomeostaticLr = 0.3, // adjustment rate for homeostaticCurrent;
    // much larger than LIF's homeostaticLr (0.002) because Izhikevich's
    // current scale (units of ~10s) is far larger than LIF's threshold
    // scale (bounded to [0.3, 1.0]) -- these are not directly comparable rates.
    noiseAmplitude = 0, // DEFAULT IS 0 -- no change to any existing result
    // unless explicitly enabled, same backward-compatibility discipline as
    // izhikevichCurrentScale. Genuine per-tick stochastic fluctuation, added
    // to drive INDEPENDENTLY of the deterministic homeostatic bias --
    // testing the literature's actual explanation for real cortical ISI
    // irregularity (fluctuating synaptic bombardment, not smooth
    // regulation), which the homeostatic mechanisms alone cannot provide
    // (they are definitionally variance-reducing; see conversation).
    // Same units as `incoming` for LIF neurons; same units as injected
    // current (post izhikevichCurrentScale) for Izhikevich neurons -- the
    // two are NOT on the same numeric scale, matching the same reasoning
    // that made izhikevichCurrentScale a separate parameter.
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

  // Incoming drive is read first: every connection has delay >= 1 tick, so
  // this tick's incoming was fully determined by PRIOR ticks' firing --
  // never a same-tick dependency. This lets Izhikevich-mode neurons (whose
  // firing decision is computed FROM incoming, unlike LIF's threshold
  // check against already-settled state) be resolved in the same pass as
  // everything else, with no ordering conflict.
  const incoming = model.factors.pendingDrive[0];
  const pendingDrive = model.factors.pendingDrive.slice(1);
  pendingDrive.push(new Array(n).fill(0));

  // 1. Firing, per neuron's own dynamics model:
  //  - 'lif' (default): threshold crossing (refractory-gated), or a
  //    spontaneous roll for pacemakers.
  //  - 'izhikevich': the coupled (v, u) update (Izhikevich, 2003) is
  //    computed HERE, using this tick's incoming as injected current --
  //    genuinely capable of bursting limit cycles that linear-subtraction
  //    adaptation on plain LIF cannot reach, at similar computational cost.
  const izhikevichNext = {}; // neuron index -> { v, u } for izhikevich-mode neurons this tick
  const fired = model.vector.map((neuron, i) => {
    const f = neuron.factors;
    if (f.dynamicsModel === "izhikevich") {
      const I = incoming[i] * izhikevichCurrentScale + f.homeostaticCurrent + gaussianNoise(nextRandom) * noiseAmplitude;
      let v = f.v, u = f.u;
      const dv = 0.04 * v * v + 5 * v + 140 - u + I;
      v = v + dv;
      u = u + f.izhikevichA * (f.izhikevichB * v - u);
      let didFire = false;
      if (v >= 30) {
        didFire = true;
        v = f.izhikevichC;
        u = u + f.izhikevichD;
      }
      izhikevichNext[i] = { v, u };
      return didFire;
    }
    const inRefractory = f.lastSpikeTick != null && age - f.lastSpikeTick < f.refractoryPeriod;
    if (inRefractory) return false;
    if (f.membranePotential >= f.threshold) return true;
    if (f.isSpontaneous && nextRandom() < f.spontaneousRate) return true;
    return false;
  });

  // 2. Schedule THIS tick's firing for delayed future delivery -- reads
  // factors.strength (the real synaptic efficacy), NOT `weight` (which is
  // now purely a binary existence marker, unused in any internal dynamics).
  for (let i = 0; i < n; i++) {
    if (!fired[i]) continue;
    const row = model.matrix[i];
    for (let j = 0; j < n; j++) {
      if (!row[j].factors.exists) continue;
      const slot = row[j].factors.delay - 1; // delay>=1, slot 0 of `pendingDrive` = next tick
      if (slot >= 0 && slot < maxDelay) pendingDrive[slot][j] += row[j].factors.strength;
    }
  }

  // 3. Leaky integrate-and-fire MEMBRANE POTENTIAL update (factors-resident;
  // replaces the earlier, biologically-ungrounded squaring relaxation):
  // fired neurons reset to resting potential; others leak a fraction of the
  // way toward it before incoming (delayed) drive is added, with spike-
  // frequency ADAPTATION subtracted as a persistent, slowly-decaying
  // suppressive current -- abstracted from the real slow-potassium
  // mechanism, not a channel-level simulation. `newState` (top-level) is
  // computed separately below and is ALWAYS just a binary fired-indicator,
  // for both dynamics models -- never the real dynamical variable.
  const newMembranePotential = new Array(n);
  const newState = new Array(n);
  for (let i = 0; i < n; i++) {
    const f = model.vector[i].factors;
    newState[i] = fired[i] ? 1 : 0;
    if (f.dynamicsModel === "izhikevich") continue; // real state lives in factors.v/u instead
    const baseline = fired[i]
      ? f.restingPotential
      : f.membranePotential + f.leakRate * (f.restingPotential - f.membranePotential);
    newMembranePotential[i] = clip(baseline + incoming[i] - f.adaptation + gaussianNoise(nextRandom) * noiseAmplitude, 0, 1);
  }

  // 4. Homeostasis adjusts THRESHOLD now, not an additive bias -- and spike
  // bookkeeping (lastSpikeTick for STDP, recentSpikeCount for readout,
  // adaptation build-up/decay for spike-frequency adaptation). Izhikevich-
  // mode neurons skip threshold homeostasis entirely (their firing
  // threshold is the fixed v>=30 crossing, not an adjustable field) and
  // persist their own (v, u) instead.
  const newNeuronFactors = model.vector.map((neuron, i) => {
    const f = { ...neuron.factors };
    if (f.dynamicsModel === "izhikevich") {
      f.v = izhikevichNext[i].v;
      f.u = izhikevichNext[i].u;
      if (fired[i]) f.lastSpikeTick = age;
      f.recentSpikeCount = f.recentSpikeCount * spikeWindowDecay + (fired[i] ? 1 : 0);
      f.avgActivity = (1 - activityEmaAlpha) * f.avgActivity + activityEmaAlpha * (fired[i] ? 1 : 0);
      f.homeostaticCurrent = clip(
        f.homeostaticCurrent + izhikevichHomeostaticLr * (f.targetActivity - f.avgActivity),
        -15,
        15
      );
      return f;
    }
    f.avgActivity = (1 - activityEmaAlpha) * f.avgActivity + activityEmaAlpha * (fired[i] ? 1 : 0);
    f.threshold = clip(
      f.threshold + homeostaticLr * (f.avgActivity - f.targetActivity),
      thresholdBounds[0],
      thresholdBounds[1]
    );
    if (fired[i]) f.lastSpikeTick = age;
    f.recentSpikeCount = f.recentSpikeCount * spikeWindowDecay + (fired[i] ? 1 : 0);
    f.adaptation = f.adaptation * (1 - adaptationDecay) + (fired[i] ? adaptationIncrement : 0);
    f.membranePotential = newMembranePotential[i];
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

  // 7. Consolidation: TWO plasticity pathways run simultaneously, not one
  // gated pathway (see conversation -- the earlier single-rate, modulator-
  // multiplied formula produced EXACTLY zero weight change whenever
  // ambientModulator was 0, meaning the network could not learn from input
  // structure alone at all). Real cortical synapses show baseline Hebbian/
  // STDP plasticity independent of neuromodulation, with dopamine-gated
  // three-factor plasticity acting as an ADDITIONAL amplifying pathway
  // (well-established specifically at corticostriatal synapses), not a
  // strict on/off switch for all plasticity everywhere. hebbianRate is the
  // always-on, pattern-driven term; modulatoryRate scales with the current
  // ambient level on top of it. Setting hebbianRate=0 recovers the old,
  // strictly-gated behavior if that's ever wanted. Dale's Law is enforced
  // here too (not just at creation): sign is clamped to match the source
  // neuron's type, every time. Updates factors.strength -- the real
  // synaptic efficacy -- NEVER the top-level `weight`, which stays a pure
  // existence marker throughout ordinary plasticity.
  for (let i = 0; i < n; i++) {
    const isExcitatory = newNeuronFactors[i].type === "excitatory";
    for (let j = 0; j < n; j++) {
      const c = newMatrix[i][j];
      if (!c.factors.exists) continue;
      if (Math.abs(c.factors.eligibility) >= activityThreshold) c.factors.lastActive = age;
      const delta = c.factors.eligibility * (hebbianRate + modulatoryRate * ambientModulator);
      if (Math.abs(delta) < 1e-9) continue;
      c.factors.strength = isExcitatory
        ? clip(c.factors.strength + delta, 0, maxWeight)
        : clip(c.factors.strength + delta, -maxWeight, 0);
      c.factors.lastConsolidated = age;
    }
  }

  // 8. Structural plasticity: rare, slow, scaled by the model's current
  // structuralPlasticityRate. Sustained high eligibility on a NONEXISTENT
  // connection can grow a new synapse; a long-quiet EXISTING connection can
  // be pruned. Pruning is gated on `lastActive` (genuine eligibility
  // quiescence, updated above regardless of whether a weight change
  // occurred) rather than `lastConsolidated` (which only ever updates when
  // delta is non-negligible) -- with hebbianRate=0 and no feedback, the old
  // version would have made every connection look permanently "unconsoli-
  // dated" and thus indiscriminately prune-eligible, independent of whether
  // it was actually correlated or quiet. That bug is fixed by tracking
  // activity itself, separately from whether it happened to produce a
  // nonzero weight update.
  if (age % structuralPlasticityInterval === 0) {
    const rate = model.factors.structuralPlasticityRate;
    for (let i = 0; i < n; i++) {
      const isExcitatory = newNeuronFactors[i].type === "excitatory";
      for (let j = 0; j < n; j++) {
        const c = newMatrix[i][j];
        if (!c.factors.exists) {
          if (Math.abs(c.factors.eligibility) >= growthThreshold && nextRandom() < rate * growthProb) {
            c.factors.exists = true;
            c.weight = 1; // binary existence marker
            c.factors.strength = isExcitatory ? 0.1 : -0.1; // real synaptic efficacy
            c.factors.lastConsolidated = age;
            c.factors.lastActive = age;
          }
        } else if (age - c.factors.lastActive > pruneAge && nextRandom() < rate * pruneProb) {
          c.factors.exists = false;
          c.weight = 0;
          c.factors.strength = 0;
        }
      }
    }
  }

  // 9. Clamp external sensory inputs -- the one forced exception, applied
  // last, overriding whatever the dynamics above computed. CRITICAL: this
  // must write the REAL dynamical variable (factors.membranePotential for
  // LIF, factors.v for Izhikevich), not just the now-cosmetic top-level
  // `state` -- otherwise firing determination on the NEXT tick would read
  // membranePotential/v (unaffected by the clamp) and silently ignore
  // clamped sensory input entirely. `state` is still set too, for
  // consistency/compat, but it no longer drives anything internally.
  for (const idxStr of Object.keys(inputs)) {
    const idx = Number(idxStr);
    const value = inputs[idxStr];
    newState[idx] = value;
    if (newNeuronFactors[idx].dynamicsModel === "izhikevich") {
      newNeuronFactors[idx].v = value;
    } else {
      newNeuronFactors[idx].membranePotential = value;
    }
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