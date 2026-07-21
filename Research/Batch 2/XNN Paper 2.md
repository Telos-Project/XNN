# On OXNNs (Organic Cross Neural Networks) and On Lessons Learned as Applicable to Further Research

## Abstract

Cross Neural Networks (XNNs) are fully-recurrent, continuously-operating neural graphs trained without backpropagation, proposed as a substrate closer to organic cognition than conventional deep learning. This paper documents a phase of research that took that proposal literally: rebuilding the architecture from the ground up around real neuroscience, and testing it against the most direct empirical benchmark available — DishBrain, the 2022 experiment in which a dish of living cortical neurons learned to play Pong using no reward signal at all, only the tendency of biological tissue to prefer predictable stimulation over noise. We call this variant **OXNN (Organic XNN)**: spatially-embedded, Dale's-Law-respecting, spiking topology; continuous spike-timing-dependent plasticity (STDP) gated by a diffuse neuromodulator; and a dual-pathway consolidation rule that, unlike prior designs, can learn from input structure alone with zero reward, a capability we verify directly. We calibrate the substrate's spontaneous dynamics against a real, checkable signature of healthy cortical tissue — neuronal-avalanche criticality — and then attempt to replicate DishBrain's actual mechanism in miniature. The attempt fails, decisively and instructively: an initially promising result is traced to a connectivity confound and vanishes once corrected; a winner-take-all collapse pathology — the third independent appearance of this exact failure mode across this research program — is diagnosed, only partially mitigated, and shown to persist regardless of scale; and a wide, systematic search across regulatory mechanisms (synaptic scaling, inhibitory homeostasis, fast activity damping, richer stimulus design) neither reproduces the effect nor identifies a configuration that comes close. We treat this as a genuine, informative negative result rather than a dead end: it narrows a previously open-ended question to two concrete, separately testable hypotheses, and leaves behind a working, biologically-grounded local-learning engine whose core plasticity claim — pure Hebbian learning from pattern alone — is independently verified and real.

---

## 1. Introduction

An XNN is the simplest possible recurrent substrate: every neuron connects to every other neuron (including itself) through a signed weight, and the network runs continuously rather than being invoked function-call style. Two properties distinguish it from conventional neural networks: **ASNP** (asynchronous neural processing — the network never stops, has no forward/backward pass distinction, and can be queried or interrupted at any instant) and **ATNP** (atypical neural parameters — signed weights and per-neuron bias, rather than the uniform, unsigned units typical of standard architectures). Prior work in this research program established that local, non-backpropagation learning rules can train such networks at all — closing most of the gap to a gradient-checked baseline on a genuine memory task, and producing real self-play competence with zero labelled data.

This phase asks a different, harder question: not whether local learning works, but whether the *architecture itself* can be made to resemble real neural tissue closely enough to test against it directly. We call the result **OXNN**. The concrete benchmark is DishBrain (Kagan et al., 2022, *Neuron*): roughly 800,000 living cortical neurons on a multi-electrode array, taught to play Pong using no external reward channel — only the free-energy-principle claim (Friston et al.) that biological neurons intrinsically prefer predictable input over noise, and will self-organize toward actions that produce it. This paper is the account of building OXNN to attempt that replication, what happened, and what it teaches about where the real obstacles lie.

---

## 2. From XNN to OXNN: Grounding Topology in Biology

Every prior XNN in this program was fully, densely connected with no notion of physical space. Real cortical tissue is not: connectivity is sparse and falls off with distance, neurons are exclusively excitatory or inhibitory (**Dale's Law**), and signal transmission takes real time. OXNN's topology is built directly from these facts, each grounded in a specific published estimate rather than an invented constant:

| Property | OXNN default | Source |
|---|---|---|
| Excitatory/inhibitory ratio | 80% / 20% | Braitenberg & Schüz, 1998 — canonical cortical estimate |
| Inhibitory synapse strength | 2× excitatory (conservative) | Balanced-network literature cites up to ~8× |
| Connection probability | exponential decay with distance | standard cortical wiring assumption |
| Conduction velocity | ~1 m/s | unmyelinated axon range (~0.5–10 m/s); organoid/cultured tissue is unmyelinated |
| STDP time constant | τ ≈ 15ms, window ≈ 40ms | Bi & Poo, 1998; Sjöström & Gerstner, 2010 review |

**Design philosophy: dense in the abstract, sparse in implementation.** The connection matrix remains a full N×N array — a connection that doesn't exist simply has weight 0 — but *existence* is tracked as an independent, structural fact (`exists`), never conflated with weight. This keeps the public interface stable while leaving room for a future, genuinely sparse storage backend (matching the fact that real synapse counts per neuron stay roughly constant regardless of brain size, which a dense array does not respect at scale) without changing what it means to query a connection. The same principle that let earlier XNN work extend arbitrary task metadata through an open `factors` field, without ever touching the core module, is what makes this substitution possible later without disruption now.

**Data structure.** A model is a plain, JSON-serializable object with three parts: a connection matrix, a neuron vector, and — new in OXNN — a **model-level `factors` object** for ambient, shared state (a diffuse neuromodulator level, a developmental-age counter, a structural-plasticity rate that decays with age, and the model's own serializable pseudo-random state, so the network can make its own stochastic decisions — spontaneous firing, structural rewiring — while remaining pure and fully reconstructable from its own saved data). Connection-level `factors` hold `exists`, a signed STDP eligibility trace, a genuine quiescence marker (`lastActive`), and a fixed conduction delay. Neuron-level `factors` hold Dale's-Law type, spatial position, leak/resting-potential parameters, a threshold (what homeostasis now adjusts, replacing the earlier additive-bias mechanism), and a rate-coded spike-count readout.

**Public interface: `create` and `step` only.** OXNN has no separate training function. Real synaptic plasticity is not a distinct mode a neuron enters — it is continuous and intrinsic to ordinary operation — so local STDP and neuromodulator-gated consolidation both live inside `step`, running every tick regardless of whether anything externally interesting is happening:

```javascript
function create(n, options) {
  // assign each neuron a spatial position, a Dale's-Law type, a threshold;
  // for every (i, j) pair, compute connection probability from distance,
  // draw existence, and if it exists, a signed weight and a conduction
  // delay derived from distance / conductionVelocity (minimum 1 tick --
  // nothing is ever instantaneous)
  return { matrix, vector, factors };  // ambient state lives at the root
}

function step(model, inputs, feedback) {
  // 1. determine firing: threshold crossing, or a stochastic roll for
  //    designated spontaneous (pacemaker) neurons
  // 2. deliver only what was scheduled to ARRIVE this tick (a ring buffer
  //    of delayed drive, not instantaneous same-tick delivery)
  // 3. leaky integrate-and-fire state update; homeostasis nudges threshold
  // 4. continuous STDP: decay every eligibility trace, then add
  //    potentiation/depression from this tick's spike-timing coincidences
  // 5. ambient neuromodulator decays, then this tick's `feedback` is added
  // 6. consolidation: weight change = eligibility * (hebbianRate +
  //    modulatoryRate * ambientModulator) -- see Section 4
  // 7. rare structural plasticity: sustained eligibility can grow a new
  //    connection; long quiescence can prune an existing one
  // 8. clamp external sensory inputs (the one forced exception, as always)
  return newModel;  // never mutates its input
}
```

---

## 3. Algorithms: Dynamics and Plasticity

**Leaky integrate-and-fire** replaces the prior architecture's ungrounded squaring relaxation: a neuron that doesn't fire decays a fraction of the way toward a resting potential every tick, rather than following an ad hoc nonlinearity with no biological analogue.

**Rate-coded readout, not raw membrane state.** Every prior XNN experiment read a neuron's continuous pre-spike value directly for decisions — privileged information no real downstream neuron or electrode has access to. OXNN decodes from an exponentially-weighted recent spike count instead, the simplest of three real candidate codes (rate, first-spike/latency, population synchrony); rate coding was chosen for tractability, not because it is confirmed to be how real tissue decodes, and remains the least-tested design choice in this paper.

**Continuous STDP** maintains a signed, per-connection eligibility trace: potentiation when the presynaptic neuron fired shortly before the postsynaptic one, depression in the reverse order, both weighted by an exponential kernel in the time gap and decaying continuously otherwise.

**The dual-pathway consolidation fix.** The first working version gated all weight change through a single term, `consolidationRate * ambientModulator * eligibility` — meaning that with no ambient feedback delivered, weight change was **exactly zero regardless of eligibility**, confirmed directly by feeding the network strongly correlated input with feedback held at zero throughout: eligibility built up genuinely (max 0.86) while not one connection's weight moved. This is a real problem for the stated goal: purely cortical tissue (DishBrain's actual preparation) has no intrinsic dopaminergic source at all — three-factor, reward-gated plasticity is a real and specific mechanism (well-established at corticostriatal synapses), but assuming it as freely available everywhere doesn't hold for a purely cortical system. The fix splits consolidation into two independently-tunable, additive rates:

```
delta = eligibility * (hebbianRate + modulatoryRate * ambientModulator)
```

`hebbianRate` is always-on, pattern-driven learning; `modulatoryRate` scales an additional reward-amplified boost on top. Verified directly on an isolated two-neuron connection: weight climbed steadily from 0.2 to saturation purely from repeated pre-then-post firing, **with feedback held at exactly zero for the entire run** (67 repetitions to saturate), while adding periodic reward pulses on top reduced that to 4 repetitions — confirming both pathways operate simultaneously and genuinely add together, rather than one silently overriding the other.

Fixing this also surfaced a second, related bug: structural pruning had been keyed on *whether a weight update fired*, which — under the old single-gated formula — meant every connection looked permanently "unconsolidated" whenever feedback was absent, regardless of its real activity. The fix decouples pruning eligibility from consolidation entirely, tracking genuine eligibility quiescence (`lastActive`) instead.

---

## 4. Calibrating the Substrate: Criticality

A specific, checkable signature of healthy cortical (and organoid) dynamics exists in the literature: spontaneous activity organizes into cascades — "neuronal avalanches" — whose size distribution follows a power law with exponent near **−1.5** (Beggs & Plenz, 2003, extensively replicated since). This gives OXNN's spontaneous dynamics a real target to calibrate against, rather than an arbitrary "looks plausible" parameterization.

The naive default (connection probability 0.5) produced deeply **supercritical** dynamics: avalanches spanning nearly the entire simulation, no power-law signature at all. A much sparser setting (0.15) was **subcritical**: avalanches died out too fast (exponent −2.4 to −3.0, steeper than the real target). Bracketing this transition empirically:

| Connection probability | Mean avalanche size | Fitted exponent |
|---|---|---|
| 0.50 | tens of thousands | ~0 (no power law; runaway) |
| 0.30 | 8.4 | −0.81 |
| **0.25** | **4.8** | **−1.10** |
| 0.22 | 1.7 | −2.83 |
| 0.15 | 1.6 | −2.9 to −3.0 |

0.25 is the closest approach found — a real, if imperfect, signature (R² ≈ 0.68–0.97 depending on the fit), now used as the module's grounded default. We note honestly that hitting −1.5 exactly likely needs either a finer search or a genuine self-organizing mechanism: real cortical tissue is believed to *tune itself* toward criticality via regulatory processes; OXNN's threshold-based homeostasis does not do this on its own, and every calibration reported here required deliberate external tuning, not emergent self-organization.

---

## 5. Toward DishBrain: An Embodied Hello-World

DishBrain's actual mechanism is more specific than "reward for winning": ball position was encoded by which of a small set of fixed electrodes fired; a separate region's activity, read out and mapped to paddle movement; and — the genuinely distinctive part — a **hit** delivered structured, predictable stimulation, while a **miss** delivered unpredictable, randomized stimulation. No reward scalar exists anywhere in the loop. The hypothesis (grounded in the free-energy principle) is that biological networks intrinsically minimize surprise, and will self-organize toward whichever action yields predictability, with no need to be told which outcome is "good."

Our minimal recreation: a sensory population receiving either a fixed, regular pulse or a randomly-timed one (matched for average intensity, differing only in regularity); two competing "motor" pools, whichever more active at the end of a free-running window is that trial's "choice"; a fixed, undisclosed rule (pool A wins → predictable stimulation next; pool B wins → unpredictable) the network must discover through correlation alone, with **ambient feedback held at exactly zero throughout** — testing the `hebbianRate` pathway in isolation, the actual mechanism under test, not a reward-based stand-in for it.

---

## 6. Results

### 6.1 A promising result that did not survive scrutiny

An initial 20-seed sweep showed 14/20 seeds shifting toward the predictable-associated pool over training (mean shift +0.078) — encouraging, but not statistically significant on its own (t = 1.50). Structural inspection of the specific seeds driving this, comparing one that collapsed to always-B against one showing the strongest positive shift, found **near-zero or literally zero sensory-to-pool connectivity in both** — the very pathway meant to drive the effect was barely present. Re-running with sensory-to-pool connectivity explicitly guaranteed (removing the confound) made the effect **vanish**: 9/20 positive (below chance), mean shift +0.059, t = 0.767. The original result was an artifact of sparse, asymmetric initial wiring, not learning.

### 6.2 A third independent replication of a recurring collapse pathology

Once the confound was removed, a different, more informative pattern emerged: hard, fast, winner-take-all lock-in to one pool, largely independent of the actual environmental contingency. This is the **third** time this exact signature has appeared in this research program — previously in board-game self-play and in a language-timing task, each using an entirely different learning mechanism. Its reappearance here, under pure STDP with no relation to either prior mechanism, is strong convergent evidence that it is a structural property of recurrent, self-reinforcing dynamics generally, not an artifact of any one training rule.

**Diagnosis.** Homeostasis is a slow negative-feedback correction; STDP-driven recurrent self-excitation is a fast positive-feedback loop. Speeding homeostasis up substantially reduced collapse (100% → 60% of seeds, across a systematic learning-rate sweep) but plateaued rather than reaching zero — confirming the mismatch is real and causal, while showing that speed alone is not the whole story.

**A systematic search across proposed fixes**, all tested against the same ten seeds:

| Fix | Collapse rate | Verdict |
|---|---|---|
| Fast homeostasis alone | 6/10 | Partial, plateaus |
| Lateral inhibition (−0.05 to −0.3, three strengths) | 9–10/10 | **Worse at every strength** |
| Periodic perturbation of the losing pool | 7/10 | No improvement |
| Per-connection weight cap (0.15) | 3/10 (best raw rate) | Collapse↓, but late-stage choice frequency flat at chance (0.474, not different from 0.5) — suppresses commitment, not learning |
| Synaptic scaling (aggregate weight budget, not per-connection) | 3–4/10, **consistent across pool sizes 3 and 20** | Genuinely scale-invariant fix; still no learning signal |
| Richer, sequential (non-synchronized) stimulus pattern | 3/10 at small scale; **8/10 at larger scale — worse** | No improvement; possibly harmful at scale |
| Fast activity-dependent damping | 4–7/10 | No improvement, sometimes worse |
| Dedicated inhibitory homeostatic plasticity (Vogels et al., 2011-style) | 4–5/10 | No improvement |
| Both combined | 7–8/10 | **Worse than either alone** |

Lateral inhibition's failure is itself a genuine, transferable finding: mutual inhibition between competing populations is the textbook mechanism for *creating* sharp, decisive winner-take-all dynamics in computational neuroscience, not suppressing them — the intuition that "real competitive circuits use inhibition, so inhibition should help" has the causal direction backwards for this specific failure mode.

### 6.3 Scale does not rescue the effect, and can actively break a working fix

Repeating the collapse test with pools 6.7× larger (3 → 20 neurons per pool), holding every other setting fixed, produced **no meaningful improvement** (10/10 → 9/10 collapsed) — directly ruling out "insufficient averaging in a small population" as the explanation. More strikingly, the one partial fix that worked at small scale (the per-connection weight cap) got **worse**, not better, at the larger size (3/10 → 6/10) — because a fixed per-connection ceiling does not bound the *aggregate* recurrent drive a neuron receives, which grows with the number of contributing connections. Synaptic scaling (capping the total incoming weight budget, not each connection) was built specifically to correct this, and did hold constant across both scales tested — a real methodological lesson: a fix validated at one network size cannot be assumed to transfer to another without checking whether it depends on aggregate, not per-unit, quantities.

### 6.4 A clean, final negative result

Across every configuration in Sections 6.2–6.3 — two pool sizes, two stimulus designs, and roughly eight distinct regulatory mechanisms and their combinations — **not one produced a statistically credible predictability-tracking learning signal.** The best late-stage win rate found (with the collapse-suppressing weight cap) was statistically indistinguishable from chance. Every apparent positive result traced back either to a connectivity confound or to which direction a pathological collapse happened to lock, never to gradual, genuine tracking of the environment's actual structure.

---

## 7. Discussion

**What this phase genuinely establishes.** A working, biologically-grounded local-plasticity engine that provably learns from input pattern alone, with zero reward signal — verified directly and cleanly, independent of everything that follows. A substrate calibrated against a real, checkable neuroscience target (avalanche criticality) rather than an invented one. A precisely diagnosed, third-replicated instability, now understood mechanistically (a timescale mismatch between fast Hebbian reinforcement and slow homeostatic correction) even though not yet solved. And a decisive ruling-out of at least two plausible-sounding fixes (lateral inhibition at any tested strength; naive per-connection weight capping at scale) — valuable specifically because it prevents wasted future effort chasing either.

**What remains open, honestly.** The actual target phenomenon — DishBrain's predictability-driven learning — has not been reproduced at any scale or configuration tested. We do not believe this is for lack of trying reasonable fixes; the negative results are consistent and mechanistically explicable, not scattered or ambiguous. Two live, separable hypotheses remain: that genuine predictive-coding computation (an explicit internal expectation, compared against outcome) is structurally necessary and STDP correlation alone cannot approximate it; or that real scale and continuous, uninterrupted operation — DishBrain's ~800,000 neurons operating with no discrete trial structure at all, against our tested maximum of 120 across explicit trial windows — matter in a way no amount of clever calibration at our current scale can substitute for.

**Why this is a genuinely optimistic outcome, not a discouraging one.** Every negative result in Section 6 narrowed the hypothesis space concretely rather than leaving it open-ended — we know inhibition is the wrong lever, we know naive capping doesn't generalize across scale, we know it isn't simply a small-N artifact. That is real, cumulative progress on a hard question, even though the headline result is negative. The substrate itself is validated and ready; what's left is two clearly-stated, independently pursuable next questions, not an open-ended search.

---

## 8. Limitations

- All experiments used at most 120 neurons, against DishBrain's ~800,000 — several orders of magnitude below the scale of the actual benchmark, and this gap has not itself been shown to be irrelevant.
- Criticality calibration is approximate (exponent ≈ −1.1 against a target of −1.5) and was achieved through manual parameter search, not a self-organizing mechanism, unlike (believed) real cortical tissue.
- The storage backend remains dense in memory (N² even though sparse in structural meaning), which is a real, near-term ceiling on how far scale can practically be pushed with this implementation.
- Collapse-pathology and DishBrain-replication results are drawn from 10–20 seeds per condition; while the consistency of null results across many structurally different conditions makes a hidden real effect unlikely, individual condition estimates carry real sampling uncertainty.
- No glial representation, no multi-neuromodulator diversity beyond a single ambient channel, no short-term synaptic plasticity, and no dendritic computation — all flagged as open gaps between OXNN and real tissue, none addressed in this phase.
- The free-energy-principle mechanism itself was approximated through raw STDP correlation rather than any explicit predictive or generative computation; this substitution is the paper's own leading hypothesis for the negative result, not a settled fact.

---

## 9. Conclusion and Future Work

OXNN is a real, working answer to "can an XNN be made to structurally and dynamically resemble organic neural tissue" — spatially grounded, Dale's-Law-respecting, spiking, criticality-calibrated, and capable of genuine unsupervised learning from pattern alone. It is not yet a working answer to "does this produce DishBrain's specific behavior," and this paper's contribution is converting that from an open-ended question into two concrete ones:

1. **Build genuine predictive coding** — an explicit internal expectation and prediction-error signal, not an implicit hope that spike-timing correlation approximates one — as its own dedicated mechanism, tested in isolation before being layered onto anything else.
2. **Invest in a genuinely sparse storage backend** before attempting a large scale-up, so that an order-of-magnitude (or more) increase in network size is a real, tractable experiment rather than an aspiration, and re-test the same hello-world at that scale once it is.

Either path is well-defined, buildable, and directly motivated by evidence gathered here rather than speculation. That is the right place for this phase to end: not at the result we hoped for, but at a substantially clearer map of exactly what stands between here and it.
