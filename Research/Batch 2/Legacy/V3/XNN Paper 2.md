# On OXNNs (Organic Cross Neural Networks) and On Lessons Learned as Applicable to Further Research

## Abstract

Cross Neural Networks (XNNs) are fully-recurrent, continuously-operating neural graphs trained without backpropagation, proposed as a substrate closer to organic cognition than conventional deep learning. This paper documents a phase of research that took that proposal literally, rebuilding the architecture around real neuroscience — **OXNN (Organic XNN)** — and, in doing so, learned something that reshaped the whole research program: task performance was the wrong success criterion for this branch entirely. An early attempt to replicate DishBrain (Kagan et al., 2022), the experiment in which living cortical neurons learned Pong from stimulation predictability rather than reward, produced a clean, decisive negative result after ruling out a real confound and a genuine, recurring instability — valuable knowledge, but knowledge that pointed at the wrong target. We pivoted: from "can this learn a task" to "does this reproduce known biological dynamics, built bottom-up from the single cell, at minimal computational abstraction." That pivot paid off immediately and then compounded. Working from a single, specific literature target (Softky & Koch's 1993 finding that plain integrate-and-fire neurons fire too regularly compared to real cortex), we found our own substrate had the *opposite* problem — pathologically bursty — diagnosed why, and closed most of the gap with refractory and adaptation dynamics. Adding a full Izhikevich (2003) neuron as an optional per-neuron mode reproduced five distinct, real cortical firing classes, including genuine bursting our simpler mechanism structurally could not reach. Scaling this to a full network surfaced a real unit-scale bug, then a genuine tension: the same homeostatic regulation that fixed network-wide silence also suppressed the biological irregularity we were chasing. Rather than leave that open, we closed it: re-verifying our earlier criticality calibration against the richer neuron models found it had quietly drifted, and re-calibrating incidentally improved the irregularity problem too; injecting genuine stochastic fluctuation into neural drive — the literature's own explanation for real cortical irregularity, never previously tested here — then closed most of the remaining gap directly, while fully preserving network-wide activity. Every step here, including the DishBrain result that started it, turned real uncertainty into a specific, well-posed, and increasingly answered question — which is the only standard this kind of research should be held to.

---

## 1. Introduction

An XNN is the simplest possible recurrent substrate: every neuron connects to every other neuron (including itself) through a signed weight, and the network runs continuously rather than being invoked function-call style. Two properties distinguish it from conventional neural networks: **ASNP** (asynchronous neural processing — no forward/backward pass distinction, queryable or interruptible at any instant) and **ATNP** (atypical neural parameters — signed weights and per-neuron bias, unlike the uniform units of standard architectures). Earlier work in this program established that local, non-backpropagation learning rules can train such networks at all. This phase asked whether the architecture itself could be made to resemble real neural tissue — **OXNN** — closely enough to test against it directly.

That test, described in full below, changed the question being asked. We began by trying to replicate a specific behavioral phenomenon (DishBrain's reward-free, predictability-driven learning) and got a clean, well-earned negative result. Rather than treat that as a dead end, we recognized it as evidence of a **prior**, more fundamental problem: nothing about our substrate had ever been checked against real single-neuron biology at all. Every mechanism had been layered on top of a fixed threshold-crossing neuron whose own dynamics were never validated in isolation. This paper is the account of that recognition, the pivot it produced, and the concrete, quantitative progress that followed from taking dynamical fidelity — not task performance — as the actual goal.

---

## 2. From XNN to OXNN: Grounding Topology in Biology

Every prior XNN was fully, densely connected with no notion of physical space. Real cortical tissue is not: connectivity is sparse and distance-dependent, neurons are exclusively excitatory or inhibitory (**Dale's Law**), and signal transmission takes real time.

| Property | OXNN default | Source |
|---|---|---|
| Excitatory/inhibitory ratio | 80% / 20% | Braitenberg & Schüz, 1998 |
| Inhibitory synapse strength | 2× excitatory (conservative) | Balanced-network literature cites up to ~8× |
| Connection probability | exponential decay with distance | standard cortical wiring assumption |
| Conduction velocity | ~1 m/s | unmyelinated axon range (~0.5–10 m/s); organoid tissue is unmyelinated |
| STDP time constant | τ ≈ 15ms, window ≈ 40ms | Bi & Poo, 1998; Sjöström & Gerstner, 2010 |

**Design philosophy: dense in the abstract, sparse in implementation.** The connection matrix stays a full N×N array — a nonexistent connection simply has weight 0 — but existence (`exists`) is tracked independently of weight, so a future genuinely sparse storage backend (necessary for ever approaching real biological scale) could replace how the matrix is stored without changing what it means to query it. This has not yet been built; current scale is capped at a few hundred neurons.

**Data structure and interface, unchanged since first introduced**: a model is `{ matrix, vector, factors }` — connections, neurons, and shared ambient state — manipulated through exactly two functions, `create` and `step`, with no separate training call. Real synaptic plasticity is not a mode a neuron enters; it is continuous and intrinsic, so it lives inside `step` itself, running every tick.

---

## 3. Algorithms: Dynamics and Plasticity

**Leaky integrate-and-fire**, with signal delivery on a real, distance-derived conduction delay (minimum one tick — nothing is ever instantaneous), replaced an earlier, biologically ungrounded relaxation rule.

**Continuous STDP** maintains a signed, per-connection eligibility trace: potentiation when the presynaptic neuron fired shortly before the postsynaptic one, depression in reverse order.

**The dual-pathway consolidation fix.** An early version gated all weight change through a single term multiplying eligibility by an ambient neuromodulator level — meaning weight change was *exactly* zero with no ambient feedback delivered, confirmed directly: eligibility built genuinely (0.86) while not one connection moved. This mattered specifically because purely cortical tissue has no intrinsic dopaminergic source at all. The fix splits consolidation into two additive rates:

```
delta = eligibility * (hebbianRate + modulatoryRate * ambientModulator)
```

Verified on an isolated connection: weight climbed from 0.2 to saturation from repeated correlated firing alone, feedback held at exactly zero (67 repetitions to saturate; periodic reward on top reduced this to 4) — both pathways genuinely add, neither silently overrides the other.

---

## 4. Calibrating the Substrate: Criticality

Spontaneous cortical activity organizes into cascades ("neuronal avalanches") whose size distribution follows a power law with exponent near **−1.5** (Beggs & Plenz, 2003). The naive default (connection probability 0.5) was deeply supercritical (runaway, no power law at all); a much sparser setting (0.15) was subcritical (exponent −2.4 to −3.0, too steep). Bracketing the transition empirically landed on **0.25** as the closest approach found (exponent ≈ −1.1) — real, if imperfect, and still requiring deliberate external tuning rather than the self-organization believed to produce this in real tissue. This calibration was measured on the neuron model as it existed at this point in the research; Section 9 revisits and corrects it once that model changed.

---

## 5. A Methodological Pivot: From Task Performance to Dynamical Fidelity

Section 6 documents our first real test of OXNN: an attempt to replicate DishBrain, evaluated by whether the network learned a *behavior*. That attempt produced a clean negative result — genuinely useful, but only after it was complete did the deeper problem become clear: **we had spent the entire prior phase building network-level mechanisms (topology, plasticity, criticality) on top of a single-neuron model that had never once been checked against real single-cell electrophysiology.** Success had been implicitly defined as "does the network do something interesting," when the more foundational, more honest question — for a substrate whose whole premise is biological resemblance — is "does it behave like real neural tissue at all, starting from the single cell."

We adopted that as the standard going forward: reproduce known biological dynamics as faithfully as possible, at minimal computational abstraction (dynamical-systems variables, not literal channel or molecular simulation), working bottom-up from one neuron before asking anything of a network. Task learning is not abandoned — it remains the eventual integration point once a dynamically faithful substrate exists — but it is no longer the metric this branch is judged against. Section 6 is reported in full because the negative result and the mechanism behind it were real and instructive; Sections 7–9 are the work this pivot actually produced.

---

## 6. An Early Test Under the Prior Framing: DishBrain

DishBrain (Kagan et al., 2022) taught living cortical neurons to play Pong using no reward signal — a hit delivered structured, predictable stimulation; a miss delivered unpredictable noise; the free-energy-principle claim (Friston et al.) being that biological networks intrinsically minimize surprise and self-organize toward whichever action produces it. Our minimal recreation used two competing "motor" pools whose relative activity chose a trial's outcome, with ambient feedback held at exactly zero throughout.

**The result was negative, and cleanly so.** A promising-looking initial sweep (14/20 seeds shifting toward the predictable pool) traced to a connectivity confound — the pathway meant to drive the effect was structurally near-absent in the very seeds showing it — and vanished once fixed (9/20, below chance). What replaced it was a **winner-take-all collapse pathology**, the third independent appearance of this exact failure mode in this research program, under a mechanism (pure STDP) unrelated to either prior instance — strong evidence it is a structural property of self-reinforcing recurrent dynamics generally. A systematic search across eight regulatory mechanisms found lateral inhibition reliably made it *worse* at every strength tested (mutual inhibition is the textbook mechanism for *creating* decisive winner-take-all dynamics, not suppressing them), while synaptic scaling was the one fix that generalized correctly across network sizes without collapsing differently at scale — genuinely useful, transferable findings, even though no configuration ever produced credible learning. This consistent, mechanistically-explicable pattern of negative results, more than any single number, is what motivated Section 5's pivot.

---

## 7. Single-Cell Validation: Finding the Wrong Kind of Irregularity

Real cortical neurons fire irregularly at high rates — interspike-interval coefficient of variation (CV) near 1, close to Poisson — and Softky & Koch's (1993) central, well-replicated finding is that **plain leaky-integrate-fire, driven by steady input, fails to reproduce this: it fires too regularly.**

Measured directly on OXNN's spontaneous dynamics: **CV = 2.03 overall.** The opposite failure. Splitting by neuron type isolated the cause precisely: designated pacemaker neurons (independent stochastic firing) measured 1.14, close to the real target; ordinary, network-driven neurons measured 2.11. The burstiness was intrinsic to the same cascading, avalanche-style dynamics calibrated for criticality in Section 4 — a real, informative tension between two genuine biological signatures (population-level cascades and individual-neuron regularity) that a network this simple may not satisfy simultaneously without more mechanism.

**Fix: refractory period and spike-frequency adaptation**, added as ordinary abstract state variables (not channel simulation) — a hard minimum interval between spikes, and a slow recovery variable that builds with firing and subtracts from drive. A real methodological trap was caught before trusting the first result: testing "sustained current injection" via a live, plastic synapse let ordinary consolidation quietly erode the test connection's weight mid-run — a real patch-clamp injects current directly, never through a synapse subject to learning. Once isolated correctly (plasticity disabled for single-cell tests only), the mechanism cleanly reproduced two real classes: **Fast-Spiking** (minimal adaptation, sustained near-regular firing) and **Regular-Spiking** (a decelerating transient settling into a stable rate) — but not genuine bursting, for a specific, mechanistic reason: a single linearly-decaying variable settles to a fixed point, not a limit cycle, regardless of parameters.

Re-measured at the network level, this closed most of the gap: **CV = 1.30 overall, 1.33 non-spontaneous** — a real, substantial improvement, not a perfect match.

---

## 8. A Richer Neuron: Izhikevich Dynamics

Genuine bursting needs nonlinear coupling between a fast and slow variable — exactly what Izhikevich's (2003) two-equation reduction of Hodgkin-Huxley biophysics provides, at close to integrate-and-fire computational cost. Added as an **optional per-neuron mode** (`dynamicsModel: 'izhikevich'`), with zero change to any existing LIF neuron, confirmed by full regression.

Tested in isolation (plasticity disabled, one driver neuron, published parameter sets), five distinct, real classes were reproduced cleanly:

| Class | Signature |
|---|---|
| Regular Spiking | Short transient, then stable ~33-34 tick period |
| Fast Spiking | Sustained, minimal adaptation |
| **Intrinsically Bursting** | **4, 5, 26, 8, 31, 9, 34... — genuine repeating burst-silence cycle** |
| **Chattering** | **3,3,3,4,4,5,7, 36, [exact repeat] — the same cluster shape recurring precisely** |
| Low-Threshold Spiking | Smooth, distinct ramp to a moderate steady rate |

The chattering result is close to a textbook confirmation: the *identical* seven-spike cluster shape recurring after each long gap, not merely "sometimes bursty." Two further, real phenomena emerged **without any special-casing**: Regular-Spiking developed a sharper initial burst at higher injected current (a documented current-dependent transition), and chattering became more precisely periodic at weak drive but collapsed to near-continuous firing at strong drive (also independently documented). STDP was verified to work correctly regardless of which dynamics model produced a spike — eligibility tracked genuine, sensible timing correlation from an Izhikevich source, and consolidation applied it with exactly the predicted magnitude.

---

## 9. From Single Cell to Network: A Real Tension, and Its Resolution

Mapping the network's excitatory/inhibitory populations to Izhikevich Regular-/Fast-Spiking (a real, established biological correspondence) first **failed completely**: zero spikes across 142 neurons, membrane potential frozen at rest. The cause was a genuine unit-scale mismatch — our network's weights, calibrated for LIF's [0,1]-bounded drive, were never remotely close to the current magnitude (~10-25) the isolated Izhikevich tests needed. Fixed with an explicit current-scaling factor, defaulting to 1 to exactly preserve every already-validated isolated result (confirmed byte-identical), with a larger, explicitly-passed scale for network contexts.

Once genuinely active, the network told a different, more interesting story than "problem solved":

| Configuration | Overall CV | Active neurons |
|---|---|---|
| Bare LIF | 2.03 | 97/150 |
| LIF + refractory + adaptation | 1.30 | 126/150 |
| Izhikevich RS/FS | 0.71 | 46/150 |
| Izhikevich RS/FS + homeostatic current | 0.47 | **150/150** |

Adding a homeostatic bias current (Izhikevich's analogue of LIF's threshold homeostasis — necessary since Izhikevich's firing threshold is a fixed biophysical constant, not something that should drift without blurring the class-defining parameters) **completely solved under-activity** (46→150) but **pushed CV further from the target, not closer.** This is a real, structural tension, not a tuning failure: homeostatic stabilization is, by definition, variance-reducing.

Closing it took two separate steps, not one. First, an overdue check: the criticality calibration (Section 4) predated refractory period, adaptation, and Izhikevich mode entirely, and re-measuring it against the current substrate found real drift — 0.25 now measured exponent **−0.47** (supercritical, not the original −1.1), with the critical transition compressed into a far narrower window than before. Recalibrating found **0.245 → −1.59**, tighter than the original approximation — and, incidentally, brought the LIF network's CV down to **1.06** with no other change at all, a useful reminder that these targets are not independent. Second, the actual hypothesis test: genuine stochastic fluctuation, added to injected drive as an *independent* term alongside (not replacing) the deterministic homeostatic bias — the literature's own explanation for real cortical irregularity, never previously tested here.

| Noise amplitude | CV | Active neurons |
|---|---|---|
| 0 | 0.53 | 150/150 |
| 8 | 0.70 | 150/150 |
| 16 | 0.90 | 150/150 |
| **18** | **0.93** | **150/150** |
| 24 | 0.73 | 150/150 |

Noise closed most of the remaining gap directly, **while fully preserving network-wide activity across the entire tested range** — confirming the literature's explanation rather than merely finding it plausible. The relationship is a real sweet spot, not a monotonic dial (CV declines again well past amplitude 18), and the right scale had to be found per context, the same way current-scaling did. Both fixes are now the module's defaults, verified against every prior single-cell result with zero regression.

---

## 10. Discussion

**The pivot was the right call, and the evidence for that is cumulative, not a single result.** DishBrain's negative outcome was real and well-earned, but the deeper finding was that it was the wrong question for this branch — nothing beneath it had been validated. Once redirected to single-cell fidelity, every subsequent step produced genuine, checkable progress: an unexpected failure direction (too bursty, not too regular), a real methodological trap caught before it could contaminate results, a clean mechanistic explanation for why linear adaptation cannot burst, five real classes reproduced by a richer model, a genuine tension between two desirable properties — and then, distinctively, an actual resolution of that tension rather than a plausible-sounding fix left untested.

**Where optimism is earned, and it compounds.** The substrate now has real, checkable answers at the level the whole enterprise depends on. Two of three major single-cell classes were reached with minimal added mechanism; all five tested, including genuine bursting, were reached with a still-abstract, still cheap richer model. The one clean open failure from the prior phase — CV suppressed by homeostasis — is now substantially closed, and closed by confirming the literature's actual mechanism directly rather than assuming it would work. The recalibration that helped get there also improved an unrelated result (the LIF network's CV) for free, a real sign the substrate's pieces are coherent with each other rather than independently patched.

**Where it isn't yet.** The Softky-Koch match is close, not exact, and the noise-CV relationship is empirically found, not derived from first principles. Firing-rate distribution shape and small-world topology remain entirely unmeasured; genuinely sparse storage does not exist yet; and DishBrain remains formally unanswered, deferred rather than resolved — now on a foundation this phase spent real effort validating rather than assuming.

---

## 11. Limitations

- All network-scale results use at most 150 neurons; single-cell classification claims are qualitative matches to published parameter sets, not verified against exact quantitative spike-train data from a specific paper.
- Izhikevich-mode neurons currently have no equivalent of LIF's stochastic pacemaker mechanism.
- The noise-CV relationship (Section 9) is characterized empirically across one sweep, not derived; the mechanism behind its non-monotonicity (why irregularity declines again at high amplitude) is not understood, and the right amplitude must be found per network context rather than computed.
- Firing-rate distribution shape and small-world connectivity structure — both real, checkable, already-identified targets — remain completely unmeasured.
- The storage backend remains dense in memory; genuine biological scale is not currently reachable regardless of any other fix.
- Recalibration (Section 9) was re-verified once, after adding refractory/adaptation/Izhikevich mode; it has not been re-checked again since noise injection was added on top, which itself changes spontaneous dynamics.

---

## 12. Conclusion and Future Work

This phase's real result is not a number — it is the discovery that the right question for OXNN was never "can it learn," but "does it behave like real neural tissue, from the single cell up," and that asking it produces answers that build on each other. Two items from this phase's own prior future-work list are now done: criticality was re-verified and corrected, and noise injection was built and shown to work. What follows from that:

1. **Characterize the noise-CV relationship properly** — find why it is non-monotonic, and whether a principled derivation of the right amplitude exists, rather than treating amplitude 18 as a found constant.
2. **Re-verify criticality once more**, now that noise injection itself changes spontaneous dynamics on top of the mechanisms it was last checked against.
3. **Measure firing-rate distribution and small-world topology** — cheap, already-identified, still entirely unmeasured targets.
4. **Build genuinely sparse storage**, the prerequisite for any real attempt at biological scale.
5. Only once the substrate is validated this thoroughly at the population level too, **return to task learning** — DishBrain included — as originally intended.

The negative result that began this phase and the positive results that followed it are the same story: asking a more honest question produced better answers. That is the right note for a computational biology program to end a chapter on.