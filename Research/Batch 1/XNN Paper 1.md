# On XNNs (Cross Neural Networks) with regards to the LFA XNN (Local Feedback Alignment Cross Neural Network) Variant and its Application to Babbler Language Models

## Abstract

Cross Neural Networks (XNNs) are a proposed bio-analogous architecture: fully recurrent, continuously-running (ASNP) integrate-and-fire networks with atypical neural parameters (ATNP) such as signed weights and per-neuron bias, trained without backpropagation. This paper documents the empirical path from that proposal to a specific, working variant we call **LFA (Local Feedback Alignment)**: column-specific local credit assignment, a fixed random feedback projection routing error into hidden layers with no weight transport, an undecayed eligibility trace for temporal credit, and continuous homeostatic regulation. Every component of LFA was earned by diagnosing a real failure — a literal reading of the original training convention collapses outright; positive-only weights are provably unable to represent inhibition; diffuse credit assignment causes emergent saturation; and, most importantly, a credit rule fully validated on one task (tic-tac-toe) was found to leave recurrent weights completely untrained (measured directly: zero change) on a task requiring genuine memory, invisible until the right task exposed it. Once assembled, LFA closes nearly the full gap to a gradient-checked backpropagation baseline on a temporal-memory task, drives strong self-play performance with zero labelled data, and produces a positive, if fragile, transfer effect across structurally unrelated tasks. Applied to **babblers** — XNNs that receive language one token at a time and must decide, unprompted, when to speak — LFA produces a network whose self-initiated timing and word choice both correlate with a scripted grammar well above chance, using a compressed but genuine subset of English. Scaling this system naively (bigger network, richer grammar, richer input, more training, all at once) underperformed; isolating the four factors individually recovered a clear win — a small addition of irrelevant background noise to the input stream, alone, nearly doubled performance and gave the best result of any configuration tested, better than the full combined scale-up. This resolves the ambiguity directly: the barrier to a competent babbler is not LFA's learning capacity but a specific, now-localized instability in scaling hidden-layer size, one clean problem rather than an unbounded one.

---

## 1. Introduction

The original XNN proposal argues that recurrent, deliberative processing — not scale alone — is the missing ingredient for genuinely intelligent behavior, and that this can be built from biologically-motivated primitives: full recurrence, continuous asynchronous operation (ASNP), and local, non-backpropagation learning. This paper treats that proposal as a real engineering target and works toward the most demanding test of it we could construct: a **babbler**, an XNN that receives language continuously, decides for itself when to speak, and responds intelligently — the long-term aim of this research program, and, we think, of any serious attempt to realize the original vision.

Getting there required first discovering what actually works. Every mechanism in this paper's final recipe — which we name **LFA (Local Feedback Alignment)** — was the product of a failure diagnosed precisely enough to fix. That process, run across gate logic, board games, memory tasks, and an embodied survival environment, is what this paper documents. The babbler is where all of it converges.

---

## 2. The XNN Substrate

An XNN is a fully-connected graph of *N* neurons: every neuron connects to every other neuron, including itself, via a signed weight matrix. Each neuron holds a continuous state in [0, 1] and a bias (an atypical neural parameter, ATNP). Processing is a sequence of discrete steps applied to a network that never resets except at genuine episode boundaries — the literal implementation of ASNP (asynchronous neural processing):

1. **Fire and reset**: a neuron with state ≥ 1 pushes its state along every outgoing edge, then resets to 0.
2. **Integrate**: each neuron's incoming drive is the sum of `weight × 1` over every neuron that just fired.
3. **Relax**: `new_state = clip(baseline + incoming + bias, 0, 1)`, then squared if positive — a decay that vanishes near 0 and 1 and is steepest near 0.5.
4. **Clamp inputs**: designated input neurons are force-set to external values, overriding step 3. This is the *only* forced exception to ordinary dynamics.

This substrate is architecturally neutral — it says nothing about how training should work. The rest of this paper is the search for a training rule that does.

---

## 3. Arriving at LFA: A History of Diagnosed Failure

### 3.1 The literal rule collapses

The original specification's proposed training rule — Hebbian-style associative scores, credited indiscriminately and updated in coarse feedback batches — was implemented exactly as written on the smallest possible test: a 5-neuron NAND gate. It does not merely fail to converge. It actively collapses to 0% accuracy within the first few feedback cycles.

### 3.2 Two structural walls, found before blaming training

A corrected version (crediting only edges whose source actually fired, signed per-trial feedback) found real signal but plateaued at exactly 75% accuracy — regardless of training method. Two structural causes, not training bugs, were confirmed directly rather than assumed:

- **Positive-only weights cannot represent inhibition.** A network with only non-negative weights is provably monotonic in its inputs; NAND is not. Twenty thousand random matrices and eight independent genetic-algorithm searches (150 individuals × 800 generations) all capped at exactly 75%.
- **The all-silent input is an inescapable fixed point.** With zero input and zero initial state, every neuron stays at exactly zero forever, for *any* weight matrix — confirmed across 50 random matrices with zero exceptions.

Adding signed weights and **bias** (escaping the fixed point) plus checkpointing (the threshold-trigger dynamics oscillate rather than converge monotonically) solved NAND completely: **15 of 15 random seeds, 100% accuracy.**

### 3.3 Scaling up exposes diffuse credit assignment

Applied to tic-tac-toe via self-play (TD(0) value learning, zero labelled data), outcome-based whole-game credit plateaued at 77% win / 20% loss against a random opponent and stayed frozen for thousands of subsequent games. Persistent state made no difference — a clean negative result, since tic-tac-toe is fully Markovian and never needed memory in the first place. Direct inspection of lost games showed a specific, fixable hole: the network had a fixed opening sequence and never once blocked an opponent's developing threat, in any orientation, even after symmetry augmentation confirmed the network *had* generalized positionally.

Direct measurement found the real cause: **86% of weights were saturated at the clip boundary**, with every neuron in a given role sharing an identical bias — a lockstep collapse from diffuse credit assignment (updating a fired neuron's *entire* outgoing row, and every neuron's bias uniformly, rather than only the specific edges relevant to the decision actually made). A weight-decay control disproved the naive "just saturation" story — decay caused saturation to swing wildly while the policy stayed completely frozen, revealing that an early-locked sign pattern, not raw magnitude, was the real culprit.

**The fix: surgical, column-specific credit** — only the incoming edges of the one neuron actually responsible for a decision are updated, and only that neuron's own bias moves. This broke the freeze. Capacity scaling (24 → 600 hidden neurons) then produced a clean, monotonic improvement (loss vs. random: 13.2% → 8.4%), and self-play alone (no oracle, no hand-designed opponent) reached **77.4% win / 13.2% loss vs. random, 64.0% win / 30.4% loss vs. a one-ply-lookahead opponent.**

### 3.4 The hole that only a harder task could reveal

A task built specifically to require genuine cross-time memory — a delayed-recall task, cue then distractor-filled delay then probe — exposed something the surgical fix could never have shown on tic-tac-toe: **measured directly, hidden-to-hidden and input-to-hidden weights changed by exactly 0.00000**, regardless of training budget (tested to 100,000 trials, 20× baseline) or capacity (tested to 300 hidden neurons, 12× baseline). The column-specific rule that fixed tic-tac-toe only ever touches columns belonging to *output* neurons — it structurally cannot train recurrent memory, and nothing about tic-tac-toe's structure could have exposed that, since tic-tac-toe never needed recurrent memory to begin with.

A hand-derived, gradient-checked backprop RNN baseline (verified to 7.7×10⁻¹⁰ relative error) confirmed the task itself was never the obstacle: **100% accuracy out to a 10-step delay, 96.7% at 20 steps**, using fewer parameters and less training than the XNN needed to achieve nothing.

### 3.5 Direct Feedback Alignment closes the gap

Three theoretically-motivated local mechanisms were tested for training hidden weights:

| Mechanism | Result |
|---|---|
| Node perturbation (REINFORCE-style) | Chance, even at 60,000 trials — variance from simultaneous multi-unit perturbation swamps the signal at this scale |
| Slowness / Hebbian rule (reward-free) | First version reproduced the exact saturation collapse from §3.3 (63.4% of weights pinned); a signed, corrected version fixed the saturation but stayed at chance |
| **Direct Feedback Alignment (DFA)** | **95.6–100% accuracy at delays of 3–10 steps, 87.5% at 20 steps** |

DFA — a *fixed*, never-updated random matrix projecting real output error into a per-hidden-neuron pseudo-error, requiring no weight transport and no backward pass — closed nearly the entire gap to the supervised RNN baseline. This is the missing piece. Combined with the column-specific credit rule (§3.3), the eligibility trace needed to span whole trials rather than single ticks (undecayed trace outperformed every decayed variant tested), and continuous homeostatic bias regulation (the standing defense against §3.3's saturation collapse), this is **LFA: Local Feedback Alignment.**

---

## 4. LFA: Definition and Interface

LFA has four components, each independently motivated by a diagnosed failure above:

1. **Signed weights + bias (ATNP)** — required representational capacity (§3.2).
2. **Column-specific local credit assignment** — only a specific neuron's incoming edges and own bias are ever updated in a single training event (§3.3).
3. **A fixed random feedback projection (DFA)** — routes real error into hidden neurons as a pseudo-error, with no weight transport (§3.5).
4. **Continuous homeostatic regulation** — every neuron's bias is nudged toward a target firing rate every tick, independent of any training signal, running whether or not training is occurring.

A model is a plain, JSON-serializable object with two fields — a connection matrix and a neuron vector — implemented as three pure functions with no classes:

```javascript
// matrix[i][j] = signed weight of the connection FROM neuron i TO neuron j
// vector[i]    = { state: [0,1], factors: { bias, trace, ...anything else } }

function create(n, options) {
  const matrix = /* n x n signed random weights, [-1, 1] */;
  const vector = /* n neurons: random bias [0,1], trace = 0 */;
  return { matrix, vector };
}

function step(model, inputs) {
  const fired = model.vector.map(neuron => neuron.state >= 1);
  // homeostasis: nudge every neuron's bias toward its target firing rate
  // incoming[j] = sum of weight[i][j] over every fired source i
  // new state = clip(baseline + incoming + bias, 0, 1), squared if unfired
  // force-clamp any externally supplied `inputs` (the only exception to the above)
  // mark trace = 1 for any neuron that fired this tick
  return newModel; // never mutates its input
}

function train(model, options) {
  // options.mode: 'supervised' (a real target vector) or 'td' (a scalar reward-prediction error)
  // output neurons: update THEIR OWN incoming edges directly from real error,
  //   weighted by each source neuron's trace
  // hidden neurons: project that SAME error through the fixed random matrix
  //   from component 3 -> a pseudo-error -> the identical local update rule
  // only the specific neuron's own column is ever touched -- never a fired
  //   neuron's whole outgoing row, never every neuron's bias at once
  return newModel; // never mutates its input
}
```

`step` and `train` are pure: they return new model objects and never mutate their input, so a caller can freely branch, checkpoint, or discard a trajectory. Everything task-specific — vocabulary, sensors, action spaces — lives entirely in `factors`, so a trained model is fully self-describing: reloading a saved file and deriving its architecture requires nothing beyond reading `factors.role` and whatever task-specific tags were attached to it, no external metadata file needed.

---

## 5. Characterizing LFA

### 5.1 Capacity and depth are real levers, but not free ones

Scaling tic-tac-toe's hidden layer (24 → 600 neurons, fixed deliberation depth) improved performance monotonically and cheaply. Scaling *deliberation depth* (recurrent steps per decision, fixed capacity) roughly halved loss for a small network (30.4% → 19.8% vs. the harder opponent) — but under a properly matched training budget, the *same* depth increase applied to the large network made performance measurably worse (loss vs. random: 8.4% → 18.8%). Capacity and depth interact; neither is safe to scale in isolation without matching the training budget to the larger effective space being explored.

### 5.2 A fixed feedback pathway beats an adaptive one

Since DFA's projection is fixed by design, a natural question is whether making it adaptive — via a biologically-plausible, weight-transport-free Hebbian rule (Kolen-Pollack style: `B += lr·outer(hidden_activity, error) − decay·B`) — would help it track a changing task. Two implementation bugs (decay silently collapsing *B* to near-zero; adapting from the very first trial destabilizing the alignment process) were found and fixed, restoring full performance on a stable task. But on a genuine context change (an input-channel relabeling, validated as a fair, equally-learnable alternative task), **adaptive B underperformed both a frozen pathway and a network trained from nothing**, given an identical retraining budget:

| Condition | Accuracy after matched retraining |
|---|---|
| Frozen B | 79.5% |
| Adaptive B | 52.7% (barely above chance) |
| From-scratch network | 75.1% |

DFA's power comes specifically from *W* having a stable target to rotate toward. A context change is exactly when the signals that would drive *B*'s adaptation are noisiest — the fixed randomness that looks like a limitation turns out to be what makes DFA robust to change, not an implementation detail worth "fixing."

### 5.3 A trained hidden core transfers, for reasons only partially understood

A tic-tac-toe-trained hidden core, transplanted into a fresh delayed-recall network (matched hidden-layer size), converged reliably to 100% across every seed tested, while random initialization left one of three seeds permanently stuck. Two explanations were directly ruled out: a homeostasis-only control (bias regulated, weights never trained) failed to replicate the benefit, and a content-unrelated but single-tick control task was messier and even regressed in one seed. A third hypothesis — that practice sustaining multi-tick dynamics specifically is what transfers — remains untested, because its own control task (a running-parity rule) was too hard to learn at this scale to serve as a valid test. What survives is narrower but real: **something about a genuinely trained LFA core reliably helps a different task**, beyond mere weight initialization statistics.

---

## 6. Toward Real-Time, Embodied Operation

Every result above used clean, well-defined tasks. The babbler's actual target — organic, continuously-operating cognition — needed two further tests before language was in scope.

**Embodied foraging.** An LFA agent (TD(0) actor-critic + DFA hidden training + homeostasis, synthesized for the first time into one architecture) learned a partially-observable survival task — directional sensing, an energy economy, pursuing threats — from zero labelled data, roughly doubling survival time over a random baseline. A scripted DFA-ablation control replicated §3.5's finding on an entirely new task (removing DFA reliably hurt: 45.1 → 36.8 mean survival ticks), a genuine positive replication. A memory-ablation control gave a **contradictory result across seeds** and is reported as genuinely unresolved rather than forced to a conclusion — the honest, if less satisfying, outcome.

**Genuine asynchronous execution.** The network was run on a real, independent OS thread against the environment, with no artificial synchronization barrier — the literal test of ASNP. Average performance matched lockstep execution closely (43.9 vs. 43.3 survival ticks), but individual seeds showed **real, non-deterministic behavioral variance** driven purely by OS scheduling — the same seed and weights producing different outcomes across repeated runs, confirmed not to be data corruption. Giving the network more relative deliberation time per decision (via environment pacing) sharply stabilized this variance, echoing the deliberation-depth finding from §5.1 through a completely different, genuinely concurrent mechanism.

Both results say the same thing: LFA's mechanisms generalize past clean, turn-based tasks to embodied, continuously-running operation. The babbler was the natural next step.

---

## 7. The Babbler

### 7.1 Design

A babbler receives one token at a time and must decide, unprompted, when to speak — separately from *what* to say — so timing and word choice can be measured independently. A dedicated **gate neuron** governs timing; word-output neurons (one per vocabulary word) govern content. The grammar is a genuine, if heavily compressed, subset of English, chosen so exchanges read as real (if terse) conversation rather than an arbitrary symbol task.

Each exchange: trigger words (one tick each, with gaps), a fixed silent pause, then a response window. Nothing marks the response window as special in the input stream — it is only distinguishable by accumulated silence, which is exactly the "timing inferred from internal dynamics, not an external cue" property being tested. Training is supervised DFA (the scripted grammar gives an oracle-known correct target at every tick — no external corpus, no human labels): a proactive correction whenever the gate fires prematurely, and a full gate-plus-word update at the response tick, using the trace accumulated across the whole exchange.

### 7.2 A real, if narrow, first result

A 65-neuron babbler (16 words, 6 rules, single-word responses) showed genuine, above-chance learning on both success criteria — timing correlating with grammar, word choice correlating with context — but far from mastery:

| | Untrained | Trained |
|---|---|---|
| Responds when a response is expected | ~0% | 20% |
| Word choice correct, given it responded | chance (~6%) | 20% |
| Correctly silent when silence is expected | — | 62.5% |

Actual exchanges showed the network had learned real word associations ("hi"→"hi", "help please"→"yes") before it had learned to wait for the pause — firing the instant a relevant word arrived rather than after the silence that should trigger a response. A specific, diagnosable gap, not incoherence.

### 7.3 A saturation problem, and two fixes that taught more by failing

Diagnostic work traced the premature-firing tendency to a **self-reinforcing recurrent clique**: a subset of hidden neurons with strong mutual excitatory weights (mean 0.291, against −0.036 for the rest of the network) that fire almost regardless of input content. This is invisible to homeostasis by construction — the network's `incoming` drive is clipped to bounds *before* bias is even applied, so sufficiently strong recurrent excitation saturates a neuron no matter how far homeostasis pushes its bias down.

Two fixes were tried and both are worth reporting for what they revealed, not just whether they worked:

- **Uniform hidden-to-hidden weight decay** made the diagnosed problem *worse* (always-on neurons: 9/32 → 19/32) — a small uniform decay is negligible against actively-reinforced clique weights but erodes the weak, genuinely useful discriminative connections faster.
- **Magnitude-proportional (cubic) decay**, targeting strong weights specifically, *looked* like a clear win on aggregate metrics (tick accuracy 0.71 → 0.88) — until a decoupled check revealed the network had collapsed to **never speaking at all**, trivially scoring well because most ticks in any exchange are supposed to be silent anyway. This directly motivated a properly principled evaluation metric (precision/recall/F1 on the gate decision, mathematically dragged toward zero by *either* a never-fires collapse or an always-fires collapse) — a real methodological contribution in its own right, since the architecture retained here (§7.4) reuses it throughout.

Both fixes were reverted. The saturation mechanism is confirmed; a working fix is not yet found.

### 7.4 Scaling everything at once underperforms; isolating the factors finds a genuine win

A scaled-up babbler — 181 neurons (4× hidden layer), 24-word vocabulary with genuine multi-word responses ("bye"→"see you later"), distractor input noise, more training — was built and trained to completion (15,000 exchanges). It performed **worse** than the original by most measures, drifting toward the opposite collapse mode (responding 94.8% of the time, but correctly silent only 3.3% of the time).

Isolating each factor against the original baseline, changing exactly one variable at a time, resolved why:

| Factor changed (alone) | F1 (gate precision/recall) | Exact response accuracy |
|---|---|---|
| Baseline (no change) | 0.120 | 11.1% |
| Bigger hidden layer only | 0.213* | 19.3% |
| Richer vocabulary/grammar only | 0.168 | 11.2% |
| **Distractor input noise only** | **0.229** | **18.8%** |
| More training only | 0.165 | 13.5% |
| *All four combined* | *0.186* | *5.5%* |

*The bigger-hidden-layer result is a confirmed always-fire collapse (response rate 1.0, correct silence 0.0) — a genuinely unstable result, not real competence, flagged directly rather than left to the number alone.

**Distractor noise alone, the cheapest possible change, produced the best result of any configuration tested — better than the full combined scale-up.** Background noise appears to push the network toward more robust, more discriminative representations, the same way regularization helps conventional learning generalize. Scaling the hidden layer, in isolation, reproduced the same instability found in §7.3 and nowhere else — **the barrier to a better babbler is now a specific, localized problem (hidden-layer scaling), not a diffuse one.**

---

## 8. Discussion

**LFA works, and the case for it keeps getting more specific rather than more hand-wavy.** Every positive result in this paper — NAND, self-play tic-tac-toe, delayed recall out to 20 steps, embodied foraging, genuine concurrent execution, a babbler with real if weak language competence — used the same four-component recipe, and every one of its components exists because a specific failure demanded it. That is a meaningfully stronger claim than "a plausible bio-inspired architecture that sometimes works."

**The recurring instability has a name and a location now, not just a symptom.** Saturation/lockstep collapse first appeared in tic-tac-toe, was fixed there, reappeared in the babbler in a different form, and factor isolation (§7.4) has now shown directly that it concentrates specifically in hidden-layer scaling — the one axis a larger, more capable babbler would need most. That is genuinely useful to know: it converts an open-ended "something about training is unstable" into a concrete, bounded engineering problem.

**Where optimism is earned and where it isn't.** It is earned in the sense that nothing in this paper suggests LFA is fundamentally incapable of what a babbler needs — every existence proof (real credit assignment across time, real self-play competence, real if fragile cross-task transfer, real above-chance conversational timing) argues the opposite. It is not yet earned in the sense of a validated scaling law: capacity has repeatedly shown it can hurt as easily as help without matching training and without first fixing the specific instability now localized above. "LFA can eventually produce a competent babbler with enough refinement and scale" is a defensible hypothesis this paper's evidence is consistent with — not yet a demonstrated trend.

---

## 9. Limitations

- Every quantitative result here is small-scale (tens to a few hundred neurons) and, outside the tic-tac-toe capacity sweeps, mostly 1–3 seeds; effect sizes large enough to be confident in direction (DFA's jump from chance to >95%; distractor noise nearly doubling F1) should still not be read as precise percentages.
- No conventional baseline exists for the babbler task the way the gradient-checked RNN existed for delayed recall — meaning current babbler performance cannot yet be split into "genuinely hard task" versus "still-fixable LFA limitation."
- Every experiment in this paper, including the ones described as "persistent" or "continuous," resets to zero at an episode boundary. Nothing has tested indefinite, boundary-less operation — the literal, strongest reading of ASNP, and arguably the most important untested claim in the whole research program.
- The recurrent-clique saturation problem is diagnosed, not solved; both attempted fixes are documented failures, one of which was initially mistaken for a success until a more careful, decoupled check caught it.
- LFA's fixed random feedback projection is a real departure from "everything is one trainable recurrent substrate" — external, untrained scaffolding, not a neuron or a trainable edge.

---

## 10. Conclusion and Future Work

LFA — signed weights and bias, column-specific local credit assignment, a fixed random feedback projection, and continuous homeostatic regulation — is not a hypothesis anymore. It is a working, repeatedly-validated recipe, arrived at by diagnosing real failures rather than guessing, and it is the first architecture in this research program to produce a language-using system, however small, entirely from local rules and zero labelled data. The babbler answers understand-produce-respond questions weakly today; it answers them, which nothing before it in this program could.

The path forward is specific, not open-ended:

1. **Solve the recurrent-clique saturation problem as its own dedicated investigation**, verified against the actual diagnostic (weight statistics, not aggregate score, which has now misled evaluation three separate times in this program) before any further scaling is attempted.
2. **Build a matched conventional baseline for the babbler task**, the same role the gradient-checked RNN played for delayed recall, to separate genuine task difficulty from remaining LFA limitations.
3. **Test genuinely boundary-less, continual operation** — no episode resets, ever — the one claim from the original ASNP premise this entire program has assumed rather than demonstrated.
4. **Scale deliberately from the validated win**: a babbler built on the original architecture plus distractor input noise, with hidden-layer scaling deferred until step 1 is resolved, rather than scaled everywhere at once.

Babblers remain the right long-term target: a real-time, self-initiating, locally-trained language user is the most demanding and most direct test of what the original XNN proposal actually claimed. This paper does not close that distance. It replaces a vague direction with a short, concrete list of what stands between here and there.
