# Local Learning in Cross Neural Networks: From Logic Gates to Self-Play, and the Failure Modes in Between

**An empirical investigation of a bio-inspired recurrent architecture, its training conventions, and what it takes to make local learning rules actually work**

---

## Abstract

Cross Neural Networks (XNNs) are a proposed bio-analogous architecture built around fully recurrent, continuously-processing (asynchronous, ASNP) integrate-and-fire neurons, extended with atypical neural parameters (ATNP) such as signed weights and per-neuron bias, and trained via local, biologically-plausible learning rules rather than backpropagation. This paper documents an empirical investigation of XNNs across three tasks: a 5-neuron NAND gate, a tic-tac-toe agent scaled from 51 up to 627 neurons, and a delayed-recall memory task specifically designed to require genuine cross-time information maintenance. We report every major failure mode encountered, since they turned out to be more informative than the successes: a literal implementation of the originally proposed associative-score training rule actively collapses; positive-only weights are structurally incapable of representing inhibitory relationships; an all-silent input is an inescapable fixed point of the dynamics; diffuse credit assignment causes an emergent lockstep saturation collapse; outcome-only reinforcement learning cannot acquire conditional, state-specific behavior without something resembling a value function; and — the paper's largest finding — the neuron-specific credit assignment rule that fixes the saturation collapse turns out to leave recurrent and encoding weights completely untrained, measured directly as exactly zero weight change regardless of training budget (up to 20×) or capacity (up to 12×). A hand-derived, gradient-checked backprop RNN baseline confirms the memory task itself was never the obstacle. Of three theoretically-motivated local fixes tested — Direct Feedback Alignment (DFA), node perturbation, and a slowness/Hebbian rule — DFA, a fixed random feedback pathway requiring no weight transport, closes nearly the entire performance gap to the fully-supervised baseline, while the other two fail for distinct, explicable reasons rather than mysteriously. Attempting to make DFA's feedback pathway itself adaptive, so it could track a changing task context, produces a counterintuitive result: adaptivity makes recovery from a genuine context change *worse* than either a frozen pathway or training from nothing, because a fixed random projection was never specialized to the old context and therefore has nothing to lose when that context changes. Separately, the tic-tac-toe architecture — combining signed weights, bias as ATNP, persistent cross-episode state, perspective-canonicalized self-play, TD(0) value learning, and continuous homeostatic regulation — reaches 77.4% win / 13.2% loss against a random opponent and 64.0% win / 30.4% loss against a one-ply-lookahead opponent with zero labelled data, and scaling its capacity and deliberation depth (independently and jointly) shows the two are not independent, additive levers but interact in ways that require matching training budget to the effective size of the resulting dynamical space. Finally, transplanting a trained tic-tac-toe hidden core into a fresh delayed-recall network reliably improves its learning (every transplant seed converges cleanly, versus at least one stuck seed in every control tried); two specific explanations for this — bias conditioning alone, and "any trained matrix helps regardless of content" — are directly ruled out, while a third, sharper hypothesis (that practice sustaining multi-tick dynamics specifically is what transfers) remains untested because its own control task proved too hard to learn at this scale, leaving the precise mechanism of this cross-task transfer effect an open question. Two further experiments deliberately move away from clean, well-defined tasks toward the architecture's stated motivation: an embodied, partially-observable foraging environment (the same TD/DFA/homeostasis synthesis learns a substantially richer task, and the DFA-ablation finding replicates independently, but a memory-ablation comparison — the one most directly testing this section's premise — returns genuinely contradictory results across seeds and is reported as unresolved rather than forced to a conclusion), and genuine OS-thread concurrent execution of the network against the environment (which produces the first result in this project where wall-clock timing itself, not the architecture or training rule, measurably and reproducibly changes behavior, verified not to be data corruption, and which is stabilized by giving the network more relative deliberation time per decision).

---

## 1. Introduction

### 1.1 Motivation

The XNN proposal starts from a specific critique of mainstream deep learning architectures: that attention alone, without genuine deliberation — initiation, hesitation, meditation, reflection over time — is insufficient for general intelligence, and that this gap stems from the non-recurrent, largely reactive nature of transformer-style models. The proposed fix is architectural: a fully-connected, signed-weight, continuously-running recurrent network (an XNN) in which every neuron connects to every other neuron (including itself), processing happens in discrete steps but without a fixed input/output boundary, and both input and output occur via direct manipulation of neuron states while the network keeps running — a mode termed **asynchronous neural processing (ASNP)**. Node and edge properties beyond simple scalar activations and weights are termed **atypical neural parameters (ATNP)** — for example, a per-neuron bias, or a per-edge "heat score" tracking recent contribution to network output.

The original specification (Training Convention 1 and its sub-conventions) proposes a specific local, associative learning rule loosely modeled on Hebbian plasticity: neurons integrate input and fire when their state reaches a threshold, firing increments an associative score on the edges that fed the firing neuron, and periodic feedback (a scalar in [-1, 1]) adjusts edge weights in proportion to their share of that associative credit.

This paper is an empirical stress test of that proposal. We do not assume the original training rule works; we implement it exactly as specified, observe what happens, and iterate.

### 1.2 Scope and method

We proceed through three tasks, the second and third each explored across several conditions:

1. **A 5-neuron NAND gate**, chosen because it is the smallest problem for which any local learning rule can be meaningfully tested to convergence, and because its solution requires genuine inhibition (NAND is not a monotonic function of its inputs), making it a useful stress test for representational capacity as well as training dynamics.
2. **Tic-tac-toe**, played first by a 51-neuron network (chosen to scale up to sequential decision-making, credit assignment across multiple moves, and a much larger effective input space, while remaining small enough to solve exactly via minimax for diagnostic purposes), then scaled up to 627 neurons and to substantially deeper per-decision deliberation once the baseline architecture and training rule were validated.
3. **A delayed-recall memory task**, built specifically to require genuine cross-time information maintenance in a way neither prior task could test — tic-tac-toe is fully Markovian and NAND has no temporal structure at all, so neither could have revealed whether persistent state was actually doing anything. This task exposed the largest finding in this paper: a training rule validated extensively on tic-tac-toe turns out to never train recurrent weights at all, and fixing that required comparing several candidate local learning mechanisms against a conventional backprop baseline.
4. **A cross-task transfer experiment**, testing whether a trained hidden core carries anything useful across genuinely different tasks, using controls built to rule out specific alternative explanations one at a time.
5. **An embodied foraging environment and a genuine-concurrency experiment**, stepping back from clean, well-defined tasks toward the architecture's stated motivation — organic, real-time cognition — and testing, for the first time in this paper, whether literal asynchronous execution (not just persistent state under otherwise-lockstep control) changes anything.

Throughout, we hold ourselves to a simple standard: when a training method produces a plateau — or, as it turns out, when a scaling experiment, a hyperparameter sweep, an adaptivity mechanism, or a memory ablation produces a *particular result at all* — we do not report it as a property of the *architecture* without first checking whether it is actually a property of the *training rule*, a *structural* impossibility, an outright *bug*, or an uncontrolled comparison; and when a result remains genuinely ambiguous after that checking, we report it as ambiguous rather than force a conclusion. This distinction turned out to matter enormously, and most of the interesting content of this paper is in disentangling these possibilities from each other.

---

## 2. Related concepts from the original XNN specification

For reference, the relevant conventions from the source specification that this paper implements, tests, or explicitly deviates from:

| Convention | Content | Status in this paper |
|---|---|---|
| 1.1 Positive Bounds | Neuron states and edge weights in [0, 1] | Neuron states kept in [0,1]; edge weights **not** — see Section 4.3 |
| Sub-convention 1-1.1 | Signed edges, [-1, 1] | Adopted; required for representational capacity |
| Sub-convention 1-1.2 | Bias as ATNP, [0, 1] | Adopted; required to escape a structural fixed point (Section 4.4) |
| 1.3 Multi-Step Input | Input persists across steps | Adopted throughout |
| 1.2.2 / 1.2.2.1 | Binary output reading + Clearing | Adopted; output neurons are the only neurons ever force-cleared, and only at read time |
| 1.4 Uniform Threshold Trigger | Fire at state == 1, push, reset | Adopted throughout |
| 1.5 / 1.5.1 State Relaxation | Squaring decay for unfired positive states | Adopted throughout |
| 1.6 / 1.7 Associative Score / Associative Training | Local Hebbian-style credit + feedback-proportional weight update | Implemented literally in Section 4.1; found to actively collapse; replaced |
| 1.7.1 Global Training | Feedback broadcast to all neurons simultaneously | Superseded by neuron-specific credit assignment (Section 5.9) |
| 2.2 (Goals) "heat scores" | ATNP tracking recent edge contribution | Precursor to the fired-neuron credit tracking used throughout; ultimately found too diffuse without further restriction (Section 5.9) |
| 2.2 (Goals) "neurotransmitter" typing | Distinct activation types for processing vs. training | Precursor to the TD-error-as-broadcast-signal used in Section 6; see Discussion |

---

## 3. Architecture

### 3.1 Base dynamics

A network of *N* neurons is represented as a full adjacency matrix *W* (signed, [-1, 1]) and a state vector *s* (bounded [0, 1]), with a per-neuron bias vector *b* (ATNP, [0, 1]). Each step:

1. Neurons with `s_i >= 1` are considered **fired**: they push their state (exactly 1) along their outgoing edges, then reset to 0 (Uniform Threshold Trigger).
2. Each neuron's incoming drive is the sum of `1 * W[i,j]` over all fired sources *i*.
3. Raw new state = `clip(baseline + incoming + bias, 0, 1)`, where `baseline` is the neuron's previous state (0 if it just fired).
4. **State relaxation**: any positive raw state is squared before being kept (`x -> x^2`), a decay that is steepest at moderate activity and vanishes near 0 or 1 — an integrate-and-fire model with a natural "leaky but sticky-near-threshold" property.
5. Input neurons are then force-clamped to their externally supplied values, overriding whatever the dynamics computed for them. This is the **only** point at which any neuron's state is forcibly overwritten outside of the ordinary dynamics, other than the Clearing of output neurons immediately after they are read.

### 3.2 Persistent state (ASNP)

In the final architecture, network state is **never reset except at genuine episode boundaries** (the start of a new game). Across an entire task episode — every move, both players' turns — the hidden "deliberation" neurons evolve continuously; nothing is zeroed between decisions. This is a deliberate, literal implementation of the ASNP premise, tested directly in Section 5.4.

### 3.3 Final ATNP scheme

- Signed edge weights, [-1, 1]
- Per-neuron bias, [0, 1], trained
- A designated subset of hidden neurons whose mean state serves as a continuously-updated **value estimate** (Section 6)
- A per-neuron running activity estimate, used for continuous **homeostatic bias regulation** (Section 6.1), independent of any reward signal

---

## 4. Experiment 1: A 5-Neuron NAND Gate

### 4.1 The literal training rule collapses

The original specification's associative-score training rule was implemented exactly as written: edges into a firing neuron accrue an associative score regardless of whether their source actually fired; periodic feedback (majority-correct-over-an-interval, ±1) adjusts each neuron's outgoing edges in proportion to their share of that neuron's total associative score.

Result: the network does not merely fail to converge — it **collapses**. Interval accuracy drops to 0% after the first few feedback cycles and stays there for the remainder of training (2000 intervals tested). The rule credits edges indiscriminately (any incoming edge of a firing neuron, whether or not the source contributed) and applies feedback in large, undifferentiated batches, giving the network no way to distinguish which specific behavior caused a given outcome.

### 4.2 A corrected local rule, still capped

Two fixes were applied: (a) credit only edges whose *source* neuron actually fired (real coincidence detection, closer to genuine Hebbian plasticity), and (b) per-trial, signed feedback (`target - output`) rather than batched majority-vote feedback. This rule reliably found *some* signal but plateaued at exactly **75% accuracy** (3 of 4 truth-table rows) across every tested variant — batched or per-trial, annealed or fixed learning rate.

### 4.3 Structural impossibility: positive-only weights

A ceiling this consistent, appearing regardless of training method, warranted a structural check rather than further tuning. With edge weights restricted to [0, 1] (no bias), a straightforward inductive argument shows every neuron's state is a monotonic non-decreasing function of the inputs: incoming drive from any fired neuron is non-negative, and every step of the dynamics (summation, clipping, squaring) preserves monotonicity. NAND is *not* monotonic — increasing one input while the other is held at 1 must *decrease* the output — so no assignment of non-negative weights can represent it, regardless of training method.

This was confirmed empirically rather than left as a purely theoretical claim: 20,000 random positive-weight matrices, and eight independent genetic-algorithm searches (150 individuals × 800 generations each, an unrestricted global optimizer), **never exceeded 75% accuracy**. The reset-on-fire mechanic introduces a narrow loophole in the strict monotonicity argument (a neuron receiving *more* input can fire *earlier* and reset, producing a *lower* instantaneous reading than a neuron that fired later), but this loophole proved empirically unexploitable at this scale.

### 4.4 Structural impossibility: the all-zero fixed point

Switching to signed weights ([-1, 1]) did not lift the same 75% ceiling — confirmed again by unrestricted genetic search (5 seeds, all capping at exactly 75%). Direct inspection of *which* case consistently failed revealed the cause: with the `(0,0)` input and no bias, every neuron starts at state 0, receives 0 incoming drive, and remains at 0 for all subsequent steps — a genuine fixed point of the dynamics for *any* weight matrix whatsoever (confirmed across 50 independent random matrices, zero exceptions). Since NAND requires `output(0,0) = 1`, this case is structurally unreachable without some source of intrinsic, input-independent activity.

### 4.5 Resolution

Adding **bias** (sub-convention 1-1.2), trained via the specification's own bias-update rule (only adjusting a neuron's bias when it has no outgoing associative credit of its own), removed the fixed point. Combined with the corrected coincidence-based delta rule (Section 4.2) and **checkpointing** (retaining the best-scoring weight snapshot observed during training, since the hard threshold-trigger produces oscillatory rather than monotonically convergent training dynamics), the network reached **100% accuracy on 15 of 15 random seeds**.

**Summary of NAND results:**

| Variant | Result |
|---|---|
| Literal associative-score rule | Collapses to 0% |
| Corrected coincidence credit, positive weights | Capped at 75% (structural: monotonicity) |
| Corrected coincidence credit, signed weights, no bias | Capped at 75% (structural: `(0,0)` fixed point) |
| + bias | 14/15 seeds reach 100%, oscillatory |
| + bias + checkpointing | **15/15 seeds, 100%** |

---

## 5. Experiment 2: Tic-Tac-Toe

### 5.1 Architecture and initial training

An 18-input / 24-hidden / 9-output network (51 neurons total: one input pair per cell for "X here" / "O here", one output per cell) was trained with outcome-based reinforcement: play a full game against a random opponent, then apply a local delta-rule update — recency-weighted across the game's moves — scaled by the game's outcome (+1 win, -0.1 draw, -1 loss).

Result: **77% win / 4% draw / 20% loss** against a random opponent (baseline for random-vs-random play: 57% / 13% / 30%), reached within roughly 500 games and then **completely frozen** for the remaining 5,500+ games of training, unchanged to the decimal.

### 5.2 Diagnosing the plateau: a real tactical hole

Direct inspection of losing games revealed a consistent, specific failure: the network plays an almost fixed opening sequence and **never blocks an opponent's developing line**, in any of the losses examined, despite the threat being visible for multiple turns beforehand.

### 5.3 Persistent state does not help (and this is informative)

Given that tic-tac-toe is fully Markovian — the current board already contains all information needed to play well — a natural hypothesis was that cross-move memory (Section 3.2) was not actually the network's bottleneck. This was tested directly: switching from a reset-every-move architecture to one where hidden state persists continuously across an entire game (with the network additionally "observing" a few passive deliberation steps after the opponent's move) produced an **identical result to sixteen significant figures on some evaluation metrics**, and the identical specific losing-game pattern. Direct inspection confirmed the persistent state was genuinely evolving (non-trivial, changing hidden-neuron values across moves) — this was not an implementation bug, but a real null result: **memory and generalization are separable capabilities, and this task's bottleneck was never about memory.**

### 5.4 Symmetry augmentation fixes positional invariance, not the ceiling

A second hypothesis was that the network had memorized a *specific*, position-locked response pattern rather than a general "block the line" concept. This was tested by training with one randomly chosen board symmetry (of the 8 dihedral transforms) fixed per game — diversifying which literal board positions a given strategic pattern appears at across training, while keeping the transform fixed *within* a game so the persistent hidden state's input-neuron-to-cell mapping stayed coherent.

This worked exactly as intended: an invariance check (evaluating the same trained network under each of the 8 fixed board orientations) showed consistent performance (75–78%) regardless of orientation, versus the untrained expectation of orientation-dependent performance. But the *aggregate* win rate did not move. Inspection of losing games post-augmentation showed the network was now failing to block threats on *every* line (rows, columns, diagonals) rather than specifically the one orientation seen before — confirming the invariance fix had generalized correctly, while revealing that the deeper problem was that **no defensive concept had been learned in the first place**, in any orientation.

### 5.5 A harder training opponent does not help either

Replacing the random training opponent with a one-ply-lookahead opponent (takes an immediate winning move if available, else random) — intended to make every unblocked threat costly and thereby supply a stronger defensive training signal — again produced **numerically frozen results**: identical win/draw/loss and identical "optimal move rate" (72%, measured against a full minimax oracle used only for diagnosis) from roughly game 2,000 onward through the remaining 18,000 games of training.

### 5.6 Diagnosing the plateau, again: saturation

The recurrence of an *exactly frozen* metric across three structurally different training setups (whole-game credit, symmetry-augmented whole-game credit, harder-opponent whole-game credit) motivated a direct check of the weight matrix. Result: **86% of all weights sat at exactly ±1** (the clip boundary), and every neuron within each structural role — all 18 input neurons, all 24 hidden neurons, all 9 output neurons — had converged to an **identical** bias value within its group. This is consistent with a training rule whose updates had saturated nearly the entire parameter space early in training, after which further updates in the same direction became no-ops (clipped away).

### 5.7 A control that falsified the saturation hypothesis

A direct fix — a smaller step size plus mild weight decay, to prevent permanent boundary lock-in — was tested. The result **disproved the saturation hypothesis as originally framed**: weight saturation now oscillated dramatically across training checkpoints (from 0% to 75% and back, repeatedly, as decay and re-training fought each other), yet the **win/draw/loss policy remained completely unchanged** across every one of those swings. The network's actual decisions were evidently governed by something other than weight magnitude — most likely the *sign pattern* of the weights, which locked in early and never flipped, regardless of how much the magnitudes moved.

### 5.8 A precise, per-move oracle signal — and the same frozen result

To rule out "insufficiently informative reward" as the explanation, an exact minimax oracle (feasible because tic-tac-toe's state space is small enough to solve exhaustively) was used to grade every individual move against the best available alternative, training on the resulting advantage immediately rather than waiting for the game to end. This is a substantially more precise, more immediate training signal than anything used previously.

Result: **identical numbers yet again**, matching the previous experiments' win/draw/loss and optimal-move-rate to within noise. This ruled out training-signal quality/timing as the bottleneck and pointed toward something mechanistic in how the signal was being *applied*.

### 5.9 Root cause: diffuse credit assignment causes emergent output lockstep

Direct inspection of the output layer's raw (pre-Clearing) state at decision time revealed the actual mechanism: **all nine output neurons were reading exactly 1.0**, permanently, from roughly game 100 onward. With every output tied at the ceiling, `argmax` degenerates to "always select the lowest-index legal cell" — a decision rule completely disconnected from the board, the weights, or the training signal, which explains every frozen result in Sections 5.1–5.8 simultaneously.

The cause was two compounding design flaws in the credit-assignment rule inherited from the NAND experiments:

1. Weight updates touched a fired neuron's **entire outgoing row** — every target it connects to — rather than only the edge relevant to the specific decision made. A hidden neuron contributing to one output's win would have its connections to *all other outputs* boosted identically.
2. Bias updates applied to **every neuron lacking outgoing associative credit simultaneously** — which, for structural reasons, is nearly always all 9 output neurons at once (they are near-leaf nodes with little further outgoing influence within a short deliberation window), causing their biases to drift in lockstep rather than differentiate.

An initial attempted fix — hardwired lateral inhibition among the output neurons, intended to force competitive differentiation — instead reproduced the **exact same failure mode with the opposite sign**: outputs saturated to a permanent 0 rather than 1, with a bit-for-bit identical resulting policy, confirming that inhibition strength alone could not substitute for correct credit assignment.

### 5.10 A surgical fix, and its own second-order version of the same bug

Restricting weight updates to only the **column** of the specific output neuron that won the decision (not its whole row), and restricting bias updates to only that neuron (not every neuron satisfying a shared, largely output-layer-wide condition), broke the freeze: post-training biases differentiated meaningfully (e.g. `[1.0, 0.4, 1.0, 0.3, 1.0, 0.7, 1.0, 0.9, 1.0]` across the 9 output neurons, rather than a single shared value), and aggregate performance genuinely improved (loss rate against random fell from a fixed 20% to 15%; optimal-move-rate against the oracle rose from a fixed 72% to as high as 82%).

However, output *states* were still frequently saturated to `[1,1,...,1]` at decision time despite the differentiated biases — a **second-order version of the same lockstep problem**, this time via "generically active" hidden neurons that fire on nearly every board state regardless of content, and whose contributions to all nine output columns therefore accumulate roughly in parallel over the course of many games.

**Summary of tic-tac-toe diagnostic results (all evaluated against the one-ply-lookahead opponent unless noted):**

| Configuration | Win / Draw / Loss | Notes |
|---|---|---|
| Random-vs-random baseline | 57% / 13% / 30% | (vs. random opponent) |
| Outcome-based whole-game credit | 70% / 3% / 27% | frozen from ~game 2000 |
| + persistent state (ASNP) | 70% / 3% / 27% | numerically identical; confirms memory was not the bottleneck |
| + symmetry augmentation | 70% / 3% / 27% (aggregate) | orientation-invariance confirmed; ceiling unchanged |
| + one-ply-lookahead training opponent | 70% / 3% / 27% | frozen |
| + weight decay (saturation control) | 70% / 3% / 27% | saturation now oscillates 0–75%, policy still frozen |
| + minimax-oracle per-move grading | 70% / 3% / 27% | rules out signal quality/timing |
| + surgical (column-specific) credit | **65% / 4% / 32%*** | *vs. smart opponent; loss vs. random opponent fell to 15% |

---

## 6. Experiment 3: Self-Play with TD(0) Value Learning

### 6.1 Motivating constraints and design

Under a stated commitment to XNN's core premises — full recurrence, genuine ASNP, ATNP, and **no labelled data** — the minimax oracle used in Section 5.8–5.10 had to be retired: grading every move against an external, exhaustively-computed judge is in real tension with "no labelled data," even though it was a legitimate and useful *debugging* tool for isolating the credit-assignment bug in Section 5.9.

The replacement architecture:

- **Perspective-canonicalized self-play.** The same weights play both X and O. Input encoding is always "my pieces / opponent's pieces" rather than "X pieces / O pieces," so a single network can play either role from the same 18 input neurons.
- **A value readout sharing the recurrent substrate.** Four hidden neurons are designated as a continuously-updated value estimate — the network's own running judgment of "how good is this position for whoever is about to move" — read directly from the same persistent state validated in Section 5.3, giving deliberation an actual causal role in decision-making for the first time in this investigation (Section 5.3 found persistent state made no measurable difference specifically because nothing was using it for anything beyond the final move choice).
- **TD(0) bootstrapping.** After each move, the next mover's value estimate — negated, since the game is zero-sum — becomes the bootstrap target for the *previous* mover's transition. The final move of a game trains against the true terminal outcome rather than a bootstrap. No external reward shaping, oracle, or hand-designed opponent is used anywhere in training.
- **Surgical (column-specific) credit assignment**, per the fix validated in Section 5.10, applied to both the actor (the chosen action's incoming edges) and the critic (the value neurons' incoming edges), sharing the same TD error — standard actor-critic structure, implemented as a purely local per-edge update.
- **Continuous homeostatic bias regulation**, running every step independent of any reward signal: each neuron tracks its own exponential moving average of firing rate and continuously self-corrects its bias toward a target activity level. This is a standing defense against the saturation/lockstep pathology that recurred across every earlier version of this investigation, rather than a one-off patch for a specific instance of it.

### 6.2 Results

Across 20,000 self-play games (no external opponent involved in training at any point), evaluated purely for diagnostic purposes against the same fixed external opponents used throughout Section 5:

- **Mean absolute weight remained stable around 0.5 throughout training**, never drifting toward the ±1 boundary — the saturation/lockstep collapse observed in every earlier tic-tac-toe variant did not recur.
- Performance was **non-monotonic**: win rate against the random opponent ranged roughly 45–87% and against the smart opponent roughly 28–65% across evaluation checkpoints, without a clear, stable upward trend. This is consistent with the known non-stationarity of self-play — the network is simultaneously each side's training opponent, so as its own play improves, its own training curriculum shifts under it, leaving no fixed target for performance against an external, fixed benchmark to converge toward.
- **Checkpointing** (retaining the best-scoring snapshot observed during training against the fixed external benchmarks, used here purely for model selection, not as a training signal) recovered a stable, deployable network reaching:

| Opponent | Win | Draw | Loss |
|---|---|---|---|
| Random | 77.4% | 9.4% | **13.2%** |
| One-ply lookahead | 64.0% | 5.6% | **30.4%** |

Final weight saturation: 2.8%.

This is competitive with — and on the random-opponent loss rate, slightly better than — the oracle-supervised result from Section 5.10 (15% loss vs. random), achieved with **zero external supervision**: no labelled targets, no hand-designed opponent used in training, no oracle grading, purely self-referential TD bootstrapping between the network's own successive judgments.

---

## 7. Experiment 4: Scaling Capacity and Deliberation Depth

### 7.1 Motivation

The architecture from Section 6 satisfies the constraints motivating this investigation — full recurrence, genuine ASNP, ATNP, no labelled data — but everything reported so far used a fixed, fairly small network (24 hidden neurons, 3 deliberation steps per decision). The natural next question is whether the architecture actually benefits from more of the two things that differentiate it from a generic small classifier: more neurons (**capacity**) and more recurrent steps per decision (**deliberation depth**). We test each independently under matched training budgets, then jointly, using the self-play recipe from Section 6 unchanged.

Before scaling, the per-column training update (Section 6.1) was rewritten from an explicit Python loop over all *N* neurons to a vectorized (boolean-indexed) NumPy operation. This is a pure performance change with no effect on the update semantics — confirmed by re-evaluating the Section 6.2 network before and after the change and observing identical behavior — but it was necessary to keep training cost from growing needlessly as *N* increases.

### 7.2 Scaling capacity alone

Holding deliberation depth fixed at 3 steps (as in Section 6), we trained networks with 24, 150, and 600 hidden neurons via self-play, each with checkpointing as in Section 6.2. Training budgets were 20,000 games for the two smaller networks and 13,000 games for the largest, the latter reduced only to fit the larger network's per-game compute cost within a practical wall-clock budget (single-seed results; see Limitations):

| Hidden neurons | Total *N* | Loss vs. random | Loss vs. smart opponent | Final saturation |
|---|---|---|---|---|
| 24 | 51 | 13.2% | 30.4% | 2.8% |
| 150 | 177 | 12.6% | 25.8% | 0.9% |
| 600 | 627 | **8.4%** | **22.4%** | 0.2% |

The improvement is consistent and monotonic across two independent size increases (12x total), and weight saturation stayed low and, if anything, decreased with scale — confirming the homeostatic regulation and surgical credit assignment from Section 6.1 continue to prevent the lockstep collapse of Section 5.9 at substantially larger *N*, rather than merely postponing it.

### 7.3 Scaling deliberation depth alone (matched budget)

Holding the small network's capacity fixed at 24 hidden neurons, we varied `steps_per_decision` while holding the training budget fixed at exactly 20,000 games for every value — a fully matched comparison:

| Steps per decision | Loss vs. random | Loss vs. smart opponent |
|---|---|---|
| 3 | 13.2% | 30.4% |
| 6 | 8.4% | 24.6% |
| 10 | 14.0% | **19.8%** |
| 20 | 13.6% | **19.4%** |

Against the harder opponent, loss roughly halves from 3 to 10 steps, then plateaus. Notably, the 24-neuron network at 10 deliberation steps (19.8% loss vs. the smart opponent) outperforms the 600-neuron network at 3 steps (22.4%, Section 7.2) on this metric — more time to think, with two orders of magnitude less substrate, outperformed substantially more substrate given less time to think. This is a direct, controlled piece of evidence for deliberation depth (rather than raw capacity) mattering on its own terms, which nothing in Sections 4–6 tested directly.

### 7.4 An initial large-network comparison, and why it had to be redone

A first pass at testing deliberation depth on the large (600-hidden) network compared 3 steps (13,000 games, as in Section 7.2) against 10 steps trained for only 8,000 games — the number that fit the same wall-clock budget, given the larger network's higher per-step cost. This comparison suggested roughly comparable, perhaps slightly worse, performance at 10 steps. We do not report the specific numbers from this pass, because the comparison is confounded: the two configurations received different amounts of training experience, and the difference observed is equally consistent with "10 steps doesn't help the large network" and "the 10-step network was simply undertrained relative to its baseline." We flag this explicitly because it is exactly the kind of premature-conclusion mistake this paper's own methodology (Section 1.2) commits to avoiding, and it very nearly made it into this report without being caught.

To resolve it properly, we implemented checkpointed, resumable training (`resumable_train.py`) so a single run could span multiple time-limited invocations while training on an exact, matched number of games regardless of per-game cost.

### 7.5 Properly matched capacity x depth comparison

With both configurations trained on exactly 13,000 games:

| Network | Steps | Games | Loss vs. random | Loss vs. smart opponent |
|---|---|---|---|---|
| Small (24 hidden) | 3 | 20,000 | 13.2% | 30.4% |
| Small (24 hidden) | 10 | 20,000 | 14.0% | 19.8% |
| Large (600 hidden) | 3 | 13,000 | **8.4%** | **22.4%** |
| Large (600 hidden) | 10 | 13,000 | 18.8% | 28.4% |

With the confound removed, the result reverses direction from what the small network showed: **more deliberation steps made the large network clearly worse** — loss against the random opponent more than doubled (8.4% → 18.8%) and loss against the smart opponent also increased (22.4% → 28.4%).

### 7.6 Interpretation: capacity and deliberation depth are not independent levers

Taken together, Sections 7.3 and 7.5 show deliberation depth helping substantially for the small network and hurting for the large one, under identical training budgets. Our best explanation: a network with more neurons run for more steps per decision has a much larger effective space of internal dynamics to explore and calibrate for every single decision than a smaller network does, and the same fixed number of training games provides proportionally sparser coverage of that larger effective space. The large-and-deep configuration was not given enough training experience, relative to its own effective complexity, to calibrate the additional dynamical richness that more steps introduced — even though it received the same *nominal* number of games as its more successful 3-step counterpart.

This is a substantive revision to the naive reading of the motivating premise (Section 1.1) that more recurrent deliberation time is straightforwardly better. What the data actually support is a conditional version: deliberation depth's benefit depends on having a training budget matched to the effective size of the dynamical space that depth, combined with capacity, opens up — structurally analogous to depth/width trade-offs in conventional deep networks, where neither dimension can be scaled in isolation without accounting for the other.

Every result in this section is from a single seed per configuration. Section 6.2 already established that this self-play setup has real run-to-run variance (win rate swinging roughly 45–87% across checkpoints within one run). The effect sizes here — loss rates roughly halving in Section 7.3 and roughly doubling in Section 7.5 — are large enough that we believe the *direction* of both effects is real, but the exact percentages should not be treated as precise without repeated seeds, which we did not have time to run.

---

## 8. Experiment 5: A Genuine Memory Task — Delayed Recall

### 8.1 Motivation

Every task so far shares a property that makes it a poor test of the architecture's actual motivating premise: tic-tac-toe is Markovian (Section 5.3 found persistent state made no measurable difference, precisely because the current board already contains everything needed), and NAND has no temporal structure at all. Neither could have shown genuine cross-time memory mattering even if it did. This experiment is designed specifically so that it can't be solved without information surviving across ticks.

**Task.** A trial consists of a cue tick (one of two dedicated input neurons fires, encoding a bit to remember), *D* delay ticks (the cue neurons are silent, but six distractor input neurons fire random noise every tick — deliberately, since an all-silent input is a literal fixed point of this network's dynamics, Section 4.4, so a silent delay would let a network "solve" the task by passively doing nothing), and a probe tick (a dedicated trigger neuron fires; the network must report the *original* cue on two output neurons). Two conditions share an identical architecture and training rule: **full memory** (ordinary persistent-state XNN, as validated throughout) and **ablated memory** (a negative control whose hidden neurons are forcibly zeroed after every tick — this should fail at any D > 0, and if it doesn't, the task is leaking the answer through some channel other than genuine memory).

### 8.2 Two real bugs, caught before trusting any result

Building this task surfaced two implementation issues, both worth reporting because of what they say about how easy this kind of experiment is to get subtly wrong:

1. **The ablation control leaked.** The first version zeroed only hidden neurons, not output neurons — output-layer state is a separate index range and was never touched, so it silently carried the answer across ticks anyway. The "memory-ablated" control scored 99–100%, which should have been an immediate red flag rather than a result.
2. **A clamp-timing leak.** `XNN.tick()` only overwrites input neurons *after* each sub-step, so the first sub-step of any new tick briefly sees the *previous* tick's clamped input value — including the cue — before it gets overwritten. This let the cue leak through even in the ablated condition. The fix was applied at the task level (explicitly pre-clearing input neurons before ticks where they should be silent) rather than by changing the shared model code, since the underlying dynamics are correct and validated elsewhere; this timing detail simply needed explicit handling for a task that depends on clean silence.

With both fixed, the ablated condition sits cleanly at chance (~48–57%) across every D tested, confirming the comparison is valid.

### 8.3 A deeper problem: the training rule doesn't reach back in time

Even after both bugs were fixed, the full-memory condition **also** sat at chance — including at D=0. The reason: the existing training rule (`train_actor`, validated throughout Sections 4–7) only ever credits the firing pattern of the *current* decision tick. With only the probe tick tracked, nothing during the cue or delay ticks ever received a training signal at all — whatever hidden-layer dynamics would need to encode and maintain the cue was never shaped by training and stayed at its random initialization.

Extending credit across the whole trial (a flat, undecayed eligibility trace — any neuron that fired at *any* point from cue to probe, not just at the probe tick) fixed D=0 (73.0% vs. ablated's 52.0%) and D=1 (55.0% vs. 52.2%), but D≥3 remained at chance regardless.

### 8.4 Ruling out decay rate, training budget, and capacity — one at a time

Three plausible explanations were tested and eliminated, each with a dedicated, controlled experiment rather than assumed:

- **Decay rate**: a systematic sweep (7 decay values × D∈{3,5,10} × 3 seeds) found no decay value reliably beat chance at any of these delays — differences were within seed-to-seed noise (~0.03–0.04) throughout.
- **Training budget**: even 100,000 trials (20× the original budget) at D=3 showed no improvement (0.488 mean across 2 seeds, indistinguishable from 5,000 trials).
- **Capacity**: scaling from 24 to 300 hidden neurons (12×) at D=3, with matched training, showed no improvement either (0.490–0.514 across all sizes).

### 8.5 Root cause, measured directly

Comparing the weight matrix before and after 20,000 training trials, split by connection type:

| Connection type | Mean \|ΔW\| after training |
|---|---|
| hidden → hidden (recurrent/maintenance) | **0.00000** |
| input → hidden (encoding) | **0.00000** |
| hidden → output (readout) | 0.60495 |
| input → output (shortcut) | 0.75517 |

The recurrent and encoding pathways — the only connections that could possibly shape a memory trace to survive multiple ticks — changed by *exactly zero*, to five decimal places, regardless of training budget or capacity, because `train_actor` only ever updates columns belonging to output neurons. This fully explains the D=0 partial success (readable from residual one-hop activity, no real maintenance needed) and the collapse at D≥3 (information has to survive multiple hops through completely untrained, random recurrent weights amid constant distractor interference).

### 8.6 Is the task itself hard, or just hard for this rule? A gradient-checked RNN baseline

To answer this cleanly, a conventional backprop-through-time Elman RNN was implemented by hand (no `torch` available in this environment) on the *identical* task generator, verified against finite-difference gradients before being trusted (max relative error 7.7×10⁻¹⁰).

| Delay (D) | RNN baseline accuracy |
|---|---|
| 0–10 | 100% |
| 20 | 96.7% |
| 40 | 49.2% (chance — the RNN's own limit) |
| 80 | 66.7% (unstable across seeds) |

With **fewer parameters** (866 vs. the XNN's 1,260) and **far less training** (3,000 trials vs. the 100,000 that failed to move the XNN off chance), the RNN solves the task cleanly up to D≈20 before its own well-known limitation (vanishing gradients in a plain tanh RNN over long sequences) takes over. This settles the question: D=3 was never intrinsically hard. It was hard specifically and only because the XNN's local credit-assignment rule structurally never touches recurrent weights.

---

## 9. Experiment 6: Local Fixes for Recurrent Credit Assignment

### 9.1 Three candidates

Given the precise diagnosis in Section 8.5, three theoretically-motivated local mechanisms for training hidden-layer weights were implemented and compared, each keeping output-layer training identical (the real supervised error, as always) so the comparison isolates exactly the mechanism in question:

**A. Direct Feedback Alignment (DFA).** A *fixed*, never-updated random matrix *B* (shape hidden × output) projects the real output error into a per-hidden-neuron pseudo-error, applied via the same trace-weighted column update used everywhere else. No weight transport (B never reads W), no backward pass.

**B. Node perturbation (REINFORCE-style).** Small Gaussian noise injected into each hidden neuron's bias for the duration of a trial; the resulting reward, relative to a running baseline, gives a per-neuron score-function gradient estimate (`ε_j · (r − baseline) / σ²`).

**C. Slowness / temporal-consistency Hebbian rule.** A continuous, reward-independent local rule reinforcing a hidden neuron's incoming edges from whatever was active the tick before, in proportion to how stable its own activity was between those two ticks — entirely unsupervised, running every tick.

### 9.2 Results

| Method | D=3 | D=5 | D=10 | D=20 |
|---|---|---|---|---|
| Original rule (output-columns only) | chance | chance | chance | — |
| **DFA** | **95.6%** | **100%** | **100%** | **87.5%** |
| Node perturbation | chance | — | — | — |
| Slowness (Hebbian, signed) | chance | — | — | — |

**DFA works, dramatically.** With a properly tuned hidden-layer learning rate, it goes from chance to 95–100% across D=3–10 and remains strong at D=20 — closing almost the entire gap to the fully-supervised RNN baseline (Section 8.6), using a mechanism that never computes anything resembling a true gradient. The first hyperparameter setting tried undershot badly (D=10: 52%, chance) purely from too low a hidden-layer learning rate; a 3× increase unlocked the full result — worth flagging since it's a reminder that a negative result for a *method* can sometimes just be an under-tuned hyperparameter for that method, exactly the kind of premature conclusion this paper has tried to avoid elsewhere (Sections 5.7, 7.4).

**Node perturbation failed for a specific, explicable reason, not mysteriously.** Even at 60,000 trials (2 seeds) it stayed at or below chance across a 9-point hyperparameter grid. This is consistent with node perturbation's known theoretical weakness: perturbing 24 hidden neurons simultaneously means a single scalar reward cannot disentangle which neuron's noise actually mattered, and the resulting gradient estimate is too high-variance to be useful at this scale without dramatically more samples than were practical here.

**Slowness/Hebbian failed in a more interesting way.** The first (always-positive) version didn't just fail to help — it reproduced the lockstep saturation pathology from Sections 5.6–5.9: an unconstrained, always-positive Hebbian update has nothing pushing weights back down, and 63.4% of the weight matrix saturated to the clip boundary. A signed, differential version (reward *above-average* stability, punish *below-average*) fixed the saturation (down to ~21%) but left accuracy at flat chance (49.5% mean, 3 seeds) — a clean negative result for this specific formulation, distinguishable from a bug.

---

## 10. Experiment 7: Adaptive Feedback Alignment and Context Change

### 10.1 Motivation

DFA's fixed random matrix *B* is a genuine departure from "everything is one recurrent, locally-trained substrate" — it's external scaffolding, not a neuron or an edge in the network. A natural question: could *B* itself adapt over time via a local rule, and would that help the network track a genuinely changing task context, or is a frozen, uninformed projection actually preferable?

**Mechanism**: Kolen-Pollack-style local Hebbian co-adaptation — `B += lr_B · outer(hidden_activity, error) − decay_B · B` — using only signals already locally available where the error is computed (hidden activity, real output error). This requires no reading of *W* (no "weight transport"), staying within the same locality constraint as every other mechanism in this paper.

### 10.2 Two bugs before a fair test was possible

1. **Naive decay collapsed B.** With `lr_B` and `decay_B` of comparable magnitude, B's mean magnitude fell from 0.41 to 0.016 over 15,000 trials — as errors shrink with learning, the Hebbian term shrinks with them while decay stays constant, so decay wins by default and erodes B into a near-useless matrix, silently reproducing the Section 8.5 failure through a different door. Fixing the decay-to-learning-rate ratio (decay two orders of magnitude smaller) preserved more of B's magnitude but still didn't restore standard-task performance (0.41–0.48 accuracy across several ratios).
2. **Adapting from the start destabilizes alignment.** DFA's power comes from the forward weights *W* rotating to align with *B* over training; if *B* is also moving from the first trial, *W* is chasing a moving target with no guaranteed joint convergence. A two-phase fix — freeze *B* for a warmup period, letting *W* align first, then allow adaptation — fully restored standard-task performance (1.000, matching frozen-B, given ≥10,000 warmup trials; 5,000 warmup trials was insufficient, 0.484).

### 10.3 A genuine context-change test

A context change needed to be one that couldn't be solved by coincidence. An abrupt difficulty increase (D=10 trained, then continued at D=20) turned out to be a null test — the D=10 solution already generalized to D=20 perfectly with zero further training. Instead: swap which physical input channels serve as the cue versus the distractors (an encoding-level change — *which inputs matter* changes, not just how hard the task is). This required its own bug fix (the first version left the new cue channels exposed to the same unconditional noise-generating code as ordinary distractors, making it a strictly noisier task rather than a fair swap) and its own validation (the swapped context is independently learnable from scratch to 100% given adequate training, confirming it's a fair, comparable alternative task, not a broken one).

**Protocol**: train to a solid baseline in the original context (18,000 trials, D=10), confirm cold transfer to the swapped context is near chance (~0.46–0.61, genuine disruption), then give a *matched, limited* retraining budget (3,000 trials) on the new context to three conditions: continuing with **frozen B**, continuing with **adaptive B** (using the warmed-up two-phase mechanism from 10.2), and a **from-scratch** network given the identical budget as a reference.

| Condition (3 seeds each) | Mean accuracy after 3,000 retraining trials |
|---|---|
| Cold transfer (no retraining) | ~0.52 |
| Frozen B | **0.795** |
| Adaptive B | 0.527 |
| From-scratch reference | 0.751 |

**Adaptive B performs worse than both frozen B and training from nothing**, given the identical budget. This is the opposite of the intuitive expectation, and we think it's mechanistically explicable rather than a fluke: DFA's power specifically comes from *B* being a stable target for *W* to chase. Right after a disruptive context change is exactly when the network's error and hidden-activity signals are noisiest and least informative — precisely the worst moment to let those same signals drive changes to the one thing (B) that's supposed to stay stable. Frozen B sidesteps this entirely: it was never *about* the old context to begin with, just an arbitrary fixed rotation, so there's nothing about it to become stale.

---

## 11. Experiment 8: Cross-Task Transfer of a Trained Hidden Core

### 11.1 Motivation

Everything in Sections 4–10 asked whether a network could learn *one* task. This experiment asks a different question: does a hidden recurrent core, once trained, carry any structure that transfers to a genuinely different task — or does training only ever produce something narrowly specific to whatever it was trained on? Tic-tac-toe (Section 6) and delayed-recall (Section 8) happen to share an architecture with identical `n_hidden=24`, so the hidden-to-hidden weight submatrix and hidden biases from a trained tic-tac-toe network can be extracted and spliced directly into a fresh delayed-recall network's hidden block — leaving the input/output layers (sized differently for the two tasks) randomly initialized as usual — and the resulting network's learning curve compared against ordinary random initialization.

### 11.2 A real, reliable benefit — but from what, exactly?

At D=10, comparing three seeds each, checkpointed through training:

| Trials | TTT-trained transplant | Plain random init |
|---|---|---|
| 300 | 0.798 | 0.655 |
| 1,500 | 0.932 | 0.853 |
| 3,000 | **1.000** | 0.757 |
| 6,000 | **1.000** | 0.855 |

All three transplant seeds converge cleanly to 100% by 3,000 trials. Random init is markedly less reliable — one of three seeds never solves the task at all even by 6,000 trials, stuck around 56%, a bad random draw the transplant never exhibits in any seed tested. This is a real, if narrow, transfer effect: something about the tic-tac-toe-trained core makes it a more reliable starting point for a structurally unrelated task.

### 11.3 Ruling out two explanations, one at a time

**Is it just well-conditioned bias?** Homeostatic regulation (Section 6.1) runs continuously during tic-tac-toe training, keeping neuron activity in a moderate range — a property a raw random draw has no guarantee of. A control core was built with the *same* random weight statistics as ordinary initialization, but with bias settled via pure homeostatic regulation under generic noise input for 8,000 ticks — no task, no training signal, no learning at all (confirmed directly: hidden-to-hidden weights are unchanged before and after settling, since homeostasis only ever touches bias). This control does **not** replicate the transplant's benefit — it has its own stuck seed (0.53–0.60 throughout, never solving the task), the same pathology plain random init shows. Bias conditioning alone is ruled out.

**Is it just "any trained matrix beats random, regardless of task"?** A second control was trained on a task deliberately unrelated in content to both tic-tac-toe and delayed-recall: single-tick, memoryless classification of a random binary vector by a fixed majority rule — no sequence, no game structure, no time dimension at all. This core's transplant was messier and less reliable than the tic-tac-toe transplant, including an outright regression in one seed (0.99 accuracy at 1,500 trials falling to 0.51 by 6,000 — backsliding that never occurs with the tic-tac-toe transplant, where every seed is monotonically non-decreasing to a clean ceiling). "Any training helps equally" is also ruled out.

### 11.4 A sharper hypothesis, and an inconclusive test of it

The single-tick classification control has a confound worth taking seriously rather than glossing over: it never required the hidden layer to sustain anything *across* ticks at all, since the task has no across-tick structure. Tic-tac-toe, by contrast, is inherently multi-move — self-play necessarily shapes recurrent dynamics to behave coherently across a whole sequence of persistent-state ticks, a property tic-tac-toe and delayed-recall share structurally that a single-shot task cannot. This suggests a sharper hypothesis: what transfers may not be tic-tac-toe-specific content, but practice sustaining stable multi-tick recurrent dynamics in general.

A running-parity control was built to test this directly: at each of 10 ticks a random bit arrives, and the network must continuously update a running XOR, reported only at the final tick — genuinely multi-tick, but content-unrelated to both prior tasks, and structurally different from delayed-recall (every tick's input matters and must be incorporated into an evolving computation, rather than one early cue being protected from later distractor noise). This control turned out to be **invalid rather than informative**: the source core itself only reached 45–52% accuracy on its own training task — chance level. Parity/XOR-family tasks are a long-documented hard case for simple learning rules, and DFA evidently could not solve a 10-tick running parity within the training budget used here. A core that never successfully learned its source task cannot inform whether successfully-learned multi-tick practice transfers; this result is reported as inconclusive, not as a negative data point, and the multi-tick-practice hypothesis remains open rather than tested.

### 11.5 Honest summary

Two specific null hypotheses for the transplant benefit — bias conditioning alone, and "any training helps equally regardless of content" — are both ruled out by direct controls. A third, more specific hypothesis — that practice sustaining multi-tick recurrent dynamics, rather than task-specific content, is what transfers — was targeted by a follow-up control that failed for an independent reason (the control task itself was too hard to learn at this scale) and remains untested. What survives, at this point, is only the narrower claim: *something* about the tic-tac-toe-trained core reliably benefits delayed-recall, beyond bias conditioning and beyond mere presence-of-training, and precisely characterizing what that something is remains open.

---

## 12. Experiment 9: Embodied Foraging Under Partial Observability

### 15.1 Motivation

Every prior experiment in this paper judges XNNs the way any other machine learning model would be judged: accuracy on a clean, well-specified task with an unambiguous correct answer. That standard has been essential for the diagnostic work in Sections 4–11, but it is a narrow lens for an architecture whose stated purpose (Section 1.1) is to emulate organic cognition rather than to win at logic gates and board games. This experiment deliberately steps back from that lens: a real-time-flavored survival environment, an "animal" avatar that must collect resources while avoiding threats, evaluated less on a single accuracy number and more on whether the architecture's validated mechanisms (persistent state, DFA-trained recurrence, TD value learning, homeostasis) actually do anything useful under embodied, continuously-running conditions.

We treat this as the first of two stages, deliberately separated: **Stage 1** (this section) validates whether the architecture can learn a rich, multi-objective, partially-observable task at all, under ordinary lockstep execution — one environment tick, one network decision, sequenced, exactly like every prior experiment. **Stage 2** (Section 16) is the qualitatively different question of genuine, literal real-time concurrency. We do not conflate the two: if the richer task fails, we want to know whether it failed because of the architecture or because of asynchrony, not both at once.

A methodological note before any results: an initial implementation of this environment and agent already existed in the working environment when this experiment began, apparently from earlier in this session but outside the context available for this response. Rather than trust it, every piece was independently re-verified by direct execution before being relied upon — the same discipline this paper has applied throughout (Sections 5.7, 7.4, 8.4, 10.2, 11.4) applied here to the code itself, not just to results.

### 15.2 Environment

A 20×20 grid world. The avatar has a single energy resource (start: 100, max: 100) that decays by 1 per tick (metabolic cost) and is restored by 30 per food item eaten (auto-pickup on contact, food respawns elsewhere immediately); three enemies deal 40 damage on contact and either pursue the avatar (within a detection radius of 4) or random-walk otherwise. Death occurs at zero energy; episodes also terminate at a 500-tick cap. Critically, **the avatar does not see the whole map** — sensing is restricted to 8 directional sectors (compass bins) within a radius of 5, each reporting only food/enemy presence, not exact position or distance. This partial observability is deliberate: tic-tac-toe's full observability meant persistent memory provably made no difference (Section 5.3); a genuine test of whether memory matters requires a task where the current observation alone is not always sufficient. Reward is the raw tick-by-tick energy change — no additional shaping.

Input encoding: 16 directional sensor neurons (8 sectors × {food, enemy}) + 5 energy-level thermometer neurons = 21 inputs. Output: 5 actions (4 directions + stay).

### 15.3 Baselines confirm the environment is sensibly difficult

A random-action agent survives 24.1 ticks on average (20 episodes) out of a possible 500, dying almost entirely to predation (19/20 episodes) rather than starvation, and eating almost no food (0.30 per episode). A scripted heuristic agent (flee any sensed enemy; otherwise move toward the nearest sensed food; otherwise move randomly) — the direct analogue of the one-ply-lookahead opponent used for tic-tac-toe, decided before seeing any trained-agent result — survives 90.7 ticks on average (30 episodes), nearly 4× longer, with a much more balanced mix of death causes (14 starvation, 16 predation). This confirms the environment rewards sensible behavior without being either trivial or unsurvivable, and gives an interpretable floor and reference point before training anything.

### 15.4 Architecture: a synthesis, not a fourth mechanism

Rather than invent a new training rule, this experiment combines three previously and separately validated mechanisms:

- **TD(0) actor-critic with value neurons**, from the tic-tac-toe self-play work (Section 6) — needed because a consequential decision and its payoff can be separated by many ticks here, more than anywhere else in this project. Single-agent, so there is no adversarial sign-flip; otherwise identical in form.
- **DFA-based hidden-layer credit assignment**, from the delayed-recall work (Section 9) — Section 8.5 found that without some such mechanism, hidden-to-hidden weights receive exactly zero training signal regardless of budget or capacity. Since actor/critic training here uses a scalar TD error rather than a vector classification error, the feedback projection *B* is a fixed random **vector** (length = hidden count) rather than a matrix.
- **Continuous homeostatic regulation**, already built into `XNN.tick()` — the standing defense against the saturation collapse documented in Sections 5.6–5.9.

One new, environment-specific detail was necessary: raw environment rewards (as large as +30 or −40 in a single tick) are far larger in magnitude than the roughly ±1 signals used everywhere else in this paper, and applying them unscaled to the surgical column-update rule blew weights through their entire [-1, 1] range in a single tick, reproducing a lockstep-like collapse. A fixed reward-scaling factor (dividing by 40) and a running reward baseline (subtracted before scaling) resolved this — worth noting as a reminder that the surgical credit-assignment mechanism validated in Sections 5.9–5.10 and 9 assumes a roughly bounded reward signal, an assumption this task was the first to violate.

### 15.5 Training results

Trained via TD(0), 48 hidden neurons, checkpointed against a fixed held-out evaluation seed set (matching the practice established in Section 6.2), 4,000 episodes:

| Condition | Best mean survival (ticks) |
|---|---|
| Random baseline | 24.1 |
| DFA-ablated (output-only credit, the original pre-Section-9 rule) | 36.8 |
| Full model (TD + DFA + homeostasis) | 45.1 |
| Memory-ablated | 57.3 (seed 0) / 35.7 (seed 1) — see 15.6 |
| Scripted heuristic | 90.7 |

Real learning occurred — survival roughly doubled from random initialization — and weight saturation stayed low throughout every condition tested (0.004–0.113), confirming the homeostasis/surgical-credit combination continues to prevent the lockstep collapse pathology on a genuinely new task, not just the two it was validated on. But no trained condition approached the scripted heuristic's performance within this budget.

### 15.6 Two ablations, two different outcomes

**The DFA ablation replicates Section 9's finding on a new task.** Removing DFA (reverting to the original, output-columns-only credit rule) reduced best performance from 45.1 to 36.8 — a real, if modest, cost, in the same direction Section 9 found on delayed-recall. This is a genuine positive replication: the same fix, motivated by a completely different task's diagnostic finding, transfers.

**The memory ablation gave a genuinely inconclusive, contradictory result, and we report it as such rather than resolving it by fiat.** At seed 0, the memory-ablated condition *outperformed* the full model (57.3 vs. 45.1) — the opposite of what the deliberate partial-observability design predicted. A second seed reversed the direction entirely (full model 47.2 vs. memory-ablated 35.7). With only two seeds and a training budget that produces this much run-to-run variance, this comparison cannot currently support a conclusion in either direction. We considered, and rejected, the temptation to run additional seeds until one direction "won" — that would be exactly the premature-conclusion mistake this paper's own methodology (Section 1.2) commits to avoiding.

Two candidate explanations, both consistent with earlier findings in this paper, remain live and undistinguished: (a) added recurrent complexity without a matched increase in training budget can measurably hurt rather than help, precisely the capacity/depth interaction found in Section 7.6; or (b) the specific sensing radius and enemy/food dynamics chosen may not press on memory as hard as the design intended — a scripted, fully reactive policy already achieves 90.7 ticks using only the current observation, which is at least suggestive that most consequential decisions in this particular environment can be made from current sensory input alone, echoing the tic-tac-toe result (Section 5.3) where full observability made memory provably irrelevant, except here observability is partial by design and memory's irrelevance (if real) would be a property of the *dynamics*, not the *observability*, making it a subtler and more concerning possibility. Resolving this would need substantially more seeds, more training budget, and likely a direct measurement of how often the scripted agent's chosen action would differ if it had access to recently-out-of-view information — none of which was completed here.

---

## 13. Experiment 10: Genuine Asynchronous Execution

### 16.1 Motivation

Every experiment in this paper, including Section 15's foraging task, has been logically lockstep underneath, regardless of how "continuous" the persistent state made it feel: one network decision per environment event, sequenced deterministically. The original XNN specification's ASNP premise (Section 1.1, Section 3.2) describes something stronger — a network that runs continuously, with the world reading and writing against whatever state it happens to hold, not a function that is called and waited on. This experiment tests that premise as literally as the environment allows: the network runs on a genuine, independent OS thread, unsynchronized with the environment except through shared state.

### 16.2 An honest engineering framing, decided before results

Python's GIL means this is **not** true multi-core simultaneous execution — only one thread executes Python bytecode at a time. A timing benchmark, run before building anything further, showed the network (48 hidden neurons) completing one raw dynamical sub-tick in ~43 microseconds and one full 3-substep decision in ~152 microseconds, against ~212 microseconds for one environment step — comparable orders of magnitude, not the "thousands of ticks per environment frame" loosely speculated about when this experiment was first proposed. At this scale, real thread-scheduling and lock overhead could plausibly be comparable to the computation itself.

What genuine OS threading *does* provide, honestly, is real scheduler-determined interleaving: no artificial barrier forces either side to wait for the other to complete a "turn," and the environment can read the network's output layer at any arbitrary instant, mid-computation or not — the literal operational version of "forced interruption of a still-forming decision" that no prior lockstep experiment in this paper could test. That property, not literal multi-core parallelism, is what this experiment actually measures.

### 16.3 A real race condition, caught before trusting any result

The network thread runs a perpetual loop: read whatever observation is currently in shared state (no waiting for a fresh one), advance the XNN by one raw sub-tick, write whatever is currently in the output layer to shared state, repeat. The environment thread reads whatever action preference is available, steps the world, and writes the new observation.

The first implementation had the environment thread directly zero the network's output neurons in its persistent state array at the moment of reading them (replicating the Clearing convention used everywhere else in this paper). This is unsafe: `XNN.tick()` performs several separate numpy operations, not one atomic step, and the GIL can switch threads between any of them — a cross-thread mutation of the same array the network thread is mid-computation on could corrupt an in-progress tick. This was fixed by giving the network thread exclusive ownership of its own state array: the environment thread only sets a flag requesting that output be cleared, which the network thread applies itself at the start of its own next iteration. We verified the fix directly (finite, properly bounded weights and biases across every trial tested) rather than assuming it was sufficient.

A second, purely semantic confound was found and fixed before comparing against lockstep results: the first version of the network thread never performed Clearing at all, meaning the comparison against lockstep evaluation would have conflated "genuinely asynchronous execution" with "a different, uncleared output-reading convention" — a real methodological trap given how much of this paper's DFA and TD machinery depends on Clearing behaving consistently. Fixing this brought average performance back in line with lockstep (see 16.4), confirming the earlier, larger gap had been substantially this confound, not a real effect of asynchrony.

### 16.4 Results

Using the trained full model from Section 15 at inference time only (no training under asynchronous execution was attempted — a harder problem, since it is unclear at what point a reward arising at an arbitrary asynchronous moment should be attributed to which internal network tick, deliberately deferred rather than conflated with this test):

- **Average performance is close to lockstep**: 43.9 mean survival ticks under genuine async execution (15 seeds) versus 43.3 under lockstep evaluation of the identical trained weights — confirming that, on average, asynchronous execution does not systematically help or hurt this task once the Clearing confound is removed.
- **But individual seeds show genuine, non-deterministic behavioral variance that lockstep execution cannot produce even in principle.** One seed produced [85, 85, 15] survival ticks across three otherwise-identical repeated runs; a further 8-trial repeat of that seed gave 85 seven times and 130 once. This was directly checked for data corruption rather than assumed benign: weights and biases remained finite and properly bounded (within [-1,1] and [0,1] respectively) in every trial, and the raw number of network ticks completed varied substantially (972 to 1,317) even across runs that reached the *same* behavioral outcome — confirming this is genuine, legitimately non-corrupted timing-dependent behavior, arising purely from real OS scheduling variance, not from any explicit stochasticity in the model or environment. This is the first result in this entire project where wall-clock timing itself, rather than the architecture or training rule, is a variable that measurably changes outcomes.
- **Pacing the environment (giving the network more relative deliberation time per decision — raising the network-ticks-per-environment-step ratio from ~15–21 to ~145–150) sharply reduced this variance**: the previously unstable seed produced [108, 108, 112, 108, 108] across five repeated runs. This mirrors Section 7.3's finding that more deliberation steps improves stability, but demonstrates it through an entirely different mechanism — real relative thinking time under genuine concurrency, rather than a fixed sub-step count parameter.

---

## 14. Discussion

**The "surgical" credit assignment that fixed tic-tac-toe's saturation had a cost tic-tac-toe could never have revealed.** Sections 5.9–5.10 fixed a real lockstep-collapse bug by restricting weight updates to only the column of the specific decision-relevant neuron. That fix was correct for tic-tac-toe. But the same restriction, applied to a task with genuine temporal structure, turned out to mean the training rule *never touches recurrent or encoding weights at all* — confirmed by direct measurement (Section 8.5: exactly zero change, to five decimal places, regardless of training budget or capacity). Every improvement that made tic-tac-toe reliable was, in retrospect, quietly building an architecture that could never have learned genuine working memory, and nothing about tic-tac-toe's structure could have exposed that, because tic-tac-toe never needed recurrent maintenance in the first place. This retroactively changes how Section 5.3's persistent-state null result should be read: it wasn't just that the *task* didn't need memory — the *training rule in use at the time* couldn't have exploited memory even if the task had needed it.

**Diffuse credit assignment causes emergent collapse, independent of the specific training rule used to apply it.** Every training method tested in Sections 5.1–5.8 — recency-weighted whole-game outcome credit, symmetry-augmented credit, harder-opponent credit, decayed credit, exact-oracle per-move credit — produced numerically frozen results, because all of them shared the same underlying flaw: updates that touched an entire fired neuron's outgoing connections, or an entire layer's bias, rather than the specific pathway relevant to a specific decision. The fix that actually mattered was not a better reward signal or a harder curriculum; it was making the *credit assignment itself* neuron-specific rather than diffuse.

**Memory and generalization are separable, and conflating them wastes debugging effort.** The persistent-state experiment (Section 5.3) is one of the cleanest results in this paper precisely because it is a *negative* result obtained by directly testing a specific hypothesis rather than assuming it. Real, continuously-evolving cross-move memory made no measurable difference to tic-tac-toe performance, because the task's bottleneck was never about memory — it was a spatial-generalization gap (fixed by symmetry augmentation, Section 5.4) layered on top of a genuinely missing behavioral concept (defense) that neither memory nor generalization could supply on their own.

**Structural impossibility looks identical to a training failure until you check.** Sections 4.3 and 4.4 both found hard 75% ceilings that no amount of tuning could cross, for reasons that had nothing to do with the training rule. Section 8.4 found a similar pattern for the memory task: decay rate, training budget (20×), and capacity (12×) were each systematically ruled out, one at a time, before the real cause (Section 8.5) was found. This is now a recurring, load-bearing methodological pattern in this paper: a persistent plateau should be treated as "cause unknown" until each plausible explanation has been independently tested and eliminated, not attributed to whichever explanation seems most likely first.

**A local, no-backprop mechanism can train genuine recurrent memory — the missing piece was never fundamental.** Section 9's DFA result is arguably the single most important finding in this paper: a fixed, uninformed random projection, with no weight transport and no resemblance to a true gradient, closed nearly the entire performance gap to a fully-supervised, gradient-checked backprop baseline (Section 8.6). This means the earlier failure (Section 8.5) was a property of one specific, narrow training rule, not a fundamental limitation of local learning or of the "no labelled data, fully recurrent" premise this whole project operates under.

**Not every local fix works, and the ones that fail are informative rather than just absent.** Node perturbation and the slowness/Hebbian rule both failed, but for different, explicable, non-mysterious reasons — variance that scales badly with simultaneous multi-unit perturbation in one case, an always-positive update with no restoring force reproducing the exact saturation pathology from Sections 5.6–5.9 in the other (Section 9.2). Cataloguing *why* a mechanism fails is as useful as showing that one succeeds, particularly for a document whose original proposal (Section 2.2, "heat scores") was vague enough to admit many possible concrete implementations — this paper narrows that space considerably by showing which specific choices reproduce known failure modes and which don't.

**Stability of the feedback pathway is protective specifically during change, which is the opposite of what "adaptive should help" intuition predicts.** Section 10's central result — an adaptive, warmed-up, properly decay-tuned feedback matrix performs *worse* than a naively frozen one when the task context genuinely changes, and worse even than training a fresh network from nothing given the same budget — is counterintuitive but mechanistically explicable: DFA's power comes from *W* having a stable target to rotate toward, and a context change is exactly the moment when the signals that would drive *B*'s adaptation (error, hidden activity) are noisiest and least informative. A frozen random *B* was never *about* the old context to begin with, so it has nothing to lose by staying fixed. This is a genuine, if narrow, insight about when adaptivity is a liability rather than an asset in a local-learning system — not a general argument against adaptive feedback alignment (Section 10.3's limitations apply), but a concrete demonstration that "let more things adapt" is not a safe default assumption.

**Relation to the original XNN proposal's own suggestions.** The self-play architecture's TD-error-broadcast-as-training-signal (Section 6) is close in spirit to the original specification's vague proposal (Section 2.2 of the source document) of "assigning activations passed through the network types... intended to mimic that of neurotransmitters" — TD-error-as-reward-prediction-error is, in fact, the dominant computational account of dopaminergic signaling in biological reward learning. Similarly, the specification's proposed "heat scores" (per-edge tracking of recent contribution) is a real precursor to the fired-neuron credit tracking used throughout this paper, though Section 5.9's and Section 8.5's findings together suggest the original proposal under-specified two critical details: *which* neurons' credit should be updated in response to a given outcome matters enormously (Section 5.9), and *whether credit reaches backward in time at all* matters just as much (Section 8.5) — "any recently active edge" is both too diffuse spatially and, without deliberate extension, entirely absent temporally.

**The oracle was a legitimate debugging tool even though it couldn't be the final training signal.** Section 5.8's minimax-oracle experiment produced the *same* frozen result as everything before it — which, at the time, looked like a null result. In retrospect it was exactly the control needed to isolate that the problem was in *how* credit was being applied, not in the *quality* of the signal being used to compute it. The same pattern recurs in Section 8.6: building a fully-supervised backprop baseline (which also can't be the final "no labelled data" answer) was exactly the right control for determining whether the memory task was intrinsically hard or hard-for-this-rule. We'd recommend this as a general debugging pattern: when a learned system is stuck under some constraint, temporarily lifting the constraint (even knowing the result can't be the final answer) is a fast way to determine whether the bottleneck is the constraint's target quantity or something else entirely.

**A trained hidden core carries something transferable, but pinning down what requires ruling out explanations one at a time — and one control can itself be invalid.** Section 11's transplant experiment found a real, repeatable benefit from a tic-tac-toe-trained core on a structurally unrelated task, and two plausible explanations (bias conditioning alone; any training helping equally regardless of content) were each directly ruled out by dedicated controls. A third, sharper hypothesis — that practice sustaining multi-tick dynamics specifically is what transfers — remains untested because its control task (running parity) turned out to be too hard to learn at this scale in the first place, an important reminder that a control experiment's own validity needs checking before its result can be trusted, the same discipline this paper has applied to primary results throughout (Sections 5.7, 7.4, 8.4, 10.2).

**An inconclusive result, reported as such, is more useful than a premature one.** Section 12.6's memory ablation on the foraging task gave contradictory results across two seeds — the opposite of what the environment's deliberate partial-observability design predicted, at one seed, and consistent with it at the other. The honest response was to report this as unresolved rather than run additional seeds until one direction "won," which would have reproduced, in miniature, exactly the mistake Sections 5.7 and 7.4 warn against. This also surfaced a genuine, unresolved tension worth stating plainly: a well-intentioned environment design (partial observability, meant to force memory to matter) does not guarantee memory actually matters in practice, if the specific dynamics chosen still let a reactive policy capture most of the achievable value — the scripted heuristic's strong performance using only current sensory input is at least suggestive of this, echoing (though not confirming) the tic-tac-toe result where full observability made memory provably irrelevant.

**Wall-clock timing itself can be a variable that changes behavior, once execution is genuinely concurrent rather than logically lockstep.** Every experiment before Section 13 was deterministic given a fixed seed, regardless of how "continuous" persistent state made it feel — one decision per environment event, sequenced. Genuine OS-thread concurrency broke that: the same seed, same trained weights, and same environment produced different outcomes across repeated runs, driven purely by real scheduling variance, verified not to be data corruption. This is a different *kind* of finding than anything else in this paper — not about the architecture or training rule, but about what genuinely asynchronous execution does to a system that every other experiment here treated as if it were, and had to be, effectively synchronous. That pacing (more relative deliberation time per decision) stabilizes this variance is a reassuring echo of Section 7.3's deliberation-depth finding, arrived at through a mechanism (real concurrent execution) that Section 7.3 could not have tested.

**Capacity and deliberation depth are both real levers, but not independent ones, and not always in the helpful direction.** Section 7 is, in some sense, a second instance of the same lesson as Section 5.7's saturation-hypothesis falsification: an initially plausible, uncontrolled comparison (Section 7.4) suggested one conclusion, and a properly matched follow-up (Section 7.5) showed the opposite. Scaling either dimension without the other, or without a correspondingly scaled training budget, is not guaranteed to help and can measurably hurt.

---

## 15. Limitations

- All experiments are on toy tasks; nothing here demonstrates the architecture scales to problems where local learning rules would need to compete with backpropagation-trained baselines on a comparable footing, even at the largest scale tested (627 neurons for tic-tac-toe).
- Self-play training remains non-stationary and non-monotonic (Section 6.2); checkpointing recovers a usable network but does not resolve the underlying instability, and no principled fix (e.g. a periodically-frozen opponent snapshot, or slower-moving target network for the critic) was tested.
- All hyperparameters (learning rates, homeostatic targets, deliberation step counts, DFA's hidden-layer learning rate, Kolen-Pollack's decay ratio and warmup length) were tuned by hand through trial and error, not derived from any principled analysis. Section 9.2 found that an under-tuned hyperparameter can look exactly like a genuine negative result (DFA at D=10 went from 52% to 100% from a single 3× learning-rate change) — this means negative results throughout this paper cannot fully rule out that a different hyperparameter setting would have worked, though the sweeps for mechanisms that did fail (Sections 8.4, 9.2) were reasonably thorough.
- No comparison was made against conventional baselines for tic-tac-toe (e.g. tabular Q-learning, which would trivially reach near-perfect play). The delayed-recall task (Section 8) does have such a baseline — the gradient-checked BPTT RNN — and that comparison was informative precisely because it existed; its absence for tic-tac-toe remains a real gap.
- Homeostatic bias regulation runs continuously, including during pure evaluation (it is not gated by a "training mode" flag), so repeated evaluation of a saved model causes slow, small ongoing drift in its biases.
- The value-neuron subset (4 of 24 hidden neurons in the baseline tic-tac-toe architecture, scaled proportionally in Section 7) was chosen arbitrarily; no ablation was run on this count or on which specific neurons are designated.
- **Every result in Section 7 is a single seed per configuration.** The capacity-scaling trend (Section 7.2) spans a large enough effect, consistently in one direction across two independent size increases, that we believe it is real; the deliberation-depth results (Sections 7.3, 7.5) show large enough effect sizes (roughly halving or doubling loss rates) that we believe their *direction* is real, but none of these numbers should be treated as precise.
- **Sections 8–11 (the memory task, the three local fixes, adaptive feedback alignment, and cross-task transfer) mostly use 2–3 seeds per configuration**, on a single small architecture (24 hidden neurons, except the explicit capacity sweep in Section 8.4). The *qualitative* conclusions are well-supported by effect size — DFA's jump from chance to 95–100%, adaptive B's consistent underperformance across all 3 seeds, and the transplant's clean convergence versus every control's stuck seeds are all far too large to be noise — but exact percentages throughout these sections should be treated as indicative, not precise.
- The RNN baseline's own degradation at D≥40 (Section 8.6) means the ceiling of what's achievable on this exact task design is itself bounded by vanishing gradients in a plain Elman RNN; DFA was only tested up to D=20 and was not pushed further to find its own ceiling.
- Only one node-perturbation formulation and one slowness/Hebbian formulation were tested each (Section 9); both represent large families of possible local rules, and the negative results in this paper apply to the specific formulations tested, not to the families in general.
- The context-switch experiment (Section 10.3) used one specific kind of context change (an input-channel swap) and one specific retraining budget (3,000 trials); whether adaptive B's disadvantage holds across other kinds of context change, other budget sizes, or other Hebbian co-adaptation formulations is untested.
- **Section 11's transfer experiment used 3 seeds per condition and one specific pair of tasks** (tic-tac-toe → delayed-recall). The core finding (tic-tac-toe transplant reliably reaches ceiling; two controls do not) is well-supported by effect size — no random-init or homeostasis-only seed avoided getting stuck, while no transplant seed ever did — but the running-parity control's failure to learn its own source task (45–52%, chance) means the "does multi-tick practice transfer" question remains genuinely open, not answered either way, and would need a control task that is both multi-tick and reliably learnable at this scale to resolve.
- Training budgets differ slightly across some comparisons in Section 7.2 (20,000 vs. 13,000 games) for practical wall-clock reasons; Section 7.5's matched-budget comparison is the only fully controlled capacity/depth comparison in Section 7, and we would not have trusted the unmatched version in Section 7.4 as reported if we had not gone back and fixed it.

---

## 16. Conclusion

A literal implementation of the originally proposed XNN training convention fails outright. Every subsequent fix in this investigation was motivated by a specific, diagnosed failure — structural incapacity, a fixed point, diffuse credit assignment, missing memory-independent generalization, missing state-specific value estimation, and a credit-assignment rule that never trained recurrent weights at all — rather than general-purpose tuning, and each diagnosis was confirmed by a targeted experiment (exhaustive/semi-exhaustive search, direct state inspection, ablation-style controls, matched-budget comparisons, gradient-checked baselines) rather than inferred from aggregate metrics alone.

The tic-tac-toe architecture — persistent ASNP state, signed weights and bias as ATNP, self-play with a shared actor-critic substrate trained via TD(0), surgical local credit assignment, and continuous homeostatic regulation — satisfies the architecture's original constraints while matching the performance of an earlier, labelled-data-dependent version of the same network. But the delayed-recall task (Section 8) revealed that this same "surgical" credit assignment, which fixed tic-tac-toe's saturation collapse, had a hidden cost tic-tac-toe's structure could never have exposed: it left recurrent and encoding weights completely untrained, measured directly as exactly zero change regardless of training budget or capacity. A conventional backprop baseline confirmed the task itself was never the obstacle. Of three theoretically-motivated local fixes tested, one — Direct Feedback Alignment, a fixed random feedback pathway requiring no weight transport — closed nearly the entire gap to that fully-supervised baseline, while the other two failed for clear, explicable reasons rather than mysteriously. Attempting to make that feedback pathway itself adaptive, so it could track a genuinely changing task context, produced the most counterintuitive result in the paper: adaptivity made recovery from a context change *worse* than either staying frozen or starting from nothing, because the very stability that seems like a limitation of a fixed random projection turns out to be exactly what makes it robust to change.

A final experiment asked a question none of the above could answer: does a trained hidden core carry anything transferable across genuinely different tasks? A tic-tac-toe-trained core reliably improved delayed-recall learning — every transplant seed converged cleanly, while every control condition (bias-only conditioning, a content-unrelated but single-tick source task) had at least one seed that never escaped a bad initialization. Two specific explanations for that benefit were ruled out directly; a third, sharper hypothesis was targeted by a follow-up control that itself failed to learn its source task, leaving the precise mechanism of transfer an open question rather than a resolved one.

The final two experiments deliberately stepped back from clean, well-defined tasks toward something closer to the architecture's stated motivation: an embodied, partially-observable survival environment, and genuine real-time concurrent execution. The foraging task confirmed the same TD/DFA/homeostasis synthesis learns a substantially richer, multi-objective task — real learning occurred, and the DFA-ablation finding replicated independently on new task structure — but the memory-ablation result, the one comparison most directly testing this section's own motivating premise, came back genuinely contradictory across seeds, and we report that as unresolved rather than resolve it by running more seeds until one direction won. Genuine OS-thread concurrency, tested as literally as the environment allows, produced the first result in this entire project where wall-clock timing itself — not the architecture, not the training rule — measurably changes behavior: the same seed, same trained weights, and same environment produced different outcomes across repeated runs, verified to be real scheduling variance rather than data corruption, and stabilized by giving the network more relative deliberation time per decision.

Scaling capacity improves tic-tac-toe performance consistently and cheaply; scaling deliberation depth improves performance substantially, but only when the training budget is matched to the larger effective dynamical space that depth and capacity jointly create — the two are not independent knobs, and treating them as such produced one of this paper's most misleading intermediate results (Section 7.4) before a more careful comparison corrected it. That same pattern — an uncontrolled comparison suggesting one conclusion, a properly matched follow-up showing another, or a plausible-looking result reported as unresolved rather than forced to a conclusion — recurs throughout this paper (Sections 5.7, 7.4, 8.4, 10.2, 11.4, 12.6) and is, if anything, the closest thing to a unifying finding here: almost every result in this investigation needed to be independently stress-tested before it could be trusted, and several of the most important conclusions (DFA works, on two independent tasks; adaptive B hurts; the memory task was never intrinsically hard; something beyond mere training transfers across tasks; genuine concurrency produces genuine non-determinism) are the ones that survived that process, not the ones that looked right first — while at least one (does persistent memory help under partial observability) has not yet been resolved either way, and is reported as such.

Whether any of these specific recipes — the tic-tac-toe architecture, DFA-trained recurrent memory, the capacity/depth and cross-task transfer effects, or the embodied and asynchronous mechanisms explored last — generalize past small toy tasks remains untested and is the natural next question.

---

## Code and Data Availability

**Tic-tac-toe (Sections 4–7):**
- `xnn_model.py` — task-agnostic XNN model: architecture, dynamics, local training rules, JSON serialization.
- `xnn_tictactoe.py` — tic-tac-toe application layer: game logic, self-play training loop, evaluation harness, CLI entry point.
- `resumable_train.py` — checkpointed, resumable self-play training, used to obtain the matched-budget comparison in Section 7.5.
- `model.json` / `model_history.json` — baseline network (24 hidden neurons, 3 deliberation steps) and its training history (Section 6.2).
- `model_medium150.json`, `model_big600.json` — capacity-scaling networks, 3 deliberation steps (Section 7.2).
- `model_small_steps6.json`, `model_small_steps10.json`, `model_small_steps20.json` — deliberation-depth scaling, 24 hidden neurons, matched 20,000-game budget (Section 7.3).
- `model_large_steps10_matched.json` — 600-hidden-neuron network, 10 deliberation steps, matched 13,000-game budget via `resumable_train.py` (Section 7.5).

**Delayed-recall memory task (Sections 8–10):**
- `xnn_delayed_recall.py` — task generator (cue/delay/probe trial structure, context-swap variant), full-memory and memory-ablated XNN conditions, trial-wide eligibility-trace training.
- `rnn_baseline.py` — hand-derived, gradient-checked BPTT Elman RNN baseline on the identical task generator (Section 8.6).
- `fix_attempts.py` — the three candidate local mechanisms for training hidden-layer weights (DFA, node perturbation, slowness/Hebbian; Section 9) and the Kolen-Pollack adaptive-B mechanism plus context-swap experiment (Section 10).
- `dfa_model_D10.json`, `dfa_feedback_matrix_D10.json` — a representative trained DFA network and its fixed feedback matrix, D=10, 100% eval accuracy (Section 9.2).
- `rnn_baseline_D20.json` — the RNN baseline's trained weights at D=20, near its own performance ceiling (Section 8.6).

**Cross-task transfer (Section 11):**
- `transfer_experiment.py` — extracts a trained tic-tac-toe hidden core and splices it into a fresh delayed-recall network; includes the homeostasis-only control (bias regulated under generic noise, no training), the single-tick classification control (content-unrelated, no temporal structure), and the running-parity control (multi-tick, content-unrelated, but found to be an invalid control since it failed to learn its own source task) — all compared against plain random initialization on identical learning-curve checkpoints.

**Embodied foraging and genuine asynchronous execution (Sections 12–13):**
- `foraging_env.py` — the 20×20 grid environment: directional sensing, energy economy, pursuing/wandering enemies, respawning food.
- `scripted_agent.py` — the flee/seek heuristic baseline (Section 12.3), decided and run before any trained-agent result.
- `foraging_agent.py` — the XNN agent: TD(0) actor-critic with value neurons, DFA-based hidden-layer credit assignment (as a vector, matching the scalar TD error), reward scaling, and `memory_ablate` / `dfa_ablate` flags for the two ablation conditions (Section 12.6).
- `trained_full.pkl`, `trained_memablate.pkl`, `trained_dfaablate.pkl` — best-checkpoint trained weights for the three conditions reported in Section 12.5.
- `async_runner.py` — the genuine OS-thread concurrency harness (Section 13): the network runs a perpetual loop on its own thread, reading/writing shared state with no synchronization barrier beyond a lock and a single-writer state-ownership protocol (Section 13.3); includes the paced/unpaced comparison and the repeated-run non-determinism check.

To reproduce the tic-tac-toe baseline: `python3 xnn_tictactoe.py [games] [seed] [output_path]` (defaults: 20000 games, seed 0, `model.json`). To reproduce the scaling experiments in Section 7, construct an `XNNConfig` with the desired `n_hidden`, `n_value`, and `steps_per_decision`, and use `play_self_play_game` directly (see `resumable_train.py` for a worked multi-session example). To reproduce the memory-task experiments, `xnn_delayed_recall.train_and_eval` (original rule), `fix_attempts.run_trial_dfa` / `run_trial_perturb` / `run_trial_slowness` (Section 9), and `run_trial_dfa` with `adapt_B=True` plus `context='B'` (Section 10) are the relevant entry points; `rnn_baseline.gradient_check()` should be run and confirmed passing before trusting any RNN baseline result on a modified version of that code. To reproduce the transfer experiment, run `transfer_experiment.py` directly (requires `model.json` from the tic-tac-toe baseline to be present). To reproduce the foraging experiments, run `scripted_agent.py` for the baseline and use `foraging_agent.run_episode` with `memory_ablate`/`dfa_ablate` flags for training and ablation; to reproduce the async experiments, run `async_runner.py` (requires `trained_full.pkl`) and compare against `foraging_agent.run_episode` on the identical weights for the lockstep reference.