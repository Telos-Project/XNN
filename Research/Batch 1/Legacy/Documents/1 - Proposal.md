# XNN Architecture & the X-CAT Training Methodology

**A self-contained specification**

*Cross Neural Networks (XNNs), the two-timescale training program (X-CAT), and concrete implementations for the logic-gate and tic-tac-toe benchmarks.*

---

## 0. Reading guide

This document is self-contained. It defines the architecture from first principles (Part I), then the training methodology (Part II), then two fully-specified worked implementations (Part III). Notation is collected in Appendix A; default parameters in Appendix B; honest risks in Appendix C.

The one idea that organizes everything: **an XNN is a process, not a function.** It is a perpetually-running dynamical system you perturb (write to some nodes) and observe (read from some nodes), not a mapping you invoke. Every design choice below follows from refusing to give that up.

---

# PART I — ARCHITECTURE

## 1. The substrate

An XNN is a directed graph of `N` nodes ("neurons") in which **every node may connect to every node, including itself**, and connections are **one-way**, so the link `i → j` may carry properties different from `j → i`.

- **Connectivity** is a weight matrix `W ∈ ℝ^{N×N}`, where `W[j][i]` is the weight from source `j` to target `i` (row = source, column = target). Absence of a connection is simply `W[j][i] = 0`.
- **Node state** is a vector `s ∈ ℝ^N`. `s[i]` is the state ("membrane potential") of neuron `i`.

Because absent edges are zero-weights, *any* graph is a subset of the fully-connected graph, and any conceivable neural network is some graph. Therefore any architecture is **representationally reachable** inside an XNN. This is a statement about the weight space, not about trainability — see §7 and Part II, where *findability*, not reachability, is the entire problem.

## 2. Time: steps and asynchronous processing (ASNP)

The network runs in discrete **steps**. A step transfers state across connections and recomputes each node's state for the next step. Crucially, there is **no input layer, no output layer, and no forward pass**:

- **Input** = externally writing the states of chosen nodes, at any step.
- **Output** = externally reading the states of chosen nodes, at any step.
- Input and output happen *while the network keeps running*, possibly across many concurrent contexts.

This continuous read/write-during-operation regime is **Asynchronous Neural Processing (ASNP)**. Any node not currently used for I/O is a **deliberation neuron** — internal capacity whose only job is to sustain and shape ongoing dynamics.

## 3. Neuron dynamics (one step)

The base neuron is a **leaky integrate-and-fire (LIF) spiking unit** with trainable parameters. At each step `t`, for every neuron `i`, in order:

1. **Inject** bias and any external/periodic input:
   `s[i] ← s[i] + b[i] + I_i(t)`
2. **Integrate** spikes that arrived from the previous step (only neurons that *fired* push):
   `s[i] ← s[i] + Σ_j ( z_j(t−1) · W[j][i] )`
3. **Fire** if state crosses threshold and lies within the active band:
   `z_i(t) = 1  if  θ_min ≤ s[i]  and  s[i] ≥ θ_i  and  s[i] ≤ θ_max ;  else 0`
4. **Reset on fire** via the transfer-relay ratio `ρ_i ∈ [0,1]` (fraction of state *retained* after firing):
   `if z_i(t) = 1:  s[i] ← ρ_i · s[i]`
5. **Relax** sub-threshold state by exponential decay (`λ_i ∈ (0,1)`):
   `if z_i(t) = 0:  s[i] ← λ_i · s[i]`

> **Design note — relaxation.** The original convention squared the state for relaxation (`0.5 → 0.25 → 0.06` in three steps), which extinguishes activity almost immediately. **Use exponential decay with a tunable time constant instead.** `λ_i` is a per-neuron trainable parameter; near-critical dynamics require `λ` close to 1.

> **Design note — signed weights.** Weights are bounded to `[−1, 1]`, **not** `[0, 1]`. A purely excitatory network cannot compute; inhibition is non-negotiable. A neuron whose outgoing weights are predominantly negative is **inhibitory**; the excitatory/inhibitory (E/I) ratio is a controlled quantity (§6).

## 4. State as vector, weights as matrix — vectorized step

The entire network is two arrays (`s`, `W`) plus per-neuron parameter vectors (`b, θ, ρ, λ`). One step, vectorized:

```text
s        ← s + b + I(t)                  # inject
fired    ← z(t-1)                         # spike vector from previous step (0/1)
s        ← s + (fired @ W)                # integrate (row-vector times matrix)
z(t)     ← (s >= θ) & (s >= θ_min) & (s <= θ_max)   # fire
s        ← where(z(t), ρ * s, λ * s)      # reset-on-fire OR relax
```

## 5. Atypical Neural Parameters (ATNP)

ATNP are anything beyond scalar weights/biases. The ones this spec uses:

| ATNP | Symbol | Role |
|---|---|---|
| Bias | `b[i]` | added to state each step; trainable |
| Threshold | `θ[i]` | firing threshold; adjusted by intrinsic plasticity (§6) |
| Active band | `θ_min, θ_max` | state range over which a neuron may fire |
| Transfer-relay ratio | `ρ[i]` | state retained after firing |
| Decay | `λ[i]` | sub-threshold relaxation rate |
| **Eligibility trace** | `e[j][i]` | decaying memory of recent pre×post coincidence — the substrate of credit assignment (§II) |
| Heat | `heat[j][i]` | running magnitude of traffic on an edge; diagnostic / optional reward routing |

> **Design note — eligibility vs. the old "associative score."** The original associative score was an **integer count with no temporal decay**: a synapse active 100 steps ago received identical credit to one active last step, destroying all timing information. Replace it with a **decaying eligibility trace** (§9). This single change is the difference between "credit cannot be assigned correctly" and "credit assignment is a known-good form."

## 6. Homeostasis and criticality (why self-organization is possible at all)

A generic random recurrent net is almost never in a regime where useful structure can form — it saturates or dies. Productive self-organization is a **tuned** property, not a free one. The network therefore maintains homeostatic controllers, run on a slow cadence (every `H` steps):

- **Intrinsic plasticity** (target firing rate `ν*`): `θ[i] ← θ[i] + κ_θ · (rate_i − ν*)` where `rate_i` is `i`'s recent firing frequency.
- **Synaptic scaling** (target incoming drive `g*`): renormalize each column so `Σ_j |W[j][i]| = g*`.
- **Branching ratio toward criticality** (`σ* ≈ 1`): monitor `σ = (#spikes at t) / (#spikes at t−1)`, nudge global gain to hold `σ` near 1 (the edge between dying and runaway activity — the regime where reverberation can be sustained *and* settle).

The set-points `ν*, g*, σ*` are **not hand-set**; they are evolved (Part II). This is the concrete cash-value of "productive self-organization is tuned, not free."

## 7. Reachability vs. findability (the crux)

The §1 universality argument guarantees the target connectome is *reachable* in `W`-space. It says **nothing** about whether a learning process can *walk there* from a blank, homogeneous start under local rules and sparse reward. Findability is the whole problem, and it is why a homogeneous fully-connected substrate — the *least* structured object, where biology's brain is among the *most* structured — cannot simply be trained directly. Part II is the answer.

---

# PART II — TRAINING: THE X-CAT METHODOLOGY

## 8. The core principle: two timescales

A brain is never a homogeneous substrate trained from scratch by reward. It is a **pre-structured** substrate whose initial wiring and whose learning rules are themselves the frozen output of an earlier, slower optimization (evolution + development). Mind lives on two timescales:

- **Slow / outer loop** — finds the coarse scaffold, the plasticity rule itself, and the homeostatic set-points. (Biology's evolution.)
- **Fast / inner loop** — learns within that scaffold, using that rule, driven by experience. (Biology's lifetime.)

Trying to do both at once on one timescale is the maximally hard version of the findability problem. **X-CAT refuses the dichotomy by separating the timescales.**

```text
        ┌─────────────────────────  OUTER LOOP (evolution)  ──────────────────────────┐
        │  genome = ( generator G,  plasticity rule θ,  homeostat set-points φ )        │
        │                                                                              │
        │   for each candidate genome:                                                 │
        │       W0, types ← G()              # grow a coarse scaffold (indirect encoding)│
        │       ┌────────────  INNER LOOP (a "lifetime", local + online)  ───────────┐ │
        │       │  W ← W0                                                            │ │
        │       │  repeat for the lifetime:                                          │ │
        │       │     run ASNP steps                                                 │ │
        │       │     self-prediction error → dense learning signal  (primary)       │ │
        │       │     sparse task reward    → neuromodulatory signal (shaping)        │ │
        │       │     ΔW via local 3-factor rule f_θ( eligibility, signal )           │ │
        │       │     homeostasis every H steps using set-points φ                    │ │
        │       └────────────────────────────────────────────────────────────────────┘ │
        │       fitness ← lifetime performance (prediction acc. + task reward)          │
        │   evolve ( G, θ, φ )  via CMA-ES / NEAT                                        │
        └──────────────────────────────────────────────────────────────────────────────┘
```

## 9. The inner loop

### 9.1 Eligibility trace (per synapse `j → i`)

```text
e[j][i] ← γ · e[j][i] + z_j(t-1) · h_i(t)
```
- `γ ∈ (0,1)` — trace decay (the timing memory the integer count lacked).
- `h_i(t)` — post-synaptic factor. With hard spikes, use a **surrogate** (pseudo-derivative of the threshold, e.g. `h_i = max(0, 1 − |s[i] − θ_i|)`), so credit flows smoothly even though firing is discontinuous.

### 9.2 The learning signal `M`

Two sources, combined:

```text
M_pred(t) = g( prediction_error(t) )     # DENSE — available every step, no trials needed
M_rl(t)   = reward / reward-prediction-error # SPARSE — task outcome
M(t)      = α · M_pred(t) + β · M_rl(t)
```

> **Why self-prediction is primary.** RL wants episodes, returns, and credit windows; an always-running ASNP substrate has no trial boundaries. **Self-prediction does not** — predicting your own next inputs/states yields a dense error signal *every step*, with no episode structure required. So the bulk of learning is continuous and self-supervised; sparse reward only *shapes* value on top. This turns the always-on property from an obstacle into the asset that drives most of the learning. (This is the predictive-coding / active-inference move, and it is faithful to the project's own "predictive training" note.)

### 9.3 The weight update (the evolvable three-factor rule)

General evolved form (`f_θ` = a small parameterized function — low-order polynomial or tiny MLP, with coefficients in the genome):

```text
ΔW[j][i] = f_θ( e[j][i], M, z_j, z_i, W[j][i], heat[j][i] )
```

Standard instantiation the search is seeded around:

```text
ΔW[j][i] = η · M(t) · e[j][i]  −  μ · W[j][i]      # reward-modulated trace, with weight decay
W[j][i]  ← clip( W[j][i] + ΔW[j][i],  −1,  +1 )
```

Biases train by an analogous three-factor rule:
```text
Δb[i] = η_b · M(t) · ē_i        # ē_i = decaying trace of i's own firing
```

### 9.4 Reduction to the original Convention 1.7 (sanity anchor)

The legacy rule was `Δw = fs · (ae/an) · c`, then renormalize, reset score. It is the **special case** of §9.3 where:

| Legacy term | X-CAT generalization |
|---|---|
| `fs` (one global scalar) | `M(t)` — a per-context, predominantly *self-predictive* signal |
| `ae/an` (non-decaying integer count, per-neuron normalized) | `e[j][i]` — a **decaying** eligibility trace |
| `c` (fixed) | `η`, plus an evolved functional form `f_θ` |
| renormalize + reset score | weight decay `μ` + synaptic scaling (§6) |

So X-CAT is not a different rule; it is the legacy rule with (a) timing restored, (b) a dense continuous signal added, and (c) the constants promoted to evolvable parameters.

## 10. The outer loop

The genome encodes three things, all **size-agnostic** (this is what makes the program affordable — see §11):

1. **Generator `G` (indirect encoding).** A compact rule that *grows* the initial connectivity and neuron types, rather than a literal weight matrix. Use an HyperNEAT-style CPPN (queried with neuron coordinates to emit `W0[j][i]`) or a developmental/morphogen-gradient process. This hands the inner loop a **coarse scaffold** — the half-built building — while leaving fine structure to emerge through plasticity. (Structure is *relocated* from innate to emergent without asking plasticity to discover the coarse scaffold from uniform conditions.)
2. **Plasticity rule parameters `θ`** — the coefficients of `f_θ` (§9.3), seeded inside the three-factor / eligibility family so evolution tunes a known-good form rather than rediscovering calculus.
3. **Homeostat set-points `φ`** = `(ν*, g*, σ*, …)` (§6).

**Fitness** = inner-loop lifetime performance, averaged over seeds and tasks: prediction accuracy + task reward, minus a parsimony term. **Optimizer**: CMA-ES over the continuous parameters; NEAT for generator topology if it is itself evolved.

## 11. The feasibility linchpin: evolve small, deploy large

A CPPN generates connectivity for **any** network size; a per-synapse local rule applies regardless of synapse count; homeostat set-points are scale-free. Therefore:

> **Evolve `G, θ, φ` on tiny networks and cheap tasks (logic gates, small boards) over thousands of short lifetimes — then *deploy* the identical genome on a large substrate without re-evolving.** Evolution operates on the compact, heritable thing; scale is a deployment parameter.

This is the difference between "computationally hopeless" and "a cluster job." It is also the program's biggest *bet* — see Appendix C.

## 12. Deliberation is selected for, not coded

Initiation, hesitation, meditation are not modules. They are what an always-running system does when it **sustains reverberant activity near criticality and defers commitment**. You obtain them by combining:

- a substrate the homeostats hold **near `σ* ≈ 1`** (so activity can reverberate without exploding or dying), and
- **tasks that reward deferred, integrated commitment** (the counter-threshold / deliberation-cycling output of §15 is mechanized hesitation: the network must *sustain* a choice long enough to cross a threshold before it commits).

Deliberation is therefore a **dynamical regime to be tuned into**, measurable as sustained metastable activity in the deliberation neurons preceding commitment.

## 13. Measurement discipline (how not to fool yourself)

Three permanent controls. A result that does not clear them has taught you nothing about the architecture.

- **Frozen-reservoir baseline.** Random fixed `W`, train **only** a linear readout. Every claimed benefit of training the recurrent substrate must beat this. Run it *first*, always.
- **Component ablations.** Evolved scaffold vs. random init; evolved rule vs. hand-set e-prop-style rule; homeostat on vs. off. Each piece must earn its compute.
- **Minimal-structure frontier (the headline result).** The curve of *how little* innate structure must be evolved before the inner loop starts learning, and which component buys the most. This is publishable even if ACI never appears.

---

# PART III — WORKED IMPLEMENTATIONS

## 14. Implementation A — Logic gates

### 14.1 Purpose and phases

Logic gates are the smallest test that the *whole pipeline* (substrate, ASNP I/O, learning signal, readout) works. Run three phases in order:

- **Phase A — capacity proof (no learning).** Hard-code known-good weight matrices and verify the substrate computes the gate. Proves representational capacity exists before asking training to find it.
- **Phase B — frozen-reservoir control.** Random `W`, train only a linear readout. Confirms the task is learnable *from* the substrate.
- **Phase C — full inner-loop learning.** Train `W` with the local three-factor rule (§9). Optionally wrap in the outer loop, though for a 2-input gate the rule can be hand-set.

### 14.2 Phase A — manually constructed gates

Row = source, column = target. These are the project's own constructions, restated.

**AND** (inputs `0,1`; output `2`):
```text
W =  0   0   .5
     0   0   .5
     0   0    0
```
Both inputs at 1 drive neuron 2 to `0.5 + 0.5 = 1.0` → normalizes to 1; any other input → < 0.5 → 0.

**NOT / transistor analog** (input `0`; output `2`; bias on neuron 1):
```text
W =  0   0  -1
     0   0   1
     0   0   0
b =  0   1   0
```
Neuron 1's bias supplies a constant 1 to neuron 2; input 1 cancels it via the `−1` edge.

**NAND** (inputs `0,1`; output `4`; bias on neuron 3):
```text
W =  0   0   .5   0    0
     0   0   .5   0    0
     0   0    0   0   -1
     0   0    0   0    1
     0   0    0   0    0
b =  0   0    0   1    0
```
Neurons `0,1,2` form the AND sub-circuit; neurons `3,4` invert it. AND and NAND are each functionally complete, so success here implies general logic capacity.

### 14.3 Phase C — learning setup

| Item | Value |
|---|---|
| Substrate size `N` | 16 |
| Input neurons | `{0, 1}` (binary input convention: state set to 0 or 1) |
| Output neuron | `{N−1}` (per ASNP I/O placement: outputs at the tail of the vector) |
| Deliberation neurons | the other 13 |
| Input persistence | held constant across the whole output cycle (multi-step input) |
| Output read | **mean state** of the output neuron over an output cycle, then rounded at 0.5 |
| Feedback `M_rl` | `+1` if rounded output matches the AND truth table for the presented inputs, else `−1` |
| Self-prediction `M_pred` | predict next input pattern; dense per-step error (small weight `α`) |
| Trace decay `γ` | 0.8 |
| Weight decay `μ` | 1e-3 |
| Learning rate `η` | 1e-2 |
| Relaxation `λ` | 0.9 (init) |
| Threshold `θ` | 1.0 (init; intrinsic plasticity active) |
| Output cycle | 32 steps; clear output neuron to 0 after reading |

### 14.4 Pseudocode (Phase C)

```python
def step(s, W, b, theta, lam, rho, fired_prev):
    s = s + b + I_external          # inject (I_external nonzero only on input neurons)
    s = s + fired_prev @ W          # integrate spikes from previous step
    fired = (s >= theta) & (s >= THETA_MIN) & (s <= THETA_MAX)
    s = np.where(fired, rho * s, lam * s)   # reset-on-fire OR relax
    return s, fired.astype(float)

def train_gate(truth_table, episodes):
    W = small_random(N, N); b = zeros(N); theta = ones(N)
    e = zeros(N, N)                  # eligibility traces
    for ep in range(episodes):
        a, bbit = sample_inputs()    # one row of the truth table
        target = truth_table[a, bbit]
        s = zeros(N); fired = zeros(N)
        out_accum = 0.0
        for t in range(OUTPUT_CYCLE):
            set_input(s, [a, bbit])          # persist input across the cycle
            s, fired = step(s, W, b, theta, LAM, RHO, fired)
            h = surrogate(s, theta)          # post-synaptic factor
            e = GAMMA * e + np.outer(fired, h)   # update traces (pre x post)
            out_accum += s[N-1]
            s[N-1] = 0.0                     # output-priority clearing
        out = 1 if (out_accum / OUTPUT_CYCLE) > 0.5 else 0
        M = (+1 if out == target else -1)    # M_rl  (+ small M_pred term in full version)
        W += ETA * M * e - MU * W            # three-factor update
        W = np.clip(W, -1, 1)
        e[:] = 0                             # reset traces after feedback
        homeostasis(theta, W)                # every-H-steps in the async version
    return W, b, theta
```

> **Expected outcome.** Phase A: exact gate behavior. Phase B: readout reaches ~100% (confirms learnability from the substrate). Phase C: `W` converges to a sub-graph functionally equivalent to the Phase-A matrix. If Phase C fails while B succeeds, the **rule** is at fault, not the substrate — exactly the diagnostic separation the controls are for.

## 15. Implementation B — Tic-tac-toe

### 15.1 Setup

| Item | Value |
|---|---|
| Substrate size `N` | 128 (default power of two) |
| Board encoding | 18-bit: each square → 2 bits (`00` empty, `01` X, `10` O), concatenated left-to-right, top-to-bottom |
| Input neurons | `{0 … 17}` (board), states set to the 18 bits, **held across the deliberation+output cycle** |
| Output neurons | `{N−4 … N−1}` — 4 neurons, since the move index (0–8 over empty squares) needs `⌈log2 9⌉ = 4` bits |
| Deliberation neurons | the remaining 106 |
| Opponent | random mover by default; pluggable expert system |
| Network plays | X by default |
| Reward `M_rl` | `+1` win, `−1` loss, `0` draw — delivered at game end |
| Self-prediction `M_pred` | predict the **next board state** after own + opponent move; dense per-step error (primary signal) |

### 15.2 Move selection — counter cycle (default) and deliberation cycling

**Move legality.** Empty squares are listed in board order; the network's chosen move is an index into this list. Decoded indices `≥ len(empty)` are floored to the last legal move (and masked).

**Counter cycle (default).** Over a window of steps, each step decodes the 4 output neurons into a move index and increments that move's counter. The **first move whose counter crosses the count threshold (128)** is committed; counters then reset. This is mechanized deliberation: a move must be *sustained* to win.

**Deliberation cycling (optional).** Give the network a **512-step deliberation cycle** (inputs present, outputs ignored — pure internal reverberation) **before** a **128-step output cycle** (counters active). Periods of "thinking" alternate with periods of "committing." Match length: **128 games.**

### 15.3 Output decoding

```python
def decode_move(out_neurons, empty_squares):
    bits = (normalize(out_neurons) >= 0.5).astype(int)   # 4 bits, MSB first
    idx  = bits_to_int(bits)                              # 0..15
    idx  = min(idx, len(empty_squares) - 1)               # floor illegal to last legal
    return empty_squares[idx]
```

### 15.4 Game / training loop (inner loop)

```python
def play_and_train(W, b, theta, rule_params, n_games=128):
    e = zeros(N, N)
    for g in range(n_games):
        board = empty_board(); s = zeros(N); fired = zeros(N)
        history = []
        while not terminal(board):
            set_input(s, encode_board(board))      # 18 bits, persisted
            counts = zeros(9)
            # optional: 512 deliberation steps here with outputs ignored
            for t in range(OUTPUT_CYCLE):           # 128 steps
                s, fired = step(s, W, b, theta, LAM, RHO, fired)
                h = surrogate(s, theta)
                e = GAMMA * e + outer(fired, h)
                # DENSE self-prediction signal every step:
                pred_err = board_prediction_error(s, board)
                M_pred = g_fn(pred_err)
                W += ETA * (ALPHA * M_pred) * e - MU * W      # continuous shaping
                move_idx = decode_move(s[-4:], empty(board))
                counts[move_idx] += 1
                if counts[move_idx] >= COUNT_THRESHOLD:        # 128
                    break
            move = argmax(counts)
            board = apply(board, move, X); history.append(snapshot(s, e))
            if not terminal(board):
                board = apply(board, random_move(board), O)
        R = (+1 if win(board, X) else -1 if win(board, O) else 0)
        # SPARSE reward applied to the traces accumulated during the game:
        for snap in history:
            W += ETA * (BETA * R) * snap.e - MU * W
        W = clip(W, -1, 1)
        e[:] = 0
        homeostasis(theta, W)                                  # hold criticality
    return W, b, theta
```

### 15.5 Outer-loop wrapping

Evolve `(G, θ, φ)` on the **logic-gate tasks and 3×3 boards** (cheap, thousands of lifetimes), then **deploy the identical genome at `N = 128`** for full tic-tac-toe (§11). Fitness = mean over games of `(self-prediction accuracy + win rate)` minus a parsimony term, averaged over several random seeds and opponent settings.

### 15.6 Controls for this benchmark

- **Frozen reservoir + random opponent**: random `W`, linear readout on the 4 output neurons → establishes the score a *non-learning* substrate reaches against a random mover (a strong sanity floor, since random-vs-random is ~58% for the first player).
- **Ablation**: counter cycle vs. deliberation cycling — does enforced "thinking time" measurably improve play? This is the first direct, quantitative test of the deliberation thesis on a real task.

---

# Appendix A — Notation

| Symbol | Meaning |
|---|---|
| `N` | number of neurons |
| `W[j][i]` | weight, source `j` → target `i` (row=source, col=target), in `[−1,1]` |
| `s[i]` | state of neuron `i` |
| `z_i(t)` | spike indicator of `i` at step `t` (0/1) |
| `b[i], θ[i]` | bias, firing threshold |
| `θ_min, θ_max` | active-band bounds |
| `ρ[i], λ[i]` | transfer-relay (retention on fire), sub-threshold decay |
| `e[j][i]` | eligibility trace on edge `j→i` |
| `γ` | eligibility-trace decay |
| `h_i` | post-synaptic (surrogate) factor |
| `M, M_pred, M_rl` | learning signal; its self-prediction and reward components |
| `α, β` | weights of the prediction and reward components in `M` |
| `η, η_b, μ` | weight LR, bias LR, weight decay |
| `ν*, g*, σ*` | homeostatic set-points: target rate, target incoming drive, target branching ratio |
| `G, θ, φ` | evolved generator, plasticity-rule params, homeostat set-points (the genome) |

# Appendix B — Default parameters

| Parameter | Gate | Tic-tac-toe |
|---|---|---|
| `N` | 16 | 128 |
| input neurons | `{0,1}` | `{0..17}` |
| output neurons | `{N−1}` | `{N−4..N−1}` |
| output cycle | 32 | 128 |
| deliberation cycle | — | 512 (optional) |
| count threshold | — | 128 |
| games / episodes | as needed to converge | 128 |
| `γ` (trace) | 0.8 | 0.8 |
| `λ` (decay, init) | 0.9 | 0.9 |
| `θ` (init) | 1.0 | 1.0 |
| `η` | 1e-2 | 1e-2 |
| `μ` (weight decay) | 1e-3 | 1e-3 |
| `α : β` (pred : reward) | 0.2 : 1.0 | 1.0 : 0.5 |
| homeostasis cadence `H` | 32 | 128 |

# Appendix C — Honest risks

1. **Transfer across scale is the load-bearing bet.** A rule and scaffold that are productive at `N=16–128` may develop pathologies at `N=10⁶`: criticality is harder to hold, and failures can be silent and late. This assumption (§11) is what makes the program affordable *and* is the most likely thing to break it. Test transfer explicitly at several intermediate scales before trusting it.
2. **The deliberation regime may not appear.** Even with everything in place, the system may plateau at competent-but-reactive (AFI-grade) behavior and never exhibit deliberative interiority. This is not engineerable away in advance; the core bet is unfalsifiable until built. The minimal-structure frontier (§13) is the insurance — it yields a real result regardless.
3. **Local rules underperform backprop at scale.** This is the field's central unsolved tension and X-CAT does not dissolve it; it bets that *meta-learning* the local rule (rather than hand-designing it) narrows the gap enough. If the inner loop stalls, the fallback ladder is: frozen-reservoir readout → surrogate-gradient BPTT on the unrolled cycle (pretrain) → e-prop → then the evolved local rule.
4. **The substrate-sensitivity / Orch-OR seam is downstream of everything.** Hybridizability is a design *affordance to preserve* (do not close off the per-step, per-neuron insertion points), **not** an early experiment. Do not attach an analog substrate to a network that cannot yet learn a logic gate.

---

*End of specification.*