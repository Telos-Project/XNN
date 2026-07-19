"""
fix_attempts.py -- Three candidate local-learning mechanisms for training
the hidden-to-hidden and input-to-hidden weights that the original rule
(xnn_delayed_recall.train_actor_traced, output-columns only) was found to
leave completely untouched (exactly zero change after 20,000 trials).

All three keep OUTPUT-layer training identical across variants (the real
supervised delta rule, as before) -- the only thing that differs is how, or
whether, hidden-layer weights get a training signal. This isolates the
comparison to exactly the mechanism in question.

  A. Direct Feedback Alignment (DFA): a FIXED random matrix projects the
     real output error into a per-hidden-neuron pseudo-error, used as that
     neuron's local training signal. No backward pass; the projection
     matrix never changes.

  B. Node perturbation (REINFORCE-style): inject small random noise into
     each hidden neuron's bias for the duration of a trial, observe the
     resulting reward, and use the standard score-function estimator
     (perturbation * (reward - baseline)) as that neuron's training signal.

  C. Slowness / temporal-consistency Hebbian rule: a continuous,
     reward-independent local rule that reinforces incoming edges to
     hidden neurons whose activity is stable (similar to itself) from one
     tick to the next -- unsupervised, runs every tick, never sees the
     task's reward signal at all.
"""
from __future__ import annotations

from typing import Tuple

import numpy as np

from xnn_model import XNN
from xnn_delayed_recall import make_config, build_input, N_OUTPUT


# ---------------------------------------------------------------------------
# Shared helper: same column-update mechanism used everywhere else in this
# project, exposed here so all three variants can reuse it identically.
# ---------------------------------------------------------------------------

def _traced_column_update(net: XNN, trace: np.ndarray, col_idx: int, signal: float,
                            c_w: float, c_b: float, train_bias: bool = True) -> None:
    net.W[:, col_idx] += signal * c_w * trace
    net.W[:, col_idx] = np.clip(net.W[:, col_idx], -1.0, 1.0)
    if train_bias:
        net.bias[col_idx] += signal * c_b
        net.bias[col_idx] = np.clip(net.bias[col_idx], 0.0, 1.0)


def _run_ticks(net: XNN, D: int, cue: int, rng: np.random.Generator, steps_per_tick: int,
               context: str = 'A'):
    """Shared trial-running logic: cue tick, D delay ticks, probe tick.
    Returns (out_states, trace, per_tick_hidden_states) where
    per_tick_hidden_states[t] is the hidden-neuron state vector at tick t
    (needed by the slowness variant; harmless overhead for the others)."""
    cfg = net.config
    state = net.new_state()
    trace = np.zeros(cfg.n_total)
    hidden_states = []

    input_vec = build_input('cue', cue, rng, context=context)
    state, fired = net.tick(state, input_vec, steps_per_tick, track=True)
    trace[fired] = 1.0
    hidden_states.append(state[cfg.hidden_idx].copy())

    for _ in range(D):
        state[cfg.input_idx] = 0.0
        input_vec = build_input('delay', cue, rng, context=context)
        state, fired = net.tick(state, input_vec, steps_per_tick, track=True)
        trace[fired] = 1.0
        hidden_states.append(state[cfg.hidden_idx].copy())

    state[cfg.input_idx] = 0.0
    input_vec = build_input('probe', cue, rng, context=context)
    state, fired = net.tick(state, input_vec, steps_per_tick, track=True)
    trace[fired] = 1.0
    hidden_states.append(state[cfg.hidden_idx].copy())

    out_states = state[cfg.output_idx].copy()
    return out_states, trace, hidden_states


# ---------------------------------------------------------------------------
# A. Direct Feedback Alignment
# ---------------------------------------------------------------------------

def make_dfa_matrix(n_hidden: int, n_output: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.uniform(-1, 1, size=(n_hidden, n_output))


def run_trial_dfa(net: XNN, B: np.ndarray, D: int, rng: np.random.Generator, train: bool,
                    steps_per_tick: int, c_w: float, c_b: float, c_w_hidden: float,
                    adapt_B: bool = False, lr_B: float = 0.02, decay_B: float = 0.001,
                    context: str = 'A') -> Tuple[bool, int]:
    """If adapt_B is True, B is co-adapted via a local Hebbian rule (Kolen-
    Pollack style): B += lr_B * outer(hidden_activity, error) - decay_B * B.
    This uses only signals already locally available where the error is
    computed (hidden activity, real output error) -- no reading of W, no
    weight transport, staying within the same locality constraint as
    everything else in this project. B is modified in place."""
    cfg = net.config
    cue = int(rng.integers(2))
    out_states, trace, hidden_states = _run_ticks(net, D, cue, rng, steps_per_tick, context=context)

    predicted = int(np.argmax(out_states))
    correct = predicted == cue

    if train:
        target = np.zeros(N_OUTPUT)
        target[cue] = 1.0
        error = target - out_states

        for i, global_idx in enumerate(cfg.output_idx):
            _traced_column_update(net, trace, global_idx, error[i], c_w, c_b)

        pseudo_error = B @ error
        for j, global_idx in enumerate(cfg.hidden_idx):
            _traced_column_update(net, trace, global_idx, pseudo_error[j], c_w_hidden, c_b)

        if adapt_B:
            hidden_activity = hidden_states[-1]   # locally available, no weight transport
            B += lr_B * np.outer(hidden_activity, error) - decay_B * B
            np.clip(B, -1.0, 1.0, out=B)

    return correct, cue


# ---------------------------------------------------------------------------
# B. Node perturbation
# ---------------------------------------------------------------------------

class RunningBaseline:
    def __init__(self, alpha: float = 0.05):
        self.value = 0.0
        self.alpha = alpha

    def update(self, r: float) -> float:
        self.value = (1 - self.alpha) * self.value + self.alpha * r
        return self.value


def run_trial_perturb(net: XNN, D: int, rng: np.random.Generator, train: bool,
                        steps_per_tick: int, c_w: float, c_b: float, c_w_hidden: float,
                        sigma: float, baseline: RunningBaseline) -> Tuple[bool, int]:
    cfg = net.config
    cue = int(rng.integers(2))

    perturbation = np.zeros(cfg.n_total)
    original_bias = net.bias.copy()
    if train:
        eps = rng.normal(0, sigma, size=len(cfg.hidden_idx))
        perturbation[cfg.hidden_idx] = eps
        net.bias = np.clip(net.bias + perturbation, 0.0, 1.0)

    out_states, trace, _ = _run_ticks(net, D, cue, rng, steps_per_tick)

    if train:
        net.bias = original_bias   # restore -- the perturbation was a probe, not a real update

    predicted = int(np.argmax(out_states))
    correct = predicted == cue

    if train:
        target = np.zeros(N_OUTPUT)
        target[cue] = 1.0
        error = target - out_states
        for i, global_idx in enumerate(cfg.output_idx):
            _traced_column_update(net, trace, global_idx, error[i], c_w, c_b)

        r = 1.0 if correct else -1.0
        b = baseline.update(r)
        advantage = r - b

        for j, global_idx in enumerate(cfg.hidden_idx):
            eps_j = perturbation[global_idx]
            signal = eps_j * advantage / (sigma ** 2)
            _traced_column_update(net, trace, global_idx, signal, c_w_hidden, c_b=0.0, train_bias=False)
            # NOTE: bias is intentionally left untrained by this signal (it was
            # already used AS the perturbation channel above); only incoming
            # edges get the node-perturbation update.

    return correct, cue


# ---------------------------------------------------------------------------
# C. Slowness / temporal-consistency Hebbian rule (unsupervised, reward-free)
# ---------------------------------------------------------------------------

def run_trial_slowness(net: XNN, D: int, rng: np.random.Generator, train: bool,
                         steps_per_tick: int, c_w: float, c_b: float,
                         c_hebb: float) -> Tuple[bool, int]:
    cfg = net.config
    cue = int(rng.integers(2))
    out_states, trace, hidden_states = _run_ticks(net, D, cue, rng, steps_per_tick)

    predicted = int(np.argmax(out_states))
    correct = predicted == cue

    if train:
        target = np.zeros(N_OUTPUT)
        target[cue] = 1.0
        error = target - out_states
        for i, global_idx in enumerate(cfg.output_idx):
            _traced_column_update(net, trace, global_idx, error[i], c_w, c_b)

        # continuous, reward-independent slowness pressure across all ticks:
        # reinforce a hidden neuron's incoming edges from whatever was active
        # the tick before in proportion to how much MORE stable than average
        # its activity was between those two ticks, and PUNISH below-average
        # stability -- a signed differential rule. An always-positive version
        # of this (reward stability, never punish instability) was tried
        # first and reproduced the lockstep saturation pathology from
        # Section 5.6-5.9: unconstrained positive-only Hebbian updates have
        # nothing pushing weights back down and saturate the same way.
        for t in range(1, len(hidden_states)):
            h_prev, h_now = hidden_states[t - 1], hidden_states[t]
            stability = 1.0 - np.abs(h_now - h_prev)          # can be negative if very unstable
            signed_stability = stability - np.mean(stability)  # differential: signed around the mean
            active_now = h_now > 0.01
            for j_local, j_global in enumerate(cfg.hidden_idx):
                if active_now[j_local]:
                    net.W[:, j_global] += c_hebb * signed_stability[j_local] * trace
            net.W = np.clip(net.W, -1.0, 1.0)

    return correct, cue
