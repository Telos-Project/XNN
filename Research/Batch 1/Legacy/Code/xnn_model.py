"""
xnn_model.py -- Generic Cross Neural Network (XNN) model utilities.

Implements the final architecture from the accompanying paper: a fully
recurrent, continuously-running (ASNP-style) integrate-and-fire network with
signed weights and bias as an atypical neural parameter (ATNP), trained via
local, per-edge credit assignment (no backpropagation, no global gradient)
and regulated by continuous homeostatic bias adjustment.

This module is task-agnostic. It knows nothing about tic-tac-toe or any
other application -- it operates on raw input vectors, index sets for
input/output/value neurons, and legal-action lists supplied by the caller.
See xnn_tictactoe.py for the task-specific application layer.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from typing import List, Optional, Tuple

import numpy as np


# --------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------

@dataclass
class XNNConfig:
    n_input: int
    n_hidden: int
    n_output: int
    n_value: int = 4                 # number of hidden neurons designated as the value readout
    steps_per_decision: int = 3      # deliberation steps run before reading a decision
    target_activity: float = 0.25    # homeostatic target firing rate per neuron
    homeostatic_lr: float = 0.002    # how fast bias chases the activity target
    activity_ema_alpha: float = 0.02 # smoothing rate of the running activity estimate

    @property
    def n_total(self) -> int:
        return self.n_input + self.n_hidden + self.n_output

    @property
    def input_idx(self) -> List[int]:
        return list(range(0, self.n_input))

    @property
    def hidden_idx(self) -> List[int]:
        return list(range(self.n_input, self.n_input + self.n_hidden))

    @property
    def output_idx(self) -> List[int]:
        return list(range(self.n_total - self.n_output, self.n_total))

    @property
    def value_idx(self) -> List[int]:
        return self.hidden_idx[: self.n_value]


# --------------------------------------------------------------------------
# Model
# --------------------------------------------------------------------------

class XNN:
    """
    ATNP scheme (see paper Section 3 for full derivation and justification):
      - neuron states:  [0, 1]                    (Positive Bounds, base Convention 1.1.1)
      - edge weights:   [-1, 1]                    (signed, sub-convention 1-1.1;
                                                      required -- positive-only weights are
                                                      provably unable to represent inhibitory
                                                      relationships, see paper Section 4.3)
      - bias:           [0, 1], per neuron          (ATNP, sub-convention 1-1.2; required to
                                                      escape the all-zero-input fixed point,
                                                      see paper Section 4.4)
      - dynamics:       uniform threshold trigger (fire at state==1, push, reset to 0)
                         + state relaxation via squaring (sub-convention 1.5.1)
      - state:          persists continuously across the whole task episode (real ASNP --
                         see paper Section 5.4); only input neurons are ever force-clamped,
                         and only output/decision neurons are ever force-cleared, and only
                         at the instant they are read (Clearing, 1.2.2.1)
    """

    def __init__(self, config: XNNConfig, seed: Optional[int] = None,
                 W: Optional[np.ndarray] = None, bias: Optional[np.ndarray] = None,
                 avg_activity: Optional[np.ndarray] = None):
        self.config = config
        n = config.n_total
        if W is not None:
            self.W = np.array(W, dtype=float)
        else:
            rng = np.random.default_rng(seed)
            self.W = rng.uniform(-1, 1, size=(n, n))
        if bias is not None:
            self.bias = np.array(bias, dtype=float)
        else:
            rng = np.random.default_rng(None if seed is None else seed + 1)
            self.bias = rng.uniform(0, 1, size=n)
        self.avg_activity = (np.array(avg_activity, dtype=float) if avg_activity is not None
                              else np.full(n, config.target_activity))

    # -- state lifecycle --------------------------------------------------

    def new_state(self) -> np.ndarray:
        """A fresh all-zero state. Should only be called at genuine episode
        boundaries (e.g. the start of a new game) -- never mid-task. See
        paper Section 5.4 on why mid-task resets undermine the ASNP premise."""
        return np.zeros(self.config.n_total)

    # -- core dynamics ------------------------------------------------------

    def tick(self, state: np.ndarray, input_vec: np.ndarray, steps: int,
              track: bool = True) -> Tuple[np.ndarray, np.ndarray]:
        """
        Advance the network `steps` steps. State persists in/out; the ONLY
        neurons ever force-set are the input neurons, clamped to input_vec
        every step (Multi-Step Input, convention 1.3). Homeostatic bias
        regulation runs every step, independent of any training signal.

        Returns (new_state, fired_log) where fired_log[i] is True if neuron i
        reached threshold at any point during this tick.
        """
        cfg = self.config
        n = cfg.n_total
        fired_log = np.zeros(n, dtype=bool)

        for _ in range(steps):
            fired = (state >= 1.0)
            if track:
                fired_log |= fired

            # continuous homeostatic regulation (not reward-driven)
            self.avg_activity = ((1 - cfg.activity_ema_alpha) * self.avg_activity +
                                  cfg.activity_ema_alpha * fired.astype(float))
            self.bias += cfg.homeostatic_lr * (cfg.target_activity - self.avg_activity)
            self.bias = np.clip(self.bias, 0.0, 1.0)

            incoming = fired.astype(float) @ self.W
            baseline = np.where(fired, 0.0, state)
            raw = np.clip(baseline + incoming + self.bias, 0.0, 1.0)
            new_state = np.where(raw > 0, raw ** 2, 0.0)          # state relaxation (squaring)
            new_state[cfg.input_idx] = input_vec                   # ONLY the IO exception
            state = new_state

        return state, fired_log

    # -- decision + value readout -----------------------------------------

    def choose_action(self, state: np.ndarray, input_vec: np.ndarray,
                       legal_actions: List[int], epsilon: float,
                       rng: np.random.Generator) -> Tuple[int, np.ndarray, np.ndarray, int, float]:
        """
        Run one decision tick, select an action among legal_actions (argmax
        of raw, non-normalized output states -- convention 2.3.1), read the
        value estimate, and clear only the output neurons (Clearing, 1.2.2.1).

        Returns (action_index, new_state, fired_log, chosen_global_neuron_idx, value_estimate)
        where action_index indexes into the output layer (0..n_output-1) and
        value_estimate is in [-1, 1].
        """
        cfg = self.config
        state, fired_log = self.tick(state, input_vec, cfg.steps_per_decision, track=True)

        out_states = state[cfg.output_idx].copy()
        state[cfg.output_idx] = 0.0   # Clearing -- ONLY the decision-readout neurons

        value = 2.0 * state[cfg.value_idx].mean() - 1.0   # map [0,1] average -> [-1,1]

        if rng.random() < epsilon:
            action = int(rng.choice(legal_actions))
        else:
            scores = np.full(cfg.n_output, -np.inf)
            for a in legal_actions:
                scores[a] = out_states[a]
            action = int(np.argmax(scores))

        chosen_global_idx = cfg.output_idx[action]
        return action, state, fired_log, chosen_global_idx, value

    # -- training: local, surgical, column-specific credit assignment ------

    def _train_column(self, col_idx: int, fired_log: np.ndarray, signal: float,
                       c_w: float, c_b: float) -> None:
        """Update ONLY the incoming edges of a single neuron (col_idx), from
        whichever neurons actually fired, plus that neuron's own bias. This
        is the fix for the diffuse-credit / lockstep-saturation failure mode
        documented in the paper (Section 5.7-5.9): updating a fired neuron's
        entire outgoing row, or every neuron's bias uniformly, causes
        unrelated outputs to drift together and saturate as a block.
        Vectorized (boolean-indexed) rather than a Python loop over all N
        neurons, so this stays cheap as N grows."""
        self.W[fired_log, col_idx] += signal * c_w
        self.W[:, col_idx] = np.clip(self.W[:, col_idx], -1.0, 1.0)
        self.bias[col_idx] += signal * c_b
        self.bias[col_idx] = np.clip(self.bias[col_idx], 0.0, 1.0)

    def train_actor(self, fired_log: np.ndarray, chosen_idx: int, advantage: float,
                     c_w: float = 0.05, c_b: float = 0.05) -> None:
        """Train the chosen action's incoming edges toward/away from having
        been taken, scaled by advantage (e.g. a TD error)."""
        self._train_column(chosen_idx, fired_log, advantage, c_w, c_b)

    def train_critic(self, fired_log: np.ndarray, td_error: float,
                      c_w: float = 0.05, c_b: float = 0.05) -> None:
        """Train the value neurons' incoming edges toward a better value
        estimate, scaled by the TD error."""
        for v in self.config.value_idx:
            self._train_column(v, fired_log, td_error, c_w, c_b)

    def train_transition(self, fired_log: np.ndarray, chosen_idx: int, td_error: float,
                          c_w: float = 0.05, c_b: float = 0.05) -> None:
        """Convenience wrapper: train both actor and critic from the same
        TD error (standard actor-critic sharing; see paper Section 6)."""
        self.train_actor(fired_log, chosen_idx, td_error, c_w, c_b)
        self.train_critic(fired_log, td_error, c_w, c_b)

    # -- serialization -------------------------------------------------------

    def to_dict(self) -> dict:
        return {
            "config": asdict(self.config),
            "W": self.W.tolist(),
            "bias": self.bias.tolist(),
            "avg_activity": self.avg_activity.tolist(),
        }

    @classmethod
    def from_dict(cls, d: dict) -> "XNN":
        config = XNNConfig(**d["config"])
        return cls(config, W=np.array(d["W"]), bias=np.array(d["bias"]),
                    avg_activity=np.array(d["avg_activity"]))

    def save(self, path: str) -> None:
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load(cls, path: str) -> "XNN":
        with open(path) as f:
            d = json.load(f)
        return cls.from_dict(d)

    # -- diagnostics ----------------------------------------------------------

    def saturation_fraction(self, threshold: float = 0.999) -> float:
        """Fraction of weights sitting at (or past) the clip boundary --
        useful for detecting the saturation/lockstep pathology described in
        the paper before it silently freezes training."""
        return float(np.mean(np.abs(self.W) >= threshold))
