"""
rnn_baseline.py -- A conventional backprop-through-time RNN baseline on the
IDENTICAL delayed-recall task used for the XNN pilot (xnn_delayed_recall.py),
using the same task generator (build_input, cue/delay/probe structure) so
the comparison is apples-to-apples, not just similar-in-spirit.

No torch available in this environment, so this is a hand-derived BPTT
implementation for a simple Elman (tanh) RNN, verified against finite-
difference gradient checks before being trusted for the real experiment --
BPTT is easy to get subtly wrong, and this project has already found
several bugs in less error-prone code than this.

Unlike the XNN's local, column-specific training rule (which was found to
leave hidden-to-hidden and input-to-hidden weights completely untouched --
literally zero change after 20,000 trials, see paper discussion), ordinary
backprop trains every weight in the network, including the recurrent
connections that would need to be shaped for a memory trace to survive
multiple ticks. This baseline exists to answer a direct question: is D>=3
genuinely hard on this task, or only hard for the XNN's specific training
rule?
"""
from __future__ import annotations

from typing import Tuple

import numpy as np

from xnn_delayed_recall import build_input, N_INPUT, N_OUTPUT, CUE_IDX


class ElmanRNN:
    def __init__(self, hidden_dim: int, seed: int = 0, lr: float = 0.01):
        rng = np.random.default_rng(seed)
        scale_xh = 1.0 / np.sqrt(N_INPUT)
        scale_hh = 1.0 / np.sqrt(hidden_dim)
        scale_hy = 1.0 / np.sqrt(hidden_dim)

        self.hidden_dim = hidden_dim
        self.W_xh = rng.uniform(-scale_xh, scale_xh, size=(hidden_dim, N_INPUT))
        self.W_hh = rng.uniform(-scale_hh, scale_hh, size=(hidden_dim, hidden_dim))
        self.b_h = np.zeros(hidden_dim)
        self.W_hy = rng.uniform(-scale_hy, scale_hy, size=(N_OUTPUT, hidden_dim))
        self.b_y = np.zeros(N_OUTPUT)

        self.lr = lr
        # Adam optimizer state
        self._m = {}
        self._v = {}
        self._t = 0
        for name in ["W_xh", "W_hh", "b_h", "W_hy", "b_y"]:
            p = getattr(self, name)
            self._m[name] = np.zeros_like(p)
            self._v[name] = np.zeros_like(p)

    def param_count(self) -> int:
        return (self.W_xh.size + self.W_hh.size + self.b_h.size +
                self.W_hy.size + self.b_y.size)

    def forward(self, inputs: list) -> Tuple[np.ndarray, list, np.ndarray]:
        """inputs: list of T input vectors. Returns (probs, hidden_states, final_h).
        hidden_states[t] is h_t for t=0..T-1 (hidden_states does NOT include h_{-1}=0)."""
        h = np.zeros(self.hidden_dim)
        hidden_states = []
        for x_t in inputs:
            a = self.W_xh @ x_t + self.W_hh @ h + self.b_h
            h = np.tanh(a)
            hidden_states.append(h)

        y = self.W_hy @ hidden_states[-1] + self.b_y
        y_shifted = y - np.max(y)
        exp_y = np.exp(y_shifted)
        probs = exp_y / exp_y.sum()
        return probs, hidden_states, hidden_states[-1]

    def compute_loss_and_grads(self, inputs: list, target_class: int):
        """Full forward + BPTT backward. Returns (loss, probs, grads_dict)."""
        T = len(inputs)
        h_prev_list = [np.zeros(self.hidden_dim)] + [None] * (T - 1)
        h = np.zeros(self.hidden_dim)
        hidden_states = []
        for t, x_t in enumerate(inputs):
            a = self.W_xh @ x_t + self.W_hh @ h + self.b_h
            h = np.tanh(a)
            hidden_states.append(h)
            if t + 1 < T:
                h_prev_list[t + 1] = h

        y = self.W_hy @ hidden_states[-1] + self.b_y
        y_shifted = y - np.max(y)
        exp_y = np.exp(y_shifted)
        probs = exp_y / exp_y.sum()
        loss = -np.log(max(probs[target_class], 1e-12))

        # backward
        target_onehot = np.zeros(N_OUTPUT)
        target_onehot[target_class] = 1.0
        dy = probs - target_onehot

        grads = {name: np.zeros_like(getattr(self, name))
                 for name in ["W_xh", "W_hh", "b_h", "W_hy", "b_y"]}

        grads["W_hy"] = np.outer(dy, hidden_states[-1])
        grads["b_y"] = dy.copy()

        dh = self.W_hy.T @ dy   # gradient w.r.t. h_{T-1}
        for t in range(T - 1, -1, -1):
            h_t = hidden_states[t]
            h_prev = h_prev_list[t]
            da = dh * (1.0 - h_t ** 2)
            grads["W_xh"] += np.outer(da, inputs[t])
            grads["W_hh"] += np.outer(da, h_prev)
            grads["b_h"] += da
            dh = self.W_hh.T @ da   # propagate to t-1

        return loss, probs, grads

    def apply_grads(self, grads: dict, beta1=0.9, beta2=0.999, eps=1e-8):
        self._t += 1
        for name, g in grads.items():
            self._m[name] = beta1 * self._m[name] + (1 - beta1) * g
            self._v[name] = beta2 * self._v[name] + (1 - beta2) * (g ** 2)
            m_hat = self._m[name] / (1 - beta1 ** self._t)
            v_hat = self._v[name] / (1 - beta2 ** self._t)
            update = self.lr * m_hat / (np.sqrt(v_hat) + eps)
            p = getattr(self, name)
            p -= update


def gradient_check(hidden_dim: int = 6, T: int = 4, eps: float = 1e-5, seed: int = 0) -> float:
    """Finite-difference gradient check on a small random instance. Returns
    the max relative error across all checked parameters; should be tiny
    (< 1e-4) if BPTT is implemented correctly."""
    rng = np.random.default_rng(seed)
    net = ElmanRNN(hidden_dim=hidden_dim, seed=seed)
    inputs = [rng.uniform(-1, 1, size=N_INPUT) for _ in range(T)]
    target = 0

    loss0, _, grads = net.compute_loss_and_grads(inputs, target)

    max_rel_err = 0.0
    for name in ["W_xh", "W_hh", "b_h", "W_hy", "b_y"]:
        p = getattr(net, name)
        analytic = grads[name]
        # check a handful of random entries, not the whole tensor (cheap but sufficient)
        flat_idx = rng.choice(p.size, size=min(5, p.size), replace=False)
        for idx in flat_idx:
            multi_idx = np.unravel_index(idx, p.shape)
            orig = p[multi_idx]

            p[multi_idx] = orig + eps
            loss_plus, _, _ = net.compute_loss_and_grads(inputs, target)
            p[multi_idx] = orig - eps
            loss_minus, _, _ = net.compute_loss_and_grads(inputs, target)
            p[multi_idx] = orig

            numeric = (loss_plus - loss_minus) / (2 * eps)
            a = analytic[multi_idx]
            rel_err = abs(numeric - a) / max(abs(numeric), abs(a), 1e-8)
            max_rel_err = max(max_rel_err, rel_err)

    return max_rel_err


def run_trial(net: ElmanRNN, D: int, rng: np.random.Generator, train: bool) -> Tuple[bool, int]:
    cue = int(rng.integers(2))
    inputs = [build_input('cue', cue, rng)]
    for _ in range(D):
        inputs.append(build_input('delay', cue, rng))
    inputs.append(build_input('probe', cue, rng))

    if train:
        loss, probs, grads = net.compute_loss_and_grads(inputs, cue)
        net.apply_grads(grads)
    else:
        probs, _, _ = net.forward(inputs)

    predicted = int(np.argmax(probs))
    return predicted == cue, cue


def train_and_eval_rnn(D: int, hidden_dim: int, train_trials: int = 5000,
                        eval_trials: int = 400, seed: int = 0, lr: float = 0.01
                        ) -> Tuple[ElmanRNN, float]:
    net = ElmanRNN(hidden_dim=hidden_dim, seed=seed, lr=lr)
    rng = np.random.default_rng(seed + 1000)

    for _ in range(train_trials):
        run_trial(net, D, rng, train=True)

    eval_rng = np.random.default_rng(seed + 999999)
    correct = 0
    for _ in range(eval_trials):
        c, _ = run_trial(net, D, eval_rng, train=False)
        correct += c

    return net, correct / eval_trials


if __name__ == "__main__":
    err = gradient_check()
    print(f"Gradient check max relative error: {err:.2e}  ({'PASS' if err < 1e-4 else 'FAIL'})")
