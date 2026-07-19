"""
foraging_agent.py -- XNN agent for the foraging task, synthesizing three
previously-separately-validated mechanisms rather than inventing a fourth:

  - TD(0) actor-critic with value neurons sharing the recurrent substrate
    (from the tic-tac-toe self-play work) -- needed because a consequential
    decision and its payoff can be separated by many ticks here, more than
    anywhere else in this project.
  - DFA-based hidden-layer credit assignment (from the delayed-recall work)
    -- Section 8 found that without this, hidden-to-hidden weights receive
    EXACTLY ZERO training signal regardless of budget or capacity. Since B
    projects a single scalar TD error (not a vector classification error),
    it's a fixed random VECTOR here rather than a matrix.
  - Continuous homeostatic regulation (built into xnn_model.XNN.tick()
    already) -- the standing defense against saturation collapse.

State persists continuously across an entire episode (real ASNP): it is
reset only at true episode boundaries (death or max ticks), never between
individual environment ticks.
"""
from __future__ import annotations

from typing import Tuple, Optional

import numpy as np

from xnn_model import XNN, XNNConfig
from foraging_env import ForagingEnv, N_INPUT, N_OUTPUT


def make_config(n_hidden: int = 48, steps_per_decision: int = 3) -> XNNConfig:
    return XNNConfig(n_input=N_INPUT, n_hidden=n_hidden, n_output=N_OUTPUT,
                      n_value=4, steps_per_decision=steps_per_decision,
                      target_activity=0.25, homeostatic_lr=0.002, activity_ema_alpha=0.02)


def make_dfa_vector(n_hidden: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.uniform(-1, 1, size=n_hidden)


def choose_action(net: XNN, state: np.ndarray, obs: np.ndarray, epsilon: float,
                    rng: np.random.Generator) -> Tuple[int, np.ndarray, np.ndarray, int, float]:
    cfg = net.config
    state, fired_log = net.tick(state, obs, cfg.steps_per_decision, track=True)

    out_states = state[cfg.output_idx].copy()
    state[cfg.output_idx] = 0.0   # Clearing -- only the decision-readout neurons

    value = 2.0 * state[cfg.value_idx].mean() - 1.0

    if rng.random() < epsilon:
        action = int(rng.integers(N_OUTPUT))
    else:
        action = int(np.argmax(out_states))

    chosen_global_idx = cfg.output_idx[action]
    return action, state, fired_log, chosen_global_idx, value


def train_step(net: XNN, B_vec: Optional[np.ndarray], fired_log: np.ndarray, chosen_idx: int,
                td_error: float, c_w: float = 0.05, c_b: float = 0.05,
                c_w_hidden: float = 0.1, dfa_ablate: bool = False) -> None:
    """actor + critic: direct real TD error on output-facing columns (chosen
    action, value neurons) -- known-correct signal, no projection needed.
    hidden: DFA-projected pseudo-error, UNLESS dfa_ablate=True, in which case
    hidden columns receive no training at all (the ORIGINAL, pre-Section-9
    rule, kept here as a direct ablation of the DFA fix on this new task)."""
    net.train_actor(fired_log, chosen_idx, td_error, c_w, c_b)
    net.train_critic(fired_log, td_error, c_w, c_b)

    if not dfa_ablate and B_vec is not None:
        pseudo_error = B_vec * td_error
        for j, gidx in enumerate(net.config.hidden_idx):
            net._train_column(gidx, fired_log, pseudo_error[j], c_w_hidden, c_b)


class RunningBaseline:
    def __init__(self, alpha: float = 0.01):
        self.value = 0.0
        self.initialized = False
        self.alpha = alpha

    def update(self, r: float) -> float:
        if not self.initialized:
            self.value = r
            self.initialized = True
        else:
            self.value = (1 - self.alpha) * self.value + self.alpha * r
        return self.value


def run_episode(net: XNN, B_vec: Optional[np.ndarray], epsilon: float, seed: int,
                  train: bool = True, c_w: float = 0.05, c_b: float = 0.05,
                  c_w_hidden: float = 0.1, memory_ablate: bool = False,
                  dfa_ablate: bool = False, baseline: Optional[RunningBaseline] = None,
                  reward_scale: float = 1.0 / 40.0) -> dict:
    """reward_scale brings the raw environment reward (which can be as large
    as +30/-40 in a single tick -- much bigger than the +-1-ish signals used
    everywhere else in this project) down to a comparable magnitude before
    it drives any weight update. Without this, a single high-magnitude
    reward event can blow a weight through its entire [-1,1] range in one
    tick, which turned out to be the actual cause of the lockstep collapse
    seen when this was first tried unscaled."""
    cfg = net.config
    env = ForagingEnv(seed=seed)
    obs = env.reset()
    state = net.new_state()
    rng = np.random.default_rng(seed + 500_000)

    pending = None   # (fired_log, chosen_idx, value, reward) awaiting a bootstrap target

    while True:
        action, state, fired_log, chosen_idx, value = choose_action(net, state, obs, epsilon, rng)

        if memory_ablate:
            state[cfg.hidden_idx] = 0.0
            state[cfg.output_idx] = 0.0   # avoid the exact leak found in Section 8.2

        if pending is not None and train:
            shaped_reward = (pending['reward'] - (baseline.value if baseline else 0.0)) * reward_scale
            td_error = shaped_reward + value - pending['value']
            train_step(net, B_vec, pending['fired_log'], pending['chosen_idx'], td_error,
                       c_w, c_b, c_w_hidden, dfa_ablate)

        next_obs, reward, done, info = env.step(action)
        if train and baseline is not None:
            baseline.update(reward)

        if done:
            if train:
                shaped_reward = (reward - (baseline.value if baseline else 0.0)) * reward_scale
                td_error = shaped_reward - value
                train_step(net, B_vec, fired_log, chosen_idx, td_error,
                           c_w, c_b, c_w_hidden, dfa_ablate)
            return info

        pending = {'fired_log': fired_log, 'chosen_idx': chosen_idx, 'value': value, 'reward': reward}
        obs = next_obs
