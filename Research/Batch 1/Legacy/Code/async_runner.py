"""
async_runner.py -- Stage 3: genuine OS-thread concurrency between the XNN
and the environment, as literally as Python allows.

Honest framing, decided before looking at any results: Python's GIL means
this is NOT true multi-core simultaneous execution -- only one thread runs
Python bytecode at a time, and at this network's size (48 hidden neurons,
tiny matrix ops), numpy's per-op GIL release windows are likely too short
for meaningful overlap anyway (see the timing benchmark: one network
sub-tick is ~43 microseconds, only ~5x cheaper than one environment step).
What this DOES give, genuinely, is real OS-scheduler-determined
interleaving: no artificial barrier forces either side to wait for the
other to finish a "turn," and the environment can read the network's
output layer at any arbitrary instant, mid-computation or not -- exactly
the "forced interruption of a still-forming decision" property that every
prior experiment in this project (lockstep by construction) could not
test. That property, not literal multi-core execution, is what's actually
being measured here.

The network thread runs a perpetual loop: read whatever observation is
CURRENTLY in shared state (no waiting for a "fresh" one), advance the XNN
by one raw dynamical sub-tick, write whatever is CURRENTLY in the output
layer to shared state, repeat -- forever, at whatever rate the scheduler
allows. The environment thread runs its own loop at a controlled pace,
reading whatever action preference is currently available and stepping
the world accordingly.

This is inference-only (a fixed, already-trained network from the Stage 1
lockstep experiment). Training under genuine asynchronous execution raises
a harder question -- when does a reward arising at an arbitrary async
moment get attributed to which internal network tick -- deliberately
deferred rather than conflated with this test.
"""
from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from xnn_model import XNN
from foraging_env import ForagingEnv, N_OUTPUT
from foraging_agent import make_config


@dataclass
class SharedState:
    obs: np.ndarray
    out_states: np.ndarray
    lock: threading.Lock = field(default_factory=threading.Lock)
    network_ticks: int = 0
    env_steps: int = 0
    stop: bool = False
    clear_output_requested: bool = False


def network_thread_fn(net: XNN, shared: SharedState, state_holder: list):
    """Runs forever until shared.stop is set. state_holder[0] carries the
    XNN's persistent internal state across iterations (a list so this
    closure can mutate it). Only THIS thread ever mutates state_holder[0]
    -- the environment thread requests output-clearing via a flag rather
    than mutating the array directly, since tick() makes several separate
    numpy calls (not one atomic operation), so a cross-thread mutation
    mid-computation could otherwise corrupt an in-progress tick."""
    cfg = net.config
    while True:
        with shared.lock:
            if shared.stop:
                return
            obs = shared.obs.copy()
            do_clear = shared.clear_output_requested
            shared.clear_output_requested = False

        if do_clear:
            state_holder[0][cfg.output_idx] = 0.0

        new_state, _ = net.tick(state_holder[0], obs, steps=1, track=False)
        state_holder[0] = new_state
        out = new_state[cfg.output_idx].copy()

        with shared.lock:
            shared.out_states = out
            shared.network_ticks += 1


def run_async_episode(net: XNN, env_seed: int, max_wall_seconds: float = 5.0,
                        env_pace_seconds: Optional[float] = None) -> dict:
    """env_pace_seconds: if set, the environment thread sleeps this long
    between steps (simulating a fixed frame rate); if None, the
    environment steps as fast as it can, same as the network.

    Clearing (zeroing output neurons once read) is requested at the moment
    the ENVIRONMENT thread reads them -- the natural asynchronous analogue
    of "reading is the consumption event" used everywhere else in this
    project -- but applied by the network thread itself at the start of
    its next iteration, preserving single-writer ownership of its own
    state array."""
    cfg = net.config
    env = ForagingEnv(seed=env_seed)
    obs = env.reset()

    shared = SharedState(obs=obs.copy(), out_states=np.zeros(N_OUTPUT))
    state_holder = [net.new_state()]

    net_thread = threading.Thread(target=network_thread_fn, args=(net, shared, state_holder))
    net_thread.start()

    t_start = time.time()
    done = False
    info = {}
    while not done and (time.time() - t_start) < max_wall_seconds:
        with shared.lock:
            out = shared.out_states.copy()
            shared.clear_output_requested = True
        action = int(np.argmax(out))

        obs, reward, done, info = env.step(action)

        with shared.lock:
            shared.obs = obs.copy()
            shared.env_steps += 1

        if env_pace_seconds:
            time.sleep(env_pace_seconds)

    with shared.lock:
        shared.stop = True
    net_thread.join(timeout=2.0)

    info = dict(info)
    info['network_ticks'] = shared.network_ticks
    info['env_steps'] = shared.env_steps
    info['wall_seconds'] = time.time() - t_start
    return info


if __name__ == "__main__":
    import pickle
    with open('trained_full.pkl', 'rb') as f:
        d = pickle.load(f)
    from xnn_model import XNNConfig
    cfg = XNNConfig(**d['config'])
    net = XNN(cfg, W=d['W'], bias=d['bias'])

    print("Sanity check: single async episode, unpaced (both sides run as fast as possible)")
    info = run_async_episode(net, env_seed=0, max_wall_seconds=3.0)
    print(info)
