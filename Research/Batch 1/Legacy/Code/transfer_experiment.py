"""
transfer_experiment.py -- Does a hidden recurrent core trained on tic-tac-toe
give a head start (or a handicap) on a structurally unrelated task
(delayed-recall), compared to random initialization?

Both architectures share n_hidden=24, so the hidden-to-hidden weight
submatrix and hidden biases can be extracted from the trained tic-tac-toe
model and spliced directly into a fresh delayed-recall network's hidden
block, leaving everything else (input/output layers, sized differently for
the two tasks) randomly initialized as usual.
"""
import numpy as np
import json

from xnn_model import XNN, XNNConfig
from xnn_delayed_recall import make_config
from fix_attempts import make_dfa_matrix, run_trial_dfa


def load_tictactoe_hidden_core(path: str):
    d = json.load(open(path))
    cfg = XNNConfig(**d['config'])
    W = np.array(d['W'])
    bias = np.array(d['bias'])
    h_idx = cfg.hidden_idx
    hidden_to_hidden = W[np.ix_(h_idx, h_idx)]
    hidden_bias = bias[h_idx]
    return hidden_to_hidden, hidden_bias


def make_transplanted_net(recall_cfg: XNNConfig, hidden_to_hidden: np.ndarray,
                            hidden_bias: np.ndarray, seed: int) -> XNN:
    net = XNN(recall_cfg, seed=seed)   # normal random init for everything
    h_idx = recall_cfg.hidden_idx
    net.W[np.ix_(h_idx, h_idx)] = hidden_to_hidden   # splice in the trained core
    net.bias[h_idx] = hidden_bias
    return net


def train_with_checkpoints(net: XNN, B: np.ndarray, D: int, seed: int,
                             checkpoints: list, c_w_hidden: float = 0.3):
    rng = np.random.default_rng(seed + 1000)
    results = []
    trials_done = 0
    for target in checkpoints:
        n_more = target - trials_done
        for _ in range(n_more):
            run_trial_dfa(net, B, D, rng, train=True, steps_per_tick=2, c_w=0.1, c_b=0.1,
                           c_w_hidden=c_w_hidden)
        trials_done = target
        eval_rng = np.random.default_rng(seed + 999999)
        correct = sum(run_trial_dfa(net, B, D, eval_rng, train=False, steps_per_tick=2,
                                     c_w=0.1, c_b=0.1, c_w_hidden=c_w_hidden)[0]
                      for _ in range(200))
        results.append((target, correct / 200))
    return results


def make_homeostasis_only_core(n_hidden: int = 24, seed: int = 0, settle_ticks: int = 8000):
    """A control core with the SAME random weight statistics as ordinary init,
    but bias settled via pure homeostatic regulation under generic noise input
    -- no task, no training signal, no learning at all. Weights are
    guaranteed unchanged (homeostasis only ever touches bias); this isolates
    whether an apparent transfer benefit is really about task-specific
    learned structure or just about homeostatically well-conditioned bias."""
    cfg = XNNConfig(n_input=18, n_hidden=n_hidden, n_output=9, n_value=4, steps_per_decision=1)
    net = XNN(cfg, seed=seed)
    rng = np.random.default_rng(seed + 3000)
    state = net.new_state()
    for _ in range(settle_ticks):
        noise_input = (rng.random(cfg.n_input) < 0.3).astype(float)
        state, _ = net.tick(state, noise_input, steps=1, track=False)
    h_idx = cfg.hidden_idx
    return net.W[np.ix_(h_idx, h_idx)].copy(), net.bias[h_idx].copy()


def train_unrelated_classification_core(n_hidden: int = 24, seed: int = 0,
                                          train_trials: int = 8000) -> tuple:
    """A task deliberately unrelated to both tic-tac-toe (sequential,
    adversarial, multi-move) and delayed-recall (temporal memory, distractor
    interference): single-tick, memoryless classification of a random binary
    vector by a fixed majority rule. No sequence, no game structure, no time
    dimension. If a hidden core trained on THIS shows the same reliability
    benefit as the tic-tac-toe core, that points to 'any trained recurrent
    matrix is better-conditioned than random' rather than genuine
    tic-tac-toe-relevant structure transferring."""
    n_input = 8
    n_output = 2
    cfg = XNNConfig(n_input=n_input, n_hidden=n_hidden, n_output=n_output,
                     n_value=4, steps_per_decision=2)
    net = XNN(cfg, seed=seed)
    B = make_dfa_matrix(n_hidden, n_output, seed=seed + 500)
    rng = np.random.default_rng(seed + 1000)

    for _ in range(train_trials):
        x = (rng.random(n_input) < 0.5).astype(float)
        label = 1 if x.sum() > n_input / 2 else 0

        state = net.new_state()
        state, fired = net.tick(state, x, steps=2, track=True)
        out_states = state[cfg.output_idx].copy()
        state[cfg.output_idx] = 0.0

        target = np.zeros(n_output)
        target[label] = 1.0
        error = target - out_states
        for i, gidx in enumerate(cfg.output_idx):
            net.W[fired, gidx] += error[i] * 0.1
            net.W[:, gidx] = np.clip(net.W[:, gidx], -1.0, 1.0)
            net.bias[gidx] += error[i] * 0.1
            net.bias[gidx] = np.clip(net.bias[gidx], 0.0, 1.0)
        pseudo_error = B @ error
        for j, gidx in enumerate(cfg.hidden_idx):
            net.W[fired, gidx] += pseudo_error[j] * 0.3
            net.W[:, gidx] = np.clip(net.W[:, gidx], -1.0, 1.0)
            net.bias[gidx] += pseudo_error[j] * 0.1
            net.bias[gidx] = np.clip(net.bias[gidx], 0.0, 1.0)

    # quick check that this control task was actually learned
    eval_rng = np.random.default_rng(seed + 999999)
    correct = 0
    for _ in range(200):
        x = (eval_rng.random(n_input) < 0.5).astype(float)
        label = 1 if x.sum() > n_input / 2 else 0
        state = net.new_state()
        state, _ = net.tick(state, x, steps=2, track=False)
        pred = int(np.argmax(state[cfg.output_idx]))
        correct += (pred == label)
    print(f"  [unrelated-task core, seed={seed}] trained classification accuracy: {correct/200:.3f}")

    h_idx = cfg.hidden_idx
    return net.W[np.ix_(h_idx, h_idx)].copy(), net.bias[h_idx].copy()


def train_running_parity_core(n_hidden: int = 24, seed: int = 0, T: int = 10,
                                train_trials: int = 8000) -> tuple:
    """A task multi-tick like tic-tac-toe and delayed-recall, but structurally
    unrelated in content to either: at each of T ticks a random bit arrives
    (one-hot on 2 channels), and the network must continuously update a
    running parity (XOR) of everything seen, reported only at the final
    tick (marked by a third trigger channel). Unlike delayed-recall, EVERY
    tick's input matters and must be incorporated into an evolving running
    computation -- there is no single early cue to protect from distractor
    noise, no 'store once, retrieve once' structure. If this core replicates
    the tic-tac-toe transplant's reliability, that points to 'practice
    sustaining multi-tick dynamics' as what transfers, not tic-tac-toe-
    specific content -- separating that from the single-tick control in the
    previous round, which could not have developed genuine multi-tick
    dynamics at all."""
    n_input = 3   # bit=0, bit=1, final-tick trigger
    n_output = 2
    cfg = XNNConfig(n_input=n_input, n_hidden=n_hidden, n_output=n_output,
                     n_value=4, steps_per_decision=2)
    net = XNN(cfg, seed=seed)
    B = make_dfa_matrix(n_hidden, n_output, seed=seed + 500)
    rng = np.random.default_rng(seed + 1000)

    def run_parity_trial(train: bool):
        state = net.new_state()
        parity = 0
        trace = np.zeros(cfg.n_total)
        for t in range(T):
            bit = int(rng.random() < 0.5)
            parity ^= bit
            vec = np.zeros(n_input)
            vec[bit] = 1.0
            if t == T - 1:
                vec[2] = 1.0   # final-tick trigger
            state, fired = net.tick(state, vec, steps=2, track=True)
            trace[fired] = 1.0

        out_states = state[cfg.output_idx].copy()
        state[cfg.output_idx] = 0.0
        pred = int(np.argmax(out_states))

        if train:
            target = np.zeros(n_output)
            target[parity] = 1.0
            error = target - out_states
            for i, gidx in enumerate(cfg.output_idx):
                net.W[trace.astype(bool), gidx] += error[i] * 0.1
                net.W[:, gidx] = np.clip(net.W[:, gidx], -1.0, 1.0)
                net.bias[gidx] += error[i] * 0.1
                net.bias[gidx] = np.clip(net.bias[gidx], 0.0, 1.0)
            pseudo_error = B @ error
            for j, gidx in enumerate(cfg.hidden_idx):
                net.W[trace.astype(bool), gidx] += pseudo_error[j] * 0.3
                net.W[:, gidx] = np.clip(net.W[:, gidx], -1.0, 1.0)
                net.bias[gidx] += pseudo_error[j] * 0.1
                net.bias[gidx] = np.clip(net.bias[gidx], 0.0, 1.0)

        return pred == parity

    for _ in range(train_trials):
        run_parity_trial(train=True)

    correct = sum(run_parity_trial(train=False) for _ in range(200))
    print(f"  [running-parity core, seed={seed}] trained accuracy: {correct/200:.3f}")

    h_idx = cfg.hidden_idx
    return net.W[np.ix_(h_idx, h_idx)].copy(), net.bias[h_idx].copy()


if __name__ == "__main__":
    hidden_to_hidden, hidden_bias = load_tictactoe_hidden_core(
        '/home/claude/xnn_package/model.json')
    print(f"Loaded tic-tac-toe hidden core: {hidden_to_hidden.shape}, "
          f"mean|W|={np.mean(np.abs(hidden_to_hidden)):.3f}")

    recall_cfg = make_config(24, steps_per_tick=2)
    D = 10
    checkpoints = [300, 800, 1500, 3000, 6000]
    seeds = [0, 1, 2]

    print(f"\nLearning curves at D={D}, checkpoints={checkpoints}, {len(seeds)} seeds\n")

    print("Training running-parity cores first:")
    parity_cores = {}
    for seed in seeds:
        parity_cores[seed] = train_running_parity_core(seed=seed + 600)
    print()

    transplant_curves = []
    parity_curves = []
    random_curves = []

    for seed in seeds:
        B_t = make_dfa_matrix(24, 2, seed=seed + 500)
        net_t = make_transplanted_net(recall_cfg, hidden_to_hidden, hidden_bias, seed=seed)
        curve_t = train_with_checkpoints(net_t, B_t, D, seed, checkpoints)
        transplant_curves.append([acc for _, acc in curve_t])
        print(f"seed={seed}  tic-tac-toe transplant:   " +
              "  ".join(f"{t}:{a:.2f}" for t, a in curve_t))

        B_p = make_dfa_matrix(24, 2, seed=seed + 500)
        par_W, par_bias = parity_cores[seed]
        net_p = make_transplanted_net(recall_cfg, par_W, par_bias, seed=seed)
        curve_p = train_with_checkpoints(net_p, B_p, D, seed, checkpoints)
        parity_curves.append([acc for _, acc in curve_p])
        print(f"seed={seed}  running-parity transplant:" +
              "  ".join(f"{t}:{a:.2f}" for t, a in curve_p))

        B_r = make_dfa_matrix(24, 2, seed=seed + 500)
        net_r = XNN(recall_cfg, seed=seed)
        curve_r = train_with_checkpoints(net_r, B_r, D, seed, checkpoints)
        random_curves.append([acc for _, acc in curve_r])
        print(f"seed={seed}  plain random init:        " +
              "  ".join(f"{t}:{a:.2f}" for t, a in curve_r))
        print()

    transplant_curves = np.array(transplant_curves)
    parity_curves = np.array(parity_curves)
    random_curves = np.array(random_curves)

    print("Mean across seeds:")
    for i, t in enumerate(checkpoints):
        print(f"  trials={t:5d}   ttt_transplant={transplant_curves[:,i].mean():.3f}"
              f"   running_parity={parity_curves[:,i].mean():.3f}"
              f"   plain_random={random_curves[:,i].mean():.3f}")
