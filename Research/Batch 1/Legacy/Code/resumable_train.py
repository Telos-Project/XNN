"""Resumable self-play training so a single run can span multiple time-
limited invocations while training on an exact, matched number of games."""
import json
import numpy as np
from xnn_model import XNN, XNNConfig
import xnn_tictactoe as ttt
from xnn_tictactoe import evaluate_vs_external, opponent_move_random, opponent_move_smart


def save_checkpoint(path, net, rng, g, games_total, best_score, best_W, best_bias, eval_every):
    ckpt = {
        "config": net.config.__dict__,
        "W": net.W.tolist(),
        "bias": net.bias.tolist(),
        "avg_activity": net.avg_activity.tolist(),
        "rng_state": rng.bit_generator.state,
        "g": g,
        "games_total": games_total,
        "best_score": best_score,
        "best_W": best_W.tolist() if best_W is not None else None,
        "best_bias": best_bias.tolist() if best_bias is not None else None,
        "eval_every": eval_every,
    }
    with open(path, "w") as f:
        json.dump(ckpt, f)


def load_checkpoint(path):
    with open(path) as f:
        ckpt = json.load(f)
    cfg = XNNConfig(**ckpt["config"])
    net = XNN(cfg, W=np.array(ckpt["W"]), bias=np.array(ckpt["bias"]),
              avg_activity=np.array(ckpt["avg_activity"]))
    rng = np.random.default_rng()
    rng.bit_generator.state = ckpt["rng_state"]
    best_W = np.array(ckpt["best_W"]) if ckpt["best_W"] is not None else None
    best_bias = np.array(ckpt["best_bias"]) if ckpt["best_bias"] is not None else None
    return net, rng, ckpt["g"], ckpt["games_total"], ckpt["best_score"], best_W, best_bias, ckpt["eval_every"]


def run_chunk(checkpoint_path, n_hidden=None, n_value=None, steps=None, games_total=None,
              eval_every=1500, seed=0, n_games_this_chunk=None, c_w=0.05, c_b=0.05):
    import os
    if os.path.exists(checkpoint_path):
        net, rng, g_start, games_total, best_score, best_W, best_bias, eval_every = \
            load_checkpoint(checkpoint_path)
    else:
        cfg = XNNConfig(n_input=18, n_hidden=n_hidden, n_output=9, n_value=n_value, steps_per_decision=steps)
        net = XNN(cfg, seed=seed)
        rng = np.random.default_rng(seed + 5000)
        g_start, best_score, best_W, best_bias = 0, -1e9, None, None

    g_end = min(g_start + n_games_this_chunk, games_total)

    for g in range(g_start, g_end):
        epsilon = 0.4 + (0.05 - 0.4) * (g / games_total)
        ttt.play_self_play_game(net, epsilon, rng, train=True, c_w=c_w, c_b=c_b)

        if g % eval_every == 0 or g == games_total - 1:
            wr_r, dr_r, lr_r = evaluate_vs_external(net, opponent_move_random, n_games=80, seed=999)
            wr_s, dr_s, lr_s = evaluate_vs_external(net, opponent_move_smart, n_games=80, seed=999)
            score = wr_r + wr_s - lr_r - lr_s
            if score > best_score:
                best_score, best_W, best_bias = score, net.W.copy(), net.bias.copy()
            print(f"  game {g:6d}/{games_total}  vs_random W{wr_r:.2f}/L{lr_r:.2f}"
                  f"  vs_smart W{wr_s:.2f}/L{lr_s:.2f}  score={score:.2f}  sat={net.saturation_fraction():.3f}")

    save_checkpoint(checkpoint_path, net, rng, g_end, games_total, best_score, best_W, best_bias, eval_every)
    print(f"chunk done: now at game {g_end}/{games_total}")
    return g_end >= games_total
