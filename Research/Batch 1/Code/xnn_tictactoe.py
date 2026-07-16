"""
xnn_tictactoe.py -- Applying the XNN model (xnn_model.py) to tic-tac-toe via
self-play, with no external opponent, oracle, or labelled data of any kind
used in training. See the accompanying paper for the full derivation and the
series of failed intermediate designs this one replaced.

Architecture: 18 input neurons (9 cells x {mine, opponent's}), 24 hidden
"deliberation" neurons (4 of which double as the value readout), 9 output
neurons (one per cell). The same weights play both X and O via perspective-
canonicalized input -- "my pieces" and "opponent's pieces" always occupy the
same input neurons regardless of which symbol the network is currently
playing.
"""
from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np

from xnn_model import XNN, XNNConfig

# --------------------------------------------------------------------------
# Board mechanics
# --------------------------------------------------------------------------

WIN_LINES = [
    (0, 1, 2), (3, 4, 5), (6, 7, 8),
    (0, 3, 6), (1, 4, 7), (2, 5, 8),
    (0, 4, 8), (2, 4, 6),
]


def check_winner(board: List[int]) -> Optional[int]:
    """Returns 1 (X wins), 2 (O wins), 0 (draw), or None (game ongoing)."""
    for a, b, c in WIN_LINES:
        if board[a] != 0 and board[a] == board[b] == board[c]:
            return board[a]
    if 0 not in board:
        return 0
    return None


def canonicalize(board: List[int], mover: int, n_input: int) -> np.ndarray:
    """Perspective-invariant board encoding: the first half of the input
    neurons are always 'my pieces', the second half always 'opponent's
    pieces', regardless of whether `mover` is X (1) or O (2). This is what
    lets a single set of weights play both sides in self-play."""
    other = 2 if mover == 1 else 1
    vec = np.zeros(n_input)
    half = n_input // 2
    for i, c in enumerate(board):
        if c == mover:
            vec[i] = 1.0
        elif c == other:
            vec[half + i] = 1.0
    return vec


def default_config() -> XNNConfig:
    return XNNConfig(n_input=18, n_hidden=24, n_output=9, n_value=4,
                      steps_per_decision=3, target_activity=0.25,
                      homeostatic_lr=0.002, activity_ema_alpha=0.02)


# --------------------------------------------------------------------------
# Self-play training (no labels, no external opponent)
# --------------------------------------------------------------------------

def play_self_play_game(net: XNN, epsilon: float, rng: np.random.Generator,
                          train: bool = False, c_w: float = 0.05, c_b: float = 0.05
                          ) -> Tuple[int, int]:
    """
    Plays one game with the SAME network making decisions for both sides.
    Trains via TD(0): after each move, the next mover's value estimate
    (negated, since the game is zero-sum) becomes the bootstrap target for
    the previous mover's transition. The terminal move in the game trains
    against the actual outcome instead of a bootstrap.

    Returns (winner, last_mover) where winner is 1 (X), 2 (O), or 0 (draw).
    """
    n_input = net.config.n_input
    board = [0] * 9
    state = net.new_state()
    mover = 1
    pending = None   # (fired_log, chosen_idx, v_before) awaiting a bootstrap target

    while True:
        legal = [i for i, c in enumerate(board) if c == 0]
        input_vec = canonicalize(board, mover, n_input)
        move, state, fired_log, chosen_idx, v_before = net.choose_action(
            state, input_vec, legal, epsilon, rng)

        if pending is not None and train:
            target = -v_before   # zero-sum: good for the current mover now = bad for whoever just moved
            td_error = target - pending[2]
            net.train_transition(pending[0], pending[1], td_error, c_w, c_b)

        board[move] = mover
        winner = check_winner(board)

        if winner is not None:
            r = 1.0 if winner == mover else 0.0   # a mover cannot lose on their own move
            if train:
                td_error = r - v_before
                net.train_transition(fired_log, chosen_idx, td_error, c_w, c_b)
            return winner, mover

        pending = (fired_log, chosen_idx, v_before)
        mover = 2 if mover == 1 else 1


def train_selfplay(games: int = 20000, seed: int = 0, eps_start: float = 0.4,
                    eps_end: float = 0.05, c_w: float = 0.05, c_b: float = 0.05,
                    eval_every: int = 1000, checkpoint: bool = True,
                    verbose: bool = True) -> Tuple[XNN, list]:
    """Trains a fresh XNN purely via self-play. If checkpoint=True, returns
    the best-scoring snapshot seen during training (self-play performance
    against fixed external opponents is non-monotonic -- see paper Section
    6.2 -- so checkpointing is used to recover a stable deployable network)."""
    net = XNN(default_config(), seed=seed)
    rng = np.random.default_rng(seed + 5000)
    history = []
    best_score, best_W, best_bias = -1e9, None, None

    for g in range(games):
        epsilon = eps_start + (eps_end - eps_start) * (g / games)
        play_self_play_game(net, epsilon, rng, train=True, c_w=c_w, c_b=c_b)

        if g % eval_every == 0 or g == games - 1:
            wr_r, dr_r, lr_r = evaluate_vs_external(net, opponent_move_random, n_games=100, seed=999)
            wr_s, dr_s, lr_s = evaluate_vs_external(net, opponent_move_smart, n_games=100, seed=999)
            score = wr_r + wr_s - lr_r - lr_s
            history.append(dict(game=g, epsilon=epsilon, win_random=wr_r, draw_random=dr_r,
                                 loss_random=lr_r, win_smart=wr_s, draw_smart=dr_s,
                                 loss_smart=lr_s, score=score, saturation=net.saturation_fraction()))
            if checkpoint and score > best_score:
                best_score = score
                best_W, best_bias = net.W.copy(), net.bias.copy()
            if verbose:
                marker = " <- new best" if (checkpoint and score == best_score) else ""
                print(f"game {g:6d}  eps={epsilon:.2f}  vs_random W{wr_r:.2f}/L{lr_r:.2f}"
                      f"   vs_smart W{wr_s:.2f}/L{lr_s:.2f}   score={score:.2f}{marker}")

    if checkpoint and best_W is not None:
        net.W, net.bias = best_W, best_bias

    return net, history


# --------------------------------------------------------------------------
# Evaluation against fixed external opponents (evaluation only -- never used
# to produce a training signal)
# --------------------------------------------------------------------------

def opponent_move_random(board: List[int], rng: np.random.Generator) -> int:
    legal = [i for i, c in enumerate(board) if c == 0]
    return int(rng.choice(legal))


def opponent_move_smart(board: List[int], rng: np.random.Generator) -> int:
    """One-ply lookahead: takes an immediate win if available, else random."""
    legal = [i for i, c in enumerate(board) if c == 0]
    for m in legal:
        trial = board.copy()
        trial[m] = 2
        if check_winner(trial) == 2:
            return m
    return int(rng.choice(legal))


def evaluate_vs_external(net: XNN, opponent_fn, n_games: int = 300, seed: int = 999
                          ) -> Tuple[float, float, float]:
    """Network always plays X against the given opponent function playing O.
    Greedy (epsilon=0) policy. Returns (win_rate, draw_rate, loss_rate)."""
    rng = np.random.default_rng(seed)
    n_input = net.config.n_input
    wins, draws, losses = 0, 0, 0

    for _ in range(n_games):
        board = [0] * 9
        state = net.new_state()
        turn = 1
        while True:
            if turn == 1:
                legal = [i for i, c in enumerate(board) if c == 0]
                input_vec = canonicalize(board, 1, n_input)
                move, state, _, _, _ = net.choose_action(state, input_vec, legal, 0.0, rng)
                board[move] = 1
            else:
                move = opponent_fn(board, rng)
                board[move] = 2
                input_vec = canonicalize(board, 1, n_input)
                state, _ = net.tick(state, input_vec, net.config.steps_per_decision, track=False)

            result = check_winner(board)
            if result is not None:
                if result == 1:
                    wins += 1
                elif result == 0:
                    draws += 1
                else:
                    losses += 1
                break
            turn = 2 if turn == 1 else 1

    return wins / n_games, draws / n_games, losses / n_games


# --------------------------------------------------------------------------
# CLI entry point: train, evaluate, save
# --------------------------------------------------------------------------

if __name__ == "__main__":
    import json
    import sys

    games = int(sys.argv[1]) if len(sys.argv) > 1 else 20000
    seed = int(sys.argv[2]) if len(sys.argv) > 2 else 0
    out_path = sys.argv[3] if len(sys.argv) > 3 else "model.json"

    net, history = train_selfplay(games=games, seed=seed)

    print("\n=== Final evaluation (checkpointed best network), 500 games each ===")
    wr_r, dr_r, lr_r = evaluate_vs_external(net, opponent_move_random, n_games=500, seed=42)
    wr_s, dr_s, lr_s = evaluate_vs_external(net, opponent_move_smart, n_games=500, seed=42)
    print(f"vs random opponent: win={wr_r:.3f} draw={dr_r:.3f} loss={lr_r:.3f}")
    print(f"vs smart  opponent: win={wr_s:.3f} draw={dr_s:.3f} loss={lr_s:.3f}")
    print(f"final weight saturation: {net.saturation_fraction():.3f}")

    net.save(out_path)
    with open(out_path.replace(".json", "_history.json"), "w") as f:
        json.dump(history, f, indent=2)
    print(f"\nSaved model to {out_path}")
