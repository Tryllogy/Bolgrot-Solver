"""Scripted baseline policies and a reward-decomposition benchmark.

Balancing the ``REWARD_*`` weights (see ``constant.py``) is about getting an
*ordering* right, not absolute numbers: winning must beat clearing-then-dying,
which must beat sitting still. This module makes that ordering observable.

It scores a handful of hand-written policies — plus, optionally, a trained
network — over the fixed :data:`EVAL_SEEDS`, and prints each policy's mean
return *broken down by reward term* (:data:`REWARD_PARTS`). Read the table two
ways:

* down the ``return`` column — the ordering invariant should hold
  (``win`` policy ≫ ``greedy`` ≫ ``survive``/``end_turn`` ≫ ``random``);
* across a row — in a policy that is *not* winning, whichever term dominates is
  the term your weights are over-paying. If ``approach`` or ``cast`` outweighs
  ``clear``/``terminal``, the agent is being paid to fidget.

Run: ``uv run --extra ai python -m src.ai.baselines [--policy src/ai/ppo.pt]``.
"""
from __future__ import annotations

import argparse
import os
import random

from ..game import Game, REWARD_PARTS
from .policy import (
    ActionFn,
    EVAL_SEEDS,
    eval_seed,
    net_action_fn,
    run_episode_decomposed,
)


def _end_turn_index(game: Game) -> int:
    """Index of the end-turn action (``(None, (0, 0))``) in ``ACTIONS``."""
    for i, (spell_index, _) in enumerate(game.ACTIONS):
        if spell_index is None:
            return i
    raise ValueError("no end-turn action in ACTIONS")


def _first_legal(game: Game) -> int:
    """Fallback: the first legal action index (games always have one live)."""
    for i, ok in enumerate(game.legal_actions()):
        if ok:
            return i
    return 0


# --- Scripted policies (each an ActionFn: Game -> action index) -------------

def always_end_turn(game: Game) -> int:
    """Never cast — just end every turn (the passive "do nothing" floor)."""
    return _end_turn_index(game)


def random_legal(seed: int = 0) -> ActionFn:
    """A policy that picks uniformly among the currently legal actions."""
    rng = random.Random(seed)

    def act(game: Game) -> int:
        legal = [i for i, ok in enumerate(game.legal_actions()) if ok]
        return rng.choice(legal) if legal else 0

    return act


def greedy_clear(game: Game) -> int:
    """Jump onto a flame when possible, else jump toward the nearest one.

    Only the two jumps (ShortJump/LongJump) reposition the player, so only they
    can land on — and thereby remove — a flame. When no flame is reachable this
    turn we end it to advance the wave counter toward the win condition.
    """
    legal = game.legal_actions()
    px, py = game.player.pos_x, game.player.pos_y
    flames = set(game._flame_positions())

    best: tuple[tuple[int, int], int] | None = None
    for idx, ok in enumerate(legal):
        if not ok:
            continue
        spell_index, (dx, dy) = game.ACTIONS[idx]
        if spell_index not in (0, 1):        # only the player-moving jumps
            continue
        target = (px + dx, py + dy)
        lands_on_flame = target in flames
        remaining = list(flames - {target})
        dist_after = Game._nearest_target_dist(remaining, target)
        # Prefer landing on a flame (removes it); then minimise the distance to
        # whatever flame is left. Tuple compares lexicographically.
        key = (0 if lands_on_flame else 1, dist_after)
        if best is None or key < best[0]:
            best = (key, idx)

    if best is not None and flames:
        return best[1]
    return _end_turn_index(game) if legal[_end_turn_index(game)] \
        else _first_legal(game)


# --- Benchmark --------------------------------------------------------------

def score_policy(
    act_fn: ActionFn,
    seeds: list[int] = EVAL_SEEDS,
) -> dict[str, float]:
    """Mean return, win rate, length and per-term reward for a policy."""
    totals = 0.0
    wins = 0
    steps = 0
    parts = {part: 0.0 for part in REWARD_PARTS}
    for seed in seeds:
        total, won, n, ep_parts = run_episode_decomposed(act_fn, seed)
        totals += total
        wins += int(won)
        steps += n
        for part, value in ep_parts.items():
            parts[part] += value
    n_seeds = len(seeds)
    row = {"return": totals / n_seeds,
           "win%": 100.0 * wins / n_seeds,
           "steps": steps / n_seeds}
    row.update({part: parts[part] / n_seeds for part in REWARD_PARTS})
    return row


def _print_table(rows: dict[str, dict[str, float]]) -> None:
    """Pretty-print the policy × reward-term table, sorted by return."""
    cols = ["return", "win%", "steps", *REWARD_PARTS]
    name_w = max(len("policy"), *(len(n) for n in rows))
    header = "policy".ljust(name_w) + "  " + \
        "  ".join(c.rjust(8) for c in cols)
    print(header)
    print("-" * len(header))
    for name, row in sorted(rows.items(),
                            key=lambda kv: kv[1]["return"], reverse=True):
        line = name.ljust(name_w) + "  " + \
            "  ".join(f"{row[c]:8.2f}" for c in cols)
        print(line)


def benchmark(policy_path: str | None = None,
              seeds: list[int] = EVAL_SEEDS,
              mcts_sims: int | None = None,
              az_path: str | None = None,
              az_sims: int = 120,
              az_value_mode: str = "shaped") -> dict[str, dict[str, float]]:
    """Score every baseline (plus a trained policy if given); print a table.

    ``mcts_sims`` (opt-in — the MCTS agent is orders of magnitude slower than
    the scripted baselines) adds a lookahead-planning row with that per-move
    simulation budget; pair it with a small ``seeds`` subset. ``az_path`` adds
    an AlphaZero row (net-guided PUCT at ``az_sims`` sims/move) — also slow.
    """
    policies: dict[str, ActionFn] = {
        "end_turn": always_end_turn,
        "random": random_legal(seed=0),
        "greedy": greedy_clear,
    }
    if policy_path and os.path.exists(policy_path):
        from .policy import load_policy
        net = load_policy(policy_path)
        policies["trained"] = net_action_fn(net, deterministic=True)
    elif policy_path:
        print(f"(no policy at {policy_path}; skipping trained baseline)\n")
    if mcts_sims:
        from .mcts import mcts_action_fn
        policies["mcts"] = mcts_action_fn(simulations=mcts_sims)
    if az_path and os.path.exists(az_path):
        from .policy import load_policy
        from .alphazero import az_action_fn
        policies["alphazero"] = az_action_fn(load_policy(az_path),
                                             simulations=az_sims,
                                             value_mode=az_value_mode)
    elif az_path:
        print(f"(no AZ policy at {az_path}; skipping alphazero row)\n")

    rows = {name: score_policy(fn, seeds) for name, fn in policies.items()}
    _print_table(rows)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy", default="src/ai/ppo.pt",
                        help="trained policy checkpoint to include (optional)")
    parser.add_argument("--mcts-sims", type=int, default=None,
                        help="add an MCTS row with this per-move sim budget "
                             "(slow; use --seeds to shrink the seed set)")
    parser.add_argument("--seeds", type=int, default=None,
                        help="score on N EVAL_SEEDS only (default: all), "
                             "starting at --seed-offset")
    parser.add_argument("--seed-offset", type=int, default=0,
                        help="start the --seeds window at this index. The "
                             "training gate selects the saved net on "
                             "EVAL_SEEDS[:16], which *contains* the default "
                             "--seeds 6 window, so benchmarking a gate-picked "
                             "net there scores it on the seeds it was chosen "
                             "on (measured inflation: 83%% vs 37.5%% for one "
                             "net). Use --seed-offset 16 for seeds no gate "
                             "ever saw; the window may run past EVAL_SEEDS's "
                             "32 entries (seeds are generated, not listed), so "
                             "`--seeds 32 --seed-offset 16` is a 32-seed "
                             "held-out set. Prefer that to `--seeds 8`: at 8 "
                             "seeds a 2-win gap is indistinguishable from "
                             "noise (measured 2026-07-16), which makes small "
                             "real effects unreadable.")
    parser.add_argument("--az", default=None,
                        help="add an AlphaZero row from this policy "
                             "checkpoint (net-guided PUCT; slow, use --seeds)")
    parser.add_argument("--az-sims", type=int, default=120,
                        help="PUCT sims/move for the --az row")
    parser.add_argument("--az-value-mode",
                        choices=("shaped", "winloss", "blend", "twohead"),
                        default="shaped",
                        help="value mode the --az net was trained with; the "
                             "eval search must match (a P(win) head searched "
                             "with shaped backup is incoherent)")
    parser.add_argument("--reward-win", type=float, default=None,
                        help="override constant.REWARD_WIN to match a net "
                             "trained with `az_train --reward-win` (the search "
                             "backup uses env rewards, so the eval must use the "
                             "same value; the win *rate* is unaffected, but the "
                             "search behaviour and the reward columns are not)")
    parser.add_argument("--value-scale", type=float, default=None,
                        help="override alphazero.VALUE_SCALE to match a net "
                             "trained with `az_train --value-scale` (the eval "
                             "search scales the value head back up by it)")
    args = parser.parse_args()
    if args.reward_win is not None:
        from .. import constant
        constant.REWARD_WIN = args.reward_win
    if args.value_scale is not None:
        from . import alphazero
        alphazero.VALUE_SCALE = args.value_scale
    start = args.seed_offset
    seeds = ([eval_seed(i) for i in range(start, start + args.seeds)]
             if args.seeds else EVAL_SEEDS[start:])
    benchmark(args.policy, seeds=seeds, mcts_sims=args.mcts_sims,
              az_path=args.az, az_sims=args.az_sims,
              az_value_mode=args.az_value_mode)


if __name__ == "__main__":
    main()
