"""Agent-portfolio cascade — the deployable form of the win-union.

No single agent beats this game reliably: AlphaZero nets and MCTS both plateau
around 50-72% (a *coverage* limit, not a strength wall — every seed is winnable,
but different agents win *different* seeds). The remedy is not a stronger single
agent; it is to combine agents that fail on *disjoint* seeds.

This game makes that deployable, not just an oracle bound: a loss is OBSERVABLE
(the agent dies, ``hp`` -> 0) and the env is deterministic and resettable per
seed. So a **cascade** realises the union exactly — order the agents best-first,
replay the seed with each until one wins:

    for agent in agents_ordered_by_strength:
        if agent wins the seed: stop        # cost = agents tried so far
    else: lost                              # cost = len(agents)

A seed is won iff *some* agent wins it (= the union of their win-sets), and the
cost is the number of agents replayed until the first win — you pay compute,
never an oracle (you never need to know in advance which agent wins a seed).
Because most seeds fall to the best agent first-try, the mean cost stays near
1, so a portfolio costs ~1.3-1.4x a single agent, not Nx.

Measured (2026-07-18) on two independent held-out seed windows (16-47, 48-79),
the default portfolio — two AlphaZero nets at 1000 sims plus a few MCTS@300
rollout-RNG voters — reaches **union 32/32 (100%)** on both, at ~1.34-1.44x the
cost of one agent. The AZ nets miss 2-3 orphan seeds each window; the MCTS RNG
voters, each winning a *different* ~40%, cover them. See ``report`` for the
per-window analysis this reproduces (individual win rates, orphan coverage,
cascade cost, and the minimal RNG covering subset — the "cost floor").

Whole-game diversity only: per-move vote-averaging FAILS here (it destroys the
commitment a ~60-move winning line needs — measured 4/8), so the portfolio
combines agents at whole-game granularity, never per move.

CLI::

    # run every agent fresh on a held-out window, dump per-seed win-sets:
    python -m src.ai.portfolio --seed-offset 16 --seeds 32 \\
        --nets src/ai/az.pt src/ai/az_deep2.pt \\
        --az-sims 1000 --mcts-rngs 3 --json tmp/portfolio_16.json

    # recompute the analysis from a saved dump (no games replayed):
    python -m src.ai.portfolio --from-json tmp/portfolio_16.json

This module contains no game logic; it drives ``Game`` only through the policy
glue. Requires the optional ``ai`` dependency group (``torch``).
"""
from __future__ import annotations

import argparse
import itertools
import json
import multiprocessing as mp
import os
from dataclasses import dataclass

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

DEFAULT_NETS = ["src/ai/az.pt", "src/ai/az_deep2.pt"]


@dataclass(frozen=True)
class Agent:
    """One portfolio member. ``kind`` picks the search; ``ref`` is a net path
    (AZ) or a rollout-RNG seed (MCTS); ``sims`` is its per-move budget."""
    name: str
    kind: str          # "az" | "mcts"
    ref: str | int
    sims: int


def default_portfolio(nets: list[str], az_sims: int, mcts_sims: int,
                      mcts_rngs: int) -> list[Agent]:
    """The proven portfolio: each AZ net at ``az_sims``, then ``mcts_rngs``
    MCTS voters at ``mcts_sims`` (one per rollout RNG, each winning a different
    subset). AZ nets are the strong members; the MCTS RNGs cover the orphans."""
    agents = [Agent(f"{os.path.basename(n)}@{az_sims}", "az", n, az_sims)
              for n in nets]
    agents += [Agent(f"mcts@{mcts_sims}/rng{r}", "mcts", r, mcts_sims)
               for r in range(mcts_rngs)]
    return agents


# ---------------------------------------------------------------------------
# Running (one deterministic game per worker process)
# ---------------------------------------------------------------------------
_NET_CACHE: dict = {}


def _play_task(task: tuple) -> tuple[str, int, bool]:
    """Play one (agent, seed) game and return (agent_name, seed, won).

    Deterministic given the agent and seed: AZ eval takes the most-visited root
    action with Dirichlet noise disabled, and MCTS is seeded — so parallel
    results are identical to serial, only faster. torch is imported here (not at
    module top) so importing this module never pulls in the ``ai`` group."""
    kind, name, ref, sims, seed = task
    import torch
    torch.set_num_threads(1)
    from src.ai.policy import run_episode_decomposed
    if kind == "az":
        from src.ai.alphazero import az_action_fn
        from src.ai.policy import load_policy
        if ref not in _NET_CACHE:
            _NET_CACHE[ref] = load_policy(ref)
        act = az_action_fn(_NET_CACHE[ref], simulations=sims)
    else:
        from src.ai.mcts import mcts_action_fn
        act = mcts_action_fn(simulations=sims, seed=ref)
    _total, won, _steps, _parts = run_episode_decomposed(act, seed)
    return name, seed, won


def run_portfolio(agents: list[Agent], seeds: list[int],
                  workers: int = 11) -> dict[str, set[int]]:
    """Play every (agent, seed) game in a pool; return each agent's win-set."""
    win_sets: dict[str, set[int]] = {a.name: set() for a in agents}
    tasks = [(a.kind, a.name, a.ref, a.sims, s) for a in agents for s in seeds]
    n = len(tasks)
    print(f"=== PORTFOLIO | {len(agents)} agents x {len(seeds)} seeds = {n} "
          f"games | {workers} workers ===", flush=True)
    with mp.Pool(workers) as pool:
        done = 0
        for name, seed, won in pool.imap_unordered(_play_task, tasks):
            if won:
                win_sets[name].add(seed)
            done += 1
            if done % 16 == 0 or done == n:
                print(f"  [{done:3d}/{n}] ...", flush=True)
    return win_sets


# ---------------------------------------------------------------------------
# Analysis (pure — reusable on saved win-sets)
# ---------------------------------------------------------------------------
def cascade_stats(win_sets: dict[str, set[int]],
                  seeds: list[int]) -> tuple[set[int], float]:
    """Return (union win-set, mean agents-replayed-per-seed). Agents are tried
    best-first (descending individual win count); cost is the rank of the first
    agent that wins a seed, or ``len(agents)`` on a seed nobody wins."""
    order = sorted(win_sets, key=lambda a: len(win_sets[a]), reverse=True)
    union: set[int] = set().union(*win_sets.values()) if win_sets else set()
    costs = [next((i for i, a in enumerate(order, 1) if s in win_sets[a]),
                  len(order)) for s in seeds]
    return union, (sum(costs) / len(costs) if costs else 0.0)


def minimal_cover(voter_sets: dict[str, set[int]],
                  orphans: list[int]) -> tuple[int, tuple[str, ...]]:
    """Smallest subset of voters whose union covers every orphan (set-cover,
    brute-forced — the portfolio has a handful of voters). Returns (k, names).
    This is the optimal-selection cost floor; the blind-selection floor is
    larger because a randomly-drawn voter may cover no orphan at all."""
    names = list(voter_sets)
    for k in range(len(names) + 1):
        for combo in itertools.combinations(names, k):
            cov = set().union(*(voter_sets[c] for c in combo)) if combo else set()
            if all(o in cov for o in orphans):
                return k, combo
    return len(names), tuple(names)


def report(win_sets: dict[str, set[int]], az_names: list[str],
           seeds: list[int]) -> None:
    """Print the full portfolio analysis: individual win rates, the AZ-only
    union and its orphan seeds, which voter covers each orphan, the cascade
    win rate + cost, and the minimal covering subset (the cost floor)."""
    n = len(seeds)
    az_sets = {nm: win_sets[nm] for nm in az_names}
    voter_sets = {nm: win_sets[nm] for nm in win_sets if nm not in az_names}

    print("\n=== individual (held-out, "
          f"{n} seeds) ===")
    for nm in sorted(win_sets, key=lambda a: len(win_sets[a]), reverse=True):
        print(f"  {nm:32s} {len(win_sets[nm]):2d}/{n} "
              f"({100*len(win_sets[nm])/n:4.1f}%)")

    az_union = set().union(*az_sets.values()) if az_sets else set()
    orphans = [s for s in seeds if s not in az_union]
    print(f"\n=== AZ-only union = {len(az_union)}/{n} | "
          f"orphans {orphans} ===")
    for nm in voter_sets:
        covers = [o for o in orphans if o in voter_sets[nm]]
        print(f"  {nm} covers orphans {covers}")

    union, cost = cascade_stats(win_sets, seeds)
    orphans_all = [s for s in seeds if s not in union]
    print(f"\n=== cascade (whole-game union) = {len(union)}/{n} "
          f"({100*len(union)/n:.1f}%) ===")
    print(f"  seeds NO agent wins: {orphans_all}")
    print(f"  mean agents replayed / seed: {cost:.2f}  "
          f"(a single best agent: "
          f"{max(len(v) for v in win_sets.values())}/{n} at cost 1.00)")

    k, combo = minimal_cover(voter_sets, orphans)
    solo = [nm for nm in voter_sets
            if all(o in voter_sets[nm] for o in orphans)]
    print(f"\n=== cost floor (minimal voter subset covering the orphans) ===")
    print(f"  minimal covering subset: k={k} via {list(combo)}")
    print(f"  voters that alone cover all orphans (k=1 works): {solo or 'none'}")


def _load(path: str) -> tuple[dict[str, set[int]], list[str], list[int]]:
    """Load a dumped win-set JSON back into (win_sets, az_names, seeds)."""
    d = json.load(open(path))
    seeds = d["seeds"]
    win_sets = {nm: set(v) for nm, v in d["az"].items()}
    az_names = list(d["az"])
    for r, v in d["mcts"].items():
        win_sets[f"mcts/rng{r}"] = set(v)
    return win_sets, az_names, seeds


def _dump(path: str, win_sets: dict[str, set[int]], az_names: list[str],
          seeds: list[int]) -> None:
    az = {nm: sorted(win_sets[nm]) for nm in az_names}
    mcts = {nm.split("rng")[-1]: sorted(win_sets[nm])
            for nm in win_sets if nm not in az_names}
    json.dump({"seeds": seeds, "az": az, "mcts": mcts},
              open(path, "w"), indent=0)
    print(f"per-seed win-sets -> {path}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--from-json", metavar="PATH",
                   help="recompute the analysis from a saved dump (no games)")
    p.add_argument("--seed-offset", type=int, default=16,
                   help="first held-out seed index (gate selects on [:16])")
    p.add_argument("--seeds", type=int, default=32)
    p.add_argument("--nets", nargs="+", default=DEFAULT_NETS,
                   help="AlphaZero checkpoints (the strong portfolio members)")
    p.add_argument("--az-sims", type=int, default=1000)
    p.add_argument("--mcts-sims", type=int, default=300)
    p.add_argument("--mcts-rngs", type=int, default=3,
                   help="MCTS voters, one per rollout RNG")
    p.add_argument("--json", metavar="PATH", help="dump per-seed win-sets here")
    p.add_argument("--workers", type=int, default=11)
    a = p.parse_args()

    if a.from_json:
        win_sets, az_names, seeds = _load(a.from_json)
        report(win_sets, az_names, seeds)
        return

    from src.ai.policy import eval_seed
    seeds = [eval_seed(i) for i in range(a.seed_offset, a.seed_offset + a.seeds)]
    agents = default_portfolio(a.nets, a.az_sims, a.mcts_sims, a.mcts_rngs)
    az_names = [ag.name for ag in agents if ag.kind == "az"]
    win_sets = run_portfolio(agents, seeds, a.workers)
    report(win_sets, az_names, seeds)
    if a.json:
        _dump(a.json, win_sets, az_names, seeds)


if __name__ == "__main__":
    main()
