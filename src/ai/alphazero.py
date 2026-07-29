"""AlphaZero-style PUCT search + self-play trainer (Phase 2 of the planner).

The pure-planning MCTS in :mod:`~src.ai.mcts` wins the full game but is slow:
it rebuilds a fresh tree with random/greedy *rollouts* every move. AlphaZero
replaces the two costly, hand-tuned parts of that search with a learned
:class:`~src.ai.network.ActorCritic`:

* the **rollout** becomes a single value-head evaluation of the leaf (``V(s)``,
  the return-to-go estimate), so no simulation to the end is needed;
* the **UCB1 exploration** becomes **PUCT**, whose exploration term is weighted
  by the policy head's prior ``P(a|s)`` — the net biases the search toward the
  moves it already likes, so far fewer simulations reach the same strength.

The net is trained by **self-play**: play games choosing moves in proportion to
the search's visit counts, then regress the policy head onto those visit
distributions (the search is a stronger policy than the raw net — this is the
policy-improvement step) and the value head onto the discounted return-to-go.

Two carry-overs from :mod:`~src.ai.mcts` that this game forces:

* **Shaped value, not win/loss.** Wins are far too deep for a fresh value head
  to label leaves usefully, so the value target is the *discounted shaped
  return-to-go* (same quantity PPO's value head already represents), which
  exposes ``clear`` (+10/flame) and the terminal signals. Because that return
  is unbounded and large (a full win is ~+470), PUCT normalises ``Q`` with a
  running :class:`_MinMaxStats` (MuZero's trick) so ``c_puct`` stays scaleless.
* **A curriculum.** An untrained value head is noise, so on the full game early
  self-play has no more signal than PPO did. Self-play therefore runs on the
  same difficulty ramp (:data:`~src.ai.train.DEFAULT_CURRICULUM`), promoting a
  stage once the search reliably wins it.

Everything is driven through the env API and the existing
:mod:`~src.ai.policy` helpers, so :func:`az_action_fn` is an
:data:`~src.ai.policy.ActionFn` that drops straight into ``baselines`` /
``play`` / ``evaluate`` like every other policy here.
"""
from __future__ import annotations

import math
import os
import random
from collections import deque
from dataclasses import dataclass, field

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn, optim

from ..entity import Player
from ..game import Game
from .. import constant
from .network import ActorCritic, MASK_FILL, build_net
from .policy import (
    ActionFn,
    EVAL_SEEDS,
    format_reward_parts,
    save_policy,
)


# Value-target scale. The value head is trained on shaped returns, which run to
# O(100) (REWARD_WIN alone is +100); squared, that MSE is O(10^4), whose
# gradient — after the global grad-norm clip — swamps the O(1) policy
# cross-entropy and leaves the policy head at its uniform init (measured: prior
# entropy within 1% of uniform, p_loss flat, so PUCT loses its prior and
# degenerates to unguided search). We divide the value *target* by VALUE_SCALE
# so both losses are O(1) and share the trunk's gradient fairly; the head's
# output is then in units of VALUE_SCALE, so the search multiplies it back up
# to raw reward units before mixing it with per-edge rewards in the backup.
VALUE_SCALE = 100.0


# Value-target modes (the `--value-target` A/B). "shaped" is the historical
# recipe: the value head regresses the discounted *shaped* return-to-go and the
# search backs up per-step shaped rewards. Both 64- and 128-wide nets plateau at
# it (= MCTS ~50%) because that target is really a *flame-count* predictor —
# clears (+10 each, up to +360) swamp REWARD_WIN (+100), so a leaf that "clears
# 30 then dies" scores nearly as high as one that wins, and PUCT can't rank the
# winning branch above "die late clearing more".
#
# "winloss" replaced the target and the search backup with the discounted *game
# outcome* (+1 win / -1 loss, terminal only). Measured (2026-07-13): it climbs
# the curriculum but *collapses on the full game* — self-play never wins there,
# so every full-game target is -1, the value head confidently predicts "all
# lost", and with the dense shaped guidance gone the search has no gradient to
# find flames and degenerates to passing turns (stall -480, 0% for 26 straight
# iters). Sparse outcome removes the exploration signal the game needs.
#
# "blend" keeps the dense shaped reward (so the search still explores toward
# flames) and *adds* a large terminal outcome bonus ±BLEND_TERMINAL_BONUS to
# both the search edges and the value target — enough to lift a win clear of the
# best near-win, which is the separation shaped alone lacks. Dense guidance of
# shaped + the win/near-win ranking winloss was after, without winloss's sparsity.
# Measured (2026-07-13): blend avoids winloss's passive collapse (stall stayed
# modest, clear ~100, self-play won the full game ~12%) but at equal budget
# scores 0/6 — *worse* than shaped's 3/6. The terminal ±300 spike folded into
# the *single* shaped value target poisons the value estimate near a terminal
# (a state one avoidable-death away is dragged toward -300·gamma), so the net
# learns to avoid death without learning to finish; it clears fewer flames than
# plain shaped.
#
# "twohead" is the fix for that: keep the shaped value target *clean* (no
# terminal spike) on one value head, and learn P(win) (the ±1 outcome) on a
# *second* head. The search combines them at a leaf as
# ``V_shaped·VALUE_SCALE + TWOHEAD_WIN_WEIGHT·P(win)`` and adds the same
# ±TWOHEAD_WIN_WEIGHT to a terminal edge, so a win still outranks the best
# near-win (the winloss/blend goal) — but the ranking pressure lives in the
# bounded P(win) head and never distorts the dense shaped estimate that drives
# exploration. Needs a fresh run: the value head has 2 outputs, so the width is
# baked into the checkpoint (value_dim), like `hidden`.
VALUE_MODES = ("shaped", "winloss", "blend", "twohead")

# Raw-reward bonus added to the terminal step in "blend" mode (+win / -loss).
# Sized above the clear-count spread (~±360) so the outcome dominates the value
# head's clear-count noise — the whole point is that a win outranks "clear 32
# then die". The value scale is raised in step so the (larger) target stays O(1).
BLEND_TERMINAL_BONUS = 300.0
BLEND_VALUE_SCALE = 300.0

# "twohead" mixing weight λ: raw-reward weight on the P(win) head in the leaf
# value and on the terminal edge. Same magnitude as BLEND_TERMINAL_BONUS (above
# the clear spread) so a near-certain win lifts a leaf clear of the best
# near-win. The shaped head keeps VALUE_SCALE; the P(win) head is O(1) already.
TWOHEAD_WIN_WEIGHT = 300.0


def _value_scale(value_mode: str) -> float:
    """Target/leaf scale for a value mode.

    Shaped returns are O(100) so they are divided by :data:`VALUE_SCALE` to keep
    the value loss O(1); outcome targets are already O(1), so no scaling; blend
    returns reach ~±770 (shaped + the terminal bonus) so they use a larger scale.
    """
    if value_mode == "winloss":
        return 1.0
    if value_mode == "blend":
        return BLEND_VALUE_SCALE
    return VALUE_SCALE


# --- Q-value normalisation (MuZero-style) -----------------------------------

class _MinMaxStats:
    """Running min/max of backed-up Q-values to normalise PUCT exploitation.

    Shaped returns here span roughly ``[-40, +470]``; mapping the observed
    range onto ``[0, 1]`` keeps the PUCT exploration constant ``c_puct``
    independent of the reward scale.
    """

    __slots__ = ("lo", "hi")

    def __init__(self) -> None:
        self.lo = math.inf
        self.hi = -math.inf

    def update(self, value: float) -> None:
        self.lo = min(self.lo, value)
        self.hi = max(self.hi, value)

    def normalize(self, value: float) -> float:
        if self.hi > self.lo:
            return (value - self.lo) / (self.hi - self.lo)
        return value


# --- Search tree ------------------------------------------------------------

class _Node:
    """A game state in the PUCT tree (a snapshot plus per-edge statistics).

    Edge stats are kept as dicts keyed by action index over the legal actions
    at this state. A node is *expanded* the first time it is evaluated by the
    net (priors filled in); terminal nodes are never expanded.
    """

    __slots__ = ("state", "done", "won", "expanded", "legal",
                 "P", "N", "W", "edge_reward", "children")

    def __init__(self, game: Game) -> None:
        self.state = game.get_state()
        self.done = game.done
        self.won = game.won
        self.expanded = False
        self.legal: list[int] = []
        self.P: dict[int, float] = {}
        self.N: dict[int, int] = {}
        self.W: dict[int, float] = {}
        self.edge_reward: dict[int, float] = {}
        self.children: dict[int, "_Node"] = {}


def _leaf_value(value: torch.Tensor, value_mode: str) -> float:
    """Raw-reward leaf value from the net's value output for ``value_mode``.

    Single-head modes ("shaped"/"winloss"/"blend") scale the scalar head back up
    to raw reward units. "twohead" combines the two heads:
    ``V_shaped·VALUE_SCALE + TWOHEAD_WIN_WEIGHT·P(win)`` — the dense shaped
    estimate plus a bounded win-ranking term.
    """
    if value_mode == "twohead":
        return (float(value[0]) * VALUE_SCALE
                + TWOHEAD_WIN_WEIGHT * float(value[1]))
    return float(value) * _value_scale(value_mode)


@torch.no_grad()
def _evaluate(node: _Node, work: Game, net: ActorCritic,
              value_mode: str = "shaped") -> float:
    """Expand ``node`` with the net's masked priors; return its value estimate.

    ``work`` must already be at ``node``'s state. Fills ``node.P/N/W`` over
    the legal actions and marks it expanded; returns ``V(s)`` (the
    return-to-go / outcome estimate used as the leaf value in backup).
    """
    obs = torch.as_tensor(work.get_observation(), dtype=torch.float32)
    mask = torch.as_tensor(work.legal_actions(), dtype=torch.bool)
    logits, value = net(obs)
    probs = torch.softmax(logits.masked_fill(~mask, MASK_FILL), dim=-1)
    node.legal = [i for i, ok in enumerate(mask.tolist()) if ok]
    for a in node.legal:
        node.P[a] = float(probs[a])
        node.N[a] = 0
        node.W[a] = 0.0
    node.expanded = True
    # Head(s) trained in scaled units; the backup mixes this leaf value with
    # per-edge rewards in raw units, so convert back to raw units here.
    return _leaf_value(value, value_mode)


def _puct_select(node: _Node, c_puct: float, minmax: _MinMaxStats) -> int:
    """Pick the action maximising the PUCT score at ``node``.

    ``Q(a) + c_puct·P(a)·sqrt(1+ΣN)/(1+N(a))`` with ``Q`` min-max normalised.
    Unvisited edges take ``Q=0`` and rely on the prior-weighted exploration
    term, so the first simulations follow the policy head.
    """
    total = sum(node.N.values())
    sqrt_total = math.sqrt(1 + total)
    best_a = node.legal[0]
    best_score = -math.inf
    for a in node.legal:
        n = node.N[a]
        q = node.W[a] / n if n else 0.0
        u = c_puct * node.P[a] * sqrt_total / (1 + n)
        score = minmax.normalize(q) + u
        if score > best_score:
            best_score, best_a = score, a
    return best_a


def _simulate(root: _Node, work: Game, net: ActorCritic,
              c_puct: float, gamma: float, minmax: _MinMaxStats,
              value_mode: str = "shaped") -> None:
    """One PUCT iteration: select to a leaf, expand+evaluate, back up return.

    Selection uses only cached edge stats (no stepping) until it chooses an
    action with no child yet; that action is expanded by stepping ``work`` once
    and evaluating the new leaf with the net. The discounted return
    (edge rewards + leaf value) then updates every edge on the path.

    In ``"winloss"`` mode the per-edge reward is the discounted *outcome*
    (``+1``/``-1`` on the terminal step, ``0`` otherwise) rather than the shaped
    step reward, and the leaf value is the net's P(win) at unit scale — so the
    whole backup measures "does this line win", matching the value target.
    In ``"blend"`` mode edges stay shaped (dense guidance) but a terminal step
    gets an extra ``±BLEND_TERMINAL_BONUS`` so wins outrank near-wins.
    ``"twohead"`` likewise keeps shaped edges and adds ``±TWOHEAD_WIN_WEIGHT``
    on a terminal step, matching the P(win) leaf term's scale.
    """
    node = root
    path: list[tuple[_Node, int]] = []
    leaf_value = 0.0

    while node.expanded and not node.done:
        a = _puct_select(node, c_puct, minmax)
        path.append((node, a))
        child = node.children.get(a)
        if child is not None:
            node = child
            continue
        # Expand this edge: step the working game once and evaluate the leaf.
        work.set_state(node.state)
        reward, _ = work.step(a)
        if value_mode == "winloss":
            reward = (1.0 if work.won else -1.0) if work.done else 0.0
        elif value_mode == "blend" and work.done:
            reward += BLEND_TERMINAL_BONUS if work.won else -BLEND_TERMINAL_BONUS
        elif value_mode == "twohead" and work.done:
            reward += TWOHEAD_WIN_WEIGHT if work.won else -TWOHEAD_WIN_WEIGHT
        node.edge_reward[a] = reward
        child = _Node(work)
        node.children[a] = child
        leaf_value = (0.0 if child.done
                      else _evaluate(child, work, net, value_mode))
        break
    # Loop fell through without expanding (terminal reached): leaf stays 0.

    # Backup: fold edge rewards into the leaf value from the bottom up.
    g = leaf_value
    for parent, a in reversed(path):
        g = parent.edge_reward[a] + gamma * g
        parent.N[a] += 1
        parent.W[a] += g
        minmax.update(parent.W[a] / parent.N[a])


def _visit_counts(root: _Node) -> dict[int, int]:
    return {a: root.N[a] for a in root.legal}


# --- ActionFn (evaluation / play) -------------------------------------------

def az_action_fn(net: ActorCritic, simulations: int = 100, c_puct: float = 1.5,
                 gamma: float = 0.99, seed: int = 0,
                 value_mode: str = "shaped") -> ActionFn:
    """Wrap ``net`` as an :data:`ActionFn` that picks moves by PUCT search.

    Deterministic (most-visited root action), no root noise — the evaluation /
    viewer form. Reuses the same search as self-play with exploration disabled.
    ``value_mode`` must match the net's training mode (a P(win) head searched
    with shaped backup, or vice-versa, is incoherent).
    """
    rng = random.Random(seed)

    def act(game: Game) -> int:
        if game.done:
            return 0
        root, _ = _search(game, net, simulations, c_puct, gamma, rng,
                          value_mode=value_mode)
        counts = _visit_counts(root)
        if not counts:
            legal = game.legal_actions()
            return next((i for i, ok in enumerate(legal) if ok), 0)
        return max(counts, key=lambda a: counts[a])

    return act


def _search(game: Game, net: ActorCritic, simulations: int, c_puct: float,
            gamma: float, rng: random.Random,
            dirichlet_alpha: float = 0.0,
            value_mode: str = "shaped") -> tuple[_Node, _MinMaxStats]:
    """Run ``simulations`` PUCT iterations rooted at ``game``; return root."""
    work = game.clone()
    root = _Node(work)
    _evaluate(root, work, net, value_mode)
    if dirichlet_alpha > 0.0 and len(root.legal) > 1:
        samples = [rng.gammavariate(dirichlet_alpha, 1.0) for _ in root.legal]
        s = sum(samples) or 1.0
        for a, g in zip(root.legal, samples):
            root.P[a] = 0.75 * root.P[a] + 0.25 * (g / s)
    minmax = _MinMaxStats()
    for _ in range(simulations):
        _simulate(root, work, net, c_puct, gamma, minmax, value_mode)
    return root, minmax


# --- Gumbel self-play (optional "stronger" lever) ---------------------------
# Danihelka et al. 2022, "Policy improvement by planning with Gumbel". Below the
# root this stays ordinary PUCT; at the root it does Gumbel-Top-k candidate
# selection + sequential halving, and — the point for TRAINING — it produces an
# *improved-policy* target (softmax of logit + sigma(completedQ)) that is less
# noisy than visit counts at the modest self-play sim budget. All nine prior
# training levers kept plain-PUCT self-play with a visit-count target; this is
# the one that changes the policy-target quality itself. Off by default
# (``use_gumbel=False``) — the standard path above is untouched.

def _gumbel_noise(rng: random.Random) -> float:
    u = rng.random()
    return -math.log(-math.log(u + 1e-12) + 1e-12)


def _simulate_forced(root: _Node, work: Game, net: ActorCritic, forced_a: int,
                     c_puct: float, gamma: float, minmax: _MinMaxStats,
                     value_mode: str) -> None:
    """One simulation whose FIRST root action is ``forced_a`` (below root:
    ordinary PUCT). Mirrors :func:`_simulate` in every other respect."""
    node = root
    path: list[tuple[_Node, int]] = []
    leaf_value = 0.0
    first = True
    while node.expanded and not node.done:
        a = forced_a if first else _puct_select(node, c_puct, minmax)
        first = False
        path.append((node, a))
        child = node.children.get(a)
        if child is not None:
            node = child
            continue
        work.set_state(node.state)
        reward, _ = work.step(a)
        if value_mode == "winloss":
            reward = (1.0 if work.won else -1.0) if work.done else 0.0
        elif value_mode == "blend" and work.done:
            reward += BLEND_TERMINAL_BONUS if work.won else -BLEND_TERMINAL_BONUS
        elif value_mode == "twohead" and work.done:
            reward += TWOHEAD_WIN_WEIGHT if work.won else -TWOHEAD_WIN_WEIGHT
        node.edge_reward[a] = reward
        child = _Node(work)
        node.children[a] = child
        leaf_value = (0.0 if child.done
                      else _evaluate(child, work, net, value_mode))
        break
    g = leaf_value
    for parent, a in reversed(path):
        g = parent.edge_reward[a] + gamma * g
        parent.N[a] += 1
        parent.W[a] += g
        minmax.update(parent.W[a] / parent.N[a])


def _gumbel_search(game: Game, net: ActorCritic, simulations: int, m: int,
                   c_puct: float, gamma: float, rng: random.Random,
                   num_actions: int, value_mode: str,
                   c_visit: float = 50.0, c_scale: float = 1.0
                   ) -> tuple[int, list[float]]:
    """Gumbel-root search. Returns ``(action, improved_pi)``.

    ``improved_pi`` (length ``num_actions``) is the policy target: softmax over
    legal actions of ``logit(a) + sigma(completedQ(a))`` with **no** Gumbel noise
    (noise is for action selection only). ``completedQ`` uses the net's root
    value for unvisited actions (value completion), so every legal action gets a
    coherent target even at a small sim budget.
    """
    work = game.clone()
    root = _Node(work)
    v_root = _evaluate(root, work, net, value_mode)
    legal = root.legal
    pi = [0.0] * num_actions
    if not legal:
        return 0, pi
    if len(legal) == 1:
        pi[legal[0]] = 1.0
        return legal[0], pi

    obs = torch.as_tensor(work.get_observation(), dtype=torch.float32)
    with torch.no_grad():
        logits_full, _ = net(obs)
    logit = {a: float(logits_full[a]) for a in legal}
    gv = {a: _gumbel_noise(rng) for a in legal}

    m = min(m, len(legal))
    cand = sorted(legal, key=lambda a: logit[a] + gv[a], reverse=True)[:m]
    minmax = _MinMaxStats()

    def cq(a: int) -> float:
        n = root.N[a]
        return root.W[a] / n if n else v_root

    phases = max(1, math.ceil(math.log2(m)))
    remaining = cand
    for _ in range(phases):
        per = max(1, simulations // (phases * len(remaining)))
        for a in remaining:
            for _ in range(per):
                _simulate_forced(root, work, net, a, c_puct, gamma, minmax,
                                 value_mode)
        if len(remaining) == 1:
            break
        max_n = max((root.N[a] for a in cand if root.N[a]), default=1)
        remaining = sorted(
            remaining,
            key=lambda a: gv[a] + logit[a]
            + (c_visit + max_n) * c_scale * minmax.normalize(cq(a)),
            reverse=True)[:max(1, len(remaining) // 2)]

    max_n = max((root.N[a] for a in cand if root.N[a]), default=1)
    action = max(remaining, key=lambda a: gv[a] + logit[a]
                 + (c_visit + max_n) * c_scale * minmax.normalize(cq(a)))

    scores = [logit[a] + (c_visit + max_n) * c_scale * minmax.normalize(cq(a))
              for a in legal]
    mx = max(scores)
    exps = [math.exp(s - mx) for s in scores]
    z = sum(exps) or 1.0
    for a, e in zip(legal, exps):
        pi[a] = e / z
    return action, pi


# --- Negamax operator (optional "Stockfish-style" self-play/deploy) ---------
# Depth-limited MAX-backup (1-player, so no min ply), selective to the top-k
# children by the net's policy prior; net value*scale at the horizon; dominant
# ±terminal so a forced win/death within d plies is chosen/avoided. Unlike PUCT
# it *backs up* the value exactly (no averaging), and unlike the beam solver it
# is the search AZ trains AGAINST when search="negamax": the eval is then
# co-adapted to it (the true Stockfish test — an evaluator trained FOR a deep
# backup search, not borrowed from a PUCT run). Deploy/eval via nm_action_fn.
_NM_TERMINAL = 1e5


@torch.no_grad()
def _negamax_value(work: Game, state: dict, depth: int, k: int,
                   net: ActorCritic, gamma: float, value_mode: str) -> float:
    work.set_state(state)
    obs = torch.as_tensor(work.get_observation(), dtype=torch.float32)
    logits, value = net(obs)
    if depth == 0:
        return _leaf_value(value, value_mode)
    mask = work.legal_actions()
    m = torch.as_tensor(mask, dtype=torch.bool)
    probs = torch.softmax(logits.masked_fill(~m, MASK_FILL), dim=-1)
    legal = [i for i, ok in enumerate(mask) if ok]
    topk = sorted(legal, key=lambda a: float(probs[a]), reverse=True)[:k]
    best = -1e18
    for a in topk:
        work.set_state(state)
        r, _ = work.step(a)
        if work.done:
            v = r + (_NM_TERMINAL if work.won else -_NM_TERMINAL)
        else:
            v = r + gamma * _negamax_value(work, work.get_state(), depth - 1,
                                           k, net, gamma, value_mode)
        if v > best:
            best = v
    return best


@torch.no_grad()
def _negamax_root(game: Game, net: ActorCritic, depth: int, k: int,
                  gamma: float, value_mode: str) -> dict[int, float]:
    """Backed-up value of each top-k root action. Empty dict if no legal move."""
    work = game.clone()
    root = work.get_state()
    obs = torch.as_tensor(work.get_observation(), dtype=torch.float32)
    logits, _ = net(obs)
    mask = work.legal_actions()
    m = torch.as_tensor(mask, dtype=torch.bool)
    probs = torch.softmax(logits.masked_fill(~m, MASK_FILL), dim=-1)
    legal = [i for i, ok in enumerate(mask) if ok]
    topk = sorted(legal, key=lambda a: float(probs[a]), reverse=True)[:k]
    q: dict[int, float] = {}
    for a in topk:
        work.set_state(root)
        r, _ = work.step(a)
        if work.done:
            q[a] = r + (_NM_TERMINAL if work.won else -_NM_TERMINAL)
        else:
            q[a] = r + gamma * _negamax_value(work, work.get_state(), depth - 1,
                                              k, net, gamma, value_mode)
    return q


def _negamax_policy(q: dict[int, float], num_actions: int) -> list[float]:
    """Peaked-but-soft policy target from backed-up Q (numerically safe)."""
    pi = [0.0] * num_actions
    if not q:
        return pi
    acts = list(q)
    qs = [q[a] for a in acts]
    mx = max(qs)
    scale = max(1.0, (mx - min(qs)) / 4.0)
    exps = [math.exp((v - mx) / scale) for v in qs]
    z = sum(exps) or 1.0
    for a, e in zip(acts, exps):
        pi[a] = e / z
    return pi


def nm_action_fn(net: ActorCritic, depth: int = 5, k: int = 4,
                 gamma: float = 0.99, value_mode: str = "shaped") -> ActionFn:
    """Deploy/eval form: play the argmax backed-up negamax action."""
    def act(game: Game) -> int:
        if game.done:
            return 0
        q = _negamax_root(game, net, depth, k, gamma, value_mode)
        if not q:
            legal = game.legal_actions()
            return next((i for i, ok in enumerate(legal) if ok), 0)
        return max(q, key=lambda a: q[a])
    return act


# --- Self-play --------------------------------------------------------------

@dataclass
class _Sample:
    """One training example: state, legal mask, search policy, value target(s).

    ``value`` is the value-head target (return-to-go); ``value_win`` is the
    second head's P(win) target, used only by ``value_mode="twohead"`` (left at
    ``0.0`` and ignored by the single-head modes).
    """

    obs: list[float]
    mask: list[bool]
    pi: list[float]           # visit distribution over ALL actions
    value: float = 0.0        # filled with return-to-go after the game
    value_win: float = 0.0    # twohead only: discounted ±1 outcome target


def _discounted_returns(stream: list[float], gamma: float) -> list[float]:
    """Return-to-go of ``stream`` under ``gamma`` (same length as ``stream``)."""
    g = 0.0
    returns: list[float] = [0.0] * len(stream)
    for t in reversed(range(len(stream))):
        g = stream[t] + gamma * g
        returns[t] = g
    return returns


def _fill_value_targets(samples: list["_Sample"], rewards: list[float],
                        won: bool, gamma: float, value_mode: str) -> None:
    """Fill each sample's value target(s) with the discounted return-to-go.

    ``"shaped"`` discounts the per-step shaped ``rewards``; ``"winloss"``
    discounts a stream that is ``0`` everywhere except the terminal step, which
    carries the game outcome (``+1`` won / ``-1`` otherwise) — so a sample's
    target is ``±gamma**(plies-to-end)``, a pure P(win) signal. ``"blend"``
    keeps the shaped stream but adds ``±BLEND_TERMINAL_BONUS`` to the terminal
    step, so the target is the dense shaped return lifted by a strong outcome.
    ``"twohead"`` fills *both* targets: ``value`` = the *clean* shaped return
    (no terminal spike) and ``value_win`` = the discounted ±1 outcome, so the
    two heads learn the dense estimate and P(win) independently.
    """
    outcome = 1.0 if won else -1.0
    if value_mode == "twohead":
        shaped_returns = _discounted_returns(rewards, gamma)
        win_stream = [0.0] * len(rewards)
        if win_stream:
            win_stream[-1] = outcome
        win_returns = _discounted_returns(win_stream, gamma)
        for i, sample in enumerate(samples):
            sample.value = shaped_returns[i] if i < len(shaped_returns) else 0.0
            sample.value_win = win_returns[i] if i < len(win_returns) else 0.0
        return

    if value_mode == "winloss":
        stream = [0.0] * len(rewards)
        if stream:
            stream[-1] = outcome
    elif value_mode == "blend":
        stream = list(rewards)
        if stream:
            stream[-1] += BLEND_TERMINAL_BONUS if won else -BLEND_TERMINAL_BONUS
    else:
        stream = rewards
    returns = _discounted_returns(stream, gamma)
    for i, sample in enumerate(samples):
        sample.value = returns[i] if i < len(returns) else 0.0


def self_play_game(net: ActorCritic, seed: int, simulations: int,
                   c_puct: float,
                   gamma: float, rng: random.Random, num_actions: int,
                   num_waves: int | None = None,
                   flames_per_wave: int | None = None,
                   spawn_radius: int | None = None,
                   spawn_formation: str | None = None,
                   dirichlet_alpha: float = 0.3,
                   temp_moves: int = 12,
                   value_mode: str = "shaped",
                   use_gumbel: bool = False,
                   gumbel_m: int = 16,
                   search: str = "puct",
                   nm_depth: int = 5,
                   nm_k: int = 4,
                   max_steps: int = 500) -> tuple[list[_Sample], bool]:
    """Play one self-play game; return ``(samples, won)``.

    Each move runs a PUCT search with root Dirichlet noise. The move is sampled
    from the visit counts (temperature 1 for the first ``temp_moves`` plies,
    greedy after) so early exploration is broad but the endgame is sharp. The
    per-move visit distribution is the policy target; value targets are the
    discounted return-to-go (shaped) or game outcome (``value_mode="winloss"``),
    filled in once the game ends.
    """
    game = Game(Player(*constant.BASE_PLAYER_POS), seed=seed,
                num_waves=num_waves, flames_per_wave=flames_per_wave,
                spawn_radius=spawn_radius, spawn_formation=spawn_formation)
    samples: list[_Sample] = []
    rewards: list[float] = []

    for ply in range(max_steps):
        if game.done:
            break
        if search == "negamax":
            q = _negamax_root(game, net, nm_depth, nm_k, gamma, value_mode)
            if not q:
                legal = game.legal_actions()
                action = next((i for i, ok in enumerate(legal) if ok), 0)
                reward, _ = game.step(action)
                rewards.append(reward)
                continue
            pi = _negamax_policy(q, num_actions)
            samples.append(_Sample(obs=game.get_observation(),
                                   mask=game.legal_actions(), pi=pi))
            if ply < temp_moves:
                acts = list(q)
                weights = [pi[a] for a in acts]
                action = rng.choices(acts, weights=weights, k=1)[0]
            else:
                action = max(q, key=lambda a: q[a])
            reward, _ = game.step(action)
            rewards.append(reward)
            continue
        if use_gumbel:
            action, pi = _gumbel_search(game, net, simulations, gumbel_m,
                                        c_puct, gamma, rng, num_actions,
                                        value_mode)
            if not any(pi):
                legal = game.legal_actions()
                action = next((i for i, ok in enumerate(legal) if ok), 0)
                reward, _ = game.step(action)
                rewards.append(reward)
                continue
            samples.append(_Sample(obs=game.get_observation(),
                                   mask=game.legal_actions(), pi=pi))
            reward, _ = game.step(action)
            rewards.append(reward)
            continue
        root, _ = _search(game, net, simulations, c_puct, gamma, rng,
                          dirichlet_alpha=dirichlet_alpha,
                          value_mode=value_mode)
        counts = _visit_counts(root)
        if not counts:
            legal = game.legal_actions()
            action = next((i for i, ok in enumerate(legal) if ok), 0)
            reward, _ = game.step(action)
            rewards.append(reward)
            continue
        pi = [0.0] * num_actions
        total = sum(counts.values())
        for a, n in counts.items():
            pi[a] = n / total
        samples.append(_Sample(obs=game.get_observation(),
                               mask=game.legal_actions(), pi=pi))
        if ply < temp_moves:
            actions = list(counts)
            weights = [counts[a] for a in actions]
            action = rng.choices(actions, weights=weights, k=1)[0]
        else:
            action = max(counts, key=lambda a: counts[a])
        reward, _ = game.step(action)
        rewards.append(reward)

    # Value target for every recorded state (samples are recorded before their
    # move, so sample i owns rewards[i:]).
    _fill_value_targets(samples, rewards, game.won, gamma, value_mode)
    return samples, game.won


def mcts_bootstrap_game(seed: int, simulations: int, gamma: float,
                        rng: random.Random, num_actions: int,
                        num_waves: int | None = None,
                        flames_per_wave: int | None = None,
                        spawn_radius: int | None = None,
                        spawn_formation: str | None = None,
                        rollout_depth: int = 16,
                        value_mode: str = "shaped",
                        max_steps: int = 500) -> tuple[list[_Sample], bool]:
    """Play one game with **MCTS** and record it as AlphaZero training samples.

    Self-play bootstraps the net off its own search, which only works while that
    search sometimes wins: on the full game ``selfplay_win`` sits at 0%, so the
    policy target is the visit distribution of a player who always dies and the
    value head only ever fits losing shaped returns. MCTS wins ~50% of full
    games at 300 sims, so seeding the buffer with *its* trajectories gives the
    net winning lines to imitate — the missing ingredient, not more iterations.

    Same ``_Sample`` layout as :func:`self_play_game` (visit distribution as the
    policy target, discounted return-to-go as the value target); no network is
    involved, and moves are greedy on visits (no Dirichlet noise — these games
    are meant to be strong, not exploratory).
    """
    from .mcts import mcts_root_visits

    game = Game(Player(*constant.BASE_PLAYER_POS), seed=seed,
                num_waves=num_waves, flames_per_wave=flames_per_wave,
                spawn_radius=spawn_radius, spawn_formation=spawn_formation)
    samples: list[_Sample] = []
    rewards: list[float] = []

    for _ in range(max_steps):
        if game.done:
            break
        counts = mcts_root_visits(game, simulations, rng,
                                  rollout_depth=rollout_depth, gamma=gamma)
        if not counts:
            break
        pi = [0.0] * num_actions
        total = sum(counts.values())
        for a, n in counts.items():
            pi[a] = n / total
        samples.append(_Sample(obs=game.get_observation(),
                               mask=game.legal_actions(), pi=pi))
        action = max(counts, key=lambda a: counts[a])
        reward, _ = game.step(action)
        rewards.append(reward)

    _fill_value_targets(samples, rewards, game.won, gamma, value_mode)
    return samples, game.won


def az_bootstrap_game(net: ActorCritic, seed: int, simulations: int,
                      gamma: float, rng: random.Random, num_actions: int,
                      num_waves: int | None = None,
                      flames_per_wave: int | None = None,
                      spawn_radius: int | None = None,
                      spawn_formation: str | None = None,
                      c_puct: float = 1.5,
                      value_mode: str = "shaped",
                      max_steps: int = 500) -> tuple[list[_Sample], bool]:
    """Play one game with a **fixed strong teacher net's PUCT search** and record
    it as AlphaZero training samples — the AZ analogue of
    :func:`mcts_bootstrap_game`.

    The teacher is a separate, frozen net searched at high ``simulations`` (e.g.
    ``az.pt`` @4000), *not* the net being trained. It is a stronger and more
    diverse teacher than MCTS@300 — measured held-out on 32 seeds, ``az.pt``@1000
    = 72% and @4000 = 88% versus MCTS@300 = 44%, and it wins seeds MCTS never does
    — which is exactly what the project's "the teacher is the ceiling" diagnosis
    asks for (every prior lever bootstrapped from a ~50% teacher and plateaued
    there). Distilling this teacher's visit distribution and winning lines is the
    one untried, well-motivated flywheel step: it can only raise the student's
    prior toward the teacher's search, shifting the whole budget curve left (a
    given win rate reached at fewer sims/move — a *faster* net), though a prior
    cannot fully substitute for the teacher's extra search depth.

    Same ``_Sample`` layout as :func:`self_play_game` and
    :func:`mcts_bootstrap_game` (visit distribution as the policy target,
    discounted return-to-go as the value target). Greedy on visits, no Dirichlet
    noise — these games are meant to be strong, not exploratory. ``value_mode``
    must match the student's training mode (and the teacher net's).
    """
    game = Game(Player(*constant.BASE_PLAYER_POS), seed=seed,
                num_waves=num_waves, flames_per_wave=flames_per_wave,
                spawn_radius=spawn_radius, spawn_formation=spawn_formation)
    samples: list[_Sample] = []
    rewards: list[float] = []

    for _ in range(max_steps):
        if game.done:
            break
        root, _ = _search(game, net, simulations, c_puct, gamma, rng,
                          value_mode=value_mode)
        counts = _visit_counts(root)
        if not counts:
            break
        pi = [0.0] * num_actions
        total = sum(counts.values())
        for a, n in counts.items():
            pi[a] = n / total
        samples.append(_Sample(obs=game.get_observation(),
                               mask=game.legal_actions(), pi=pi))
        action = max(counts, key=lambda a: counts[a])
        reward, _ = game.step(action)
        rewards.append(reward)

    _fill_value_targets(samples, rewards, game.won, gamma, value_mode)
    return samples, game.won


# --- Parallel game workers ---------------------------------------------------
# Self-play games, MCTS bootstrap slots and gate-eval games are all independent
# given a fixed net snapshot, and one such game saturates exactly one core (the
# bottleneck is the Python search loop; torch gains nothing from threads on an
# MLP this small). A persistent spawn Pool runs one game per core. The net
# travels by file (saved once per iteration, ~100 KB) because a spawn worker
# shares no memory; workers cache it keyed by (path, version). Module-global
# overrides (VALUE_SCALE, constant.REWARD_WIN — mutated by az_train flags in
# the parent) do not survive spawn either, so they are re-applied per worker in
# the initializer. Everything below must stay module-level picklable.

_WORKER_NET: dict = {}


def _worker_init(value_scale: float, reward_win: float) -> None:
    os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
    torch.set_num_threads(1)
    global VALUE_SCALE
    VALUE_SCALE = value_scale
    constant.REWARD_WIN = reward_win


def _worker_load_net(path: str, version: int) -> ActorCritic:
    if _WORKER_NET.get("key") != (path, version):
        from .policy import load_policy
        _WORKER_NET["net"] = load_policy(path)
        _WORKER_NET["key"] = (path, version)
    return _WORKER_NET["net"]


# The AZ bootstrap teacher is a *frozen* net (never re-shipped), so it gets its
# own cache slot — sharing `_WORKER_NET` would thrash against the training net
# that self-play/eval reload every iteration.
_WORKER_TEACHER: dict = {}


def _worker_load_teacher(path: str) -> ActorCritic:
    if _WORKER_TEACHER.get("path") != path:
        from .policy import load_policy
        _WORKER_TEACHER["net"] = load_policy(path)
        _WORKER_TEACHER["path"] = path
    return _WORKER_TEACHER["net"]


def _selfplay_task(args: tuple) -> tuple[list[_Sample], bool]:
    (net_path, version, seed, sims, c_puct, gamma, num_actions, stage,
     dirichlet_alpha, temp_moves, value_mode, use_gumbel, gumbel_m,
     search, nm_depth, nm_k) = args
    net = _worker_load_net(net_path, version)
    nw, fpw, sr, fm = stage
    return self_play_game(
        net, seed=seed, simulations=sims, c_puct=c_puct, gamma=gamma,
        rng=random.Random(seed), num_actions=num_actions,
        num_waves=nw, flames_per_wave=fpw, spawn_radius=sr, spawn_formation=fm,
        dirichlet_alpha=dirichlet_alpha, temp_moves=temp_moves,
        value_mode=value_mode, use_gumbel=use_gumbel, gumbel_m=gumbel_m,
        search=search, nm_depth=nm_depth, nm_k=nm_k)


def _bootstrap_task(args: tuple) -> tuple[list[_Sample], bool, int]:
    """One multi-seed bootstrap slot: re-roll seeds until MCTS wins (cap
    1+retries), return ``(samples_of_last_attempt, won, attempts)``."""
    slot_seed, retries, sims, gamma, num_actions, stage, value_mode = args
    rng = random.Random(slot_seed)
    nw, fpw, sr, fm = stage
    samples: list[_Sample] = []
    won = False
    attempts = 0
    for _ in range(1 + max(0, retries)):
        samples, won = mcts_bootstrap_game(
            seed=rng.randrange(1 << 30), simulations=sims, gamma=gamma,
            rng=rng, num_actions=num_actions,
            num_waves=nw, flames_per_wave=fpw, spawn_radius=sr,
            spawn_formation=fm, value_mode=value_mode)
        attempts += 1
        if won:
            break
    return samples, won, attempts


def _az_bootstrap_task(args: tuple) -> tuple[list[_Sample], bool, int]:
    """One AZ-teacher bootstrap slot: re-roll seeds until the frozen teacher net
    wins (cap 1+retries), return ``(samples_of_last_attempt, won, attempts)``.
    The teacher is loaded once per worker and cached (it never changes)."""
    (slot_seed, retries, teacher_path, sims, c_puct, gamma, num_actions, stage,
     value_mode) = args
    net = _worker_load_teacher(teacher_path)
    rng = random.Random(slot_seed)
    nw, fpw, sr, fm = stage
    samples: list[_Sample] = []
    won = False
    attempts = 0
    for _ in range(1 + max(0, retries)):
        samples, won = az_bootstrap_game(
            net, seed=rng.randrange(1 << 30), simulations=sims, gamma=gamma,
            rng=rng, num_actions=num_actions,
            num_waves=nw, flames_per_wave=fpw, spawn_radius=sr,
            spawn_formation=fm, c_puct=c_puct, value_mode=value_mode)
        attempts += 1
        if won:
            break
    return samples, won, attempts


def _eval_task(args: tuple) -> tuple[float, bool, dict]:
    (net_path, version, seed, sims, c_puct, gamma, value_mode, stage,
     search, nm_depth, nm_k) = args
    from .policy import run_episode_decomposed
    net = _worker_load_net(net_path, version)
    if search == "negamax":
        act = nm_action_fn(net, depth=nm_depth, k=nm_k, gamma=gamma,
                           value_mode=value_mode)
    else:
        act = az_action_fn(net, simulations=sims, c_puct=c_puct, gamma=gamma,
                           value_mode=value_mode)
    nw, fpw, sr, fm = stage
    total, won, _steps, parts = run_episode_decomposed(
        act, seed, num_waves=nw, flames_per_wave=fpw, spawn_radius=sr,
        spawn_formation=fm)
    return total, won, parts


# --- Trainer ----------------------------------------------------------------

@dataclass
class AZConfig:
    """Hyperparameters for an AlphaZero self-play run."""

    iterations: int = 100           # self-play + update cycles this session
    games_per_iter: int = 8         # self-play games collected per iteration
    simulations: int = 80           # PUCT sims per move during self-play
    bootstrap_games: int = 0        # MCTS games seeded into the buffer, once
    bootstrap_simulations: int = 300  # MCTS sims/move for those games
    # Multi-seed MCTS teacher. Single-seed MCTS wins ~50% of full games, but
    # *which* 3/6 it wins is set by the rollout RNG — no seed is unwinnable, a
    # different seed wins a different one (measured: aggregating configs, every
    # eval seed is won by some run). So a plain bootstrap only seeds the buffer
    # with the ~half of games MCTS happens to win. With retries > 0, each
    # bootstrap slot re-rolls a fresh game seed until MCTS *wins* (cap 1+retries
    # attempts), keeping only the winning line — converting "unlucky loss" seeds
    # into winning trajectories so the win buffer covers the pattern space, not
    # just the lucky half. 0 = off (byte-for-byte the old single-attempt path).
    bootstrap_win_retries: int = 0  # extra MCTS seeds per slot to obtain a win
    # Bootstrap teacher: "mcts" (the default, ~44-50% at 300 sims) or "az" — a
    # frozen strong AZ net (``bootstrap_teacher_path``) searched at
    # ``bootstrap_simulations`` sims/move. The AZ teacher is the flywheel step:
    # az.pt@1000/4000 is a *stronger and more diverse* teacher than MCTS@300
    # (72%/88% vs 44% held-out, and wins seeds MCTS never does), which is exactly
    # what "the teacher is the ceiling" asks for — every prior lever bootstrapped
    # from a ~50% teacher and plateaued there. "mcts" is byte-for-byte the old
    # path (empty teacher path).
    bootstrap_teacher: str = "mcts"   # "mcts" | "az"
    bootstrap_teacher_path: str = ""  # frozen AZ teacher net (bootstrap_teacher="az")
    # Gate-eval search budget. None = evaluate at `simulations`, i.e. the budget
    # the net actually plays and is benchmarked at. The old fixed 120 was the
    # gate's central defect: the metric both *understates* the net (line stage:
    # 2/16 at 120 vs 16/16 at 300 for the same net) and is noise-dominated
    # (0% -> 50% -> 6% across consecutive evals), so over ~20 evals the gate
    # reliably banked a lucky immature snapshot over the mature final (measured
    # three times: iter-95 9/32 vs final 11/32; iter-240 3/8 held-out sold as
    # 83%; iter-145 banked over iter-300). Evaluating at the real budget was
    # always the fix — it just cost too much serially; with `workers` the
    # parallel eval makes it affordable.
    eval_simulations: int | None = None
    # Game-level parallelism. Self-play games, bootstrap slots and eval games
    # are independent given the net snapshot, and the whole trainer is otherwise
    # single-core. 0 = auto (cpu_count - 1), 1 = serial (the historical path,
    # byte-for-byte). Parallel runs draw one RNG seed per game from the trainer
    # RNG, so they are seeded/reproducible but not stream-identical to serial.
    workers: int = 0
    c_puct: float = 1.5
    gamma: float = 0.99
    dirichlet_alpha: float = 0.3    # root exploration noise
    temp_moves: int = 12            # plies sampled from visits before greedy
    buffer_size: int = 20_000       # replay buffer capacity (samples)
    # Persistent win buffer (the "buffer lever"). The main buffer is a rolling
    # window: wins are ~0.3% of samples and evict within ~40 iters, so their
    # signal is washed out (measured: reward/ranking variants clear ~24 flames
    # but win 0/12 — the net learns to play, never to finish). A separate
    # win-only buffer that never evicts, from which a fixed fraction of every
    # minibatch is drawn, keeps winning trajectories (self-play *and* the MCTS
    # bootstrap's) permanently represented. 0.0 = off (byte-for-byte the old
    # behaviour: no win buffer maintained, minibatch is the plain rolling draw).
    win_buffer_fraction: float = 0.0  # share of each batch from the win buffer
    win_buffer_size: int = 40_000     # win-buffer capacity (>> a run's wins)
    batch_size: int = 128           # SGD minibatch (skips update until filled)
    update_steps: int = 64          # SGD minibatches per iteration
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    value_coef: float = 1.0         # value-loss weight relative to policy loss
    value_mode: str = "shaped"      # "shaped"|"winloss"|"blend"|"twohead"
    # Gumbel self-play (the "stronger" lever). When True, self-play selects moves
    # by Gumbel-root search and records the *improved-policy* target (softmax of
    # logit + sigma(completedQ)) instead of the visit distribution — a
    # lower-noise policy target at the self-play sim budget. Off = byte-for-byte
    # the old PUCT/visit-count self-play. Eval/bootstrap are unaffected.
    use_gumbel: bool = False
    gumbel_m: int = 16              # Gumbel-Top-k candidate count at the root
    # Search operator for self-play + eval. "puct" (default, unchanged) or
    # "negamax": depth-limited max-backup selective to top-k-by-prior. With
    # negamax the eval is co-trained FOR a deep backup search (the Stockfish
    # test). eval uses the same operator so the gate judges what it will deploy.
    search: str = "puct"           # "puct" | "negamax"
    nm_depth: int = 5              # negamax search depth
    nm_k: int = 4                  # negamax branching (top-k by prior)
    hidden: int = 64
    # Network type: "mlp" (flat obs, az.pt/deploy) or "cnn" (grid obs, needs
    # constant.OBS_MODE="grid"). Baked into the checkpoint like `hidden`, so a
    # cnn run needs a fresh --out and cannot resume an mlp checkpoint.
    net_type: str = "mlp"
    seed: int = 1
    eval_every: int = 5
    eval_seeds: list[int] = field(default_factory=lambda: EVAL_SEEDS[:16])
    device: str = "cpu"
    out_path: str = ""
    checkpoint_path: str = ""
    resume_from: str = ""
    curriculum: tuple[tuple[int | None, int | None, int | None], ...] = ()
    promote_win_rate: float = 0.6


def _winbuf_path(checkpoint_path: str) -> str:
    """Sidecar path for the persistent win buffer beside a checkpoint."""
    return checkpoint_path + ".winbuf"


def _save_az_checkpoint(path: str, net: ActorCritic,
                        optimizer: optim.Optimizer,
                        iteration: int, best_score: float,
                        stage_idx: int,
                        win_buffer: "deque | None" = None,
                        best_iteration: int = -1) -> None:
    """Resumable AZ training state (weights + optimizer + progress).

    When ``win_buffer`` is given (the buffer lever is on), its samples are
    persisted to a ``<path>.winbuf`` sidecar so a ``--resume`` carries the
    hard-won winning trajectories forward instead of rebuilding them from
    scratch each session. The samples store only ``(obs, mask, visit-π,
    return-to-go)`` — none depend on the net's weights, so a carried win stays
    valid training data (its value/policy targets don't go stale; only the
    state distribution is off-policy, as in any replay buffer).
    """
    torch.save(
        {
            "state_dict": net.state_dict(),
            "obs_size": net.obs_size,
            "num_actions": net.num_actions,
            "hidden": net.hidden,
            "value_dim": net.value_dim,
            "net_type": getattr(net, "net_type", "mlp"),
            "optimizer": optimizer.state_dict(),
            "iteration": iteration,
            "best_score": best_score,
            # Iteration of the snapshot the gate saved to out_path — WITHOUT
            # this, the only clue is decoding best_score, and a gate snapshot
            # got mislabeled with the final iteration once already (iter-145
            # named iter300, 2026-07-17).
            "best_iteration": best_iteration,
            "stage_idx": stage_idx,
        },
        path,
    )
    if win_buffer:
        torch.save(list(win_buffer), _winbuf_path(path))


def _load_win_buffer(checkpoint_path: str, win_buffer: "deque") -> int:
    """Restore a persisted win buffer (if any) into ``win_buffer``.

    Loaded with ``weights_only=False`` because the sidecar holds pickled
    ``_Sample`` objects, not tensors (it is our own file). Returns the number of
    samples added (0 if no sidecar exists)."""
    path = _winbuf_path(checkpoint_path)
    if not os.path.exists(path):
        return 0
    samples = torch.load(path, weights_only=False)
    win_buffer.extend(samples)
    return len(samples)


def _update(net: ActorCritic, optimizer: optim.Optimizer,
            buffer: deque, cfg: AZConfig, rng: random.Random,
            win_buffer: deque | None = None) -> tuple:
    """Run ``cfg.update_steps`` SGD steps; return mean (p_loss, v_loss).

    Policy loss is the cross-entropy of the net's masked policy against the
    search visit distribution; value loss is MSE against the return-to-go. In
    ``"twohead"`` mode the value loss sums the two heads' MSEs (shaped return on
    head 0, P(win) on head 1); the reported ``v_loss`` is that sum.

    When ``win_buffer`` is non-empty and ``cfg.win_buffer_fraction > 0``, that
    fraction of each minibatch is drawn (with replacement) from the persistent
    win buffer and the rest from the rolling ``buffer``; with fraction 0 or an
    empty win buffer the draw is the plain rolling sample (unchanged).
    """
    if len(buffer) < cfg.batch_size:
        return float("nan"), float("nan")
    net.train()
    twohead = cfg.value_mode == "twohead"
    data = list(buffer)
    win_data = list(win_buffer) if win_buffer else []
    n_win = round(cfg.batch_size * cfg.win_buffer_fraction) if win_data else 0
    n_main = cfg.batch_size - n_win
    p_tot = v_tot = 0.0
    for _ in range(cfg.update_steps):
        batch = rng.sample(data, n_main)
        if n_win:
            batch = batch + rng.choices(win_data, k=n_win)
        dev = cfg.device
        # np.asarray handles both obs layouts (flat list[float] and the grid
        # mode's 1-D float32 array) and avoids the slow tensor-from-list-of-
        # arrays path; result is (batch, obs_size) either way.
        obs = torch.as_tensor(
            np.asarray([s.obs for s in batch], dtype=np.float32), device=dev)
        mask = torch.tensor([s.mask for s in batch], dtype=torch.bool,
                            device=dev)
        pi = torch.tensor([s.pi for s in batch], dtype=torch.float32,
                          device=dev)
        logits, value = net(obs)
        logp = torch.log_softmax(logits.masked_fill(~mask, MASK_FILL), dim=-1)
        policy_loss = -(pi * logp).sum(dim=-1).mean()
        if twohead:
            # Head 0 regresses the clean shaped return (in VALUE_SCALE units);
            # head 1 regresses the discounted ±1 outcome (already O(1)).
            z_shaped = torch.tensor([s.value / VALUE_SCALE for s in batch],
                                    dtype=torch.float32, device=dev)
            z_win = torch.tensor([s.value_win for s in batch],
                                 dtype=torch.float32, device=dev)
            value_loss = (F.mse_loss(value[:, 0], z_shaped)
                          + F.mse_loss(value[:, 1], z_win))
        else:
            z = torch.tensor([s.value / _value_scale(cfg.value_mode)
                              for s in batch], dtype=torch.float32, device=dev)
            value_loss = F.mse_loss(value, z)
        loss = policy_loss + cfg.value_coef * value_loss
        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(net.parameters(), 0.5)
        optimizer.step()
        p_tot += policy_loss.item()
        v_tot += value_loss.item()
    return p_tot / cfg.update_steps, v_tot / cfg.update_steps


def train(cfg: AZConfig) -> ActorCritic:
    """Run AlphaZero self-play for ``cfg.iterations`` and return the network.

    Saves the best-by-eval policy to ``cfg.out_path`` (as :func:`save_policy`,
    so ``play``/``baselines`` load it like any policy) and a resumable AZ
    checkpoint to ``cfg.checkpoint_path``.
    """
    random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    rng = random.Random(cfg.seed)

    obs_size = Game.observation_size()
    num_actions = Game.num_actions()
    # "twohead" needs a second value output (shaped + P(win)); baked into the
    # checkpoint like `hidden`, so it cannot resume a single-head checkpoint.
    value_dim = 2 if cfg.value_mode == "twohead" else 1
    net = build_net(obs_size, num_actions, cfg.hidden,
                    value_dim=value_dim, net_type=cfg.net_type).to(cfg.device)
    optimizer = optim.Adam(net.parameters(), lr=cfg.learning_rate,
                           weight_decay=cfg.weight_decay)

    iter_offset = 0
    best_score = float("-inf")
    best_iter = -1
    stage_idx = 0
    if cfg.resume_from:
        if not os.path.exists(cfg.resume_from):
            raise SystemExit(f"No checkpoint to resume at {cfg.resume_from}.")
        ckpt = torch.load(cfg.resume_from, map_location=cfg.device,
                          weights_only=True)
        if (ckpt["obs_size"] != obs_size
                or ckpt["num_actions"] != num_actions
                or ckpt.get("hidden") != cfg.hidden
                or ckpt.get("value_dim", 1) != value_dim
                or ckpt.get("net_type", "mlp") != cfg.net_type):
            raise SystemExit("Checkpoint shape mismatch — train fresh.")
        net.load_state_dict(ckpt["state_dict"])
        if "optimizer" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer"])
        iter_offset = ckpt.get("iteration", 0)
        best_score = ckpt.get("best_score", float("-inf"))
        best_iter = ckpt.get("best_iteration", -1)
        stage_idx = ckpt.get("stage_idx", 0)
        print(f"Resumed from {cfg.resume_from}: iter {iter_offset}, "
              f"best {best_score:.2f} (iter {best_iter}), stage {stage_idx}",
              flush=True)

    stages: list[tuple[int | None, int | None, int | None]] = (
        list(cfg.curriculum) if cfg.curriculum else [(None, None, None)])
    stage_idx = min(stage_idx, len(stages) - 1)
    buffer: deque = deque(maxlen=cfg.buffer_size)
    # Persistent win-only buffer (the buffer lever); empty/unused when the
    # fraction is 0. Fed by winning self-play AND bootstrap games below.
    win_buffer: deque = deque(maxlen=cfg.win_buffer_size)
    use_win_buffer = cfg.win_buffer_fraction > 0.0
    # Carry a persisted win buffer across --resume: without this the "persistent"
    # win buffer is only persistent *within* a session and each resume re-warms
    # from an empty buffer (rebuilding ~30 iters of accumulated wins).
    restored = 0
    if cfg.resume_from and use_win_buffer:
        restored = _load_win_buffer(cfg.resume_from, win_buffer)
        if restored:
            print(f"Restored win buffer: {restored} samples from "
                  f"{_winbuf_path(cfg.resume_from)}", flush=True)

    # --- Worker pool (one game per core; see the worker section above) ----
    # Pool workers are daemonic, so a crashed/interrupted parent takes them
    # down with it; the normal path closes the pool before returning.
    workers = cfg.workers if cfg.workers else max(1, (os.cpu_count() or 2) - 1)
    pool = None
    net_file = ""
    net_version = 0
    if workers > 1:
        import multiprocessing as mp
        import tempfile
        fd, net_file = tempfile.mkstemp(prefix="az_worker_net_", suffix=".pt")
        os.close(fd)
        pool = mp.get_context("spawn").Pool(
            workers, initializer=_worker_init,
            initargs=(VALUE_SCALE, constant.REWARD_WIN))
        print(f"Worker pool: {workers} processes (one game per core)",
              flush=True)

    def ship_net() -> int:
        """Publish current weights for the workers; returns the version tag."""
        nonlocal net_version
        net_version += 1
        save_policy(net, net_file)
        return net_version

    def bootstrap(idx: int) -> None:
        """Seed the buffer with MCTS games at stage ``idx``.

        Run at start *and on every promotion*: a stage change swaps the game
        under the net, and the buffer is a rolling window, so the winning lines
        seeded for one stage are gone (and wrong) by the time the next is
        reached. The stage that actually needs this is the last one, where
        self-play never wins on its own.
        """
        if not cfg.bootstrap_games:
            return
        nw, fpw, sr, fm = (tuple(stages[idx]) + (None,))[:4]
        max_tries = 1 + max(0, cfg.bootstrap_win_retries)
        use_az = cfg.bootstrap_teacher == "az"
        wins = 0
        attempts = 0
        if pool is not None:
            # One slot per worker; the seed re-roll loop runs inside the task.
            if use_az:
                task_fn = _az_bootstrap_task
                tasks = [(rng.randrange(1 << 30), cfg.bootstrap_win_retries,
                          cfg.bootstrap_teacher_path, cfg.bootstrap_simulations,
                          cfg.c_puct, cfg.gamma, num_actions, (nw, fpw, sr, fm),
                          cfg.value_mode)
                         for _ in range(cfg.bootstrap_games)]
            else:
                task_fn = _bootstrap_task
                tasks = [(rng.randrange(1 << 30), cfg.bootstrap_win_retries,
                          cfg.bootstrap_simulations, cfg.gamma, num_actions,
                          (nw, fpw, sr, fm), cfg.value_mode)
                         for _ in range(cfg.bootstrap_games)]
            for samples, won, n_attempts in pool.map(task_fn, tasks):
                attempts += n_attempts
                if won:
                    buffer.extend(samples)
                    if use_win_buffer:
                        win_buffer.extend(samples)
                    wins += 1
                elif samples:
                    buffer.extend(samples)
        else:
            teacher = (_worker_load_teacher(cfg.bootstrap_teacher_path)
                       if use_az else None)
            for _ in range(cfg.bootstrap_games):
                # Multi-seed: re-roll fresh game seeds until the teacher wins (or
                # the retry budget is spent). Keep only the winning line; on
                # exhaustion fall back to the last attempt so the rolling buffer
                # still gets data.
                last_samples: list[_Sample] | None = None
                won = False
                for _try in range(max_tries):
                    if use_az:
                        samples, won = az_bootstrap_game(
                            teacher, seed=rng.randrange(1 << 30),
                            simulations=cfg.bootstrap_simulations,
                            gamma=cfg.gamma, rng=rng, num_actions=num_actions,
                            num_waves=nw, flames_per_wave=fpw, spawn_radius=sr,
                            spawn_formation=fm, c_puct=cfg.c_puct,
                            value_mode=cfg.value_mode)
                    else:
                        samples, won = mcts_bootstrap_game(
                            seed=rng.randrange(1 << 30),
                            simulations=cfg.bootstrap_simulations,
                            gamma=cfg.gamma, rng=rng, num_actions=num_actions,
                            num_waves=nw, flames_per_wave=fpw, spawn_radius=sr,
                            spawn_formation=fm, value_mode=cfg.value_mode)
                    attempts += 1
                    last_samples = samples
                    if won:
                        break
                if won:
                    buffer.extend(last_samples)
                    if use_win_buffer:
                        win_buffer.extend(last_samples)
                    wins += 1
                elif last_samples is not None:
                    buffer.extend(last_samples)
        teacher_tag = (f"AZ({os.path.basename(cfg.bootstrap_teacher_path)})"
                       if use_az else "MCTS")
        tag = (f" (multi-seed <={cfg.bootstrap_win_retries} retries, "
               f"{attempts} games)" if cfg.bootstrap_win_retries else "")
        print(f"Bootstrapped {cfg.bootstrap_games} {teacher_tag} slots{tag} at "
              f"{cfg.bootstrap_simulations} sims (stage {idx}): "
              f"{wins} won, buffer now {len(buffer)} samples", flush=True)

    # The startup bootstrap re-seeds the buffer with fresh MCTS wins — but on a
    # --resume that already carried a non-empty win buffer, those winning lines
    # are the *same* ones (MCTS targets don't go stale) and re-running the
    # (multi-seed, ~1h at stage 8) bootstrap just recomputes what we restored.
    # Skip it in that case; promotion bootstraps below still run (a new stage
    # genuinely needs fresh lines). Cold runs and non-carry resumes bootstrap as
    # before.
    if restored:
        print(f"Skipping startup bootstrap: win buffer carried "
              f"({restored} samples already seeded)", flush=True)
    else:
        bootstrap(stage_idx)

    for iteration in range(1, cfg.iterations + 1):
        total_iter = iter_offset + iteration
        # Stages are (waves, flames, radius) or (waves, flames, radius, shape).
        nw, fpw, sr, fm = (tuple(stages[stage_idx]) + (None,))[:4]

        # --- Self-play -------------------------------------------------
        wins = 0
        if pool is not None:
            version = ship_net()
            tasks = [(net_file, version, rng.randrange(1 << 30),
                      cfg.simulations, cfg.c_puct, cfg.gamma, num_actions,
                      (nw, fpw, sr, fm), cfg.dirichlet_alpha, cfg.temp_moves,
                      cfg.value_mode, cfg.use_gumbel, cfg.gumbel_m,
                      cfg.search, cfg.nm_depth, cfg.nm_k)
                     for _ in range(cfg.games_per_iter)]
            for samples, won in pool.map(_selfplay_task, tasks):
                buffer.extend(samples)
                if use_win_buffer and won:
                    win_buffer.extend(samples)
                wins += int(won)
        else:
            for g in range(cfg.games_per_iter):
                samples, won = self_play_game(
                    net, seed=rng.randrange(1 << 30),
                    simulations=cfg.simulations, c_puct=cfg.c_puct,
                    gamma=cfg.gamma, rng=rng, num_actions=num_actions,
                    num_waves=nw, flames_per_wave=fpw, spawn_radius=sr,
                    spawn_formation=fm,
                    dirichlet_alpha=cfg.dirichlet_alpha,
                    temp_moves=cfg.temp_moves,
                    value_mode=cfg.value_mode,
                    use_gumbel=cfg.use_gumbel, gumbel_m=cfg.gumbel_m,
                    search=cfg.search, nm_depth=cfg.nm_depth, nm_k=cfg.nm_k)
                buffer.extend(samples)
                if use_win_buffer and won:
                    win_buffer.extend(samples)
                wins += int(won)

        # --- Update ----------------------------------------------------
        p_loss, v_loss = _update(net, optimizer, buffer, cfg, rng,
                                 win_buffer if use_win_buffer else None)

        # --- Eval / checkpoint -----------------------------------------
        sp_win = wins / cfg.games_per_iter
        wbuf_tag = f" | wbuf {len(win_buffer):5d}" if use_win_buffer else ""
        if iteration % cfg.eval_every == 0 or iteration == cfg.iterations:
            net.eval()
            # Evaluate at the budget the net actually plays at (None = the
            # self-play budget) — see the eval_simulations comment in AZConfig.
            eval_sims = (cfg.eval_simulations if cfg.eval_simulations
                         else cfg.simulations)
            if pool is not None:
                version = ship_net()
                tasks = [(net_file, version, s, eval_sims, cfg.c_puct,
                          cfg.gamma, cfg.value_mode, (nw, fpw, sr, fm),
                          cfg.search, cfg.nm_depth, cfg.nm_k)
                         for s in cfg.eval_seeds]
                results = pool.map(_eval_task, tasks)
                from ..game import REWARD_PARTS
                n = len(results)
                mean_reward = sum(r[0] for r in results) / n
                win_rate = sum(1 for r in results if r[1]) / n
                parts = {p: sum(r[2][p] for r in results) / n
                         for p in REWARD_PARTS}
            else:
                if cfg.search == "negamax":
                    act = nm_action_fn(net, depth=cfg.nm_depth, k=cfg.nm_k,
                                       gamma=cfg.gamma, value_mode=cfg.value_mode)
                else:
                    act = az_action_fn(net, simulations=eval_sims,
                                       c_puct=cfg.c_puct, gamma=cfg.gamma,
                                       value_mode=cfg.value_mode)
                mean_reward, win_rate, parts = _eval_action_fn(
                    act, cfg.eval_seeds, nw, fpw, sr, fm)
            stage_tag = (f" | stage {stage_idx}/{len(stages) - 1} "
                         f"(w={nw},f={fpw},r={sr}"
                         f"{',' + fm if fm else ''})" if len(stages) > 1 else "")
            print(
                f"iter {total_iter:4d}{stage_tag} | buffer {len(buffer):6d}"
                f"{wbuf_tag} | "
                f"selfplay_win {sp_win:5.1%} | p_loss {p_loss:6.3f} | "
                f"v_loss {v_loss:8.2f} | eval_win@{eval_sims} {win_rate:5.1%} | "
                f"eval_reward {mean_reward:7.2f}\n"
                f"           reward parts: {format_reward_parts(parts)}",
                flush=True)
            # Lexicographic: stage, then win rate, then shaped reward. The
            # win-rate band must exceed the reward's whole range or a policy
            # that never wins can outscore one that does — with `win_rate*1000`
            # a single win in 16 (+62.5) lost to an +80 swing in shaped reward,
            # and a 0%-win policy overwrote the only full-game winner we had.
            score = (stage_idx * 10_000_000
                     + win_rate * 100_000
                     + mean_reward)
            if cfg.out_path and score > best_score:
                best_score = score
                best_iter = total_iter
                save_policy(net, cfg.out_path)
                print(f"           >> new best (iter {total_iter}) saved to "
                      f"{cfg.out_path}", flush=True)
            if (win_rate >= cfg.promote_win_rate
                    and stage_idx < len(stages) - 1):
                stage_idx += 1
                print(f"           >> promoted to stage {stage_idx} "
                      f"{stages[stage_idx]}", flush=True)
                bootstrap(stage_idx)
            if cfg.checkpoint_path:
                _save_az_checkpoint(cfg.checkpoint_path, net, optimizer,
                                    total_iter, best_score, stage_idx,
                                    win_buffer if use_win_buffer else None,
                                    best_iteration=best_iter)
        else:
            print(f"iter {total_iter:4d} | buffer {len(buffer):6d}"
                  f"{wbuf_tag} | "
                  f"selfplay_win {sp_win:5.1%} | p_loss {p_loss:6.3f} | "
                  f"v_loss {v_loss:8.2f}", flush=True)

    if cfg.out_path and best_score == float("-inf"):
        save_policy(net, cfg.out_path)
    if cfg.checkpoint_path:
        _save_az_checkpoint(cfg.checkpoint_path, net, optimizer,
                            iter_offset + cfg.iterations, best_score,
                            stage_idx, win_buffer if use_win_buffer else None,
                            best_iteration=best_iter)
    if pool is not None:
        pool.close()
        pool.join()
    if net_file and os.path.exists(net_file):
        os.unlink(net_file)
    return net


def _eval_action_fn(act: ActionFn, seeds: list[int],
                    num_waves: int | None, flames_per_wave: int | None,
                    spawn_radius: int | None,
                    spawn_formation: str | None = None) -> tuple:
    """Mean reward, win rate and parts for an ActionFn over ``seeds``."""
    from .policy import run_episode_decomposed
    from ..game import REWARD_PARTS
    results = [run_episode_decomposed(act, s, num_waves=num_waves,
                                      flames_per_wave=flames_per_wave,
                                      spawn_radius=spawn_radius,
                                      spawn_formation=spawn_formation)
               for s in seeds]
    n = len(seeds)
    mean_reward = sum(r[0] for r in results) / n
    win_rate = sum(1 for r in results if r[1]) / n
    parts = {p: sum(r[3][p] for r in results) / n for p in REWARD_PARTS}
    return mean_reward, win_rate, parts
