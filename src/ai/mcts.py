"""Monte-Carlo Tree Search planning agent (an :data:`ActionFn`).

PPO is model-free: it can only reinforce wins it stumbles into, so it never
crosses the full game's hard-exploration wall (0% win). MCTS instead *plans* —
it searches action sequences on the deterministic :class:`~src.game.Game` with
explicit lookahead, the way a human wins. The search is exposed as an
:data:`~src.ai.policy.ActionFn` so it drops into the existing evaluation and
viewer harness (``policy.py`` / ``baselines.py``) with no other changes.

For speed the whole search runs on **one working game** that is snapshotted /
restored with :meth:`Game.get_state` / :meth:`Game.set_state` (~80x cheaper
than deep-copying a game per node); only one :meth:`Game.clone` is taken per
move to avoid mutating the live game the harness owns.

Two design points that make the search actually work on this game:

* **Shaped return, not win/loss.** Rollouts almost never win, so a {win=1,
  loss=0} leaf value is ~0 everywhere and gives the tree nothing to climb. We
  back up the summed per-step reward (``game.last_reward``), which exposes
  ``clear`` (+10/flame) and the large terminal win/death signals.
* **A survival-aware rollout** (:func:`safe_greedy`). Plain ``greedy_clear``
  suicides within a few steps (landing on a flame lets ``attract_flames`` pull
  a neighbour onto the player), which makes every leaf look equally fatal.
"""
from __future__ import annotations

import math
import random

from ..game import Game
from .policy import ActionFn


def _end_turn_index(game: Game) -> int:
    """Index of the end-turn action (``(None, (0, 0))``) in ``ACTIONS``."""
    for i, (spell_index, _) in enumerate(game.ACTIONS):
        if spell_index is None:
            return i
    raise ValueError("no end-turn action in ACTIONS")


def _first_legal(game: Game) -> int:
    """Fallback: first legal action index (a live game always has one)."""
    for i, ok in enumerate(game.legal_actions()):
        if ok:
            return i
    return 0


def _legal_indices(game: Game) -> list[int]:
    return [i for i, ok in enumerate(game.legal_actions()) if ok]


def safe_greedy(game: Game) -> int:
    """Greedy flame-clearing rollout that refuses jumps ending in death.

    Jumps are ranked greedily (land-on-a-flame first, else minimise distance to
    the nearest remaining flame) *without* branching. We then return the best-
    ranked jump that survives a 1-ply probe (step + restore via
    :meth:`Game.set_state`); if flames remain but every jump is lethal we
    reposition with a safe ``MoveFlames`` (its attract is ``killable=False``);
    otherwise we end the turn. Probing restores ``game`` before returning, so
    it is side-effect free.
    """
    legal = game.legal_actions()
    px, py = game.player.pos_x, game.player.pos_y
    flames = set(game._flame_positions())
    end_idx = _end_turn_index(game)

    ranked: list[tuple[tuple[int, int], int]] = []
    for idx, ok in enumerate(legal):
        if not ok:
            continue
        spell_index, (dx, dy) = game.ACTIONS[idx]
        if spell_index not in (0, 1):        # only the player-moving jumps
            continue
        target = (px + dx, py + dy)
        lands = target in flames
        remaining = list(flames - {target})
        dist_after = Game._nearest_target_dist(remaining, target)
        ranked.append(((0 if lands else 1, dist_after), idx))
    ranked.sort(key=lambda r: r[0])

    if flames:
        save = game.get_state()
        for _, idx in ranked:
            game.step(idx)
            lethal = game.done and not game.won
            game.set_state(save)
            if not lethal:
                return idx
        for idx, ok in enumerate(legal):           # no safe jump: reposition
            if ok and game.ACTIONS[idx][0] == 2:   # MoveFlames: killable=False
                return idx
    return end_idx if legal[end_idx] else _first_legal(game)


class _Node:
    """A game state in the search tree (stores a snapshot, not a full game)."""

    __slots__ = ("state", "reward_in", "done", "children", "untried",
                 "visits", "value")

    def __init__(self, game: Game, reward_in: float,
                 rng: random.Random) -> None:
        self.state = game.get_state()            # snapshot of this state
        self.reward_in = reward_in               # reward of the edge into here
        self.done = game.done
        self.children: dict[int, "_Node"] = {}
        self.untried: list[int] = [] if game.done else _legal_indices(game)
        rng.shuffle(self.untried)
        self.visits = 0
        self.value = 0.0                         # summed backed-up return


def _ucb_child(node: _Node, c: float) -> _Node:
    """Select the child maximising UCB1 (all children are visited here)."""
    log_n = math.log(node.visits)
    best_child: _Node | None = None
    best_score = -math.inf
    for child in node.children.values():
        exploit = child.value / child.visits
        explore = c * math.sqrt(log_n / child.visits)
        score = exploit + explore
        if score > best_score:
            best_score, best_child = score, child
    assert best_child is not None
    return best_child


def _rollout(work: Game, policy: ActionFn, depth: int, gamma: float) -> float:
    """Run ``policy`` on ``work`` in place; return the shaped return."""
    total, disc = 0.0, 1.0
    for _ in range(depth):
        if work.done:
            break
        reward, _ = work.step(policy(work))
        total += disc * reward
        disc *= gamma
    return total


def _simulate(root: _Node, work: Game, rollout_policy: ActionFn,
              rollout_depth: int, c: float, gamma: float,
              rng: random.Random) -> None:
    """One MCTS iteration: select, expand, roll out, back up the return."""
    node = root
    path = [root]
    ret, disc = 0.0, 1.0

    # Selection: descend fully-expanded, non-terminal nodes by UCB1. Uses only
    # cached node stats, so no game stepping happens here.
    while not node.untried and node.children and not node.done:
        node = _ucb_child(node, c)
        ret += disc * node.reward_in
        disc *= gamma
        path.append(node)

    # Expansion + simulation: position the working game at the selected node,
    # try one untried action, then roll out from the new leaf in place.
    if node.untried and not node.done:
        work.set_state(node.state)
        action = node.untried.pop()
        reward, _ = work.step(action)
        child = _Node(work, reward_in=reward, rng=rng)
        node.children[action] = child
        ret += disc * reward
        disc *= gamma
        path.append(child)
        if not work.done:
            ret += disc * _rollout(work, rollout_policy, rollout_depth, gamma)

    # Backup: the full-trajectory return updates every node on the path.
    for visited in path:
        visited.visits += 1
        visited.value += ret


def mcts_root_visits(game: Game, simulations: int, rng: random.Random,
                     rollout_depth: int = 16, c_ucb: float = 1.4,
                     rollout_policy: ActionFn | None = None,
                     gamma: float = 1.0) -> dict[int, int]:
    """Search ``game`` and return the root's ``{action: visit count}``.

    Same search as :func:`mcts_action_fn` (which is just this plus an argmax),
    exposed so AlphaZero can use MCTS's visit distribution as a policy target —
    the search is strong enough to win the full game, so its visits are a much
    better thing to imitate than a self-play run that never wins. Empty when the
    game is over or has no legal move.
    """
    if game.done:
        return {}
    work = game.clone()
    root = _Node(work, reward_in=0.0, rng=rng)
    if not root.untried:
        return {}
    rollout = rollout_policy if rollout_policy is not None else safe_greedy
    for _ in range(simulations):
        _simulate(root, work, rollout, rollout_depth, c_ucb, gamma, rng)
    return {a: child.visits for a, child in root.children.items()}


def mcts_action_fn(simulations: int = 200, rollout_depth: int = 16,
                   c_ucb: float = 1.4, rollout_policy: ActionFn | None = None,
                   gamma: float = 1.0, seed: int = 0) -> ActionFn:
    """Build an :data:`ActionFn` that picks each move by MCTS lookahead.

    ``simulations`` search iterations are run per move on a private clone of
    the live game (never mutating the game the harness owns), then the most-
    visited root action is returned. ``rollout_policy`` defaults to
    :func:`safe_greedy`.
    """
    rng = random.Random(seed)
    rollout = rollout_policy if rollout_policy is not None else safe_greedy

    def act(game: Game) -> int:
        if game.done:
            return 0
        work = game.clone()                      # one clone per move
        root = _Node(work, reward_in=0.0, rng=rng)
        if not root.untried:
            return _first_legal(game)
        for _ in range(simulations):
            _simulate(root, work, rollout, rollout_depth, c_ucb, gamma, rng)
        if not root.children:
            return _first_legal(game)
        return max(root.children, key=lambda a: root.children[a].visits)

    return act
