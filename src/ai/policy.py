"""Glue between a trained :class:`ActorCritic` and the :class:`Game` env.

Both the trainer (``ppo.py``) and the viewer (``play.py``) drive the game
through these helpers so action selection and (de)serialisation live in exactly
one place. Nothing here contains game logic — it only reads the environment API
(``get_observation`` / ``legal_actions`` / ``step``).
"""
from __future__ import annotations

from typing import Callable

import torch

from ..game import Game, REWARD_PARTS
from ..entity import Player
from .. import constant
from .network import ActorCritic

# An action function maps the current game state to a chosen action index. Both
# neural policies (:func:`net_action_fn`) and scripted baselines
# (``src/ai/baselines.py``) expose this shape so the same rollout code scores
# them all.
ActionFn = Callable[[Game], int]

# Fixed seeds used to score a policy so results are comparable across training
# runs while still covering many flame layouts. 32 seeds keep the win-rate
# quantisation fine (~3% steps) so curriculum promotion isn't decided by a
# handful of lucky layouts.
def eval_seed(i: int) -> int:
    """The i-th evaluation seed.

    Exposed as a generator, not just a list, so a benchmark can index *past*
    :data:`EVAL_SEEDS` for a large held-out set. The training gate selects on
    ``EVAL_SEEDS[:16]``, so only indices >= 16 are unselected — and there are
    just 16 of those in the list itself, too few to resolve the differences
    that matter here (measured 2026-07-16: 4/8 vs 6/8 on the same net is noise).
    ``eval_seed(16..47)`` gives 32 seeds no gate has ever seen.
    """
    return 101 * i + 7


EVAL_SEEDS: list[int] = [eval_seed(i) for i in range(32)]

# Safety cap so a policy that never makes progress can't loop forever. HP/PA
# are finite so episodes terminate on their own; this only guards regressions.
MAX_STEPS = 500


def obs_and_mask(game: Game) -> tuple[torch.Tensor, torch.Tensor]:
    """Current observation and legal-action mask as batch-less tensors."""
    obs = torch.as_tensor(game.get_observation(), dtype=torch.float32)
    mask = torch.as_tensor(game.legal_actions(), dtype=torch.bool)
    return obs, mask


@torch.no_grad()
def choose_action(net: ActorCritic, game: Game,
                  deterministic: bool = True) -> int:
    """Pick a legal action for the current state.

    ``deterministic`` takes the highest-probability legal action (greedy, used
    for watching/evaluating); otherwise samples from the masked policy.
    """
    obs, mask = obs_and_mask(game)
    logits, _ = net(obs)
    logits = logits.masked_fill(~mask, float("-inf"))
    if deterministic:
        return int(torch.argmax(logits).item())
    return int(torch.distributions.Categorical(logits=logits).sample().item())


def net_action_fn(net: ActorCritic, deterministic: bool = True) -> ActionFn:
    """Wrap a network as an :data:`ActionFn` for the rollout helpers."""
    return lambda game: choose_action(net, game, deterministic)


def run_episode_decomposed(
    act_fn: ActionFn,
    seed: int,
    max_steps: int = MAX_STEPS,
    num_waves: int | None = None,
    flames_per_wave: int | None = None,
    spawn_radius: int | None = None,
    spawn_formation: str | None = None,
) -> tuple[float, bool, int, dict[str, float]]:
    """Play one game with ``act_fn`` and report the reward broken down by term.

    Returns ``(total_reward, won, steps, parts)`` where ``parts`` sums each of
    :data:`REWARD_PARTS` over the episode — so you can see *which* term a
    policy is accumulating, not just the scalar return. ``num_waves`` /
    ``flames_per_wave`` / ``spawn_radius`` / ``spawn_formation`` set the
    curriculum difficulty (``None`` each = full game).
    """
    game = Game(Player(*constant.BASE_PLAYER_POS), seed=seed,
                num_waves=num_waves, flames_per_wave=flames_per_wave,
                spawn_radius=spawn_radius, spawn_formation=spawn_formation)
    total = 0.0
    steps = 0
    parts = {part: 0.0 for part in REWARD_PARTS}
    for _ in range(max_steps):
        if game.done:
            break
        reward, done = game.step(act_fn(game))
        total += reward
        steps += 1
        for part, value in game.last_reward.items():
            parts[part] += value
        if done:
            break
    return total, game.won, steps, parts


def run_episode(net: ActorCritic, seed: int,
                deterministic: bool = True,
                max_steps: int = MAX_STEPS) -> tuple[float, bool]:
    """Play one full game and return ``(total_reward, won)``."""
    total, won, _, _ = run_episode_decomposed(
        net_action_fn(net, deterministic), seed, max_steps)
    return total, won


def evaluate_decomposed(
    net: ActorCritic,
    seeds: list[int] = EVAL_SEEDS,
    deterministic: bool = True,
    num_waves: int | None = None,
    flames_per_wave: int | None = None,
    spawn_radius: int | None = None,
) -> tuple[float, float, dict[str, float]]:
    """Mean reward, win rate, and per-term reward breakdown over ``seeds``.

    The ``parts`` dict sums each of :data:`REWARD_PARTS`, averaged over seeds —
    the same decomposition ``src/ai/baselines.py`` prints, so training logs and
    the offline benchmark speak the same language. ``num_waves`` /
    ``flames_per_wave`` / ``spawn_radius`` evaluate at a given curriculum
    difficulty so the win rate reflects the stage being trained on.
    """
    net.eval()
    act_fn = net_action_fn(net, deterministic)
    results = [run_episode_decomposed(act_fn, s, num_waves=num_waves,
                                      flames_per_wave=flames_per_wave,
                                      spawn_radius=spawn_radius)
               for s in seeds]
    net.train()
    n = len(seeds)
    mean_reward = sum(total for total, _, _, _ in results) / n
    win_rate = sum(1 for _, won, _, _ in results if won) / n
    parts = {p: sum(r[3][p] for r in results) / n for p in REWARD_PARTS}
    return mean_reward, win_rate, parts


def evaluate(net: ActorCritic, seeds: list[int] = EVAL_SEEDS,
             deterministic: bool = True) -> tuple[float, float]:
    """Mean total reward and win rate of ``net`` over ``seeds``."""
    mean_reward, win_rate, _ = evaluate_decomposed(net, seeds, deterministic)
    return mean_reward, win_rate


def format_reward_parts(parts: dict[str, float]) -> str:
    """Compact ``name +value`` view of a reward breakdown (nonzero terms)."""
    return "  ".join(f"{name} {parts[name]:+.2f}"
                     for name in REWARD_PARTS if abs(parts[name]) > 1e-9)


def save_policy(net: ActorCritic, path: str) -> None:
    """Serialise ``net`` together with the shapes needed to rebuild it."""
    torch.save(
        {
            "state_dict": net.state_dict(),
            "obs_size": net.obs_size,
            "num_actions": net.num_actions,
            "hidden": net.hidden,
            "value_dim": net.value_dim,
            "net_type": getattr(net, "net_type", "mlp"),
        },
        path,
    )


def load_policy(path: str, device: str = "cpu") -> ActorCritic:
    """Rebuild the net saved by :func:`save_policy` (MLP or CNN).

    ``net_type`` defaults to ``"mlp"`` so checkpoints written before the CNN
    existed (e.g. ``az.pt``) still load.
    """
    from .network import build_net
    ckpt = torch.load(path, map_location=device, weights_only=True)
    net = build_net(ckpt["obs_size"], ckpt["num_actions"], ckpt["hidden"],
                    value_dim=ckpt.get("value_dim", 1),
                    net_type=ckpt.get("net_type", "mlp"))
    net.load_state_dict(ckpt["state_dict"])
    net.to(device)
    net.eval()
    return net
