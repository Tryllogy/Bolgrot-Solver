"""PPO (Proximal Policy Optimization) trainer for the Bolgrot game.

Synchronous, single-file PPO with Generalized Advantage Estimation and a
clipped surrogate objective, following the well-known CleanRL layout. It drives
a batch of independent :class:`Game` instances purely through the environment
API (``get_observation`` / ``legal_actions`` / ``step``) and learns an
:class:`ActorCritic` with masked action selection.

Data flow per iteration:
  1. **Rollout** — step ``num_envs`` games for ``num_steps`` each, recording
     obs, action, log-prob, reward, done, value and the legal-action mask.
  2. **GAE** — turn the rewards/values into advantages and returns.
  3. **Update** — several epochs of minibatch SGD on the clipped policy loss,
     a value loss and an entropy bonus.
"""
from __future__ import annotations
import os
import random
from dataclasses import dataclass, field

import torch
from torch import nn, optim

from ..game import Game
from ..entity import Player
from .. import constant
from .network import ActorCritic
from .policy import evaluate_decomposed, format_reward_parts, save_policy


@dataclass
class PPOConfig:
    """Hyperparameters for a training run (CleanRL-style defaults)."""

    iterations: int = 300          # number of rollout+update cycles
    num_envs: int = 8              # parallel game instances per rollout
    num_steps: int = 128           # env steps collected per game per rollout
    gamma: float = 0.99            # discount factor
    gae_lambda: float = 0.95       # GAE smoothing
    update_epochs: int = 4         # SGD passes over each rollout
    num_minibatches: int = 4       # minibatches per epoch
    clip_coef: float = 0.2         # PPO surrogate clip range
    ent_coef: float = 0.01         # entropy bonus weight
    vf_coef: float = 0.5           # value loss weight
    max_grad_norm: float = 0.5     # gradient clipping
    learning_rate: float = 2.5e-4  # Adam step size
    anneal_lr: bool = True         # linearly decay LR toward lr_floor
    lr_floor: float = 0.0          # min LR as a fraction of learning_rate
    norm_adv: bool = True          # normalise advantages per minibatch
    clip_vloss: bool = True        # clipped value loss
    hidden: int = 64               # network hidden width
    seed: int = 1                  # RNG seed for torch/python
    eval_every: int = 10           # evaluate + log win rate every N iterations
    device: str = "cpu"
    out_path: str = ""             # where the best-by-eval policy is saved
    checkpoint_path: str = ""      # where the resumable training state is saved
    resume_from: str = ""          # training checkpoint to continue from
    # Curriculum: ordered (num_waves, flames_per_wave, spawn_radius) stages,
    # easy -> hard. The agent starts on stage 0 and is promoted to the next
    # once its eval win rate reaches `promote_win_rate`. Empty = fixed full
    # difficulty (no curriculum). A tiny first stage (a flame spawned next
    # to the player) is trivially winnable, giving PPO the reward gradient it
    # needs before the full game. Use `None` in the last stage's fields for the
    # true patterns / full wave count.
    curriculum: tuple[tuple[int | None, int | None, int | None], ...] = ()
    promote_win_rate: float = 0.7  # advance a stage when eval win rate >= this

    @property
    def batch_size(self) -> int:
        return self.num_envs * self.num_steps

    @property
    def minibatch_size(self) -> int:
        return self.batch_size // self.num_minibatches


class _VecGames:
    """A batch of independent games stepped in lockstep, with auto-reset.

    On an episode end the finished game is immediately reset with a fresh seed
    so rollouts stay dense. Observations and legal-action masks are returned as
    stacked tensors ready for the network.
    """

    def __init__(self, num_envs: int, rng: random.Random, device: str,
                 num_waves: int | None = None,
                 flames_per_wave: int | None = None,
                 spawn_radius: int | None = None) -> None:
        self._rng = rng
        self._device = device
        self._num_envs = num_envs
        self.num_waves = num_waves
        self.flames_per_wave = flames_per_wave
        self.spawn_radius = spawn_radius
        self.games = [self._new_game() for _ in range(num_envs)]

    def _seed(self) -> int:
        return self._rng.randrange(1 << 30)

    def _new_game(self) -> Game:
        """A fresh game at the current curriculum difficulty."""
        return Game(Player(*constant.BASE_PLAYER_POS), seed=self._seed(),
                    num_waves=self.num_waves,
                    flames_per_wave=self.flames_per_wave,
                    spawn_radius=self.spawn_radius)

    def set_difficulty(self, num_waves: int | None,
                       flames_per_wave: int | None,
                       spawn_radius: int | None) -> None:
        """Switch to a new curriculum stage; rebuild all envs immediately."""
        self.num_waves = num_waves
        self.flames_per_wave = flames_per_wave
        self.spawn_radius = spawn_radius
        self.games = [self._new_game() for _ in range(self._num_envs)]

    def observe(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Stacked ``(obs, mask)`` for every game in the batch."""
        obs = torch.tensor([g.get_observation() for g in self.games],
                           dtype=torch.float32, device=self._device)
        mask = torch.tensor([g.legal_actions() for g in self.games],
                            dtype=torch.bool, device=self._device)
        return obs, mask

    def step(self, actions: torch.Tensor
             ) -> tuple[torch.Tensor, torch.Tensor]:
        """Apply one action per game; return ``(rewards, dones)`` tensors.

        A game that finishes is reset in place so the next ``observe`` reads a
        fresh episode.
        """
        rewards, dones = [], []
        for i, game in enumerate(self.games):
            reward, done = game.step(int(actions[i].item()))
            rewards.append(reward)
            dones.append(done)
            if done:
                self.games[i] = self._new_game()
        return (
            torch.tensor(rewards, dtype=torch.float32, device=self._device),
            torch.tensor(dones, dtype=torch.float32, device=self._device),
        )


def _compute_gae(
    rewards: torch.Tensor, values: torch.Tensor, dones: torch.Tensor,
    next_value: torch.Tensor, next_done: torch.Tensor,
    gamma: float, gae_lambda: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Generalized Advantage Estimation over a ``[T, num_envs]`` rollout.

    ``dones[t]`` marks whether ``obs[t]`` began a fresh episode; ``next_value``
    / ``next_done`` bootstrap the value past the end of the rollout.
    """
    num_steps = rewards.shape[0]
    advantages = torch.zeros_like(rewards)
    last_gae = torch.zeros_like(next_value)
    for t in reversed(range(num_steps)):
        if t == num_steps - 1:
            next_nonterminal = 1.0 - next_done
            next_values = next_value
        else:
            next_nonterminal = 1.0 - dones[t + 1]
            next_values = values[t + 1]
        delta = rewards[t] + gamma * next_values * next_nonterminal - values[t]
        last_gae = delta + gamma * gae_lambda * next_nonterminal * last_gae
        advantages[t] = last_gae
    returns = advantages + values
    return advantages, returns


@dataclass
class _EpisodeStats:
    """Rolling window of completed-episode returns for logging."""

    returns: list[float] = field(default_factory=list)
    running: list[float] = field(default_factory=list)

    def start(self, num_envs: int) -> None:
        self.running = [0.0] * num_envs

    def record(self, rewards: torch.Tensor, dones: torch.Tensor) -> None:
        for i in range(len(self.running)):
            self.running[i] += float(rewards[i].item())
            if dones[i]:
                self.returns.append(self.running[i])
                self.running[i] = 0.0

    def mean(self, window: int = 100) -> float:
        recent = self.returns[-window:]
        return sum(recent) / len(recent) if recent else float("nan")


def _save_train_checkpoint(
    path: str, net: ActorCritic, optimizer: optim.Optimizer,
    iteration: int, global_step: int, best_score: float,
    stage_idx: int = 0,
) -> None:
    """Save full training state so a run can be resumed.

    A superset of the policy checkpoint (:func:`policy.save_policy`) — it adds
    the optimizer state and progress counters, so ``load_policy`` still reads it
    for playing while ``--resume`` can restore the exact training position
    (including the curriculum ``stage_idx`` reached).
    """
    torch.save(
        {
            "state_dict": net.state_dict(),
            "obs_size": net.obs_size,
            "num_actions": net.num_actions,
            "hidden": net.hidden,
            "optimizer": optimizer.state_dict(),
            "iteration": iteration,
            "global_step": global_step,
            "best_score": best_score,
            "stage_idx": stage_idx,
        },
        path,
    )


def train(cfg: PPOConfig) -> ActorCritic:
    """Run PPO for ``cfg.iterations`` and return the trained network.

    The best-by-eval policy is written to ``cfg.out_path`` (if set) whenever a
    new high win-rate/reward is reached, so an interrupted run still leaves the
    best checkpoint on disk.
    """
    random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    rng = random.Random(cfg.seed)
    device = cfg.device

    obs_size = Game.observation_size()
    num_actions = Game.num_actions()
    net = ActorCritic(obs_size, num_actions, cfg.hidden).to(device)
    optimizer = optim.Adam(net.parameters(), lr=cfg.learning_rate, eps=1e-5)

    # Resume: restore weights, optimizer moments and progress counters so a new
    # session continues where the last left off. `iteration`/`global_step` only
    # offset the logs/checkpoints; each session anneals its LR over its own
    # `cfg.iterations` (a warm restart), which helps continued learning.
    iter_offset = 0
    global_step = 0
    best_score = float("-inf")
    stage_idx = 0
    if cfg.resume_from:
        if not os.path.exists(cfg.resume_from):
            raise SystemExit(f"No checkpoint to resume at {cfg.resume_from}.")
        ckpt = torch.load(cfg.resume_from, map_location=device,
                          weights_only=True)
        if (ckpt["obs_size"] != obs_size
                or ckpt["num_actions"] != num_actions
                or ckpt.get("hidden") != cfg.hidden):
            raise SystemExit(
                "Checkpoint shape does not match the current env/config "
                "(obs/action/hidden size changed) — train a fresh policy.")
        net.load_state_dict(ckpt["state_dict"])
        if "optimizer" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer"])
        iter_offset = ckpt.get("iteration", 0)
        global_step = ckpt.get("global_step", 0)
        best_score = ckpt.get("best_score", float("-inf"))
        stage_idx = ckpt.get("stage_idx", 0)
        print(f"Resumed from {cfg.resume_from}: iteration {iter_offset}, "
              f"global_step {global_step}, best_score {best_score:.2f}, "
              f"stage {stage_idx}",
              flush=True)

    # Curriculum stages (easy -> hard). No curriculum => one full-difficulty
    # stage. `stage_idx` (possibly restored above) picks the starting stage.
    stages: list[tuple[int | None, int | None, int | None]] = (
        list(cfg.curriculum) if cfg.curriculum else [(None, None, None)])
    stage_idx = min(stage_idx, len(stages) - 1)

    envs = _VecGames(cfg.num_envs, rng, device,
                     num_waves=stages[stage_idx][0],
                     flames_per_wave=stages[stage_idx][1],
                     spawn_radius=stages[stage_idx][2])
    T, N = cfg.num_steps, cfg.num_envs

    # Rollout storage.
    obs_buf = torch.zeros((T, N, obs_size), device=device)
    mask_buf = torch.zeros((T, N, num_actions), dtype=torch.bool, device=device)
    actions_buf = torch.zeros((T, N), dtype=torch.long, device=device)
    logprobs_buf = torch.zeros((T, N), device=device)
    rewards_buf = torch.zeros((T, N), device=device)
    dones_buf = torch.zeros((T, N), device=device)
    values_buf = torch.zeros((T, N), device=device)

    next_obs, next_mask = envs.observe()
    next_done = torch.zeros(N, device=device)
    stats = _EpisodeStats()
    stats.start(N)

    for iteration in range(1, cfg.iterations + 1):
        total_iter = iter_offset + iteration
        if cfg.anneal_lr:
            # Anneal linearly toward `lr_floor` (a fraction of the base LR),
            # not all the way to 0: with a curriculum, each session must keep
            # learning to its end so the tail isn't wasted freezing the policy.
            frac = 1.0 - (iteration - 1) / cfg.iterations
            frac = max(frac, cfg.lr_floor)
            optimizer.param_groups[0]["lr"] = frac * cfg.learning_rate

        # --- Rollout ---------------------------------------------------
        for step in range(T):
            global_step += N
            obs_buf[step] = next_obs
            mask_buf[step] = next_mask
            dones_buf[step] = next_done
            with torch.no_grad():
                action, logprob, _, value = net.get_action_and_value(
                    next_obs, next_mask)
            values_buf[step] = value
            actions_buf[step] = action
            logprobs_buf[step] = logprob

            reward, done = envs.step(action)
            rewards_buf[step] = reward
            stats.record(reward, done)
            next_obs, next_mask = envs.observe()
            next_done = done

        # --- Advantages ------------------------------------------------
        with torch.no_grad():
            next_value = net.get_value(next_obs)
        advantages, returns = _compute_gae(
            rewards_buf, values_buf, dones_buf, next_value, next_done,
            cfg.gamma, cfg.gae_lambda)

        # --- PPO update ------------------------------------------------
        b_obs = obs_buf.reshape(-1, obs_size)
        b_mask = mask_buf.reshape(-1, num_actions)
        b_actions = actions_buf.reshape(-1)
        b_logprobs = logprobs_buf.reshape(-1)
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_values = values_buf.reshape(-1)

        approx_kl = torch.tensor(0.0)
        indices = torch.arange(cfg.batch_size, device=device)
        for _ in range(cfg.update_epochs):
            perm = indices[torch.randperm(cfg.batch_size, device=device)]
            for start in range(0, cfg.batch_size, cfg.minibatch_size):
                mb = perm[start:start + cfg.minibatch_size]
                _, new_logprob, entropy, new_value = net.get_action_and_value(
                    b_obs[mb], b_mask[mb], b_actions[mb])
                log_ratio = new_logprob - b_logprobs[mb]
                ratio = log_ratio.exp()
                with torch.no_grad():
                    approx_kl = ((ratio - 1) - log_ratio).mean()

                mb_adv = b_advantages[mb]
                if cfg.norm_adv:
                    mb_adv = (mb_adv - mb_adv.mean()) / (mb_adv.std() + 1e-8)

                # Clipped policy (surrogate) loss.
                pg1 = -mb_adv * ratio
                pg2 = -mb_adv * torch.clamp(
                    ratio, 1 - cfg.clip_coef, 1 + cfg.clip_coef)
                pg_loss = torch.max(pg1, pg2).mean()

                # Value loss (optionally clipped like the policy).
                if cfg.clip_vloss:
                    v_unclipped = (new_value - b_returns[mb]) ** 2
                    v_clipped = b_values[mb] + torch.clamp(
                        new_value - b_values[mb],
                        -cfg.clip_coef, cfg.clip_coef)
                    v_clipped = (v_clipped - b_returns[mb]) ** 2
                    v_loss = 0.5 * torch.max(v_unclipped, v_clipped).mean()
                else:
                    v_loss = 0.5 * ((new_value - b_returns[mb]) ** 2).mean()

                entropy_loss = entropy.mean()
                loss = (pg_loss
                        - cfg.ent_coef * entropy_loss
                        + cfg.vf_coef * v_loss)

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(net.parameters(), cfg.max_grad_norm)
                optimizer.step()

        # --- Logging / checkpointing -----------------------------------
        if iteration % cfg.eval_every == 0 or iteration == cfg.iterations:
            nw, fpw, sr = stages[stage_idx]
            mean_reward, win_rate, parts = evaluate_decomposed(
                net, num_waves=nw, flames_per_wave=fpw, spawn_radius=sr)
            stage_tag = (f" | stage {stage_idx}/{len(stages) - 1} "
                         f"(waves={nw}, flames={fpw}, radius={sr})"
                         if len(stages) > 1 else "")
            print(
                f"iter {total_iter:4d} | step {global_step:>8d}{stage_tag} | "
                f"train_return {stats.mean():7.2f} | "
                f"eval_reward {mean_reward:7.2f} | win {win_rate:5.1%} | "
                f"kl {approx_kl.item():.4f}\n"
                f"           reward parts: {format_reward_parts(parts)}",
                flush=True,
            )
            # Best-policy score: the curriculum stage dominates, so a policy
            # that reaches a harder stage always beats an easier-stage one,
            # then win rate, then mean reward break ties.
            score = stage_idx * 1_000_000 + win_rate * 1000 + mean_reward
            if cfg.out_path and score > best_score:
                best_score = score
                save_policy(net, cfg.out_path)
            # Promote to the next (harder) stage once the agent reliably wins
            # the current one. Rebuilds the envs, so refresh the cached rollout
            # state before the next iteration reads it.
            if (win_rate >= cfg.promote_win_rate
                    and stage_idx < len(stages) - 1):
                stage_idx += 1
                envs.set_difficulty(*stages[stage_idx])
                next_obs, next_mask = envs.observe()
                next_done = torch.zeros(N, device=device)
                print(f"           >> promoted to stage {stage_idx} "
                      f"{stages[stage_idx]}", flush=True)
            # Resumable training state: latest weights + optimizer + progress.
            if cfg.checkpoint_path:
                _save_train_checkpoint(cfg.checkpoint_path, net, optimizer,
                                       total_iter, global_step, best_score,
                                       stage_idx)
        else:
            print(
                f"iter {total_iter:4d} | step {global_step:>8d} | "
                f"train_return {stats.mean():7.2f} | kl {approx_kl.item():.4f}",
                flush=True,
            )

    if cfg.out_path and best_score == float("-inf"):
        # No eval happened (tiny run): still leave a playable policy.
        save_policy(net, cfg.out_path)
    if cfg.checkpoint_path:
        # Always leave the latest resumable state at the end of the run.
        _save_train_checkpoint(cfg.checkpoint_path, net, optimizer,
                               iter_offset + cfg.iterations, global_step,
                               best_score, stage_idx)
    return net
