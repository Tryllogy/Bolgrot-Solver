"""Train a PPO agent to play the game headlessly.

Run with::

    uv run --extra ai python -m src.ai.train --iterations 300

The best policy (by evaluation win rate, then mean reward) is saved to
``src/ai/ppo.pt`` as it improves, so an interrupted run still leaves the best
policy on disk. A resumable training checkpoint (weights + optimizer + progress)
is saved to ``src/ai/ppo_ckpt.pt``; pass ``--resume`` to continue from it::

    uv run --extra ai python -m src.ai.train --iterations 300 --resume

Watch a trained policy afterwards with ``python -m src.ai.play``.
"""
from __future__ import annotations
import argparse
import os

import torch

from .ppo import PPOConfig, train

# Defaults next to this file so they are found regardless of CWD.
POLICY_PATH = os.path.join(os.path.dirname(__file__), "ppo.pt")
CHECKPOINT_PATH = os.path.join(os.path.dirname(__file__), "ppo_ckpt.pt")

# Default difficulty ramp for --curriculum: (num_waves, flames_per_wave,
# spawn_radius) stages ordered by DECLINING difficulty, from a trivially
# winnable start up to the full game. The percentages are the measured ceiling
# of the scripted `greedy_clear` heuristic (1.5k seeds each) — used ONLY to
# order the stages. They are NOT PPO's ceiling: with --ent-coef 0.03 and
# --lr-floor 0.1, a trained PPO policy far exceeds them (it clears the 0.5
# promote bar on every stage below, including (3,1,1) whose greedy ceiling is
# ~9%, and climbs the whole ramp to the full game). The real wall is the LAST
# stage only: the full 6-wave game is ~0% for any policy (attract-on-jump), so
# the curriculum's job is "reach stage 7", not "win it".
DEFAULT_CURRICULUM = (
    (1, 1, 1),          # greedy ~100% — one flame adjacent, one jump wins
    (1, 1, 2),          # greedy  ~66% — flame 2 away: must approach (attract)
    (1, 2, 1),          # greedy  ~34% — two flames at the player's elbow
    (2, 1, 1),          # greedy  ~33% — a second wave starts to accumulate
    (1, 2, 2),          # greedy  ~27%
    (2, 1, 2),          # greedy  ~23%
    (3, 1, 1),          # greedy   ~9% — PPO still clears 0.5 here
    (6, None, None),    # the full game: true patterns, all waves (~0%)
)

# AlphaZero's ramp: DEFAULT_CURRICULUM plus one bridge stage before the full
# game. Stages here are 4-tuples — the extra field is `spawn_formation`.
#
# Why the bridge exists: no stage above ever puts more than ~3 flames on the
# board, and none puts two flames *adjacent to each other*, so the ramp never
# teaches the push-chain death (jumping onto a flame shoves its neighbour into
# the next one, which is instantly lethal). The full game then opens with six.
# The "line" stage is three contiguous flames on the player's row or column,
# nearest one within LongJump reach: greedy_clear scores 0/6 on it (its first
# jump kills it) while MCTS at 300 sims scores 6/6 — hard, but winnable, and
# unwinnable by the greedy reflex the earlier stages reward.
AZ_CURRICULUM = tuple(
    stage + (None,) for stage in DEFAULT_CURRICULUM[:-1]
) + (
    (1, 3, None, "line"),   # bridge: dense adjacent flames, push-chain death
    (6, None, None, None),  # the full game
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a PPO agent.")
    parser.add_argument("--iterations", type=int, default=300,
                        help="rollout+update cycles to run this session")
    parser.add_argument("--num-envs", type=int, default=8)
    parser.add_argument("--num-steps", type=int, default=128,
                        help="env steps collected per game per rollout")
    parser.add_argument("--lr", type=float, default=2.5e-4)
    parser.add_argument("--lr-floor", type=float, default=0.0,
                        help="min LR as a fraction of --lr (e.g. 0.1): anneal "
                             "toward it instead of 0 so a curriculum session "
                             "keeps learning to its end")
    parser.add_argument("--ent-coef", type=float, default=0.01,
                        help="entropy bonus weight; raise it (e.g. 0.03) if "
                             "the policy collapses to a position-blind canned "
                             "action sequence instead of tracking the flame")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--eval-every", type=int, default=10)
    parser.add_argument("--out", default=POLICY_PATH,
                        help="where to save the best-by-eval policy")
    parser.add_argument("--checkpoint", default=CHECKPOINT_PATH,
                        help="where to save the resumable training state")
    parser.add_argument(
        "--resume", nargs="?", const="", default=None,
        help="continue training from a checkpoint (defaults to --checkpoint "
             "when given without a path)")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available()
                        else "cpu")
    parser.add_argument(
        "--curriculum", action="store_true",
        help="ramp difficulty from a tiny winnable game up to the full one, "
             "promoting a stage each time eval win rate hits --promote-win")
    parser.add_argument("--promote-win", type=float, default=0.5,
                        help="eval win rate needed to advance a curriculum "
                             "stage (default 0.5; PPO clears it on every "
                             "shrunk stage, see DEFAULT_CURRICULUM)")
    args = parser.parse_args()

    # Resolve --resume: absent -> no resume; bare flag -> the --checkpoint
    # path; explicit value -> that path.
    if args.resume is None:
        resume_from = ""
    elif args.resume == "":
        resume_from = args.checkpoint
    else:
        resume_from = args.resume

    cfg = PPOConfig(
        iterations=args.iterations,
        num_envs=args.num_envs,
        num_steps=args.num_steps,
        learning_rate=args.lr,
        lr_floor=args.lr_floor,
        ent_coef=args.ent_coef,
        seed=args.seed,
        eval_every=args.eval_every,
        out_path=args.out,
        checkpoint_path=args.checkpoint,
        resume_from=resume_from,
        device=args.device,
        curriculum=DEFAULT_CURRICULUM if args.curriculum else (),
        promote_win_rate=args.promote_win,
    )
    mode = f"resuming from {resume_from}" if resume_from else "fresh run"
    print(f"Training PPO ({mode}): {cfg.iterations} iters, {cfg.num_envs} "
          f"envs x {cfg.num_steps} steps = {cfg.batch_size} samples/rollout, "
          f"device={cfg.device}", flush=True)
    train(cfg)
    print(f"Done. Best policy: {args.out} | resumable checkpoint: "
          f"{args.checkpoint}")


if __name__ == "__main__":
    main()
