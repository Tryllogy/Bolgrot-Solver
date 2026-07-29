"""Watch a trained policy play, using the normal game renderer.

Run with::

    uv run --extra ai python -m src.ai.play --delay 400
    uv run --extra ai python -m src.ai.play --policy src/ai/az.pt --sims 200

Loads a policy (``ppo.pt`` by default) and lets it drive the game while
rendering each step so you can see the strategy. With ``--sims N`` (N > 0) the
move is chosen by an AlphaZero **PUCT search** at N simulations (the strong
agent — use this for ``az.pt``); with ``--sims 0`` the raw policy head decides
(fast/reactive — the right mode for a plain PPO ``ppo.pt``).
"""
from __future__ import annotations
import argparse
import os

import pygame

from ..entity import Player
from .. import constant
from ..game import Game
from ..renderer import Renderer
from .policy import choose_action, load_policy
from .train import POLICY_PATH


def watch(policy_path: str, seed: int, delay_ms: int, sims: int = 0) -> None:
    """Render one episode driven by the policy at ``policy_path``.

    ``sims > 0`` drives the game with an AlphaZero PUCT search at that budget;
    ``sims == 0`` uses the raw (deterministic) policy head.
    """
    net = load_policy(policy_path)
    if sims > 0:
        from .alphazero import az_action_fn
        act_fn = az_action_fn(net, simulations=sims)
    else:
        def act_fn(g: Game) -> int:
            return choose_action(net, g, deterministic=True)

    pygame.init()
    screen = pygame.display.set_mode((0, 0), pygame.FULLSCREEN)
    clock = pygame.time.Clock()
    font_title = pygame.font.Font(None, 50)
    font_txt = pygame.font.Font(None, 25)

    game = Game(Player(*constant.BASE_PLAYER_POS), seed=seed)
    renderer = Renderer(screen, font_title, font_txt, game.map.cases)

    running = True
    last_step = pygame.time.get_ticks()
    while running:
        screen.fill((0, 0, 0))
        mouse_x, mouse_y = pygame.mouse.get_pos()
        for event in pygame.event.get():
            if event.type == pygame.QUIT or (
                    event.type == pygame.KEYDOWN
                    and event.key == pygame.K_ESCAPE):
                running = False

        now = pygame.time.get_ticks()
        if not game.done and now - last_step >= delay_ms:
            game.step(act_fn(game))
            last_step = now

        renderer.draw_map(mouse_x, mouse_y, game.map.cases,
                          game.previsualiation, game.spawn_pattern)
        renderer.draw_entities(game.map.cases)
        renderer.draw_spells(mouse_x, mouse_y, game.player.spells)
        renderer.draw_hp_player(game.player)
        renderer.draw_ap_player(game.player)
        pygame.display.flip()
        clock.tick(60)

    pygame.quit()
    print("Victory: all flames cleared!" if game.won
          else f"Game over — waves survived: {game.waves_spawned}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Watch a PPO policy play.")
    parser.add_argument("--policy", default=POLICY_PATH)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--delay", type=int, default=400,
                        help="milliseconds between AI moves")
    parser.add_argument("--sims", type=int, default=0,
                        help="PUCT search sims per move (0 = raw policy head; "
                             "use ~200 to watch the AlphaZero az.pt agent)")
    args = parser.parse_args()
    if not os.path.exists(args.policy):
        raise SystemExit(
            f"No policy at {args.policy}. Run `python -m src.ai.train` first.")
    watch(args.policy, args.seed, args.delay, args.sims)


if __name__ == "__main__":
    main()
