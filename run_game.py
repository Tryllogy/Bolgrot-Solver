"""Entry point for a frozen (PyInstaller) build of the game.

PyInstaller freezes a plain script, not ``python -m src``, so this thin wrapper
just calls the package's ``main()``. Run the normal game with ``python -m src``;
this file exists only for the executable build (see ``bolgrot.spec``).

``--selftest`` loads an AI net and runs one search, then exits — used by the
build to verify (non-interactively) that torch and the bundled nets actually
load inside the frozen app.
"""
import os
import sys


def _selftest() -> None:
    os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
    from src.game import Game
    from src.entity import Player
    from src import constant
    from src.hint import HintEngine
    game = Game(player=Player(*constant.BASE_PLAYER_POS))
    hint = HintEngine()
    for engine, sims in (("az", 20), ("cnn", 10)):
        action = hint._search(game.clone(), engine, sims)
        print(f"SELFTEST OK: {engine} -> action {action}")
    print("SELFTEST PASSED (torch + nets load in the frozen app)")


def main() -> None:
    if "--selftest" in sys.argv:
        _selftest()
        return
    from src.__main__ import main as run
    run()


if __name__ == "__main__":
    main()
