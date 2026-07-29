"""Entry point for a frozen (PyInstaller) build of the game.

PyInstaller freezes a plain script, not ``python -m src``, so this thin wrapper
just calls the package's ``main()``. Run the normal game with ``python -m src``;
this file exists only for the executable build (see ``bolgrot.spec``).
"""
from src.__main__ import main

if __name__ == "__main__":
    main()
