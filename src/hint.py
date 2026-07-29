"""AI move-hint for the desktop pygame UI.

Wraps the AlphaZero PUCT search (``src/ai/alphazero.py``) so the player can ask
"what should I play here?". Two engines — **Rapide** (the flat MLP ``az.pt``)
and **Fort** (the grid-CNN ``az_cnn_deep.pt``) — at a selectable search budget
(300 / 1000 / 8000 sims). Same nets and search as the online version.

Design notes:
- **torch is optional.** This module imports cleanly without the ``ai`` extra;
  torch / the network are imported lazily the first time a hint is requested,
  so the base game still runs with only pygame. A missing extra surfaces as a
  status message, not a crash.
- **The search runs in a background thread** (on a `clone()` of the game) so
  the window keeps drawing instead of freezing — an 8000-sim Fort hint takes
  tens of seconds. Results are picked up by `poll()` from the main loop.
- **`OBS_MODE` is flipped per engine** (grid for the CNN, flat for the MLP)
  around the search only — the desktop game always renders the flat default.
"""
from __future__ import annotations
import os
import threading

from . import constant
from .game import Game

# Net files live next to the AI code. "az" = flat MLP (fast), "cnn" = grid CNN
# (stronger per sim, ~4x slower).
_NET_PATHS = {
    "az": os.path.join(os.path.dirname(__file__), "ai", "az.pt"),
    "cnn": os.path.join(os.path.dirname(__file__), "ai", "az_cnn_deep.pt"),
}
ENGINE_LABEL = {"az": "Rapide", "cnn": "Fort"}
BUDGETS = (300, 1000, 8000)

# Shown when the hint is requested without the optional `ai` extra installed.
_MISSING_AI_MSG = ("Indice indisponible : installez l'extra « ai » "
                   "(uv sync --extra ai, ou pip install -e \".[ai]\").")


class HintResult:
    """A computed hint: the chosen action, its target tile, and a label."""

    def __init__(self, action: int, target: tuple[int, int] | None,
                 label: str) -> None:
        self.action = action                 # index into Game.ACTIONS
        self.target = target                 # (x, y) tile, or None = end turn
        self.label = label                   # human-readable one-liner


class HintEngine:
    """Holds the engine/budget choice and runs hints off the UI thread."""

    def __init__(self) -> None:
        self.engine = "az"                   # "az" (Rapide) | "cnn" (Fort)
        self.budget = 300                    # sims/move
        self.result: HintResult | None = None
        self.status = ""                     # status/message for the panel
        self.busy = False
        self._nets: dict[str, object] = {}   # path -> loaded net (cached)
        self._pending_id = 0                 # invalidates stale in-flight hint
        self._done: tuple | None = None      # (id, result|None, err|None)
        self._active = 0                     # search threads still running
        self._lock = threading.Lock()

    def has_active(self) -> bool:
        """True while a search thread is still running (even if invalidated).

        The caller uses this on shutdown: a daemon thread caught mid-torch at
        interpreter exit aborts the C++ teardown, so quit hard instead.
        """
        with self._lock:
            return self._active > 0

    # -- configuration (changing either invalidates a shown/pending hint) --
    def set_engine(self, engine: str) -> None:
        if engine in _NET_PATHS and engine != self.engine:
            self.engine = engine
            self.clear()

    def set_budget(self, budget: int) -> None:
        if budget in BUDGETS and budget != self.budget:
            self.budget = budget
            self.clear()

    def clear(self) -> None:
        """Drop the current hint and invalidate any in-flight search."""
        with self._lock:
            self._pending_id += 1
            self._done = None
        self.result = None
        self.busy = False
        self.status = ""

    # -- request / poll ----------------------------------------------------
    def request(self, game: Game) -> None:
        """Kick off a hint for ``game``'s current state (no-op if busy)."""
        if self.busy or game.done:
            return
        with self._lock:
            self._pending_id += 1
            my_id = self._pending_id
            self._done = None
            self._active += 1
        self.busy = True
        self.status = f"Calcul… ({ENGINE_LABEL[self.engine]} {self.budget})"
        # Snapshot everything the parse needs NOW, on the UI thread, so the
        # worker never touches the live game.
        snapshot = game.clone()
        px, py = game.player.pos_x, game.player.pos_y
        names = [s.name for s in game.player.spells]
        args = (my_id, snapshot, self.engine, self.budget, px, py, names)
        threading.Thread(target=self._worker, args=args, daemon=True).start()

    def poll(self) -> None:
        """Pick up a finished hint (call once per frame)."""
        with self._lock:
            done = self._done
            if done is None:
                return
            self._done = None
            if done[0] != self._pending_id:
                return                        # stale — newer request/clear won
        _, res, err = done
        self.busy = False
        if err is not None:
            self.result, self.status = None, f"Erreur : {err}"
        else:
            self.result, self.status = res, res.label

    # -- worker (background thread) ----------------------------------------
    def _worker(self, my_id: int, game: Game, engine: str, budget: int,
                px: int, py: int, names: list[str]) -> None:
        try:
            action = self._search(game, engine, budget)
            res = self._parse(action, px, py, names)
            payload = (my_id, res, None)
        except ModuleNotFoundError as exc:    # the `ai` extra isn't installed
            missing = (exc.name or "").split(".")[0]
            msg = _MISSING_AI_MSG if missing in ("torch", "numpy") \
                else str(exc)
            payload = (my_id, None, msg)
        except Exception as exc:              # load / search error
            payload = (my_id, None, str(exc))
        with self._lock:
            self._done = payload
            self._active -= 1

    def _search(self, game: Game, engine: str, budget: int) -> int:
        net = self._load(engine)
        from .ai.alphazero import az_action_fn   # lazy: needs torch
        prev_mode = constant.OBS_MODE
        # The CNN reads the 2D grid observation; the MLP the flat vector.
        is_cnn = getattr(net, "net_type", "mlp") == "cnn"
        constant.OBS_MODE = "grid" if is_cnn else "flat"
        try:
            return az_action_fn(net, simulations=budget)(game)
        finally:
            constant.OBS_MODE = prev_mode

    def _load(self, engine: str):
        path = _NET_PATHS[engine]
        if path not in self._nets:
            if not os.path.exists(path):
                raise FileNotFoundError(f"réseau introuvable : {path}")
            from .ai.policy import load_policy      # lazy: needs torch
            self._nets[path] = load_policy(path)
        return self._nets[path]

    @staticmethod
    def _parse(action: int, px: int, py: int,
               names: list[str]) -> HintResult:
        spell_idx, (dx, dy) = Game.ACTIONS[action]
        if spell_idx is None:
            return HintResult(action, None, "Indice : passer le tour")
        name = names[spell_idx] if spell_idx < len(names) \
            else f"Sort {spell_idx}"
        target = (px + dx, py + dy)
        return HintResult(action, target,
                          f"Indice : {name} → {target}")
