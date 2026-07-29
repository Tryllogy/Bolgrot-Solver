from __future__ import annotations
import copy
import random
from . import constant
from .BFS import BFS
from .map import Map
from .case import CaseType
from .patterns import Patterns
from .spells import Spells, ShortJump, LongJump, MoveFlames
from .entity import Flame, Player, Bolgrot

# Fixed discrete action space for the AI. Every spell target is one of the
# 4 orthogonal / 4 diagonal directions at a fixed range, so the whole game
# reduces to a small position-independent action head plus "end turn".
# Each entry is ``(spell_index | None, (dx, dy))`` where the tile targeted is
# the player's position plus the offset; end-turn is ``(None, (0, 0))``.
_LINE_DIRS = [(-1, 0), (0, 1), (1, 0), (0, -1)]
_DIAG_DIRS = [(-1, -1), (-1, 1), (1, -1), (1, 1)]
_ACTIONS: list[tuple[int | None, tuple[int, int]]] = [
    *[(0, d) for d in _LINE_DIRS],                       # ShortJump, r=1
    *[(1, (d[0] * r, d[1] * r))                          # LongJump,  r=1,2
      for r in (1, 2) for d in _LINE_DIRS],
    *[(2, d) for d in _DIAG_DIRS],                       # MoveFlames, r=1
    (None, (0, 0)),                                      # end turn
]

# Names of the additive reward terms produced by :meth:`Game.step`. Kept as an
# explicit ordered tuple so evaluators can decompose an episode's return into
# per-term sums (see ``src/ai/baselines.py``) and spot which term a policy is
# actually farming. Their sum is exactly the scalar reward ``step`` returns.
REWARD_PARTS: tuple[str, ...] = (
    "survive", "stall", "cast", "approach", "clear", "hp", "terminal",
    "illegal",
)


class Game:
    """Owns all game state and the turn/action API (no pygame dependency)."""

    ACTIONS = _ACTIONS

    @staticmethod
    def num_actions() -> int:
        """Number of discrete actions (length of :data:`Game.ACTIONS`)."""
        return len(Game.ACTIONS)

    def __init__(
        self,
        player: Player,
        seed: int | None = None,
        num_waves: int | None = None,
        flames_per_wave: int | None = None,
        spawn_radius: int | None = None,
        spawn_formation: str | None = None,
    ) -> None:
        """Set up the map, player spells, boss, spawn pattern and counters.

        ``seed`` makes the flame spawn sequence reproducible. The rest are
        **curriculum knobs** that shrink the game to an easier, winnable size
        so an RL agent has a reward gradient to climb (``None`` each = full
        difficulty):

        - ``num_waves`` — waves before the win check (default ``NB_WAVES``).
        - ``flames_per_wave`` — cap flames spawned per wave.
        - ``spawn_radius`` — when set, replace the fixed spawn *patterns* with
          ``flames_per_wave`` flames placed on random free tiles within this
          Manhattan distance of the player. Since a lone flame is a trivial win
          (jump onto it, nothing left to attract), a small radius + few flames
          makes the early stages actually winnable — the real difficulty is
          spawn *distance*, not count. The final stage leaves this ``None`` to
          train on the true game patterns.
        - ``spawn_formation`` — when set, overrides both of the above and spawns
          a *fixed shape*. ``"line"`` places ``flames_per_wave`` mutually
          adjacent flames in a straight line on the player's own row or column,
          starting ``SPAWN_LINE_OFFSET`` tiles away. This is the one shape the
          random-radius stages never produce: a dense cluster where clearing the
          near flame pushes its neighbour into the next one, which is lethal. It
          exists to teach the push-chain death before the full game.
        """
        self.map: Map = Map()
        self.player: Player = player
        self._rng: random.Random = random.Random(seed)
        self.num_waves: int = (num_waves if num_waves is not None
                               else constant.NB_WAVES)
        self.flames_per_wave: int | None = flames_per_wave
        self.spawn_radius: int | None = spawn_radius
        self.spawn_formation: str | None = spawn_formation
        bfs: BFS = BFS(self.map)
        self.player.spells = [
            ShortJump(bfs=bfs),
            LongJump(bfs=bfs),
            MoveFlames(bfs=bfs),
        ]
        self.map.place_entity([constant.BASE_PLAYER_POS], self.player)
        self.bolgrot: Bolgrot = Bolgrot(*constant.BASE_BOLGROT_POS)
        self.map.place_entity([constant.BASE_BOLGROT_POS], self.bolgrot)
        self.patterns: Patterns = Patterns(seed=seed)
        self.spawn_pattern: list[tuple[int, int]] = \
            self._make_wave(self.patterns.draw_opening())
        self.waves_spawned: int = 0
        self.previsualiation: list[tuple[int, int]] = []
        self.spell: Spells | None = None
        self.turn: int = 0
        self.done: bool = False
        self.won: bool = False
        # Per-step reward decomposition, refreshed by every ``step`` call. Sums
        # to the scalar reward returned. Lets tooling see the breakdown.
        self.last_reward: dict[str, float] = self._zero_reward()

    def clone(self) -> "Game":
        """Deep, independent copy of the full game state.

        Used by search agents (``src/ai/mcts.py``) to branch and roll back:
        the game is deterministic, so a clone can be stepped freely without
        touching the original. Headless-safe — spell sprites are lazy-loaded
        (`Spells._image` is ``None`` until rendered), so no pygame ``Surface``
        is ever in the object graph to deepcopy.
        """
        return copy.deepcopy(self)

    def get_state(self) -> dict:
        """Fast snapshot of every mutable field (see :meth:`clone`).

        ~80x cheaper than :meth:`clone` — for search agents that save/restore
        one working game thousands of times per move rather than deep-copying a
        node each time. Restore with :meth:`set_state`. The immutable map
        structure, spell configs and Bolgrot (which never moves) are shared, so
        only occupancy, counters, RNG and per-turn spell use are captured.
        """
        p = self.player
        return {
            "px": p.pos_x, "py": p.pos_y, "hp": p.hp, "pa": p.pa,
            "flames": list(self.map.flames),
            "time_used": [s.time_used for s in p.spells],
            "waves_spawned": self.waves_spawned,
            "turn": self.turn,
            "done": self.done,
            "won": self.won,
            "spawn_pattern": list(self.spawn_pattern),
            "previsu": list(self.previsualiation),
            "spell_idx": (p.spells.index(self.spell)
                          if self.spell is not None else None),
            "last_reward": dict(self.last_reward),
            "rng": self._rng.getstate(),
            "prng": self.patterns._rng.getstate(),
            "spawn_patterns": list(self.patterns.spawn_patterns),
        }

    def set_state(self, state: dict) -> None:
        """Restore a snapshot produced by :meth:`get_state` (in place)."""
        p = self.player
        # Clear the current movable occupants WITHOUT scanning every cell: the
        # flames are tracked in ``map.flames`` and the player sits at its
        # current (pre-restore) position. Bolgrot never moves. Snapshot the
        # flame set first — clearing mutates it through the entity setter. The
        # player must stay the *same* object (aliased from ``self.player``).
        for pos in tuple(self.map.flames):
            self.map.cases[pos].entity = None
        cur = self.map.cases.get((p.pos_x, p.pos_y))
        if cur is not None and cur.entity is p:
            cur.entity = None
        p.pos_x, p.pos_y = state["px"], state["py"]
        p.hp, p.pa = state["hp"], state["pa"]
        for spell, used in zip(p.spells, state["time_used"]):
            spell.time_used = used
        for fx, fy in state["flames"]:
            self.map.cases[(fx, fy)].entity = Flame(fx, fy)
        self.map.cases[(p.pos_x, p.pos_y)].entity = p
        self.waves_spawned = state["waves_spawned"]
        self.turn = state["turn"]
        self.done = state["done"]
        self.won = state["won"]
        self.spawn_pattern = list(state["spawn_pattern"])
        self.previsualiation = list(state["previsu"])
        self.spell = (p.spells[state["spell_idx"]]
                      if state["spell_idx"] is not None else None)
        self.last_reward = dict(state["last_reward"])
        self._rng.setstate(state["rng"])
        self.patterns._rng.setstate(state["prng"])
        self.patterns.spawn_patterns = list(state["spawn_patterns"])

    def reset(self, seed: int | None = None) -> None:
        """Re-initialise all game state, keeping the current difficulty."""
        self.__init__(  # type: ignore[misc]
            self.player, seed=seed,
            num_waves=self.num_waves, flames_per_wave=self.flames_per_wave,
            spawn_radius=self.spawn_radius,
            spawn_formation=self.spawn_formation)

    def _make_wave(
        self, pattern: list[tuple[int, int]]
    ) -> list[tuple[int, int]]:
        """Shape a drawn spawn wave for the current curriculum difficulty.

        ``spawn_formation`` wins over ``spawn_radius``, which wins over the
        drawn ``pattern`` (merely capped to ``flames_per_wave``).
        """
        if self.spawn_formation == "line":
            return self._line_spawn(self.flames_per_wave or 3)
        if self.spawn_radius is not None:
            return self._near_spawn(self.flames_per_wave or 1)
        if self.flames_per_wave is None:
            return pattern
        return pattern[:self.flames_per_wave]

    def _near_spawn(self, k: int, radius: int | None = None) -> \
            list[tuple[int, int]]:
        """``k`` random free tiles within ``radius`` (default ``spawn_radius``)
        of the player."""
        if radius is None:
            radius = self.spawn_radius
        px, py = self.player.pos_x, self.player.pos_y
        occupied = self._occupied_tiles()
        candidates = [
            pos for pos, case in self.map.cases.items()
            if case.case_type == CaseType.FREE
            and pos not in occupied
            and 0 < abs(pos[0] - px) + abs(pos[1] - py) <= radius
        ]
        self._rng.shuffle(candidates)
        return candidates[:k]

    def _line_spawn(self, k: int) -> list[tuple[int, int]]:
        """``k`` contiguous flames on the player's row or column.

        The line starts ``SPAWN_LINE_OFFSET`` tiles from the player and runs
        away from them, so the nearest flame is within LongJump range while the
        far one is not. Directions are tried in random order; a direction is
        usable only if all ``k`` tiles are free and unoccupied. Falls back to
        :meth:`_near_spawn` when the player is too close to an edge for any
        direction to fit.
        """
        px, py = self.player.pos_x, self.player.pos_y
        occupied = self._occupied_tiles()
        directions = [(1, 0), (-1, 0), (0, 1), (0, -1)]
        self._rng.shuffle(directions)
        for dx, dy in directions:
            line = [
                (px + dx * (constant.SPAWN_LINE_OFFSET + i),
                 py + dy * (constant.SPAWN_LINE_OFFSET + i))
                for i in range(k)
            ]
            if all(
                (case := self.map.cases.get(pos)) is not None
                and case.case_type == CaseType.FREE
                and pos not in occupied
                for pos in line
            ):
                return line
        return self._near_spawn(k, radius=constant.SPAWN_LINE_OFFSET + k)

    def select_spell(self, spell_index: int) -> None:
        """Select a spell and compute its previsualisation tiles.

        Does nothing for an out-of-range index; clears the previsualisation
        if the spell is not castable (not enough AP or uses exhausted).
        """
        if spell_index < 0 or spell_index >= len(self.player.spells):
            return
        spell: Spells = self.player.spells[spell_index]
        if not spell.is_castable(self.player):
            self.clear_previsu()
            return
        self.spell = spell
        self.previsualiation = spell.previsu(
            (self.player.pos_x, self.player.pos_y), self.map.cases)

    def play_selected_spell(
            self,
            tile_clicked: tuple[int, int] | None = None,
    ) -> None:
        """Play the selected spell on the clicked tile, if valid."""
        if not self.previsualiation or tile_clicked is None:
            return
        if self.spell is None:
            return
        self.spell.play(self.map, self.player, tile_clicked)
        self.clear_previsu()
        self._check_end()

    def clear_previsu(self) -> None:
        """Discard the current previsualisation tiles."""
        self.previsualiation = []

    def end_turn(self) -> None:
        """Advance one turn: refresh AP/spells and spawn the next wave.

        Exactly ``self.num_waves`` distinct waves spawn over a game (one per
        end-turn until the quota is reached, none after). Once every wave has
        spawned, clearing all flames from the map wins the game.
        """
        self.player.pa = self.player.base_PA
        for spell in self.player.spells:
            spell.next_turn()
        if self.spawn_pattern:
            for pos in self.spawn_pattern:
                self.map.place_entity([pos], Flame(pos[0], pos[1]))
            self.waves_spawned += 1
            self.turn += 1
            self.spawn_pattern = (
                self._make_wave(self.patterns.draw(self._occupied_tiles()))
                if self.waves_spawned < self.num_waves
                else []
            )
        if self.waves_spawned >= self.num_waves:
            self.player.hp -= 1
        self.clear_previsu()
        self._check_end()

    def _occupied_tiles(self) -> set[tuple[int, int]]:
        """Tiles currently holding a flame or the player."""
        return self.map.flames | {(self.player.pos_x, self.player.pos_y)}

    def _flames_remain(self) -> bool:
        """Whether any flame is still on the map."""
        return bool(self.map.flames)

    def _check_end(self) -> None:
        """Resolve win/loss: dead player loses; all waves cleared wins."""
        if self.player.hp <= 0:
            self.done = True
        elif self.waves_spawned >= self.num_waves and \
                not self._flames_remain():
            self.done = True
            self.won = True

    # ------------------------------------------------------------------
    # RL environment interface: fixed-size observation, discrete action space.
    # ------------------------------------------------------------------

    def _flame_positions(self) -> list[tuple[int, int]]:
        """Grid positions of every flame currently on the map.

        Iterates cells in map order and filters by the O(1) flame set — same
        order as the old full ``isinstance`` scan (the observation's K-nearest
        tie-breaks on it, so the order must be byte-identical), but with a cheap
        set-membership test instead of an ABC ``isinstance`` per cell.
        """
        flames = self.map.flames
        return [pos for pos in self.map.cases if pos in flames]

    @staticmethod
    def _nearest_target_dist(
        targets: list[tuple[int, int]],
        pos: tuple[int, int],
    ) -> int:
        """Manhattan distance from ``pos`` to the nearest tile in ``targets``.

        Returns 0 when there are no targets, so the approach reward vanishes
        rather than rewarding an arbitrary default.
        """
        if not targets:
            return 0
        px, py = pos
        return min(abs(x - px) + abs(y - py) for x, y in targets)

    def _nearest_relative(
        self,
        positions: list[tuple[int, int]],
        k: int,
    ) -> list[float]:
        """Encode the ``k`` nearest ``positions`` relative to the player.

        Returns ``3 * k`` floats: for each slot a presence flag (1/0) then the
        normalised ``dx, dy`` toward the player. Empty slots are all zeros.
        """
        px, py = self.player.pos_x, self.player.pos_y
        gx = self.map.grid_max_x or 1
        gy = self.map.grid_max_y or 1
        nearest = sorted(
            positions,
            key=lambda p: abs(p[0] - px) + abs(p[1] - py),
        )[:k]
        out: list[float] = []
        for i in range(k):
            if i < len(nearest):
                x, y = nearest[i]
                out.extend((1.0, (x - px) / gx, (y - py) / gy))
            else:
                out.extend((0.0, 0.0, 0.0))
        return out

    def _grid_observation(self):
        """Return the ``OBS_MODE == "grid"`` observation for a CNN.

        A flat ``float32`` numpy array = ``GRID_CHANNELS`` binary board planes
        (``H*W`` each, indexed ``[x, y]``) followed by ``GRID_SCALARS`` scalar
        features. Channels: 0=free-cell mask, 1=flames, 2=next-spawn tiles,
        3=player, 4=bolgrot. Returned flat (the CNN reshapes) so the replay
        buffer, masking and batching stay identical to the flat path.

        Stored as numpy (not ``list[float]``) because a ``C*H*W`` Python list
        per sample would blow the replay buffer to ~10 GB; float32 keeps it
        ~1.4 GB.
        """
        import numpy as np
        H, W, C = constant.GRID_H, constant.GRID_W, constant.GRID_CHANNELS
        plane = H * W
        grid = np.zeros(C * plane + constant.GRID_SCALARS, dtype=np.float32)

        def _set(channel: int, x: int, y: int) -> None:
            if 0 <= x < H and 0 <= y < W:
                grid[channel * plane + x * W + y] = 1.0
            elif channel != 0:  # a live entity must fit; the free-mask may not
                raise ValueError(
                    f"grid cell ({x},{y}) outside {H}x{W} — raise GRID_H/W")

        for (x, y), case in self.map.cases.items():
            if case.case_type == CaseType.FREE and 0 <= x < H and 0 <= y < W:
                grid[x * W + y] = 1.0                      # channel 0: free
        for x, y in self._flame_positions():
            _set(1, x, y)
        for x, y in self.spawn_pattern:
            _set(2, x, y)
        _set(3, self.player.pos_x, self.player.pos_y)
        _set(4, self.bolgrot.pos_x, self.bolgrot.pos_y)

        p = self.player
        tail = C * plane
        grid[tail + 0] = p.hp / constant.MAX_HP
        grid[tail + 1] = p.pa / p.base_PA if p.base_PA else 0.0
        grid[tail + 2] = self.waves_spawned / self.num_waves
        grid[tail + 3] = len(self._flame_positions()) / 20.0
        for i, spell in enumerate(p.spells[:3]):
            grid[tail + 4 + i] = 1.0 if spell.is_castable(p) else 0.0
        return grid

    def get_observation(self) -> list[float]:
        """Return the fixed-length float vector fed to the neural network.

        Layout (length ``Game.observation_size()``):
        ``[px, py, hp, pa, waves, flame_count, boss_dx, boss_dy,
        castable_0, castable_1, castable_2]`` followed by the
        ``K_NEAREST_FLAMES`` nearest flames and ``K_NEAREST_SPAWN`` nearest
        next-spawn tiles, each as ``(present, dx, dy)``. All values are
        normalised to roughly ``[-1, 1]``.

        When ``constant.OBS_MODE == "grid"`` this instead returns the 2D
        multi-channel board (:meth:`_grid_observation`) for a CNN.
        """
        if constant.OBS_MODE == "grid":
            return self._grid_observation()
        p = self.player
        gx = self.map.grid_max_x or 1
        gy = self.map.grid_max_y or 1
        flames = self._flame_positions()
        scalars: list[float] = [
            p.pos_x / gx,
            p.pos_y / gy,
            p.hp / constant.MAX_HP,
            p.pa / p.base_PA if p.base_PA else 0.0,
            self.waves_spawned / self.num_waves,
            len(flames) / 20.0,
            (self.bolgrot.pos_x - p.pos_x) / gx,
            (self.bolgrot.pos_y - p.pos_y) / gy,
        ]
        scalars.extend(
            1.0 if spell.is_castable(p) else 0.0 for spell in p.spells
        )
        obs = scalars
        obs += self._nearest_relative(flames, constant.K_NEAREST_FLAMES)
        obs += self._nearest_relative(self.spawn_pattern,
                                      constant.K_NEAREST_SPAWN)
        return obs

    @staticmethod
    def observation_size() -> int:
        """Length of the vector returned by :meth:`get_observation`."""
        if constant.OBS_MODE == "grid":
            return (constant.GRID_CHANNELS * constant.GRID_H * constant.GRID_W
                    + constant.GRID_SCALARS)
        return (11 + 3 * constant.K_NEAREST_FLAMES
                + 3 * constant.K_NEAREST_SPAWN)

    def legal_actions(self) -> list[bool]:
        """Boolean mask over :data:`Game.ACTIONS` of the currently legal moves.

        End-turn is always legal while the game is live; a spell action is
        legal only if the spell is castable and its target tile is in the
        spell's previsualisation. All actions are illegal once ``done``.
        """
        mask = [False] * len(self.ACTIONS)
        if self.done:
            return mask
        px, py = self.player.pos_x, self.player.pos_y
        previs: dict[int, set[tuple[int, int]]] = {}
        for idx, spell in enumerate(self.player.spells):
            if spell.is_castable(self.player):
                previs[idx] = set(spell.previsu((px, py), self.map.cases))
        for i, (spell_index, (dx, dy)) in enumerate(self.ACTIONS):
            if spell_index is None:
                mask[i] = True
            elif spell_index in previs and \
                    (px + dx, py + dy) in previs[spell_index]:
                mask[i] = True
        return mask

    @staticmethod
    def _zero_reward() -> dict[str, float]:
        """A fresh reward-breakdown dict with every term at zero."""
        return {part: 0.0 for part in REWARD_PARTS}

    def step(self, action_index: int) -> tuple[float, bool]:
        """Apply one discrete action and return ``(reward, done)``.

        ``action_index`` indexes :data:`Game.ACTIONS`. Illegal actions are
        no-ops with a small penalty. The reward is built so that *active*
        play beats sitting still: clearing flames is the dominant term (it is
        the only real progress toward a win), survival pays only a token
        amount, and flame *spawns* are never penalised — only genuine
        reductions in flame count earn ``REWARD_CLEAR_FLAME``.

        On top of those outcome rewards, two shaping terms guide *style*: an
        approach reward (``REWARD_APPROACH`` per tile the player moves closer to
        the nearest flame / next-spawn tile) and a per-spell cast bias
        (``REWARD_CAST[spell_index]`` — favouring the short jump over the long
        jump, and penalising the "Inaction" flame-drag). See ``constant.py``.
        """
        if self.done or not 0 <= action_index < len(self.ACTIONS):
            self.last_reward = self._zero_reward()
            return 0.0, self.done

        spell_index, (dx, dy) = self.ACTIONS[action_index]
        parts = self._zero_reward()
        if not self.legal_actions()[action_index]:
            parts["illegal"] = constant.REWARD_ILLEGAL
            self.last_reward = parts
            return constant.REWARD_ILLEGAL, self.done

        hp_before = self.player.hp
        flames_before = len(self._flame_positions())
        # Snapshot the approach targets (current flames + the next wave's spawn
        # tiles) and the player's distance to the nearest one. Measured against
        # a *fixed* snapshot with the player's old vs new position, so only the
        # player actually moving (the two jumps) earns the approach reward —
        # ending a turn or dragging flames leaves the player put.
        approach_targets = self._flame_positions() + list(self.spawn_pattern)
        dist_before = self._nearest_target_dist(
            approach_targets, (self.player.pos_x, self.player.pos_y))

        if spell_index is None:
            # Anti-stall: penalise ending a turn with flames still alive on
            # the board (measured *before* end_turn spawns the next wave, so
            # only flames the agent chose to leave count — spawns aren't hit).
            parts["stall"] = constant.REWARD_END_TURN_PER_FLAME * flames_before
            self.end_turn()
            parts["survive"] = constant.REWARD_SURVIVE_TURN
        else:
            target = (self.player.pos_x + dx, self.player.pos_y + dy)
            self.select_spell(spell_index)
            self.play_selected_spell(target)
            parts["cast"] = constant.REWARD_CAST[spell_index]

        dist_after = self._nearest_target_dist(
            approach_targets, (self.player.pos_x, self.player.pos_y))
        parts["approach"] = \
            constant.REWARD_APPROACH * (dist_before - dist_after)

        flames_after = len(self._flame_positions())
        # Reward only flame *reductions*: the wave spawned on an end-turn adds
        # flames the agent cannot prevent, so increases must not be punished.
        cleared = flames_before - flames_after
        if cleared > 0:
            parts["clear"] = constant.REWARD_CLEAR_FLAME * cleared
        parts["hp"] = constant.REWARD_HP_DELTA * (self.player.hp - hp_before)

        if self.done:
            if self.won:
                parts["terminal"] = constant.REWARD_WIN
            else:
                # Death: base cost plus a penalty per wave left unspawned, so
                # dying early (much game left) is far worse than dying on the
                # brink — clearing a few flames then dying must never pay off.
                # Scale by THIS game's wave count (``self.num_waves``), not
                # the full-game ``NB_WAVES``: on a shrunk curriculum stage,
                # surviving all its waves then dying must read as "on the
                # brink" (0 left), not as "4 left" of a 6-wave game. At full
                # difficulty ``num_waves == NB_WAVES`` so nothing changes.
                waves_left = self.num_waves - self.waves_spawned
                parts["terminal"] = (
                    constant.REWARD_DEATH
                    + constant.REWARD_DEATH_PER_WAVE_LEFT * waves_left)
        self.last_reward = parts
        return sum(parts.values()), self.done
