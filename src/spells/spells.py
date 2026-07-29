from __future__ import annotations
import copy
import os
import pygame
from abc import ABC, abstractmethod
from enum import Enum
from typing import TYPE_CHECKING

from ..BFS import BFS
from ..entity import Entity, Flame, Bolgrot, Player
from ..case import CaseType


if TYPE_CHECKING:
    from ..map import Map
    from ..case import Case


class Direction(Enum):
    """Unit step vectors plus the orthogonal/diagonal direction groups."""
    NORTH = (-1, 0)
    EAST = (0, 1)
    SOUTH = (1, 0)
    WEST = (0, -1)

    NORTH_WEST = (-1, -1)
    NORTH_EAST = (-1, 1)
    SOUTH_WEST = (1, -1)
    SOUTH_EAST = (1, 1)

    DIRECTIONS_LINE = [
        NORTH,
        EAST,
        SOUTH,
        WEST
    ]

    DIRECTIONS_DIAGONALS = [
        NORTH_WEST,
        NORTH_EAST,
        SOUTH_WEST,
        SOUTH_EAST
    ]


class TypeSpell(Enum):
    """Range shape of a spell: straight LINE, DIAGONAL or FULL area."""
    LINE = 2
    DIAGONAL = 4
    FULL = 8


class Spells(ABC):
    """Abstract spell: cost/usage, range shape, targeting and flame logic."""

    def __init__(
            self,
            name: str,
            description: str,
            cost: int,
            max_use: int,
            effects: list[str] | None,
            type_spell: list[tuple[TypeSpell, int]] | None,
            bfs: BFS | None,
            line_of_sight: bool = True,
            sprite=None
    ):
        """Store the spell's stats, range shape, BFS helper and sprite name.

        ``effects``/``type_spell`` default to empty lists when ``None``.
        """
        super().__init__()
        self.sprite = sprite
        self.name: str = name
        self.description: str = description
        self.cost: int = cost
        self.max_use: int = max_use
        self.effects: list[str] = effects if effects is not None else []
        self.type_spell: list[tuple[TypeSpell, int]] = \
            type_spell if type_spell is not None else []
        self.line_of_sight: bool = line_of_sight
        self._image: pygame.Surface | None = None
        self.BFS: BFS | None = bfs
        self.time_used: int = 0

    def __deepcopy__(self, memo: dict) -> "Spells":
        """Deep-copy the spell without its cached sprite surface.

        ``Game.clone()`` deep-copies the whole game for the search agents. The
        sprite ``_image`` is a lazily-loaded ``pygame.Surface`` (populated only
        once the game has been *rendered*) and a Surface can't be deep-copied,
        so we skip it — it reloads on next access. This keeps ``clone()`` safe
        whether or not the game has been drawn.
        """
        cls = self.__class__
        new = cls.__new__(cls)
        memo[id(self)] = new
        for key, value in self.__dict__.items():
            if key == "_image":
                new._image = None
            else:
                setattr(new, key, copy.deepcopy(value, memo))
        return new

    def is_castable(self, player: Player) -> bool:
        """True if the player has enough PA and uses left to cast."""
        return player.pa >= self.cost and self.time_used < self.max_use

    @abstractmethod
    def play(
        self,
        map: Map,
        player: Player,
        tile_clicked: tuple[int, int],
    ) -> None:
        """Apply the spell's effect for a cast on ``tile_clicked``."""
        pass

    @abstractmethod
    def next_turn(
        self,
    ) -> None:
        """Reset per-turn state (e.g. usage counters) at end of turn."""
        pass

    @property
    def image(self) -> pygame.Surface:
        """Lazily load and cache the spell's sprite surface."""
        if self._image is None:
            from .. import constant
            self._image = pygame.image.load(
                os.path.join(constant.SPRITES_DIR, self.sprite))
        return self._image

    def contains(
        self,
        mouse_x: int,
        mouse_y: int,
        x: int,
        y: int,
    ) -> bool:
        """Return True if the mouse is over the spell icon drawn at (x, y)."""
        img = self.image
        return (x <= mouse_x < x + img.get_width()
                and y <= mouse_y < y + img.get_height())

    def _draw_tooltip(
        self,
        screen: pygame.Surface,
        pos_x: int,
        pos_y: int,
        font_title: pygame.font.Font,
        font_txt: pygame.font.Font,
    ) -> None:
        """Draw the hover tooltip with the name, AP cost, effects and text."""
        from .. import constant
        height_rect = 300
        pygame.draw.rect(screen, constant.BACKGROUND_POPUP,
                         [pos_x, pos_y - height_rect, 400, height_rect])
        screen.blit(
            font_title.render(self.name, True, (255, 255, 255)),
            (pos_x, pos_y - height_rect),
        )
        screen.blit(
            font_txt.render(f"Cost: {self.cost} AP", True, (255, 255, 255)),
            (pos_x, pos_y - height_rect + 50),
        )
        for i, e in enumerate(self.effects):
            screen.blit(
                font_txt.render(str(e), True, (255, 255, 255)),
                (pos_x,
                 pos_y - height_rect + 100 + font_txt.get_height() * i),
            )
        for i, d in enumerate(self.description.split(".")):
            screen.blit(
                font_txt.render(d, True, (255, 255, 255)),
                (pos_x,
                 pos_y - height_rect + 200 + font_txt.get_height() * i),
            )

    def draw(
        self,
        screen: pygame.Surface,
        x: int,
        y: int,
        mouse_x: int,
        mouse_y: int,
        font_title: pygame.font.Font,
        font_txt: pygame.font.Font,
    ) -> None:
        """Blit the spell icon at (x, y), drawing the tooltip while hovered."""
        if self.contains(mouse_x, mouse_y, x, y):
            self._draw_tooltip(screen, x, y, font_title, font_txt)
        screen.blit(self.image, (x, y))

    def previsu(
        self,
        pos_player: tuple[int, int],
        cases: dict[tuple[int, int], Case]
    ) -> list[tuple]:
        """Return the tiles this spell can target from ``pos_player``.

        Combines the reachable tiles for each (shape, range) in
        ``type_spell`` (line / diagonal), honouring line-of-sight.
        """
        all_pos: list[tuple] = []
        for type_s in self.type_spell:
            t, r = type_s
            match t.value:
                case 2:
                    all_pos.extend(self.make_line(
                        pos_player,
                        cases,
                        r
                    ))
                case 4:
                    all_pos.extend(self.make_diag(
                        pos_player,
                        cases,
                        r
                    ))
                case 8:
                    print("FULL")
        return all_pos

    def make_line(
        self,
        pos_player: tuple[int, int],
        cases: dict[tuple[int, int], Case],
        range_s: int
    ) -> list[tuple]:
        """Return reachable tiles up to ``range_s`` along the 4 orthogonals.

        Stops including tiles past the map edge, a sight blocker, or an
        unkillable entity (line-of-sight permitting).
        """
        x, y = pos_player
        possible_pos: list[tuple] = []
        for direction in Direction.DIRECTIONS_LINE.value:
            dx, dy = direction
            nx, ny = x, y
            for _ in range(range_s):
                nx, ny = nx + dx, ny + dy
                if self.is_in_map((nx, ny), cases) and \
                        not self.is_blocked_by_sight(
                            (nx, ny), cases, direction) and \
                   self.is_entity_killable((nx, ny), cases):
                    possible_pos.append((nx, ny))
        return possible_pos

    def make_diag(
        self,
        pos_player: tuple[int, int],
        cases: dict[tuple[int, int], Case],
        range_s: int
    ) -> list[tuple]:
        """Return reachable tiles up to ``range_s`` along the 4 diagonals.

        Same edge/sight/killable rules as ``make_line``.
        """
        x, y = pos_player
        possible_pos: list[tuple] = []
        for direction in Direction.DIRECTIONS_DIAGONALS.value:
            dx, dy = direction
            nx, ny = x, y
            for _ in range(range_s):
                nx, ny = nx + dx, ny + dy
                if self.is_in_map((nx, ny), cases) and \
                        not self.is_blocked_by_sight(
                            (nx, ny), cases, direction) and \
                   self.is_entity_killable((nx, ny), cases):
                    possible_pos.append((nx, ny))
        return possible_pos

    def _find_case(
        self,
        pos: tuple[int, int],
        cases: dict[tuple[int, int], Case]
    ) -> Case | None:
        """Return the cell at ``pos`` (O(1) dict lookup), or ``None``."""
        return cases.get(pos)

    def is_entity_killable(
        self,
        pos_spell: tuple[int, int],
        cases: dict[tuple[int, int], Case]
    ) -> bool:
        """Return True if ``pos_spell`` may be targeted past its entity.

        With line-of-sight off, always True. Otherwise an entity there
        blocks targeting beyond it unless it is killable.
        """
        if self.line_of_sight is False:
            return True
        c = self._find_case(pos_spell, cases)
        if c is not None and isinstance(c.entity, Entity):
            return c.entity.killable
        return True

    def is_blocked_by_sight(
        self,
        pos_spell: tuple[int, int],
        cases: dict[tuple[int, int], Case],
        direction: tuple[int, int],
    ) -> bool:
        """Return True if the tile behind ``pos_spell`` blocks the sight line.

        Looks one step back along ``direction``; an entity there with
        ``blocks_sight`` cuts the line. Always False when line-of-sight off.
        """
        if self.line_of_sight is False:
            return False

        dx, dy = direction
        dx, dy = -1 * dx, -1 * dy
        nx, ny = pos_spell[0] + dx, pos_spell[1] + dy
        c = self._find_case((nx, ny), cases)
        return c is not None and c.entity is not None and c.entity.blocks_sight

    def is_in_map(
        self,
        pos_spell: tuple[int, int],
        cases: dict[tuple[int, int], Case]
    ) -> bool:
        """Return True if ``pos_spell`` corresponds to a cell on the map."""
        return self._find_case(pos_spell, cases) is not None

    def attract_flames(
        self,
        cases: dict[tuple[int, int], Case],
        tile_clicked: tuple[int, int] | None = None,
        player: Player | None = None,
        killable: bool = True
    ) -> None:
        """Move every flame one tile toward the target, nearest-first.

        The target is ``tile_clicked`` or the player's position (one is
        required). Each flame steps along its dominant axis (diagonally when
        equal) so it stays in its quadrant, falling back to BFS around walls.
        When ``killable`` is set, a flame reaching the player kills them.
        """
        assert self.BFS is not None
        if tile_clicked is not None:
            pos_x, pos_y = tile_clicked
        elif player is not None:
            pos_x, pos_y = player.pos_x, player.pos_y
        else:
            raise ValueError(
                "Either tile_clicked or player must be provided.")

        flame_cases = [
            case for case in cases.values()
            if type(case.entity) is Flame
            and (case.x, case.y) != (pos_x, pos_y)
        ]
        flame_cases.sort(
            key=lambda c: abs(c.x - pos_x) + abs(c.y - pos_y)
        )
        for case in flame_cases:
            dx = pos_x - case.x
            dy = pos_y - case.y
            sx = (dx > 0) - (dx < 0)
            sy = (dy > 0) - (dy < 0)
            if abs(dx) > abs(dy):
                step = (case.x + sx, case.y)
            elif abs(dy) > abs(dx):
                step = (case.x, case.y + sy)
            else:
                step = (case.x + sx, case.y + sy)
            new_case = self._find_case(step, cases)
            if new_case is None or new_case.case_type == CaseType.WALL:
                path = self.BFS.find_path(
                    (case.x, case.y),
                    (pos_x, pos_y)
                )
                new_case = self._find_case(path, cases) \
                    if path is not None else None
            if new_case is None or new_case.case_type == CaseType.WALL:
                continue
            if type(new_case.entity) is Bolgrot or type(new_case.entity) is Flame:
                continue
            elif type(new_case.entity) is Player and killable:
                if player is not None:
                    player.hp = 0
            elif new_case.entity is None:
                new_case.entity = case.entity
                case.entity = None

    def push_flames(
        self,
        player: Player,
        cases: dict[tuple[int, int], Case],
    ) -> None:
        """Push Flames away from the player position for 1 case if
        they are next to the player. (diagonals included)"""
        assert self.BFS is not None
        flame_cases = [
            case for case in cases.values()
            if type(case.entity) is Flame
            and (case.x, case.y) != (player.pos_x, player.pos_y)
            and abs(case.x - player.pos_x) <= 1
            and abs(case.y - player.pos_y) <= 1
        ]
        for case in flame_cases:
            dx = case.x - player.pos_x
            dy = case.y - player.pos_y
            sx = (dx > 0) - (dx < 0)
            sy = (dy > 0) - (dy < 0)
            if abs(dx) > abs(dy):
                step = (case.x + sx, case.y)
            elif abs(dy) > abs(dx):
                step = (case.x, case.y + sy)
            else:
                step = (case.x + sx, case.y + sy)
                for fx, fy in ((case.x + sx, case.y), (case.x, case.y + sy)):
                    flank = cases.get((fx, fy))
                    if flank is not None and type(flank.entity) is Flame:
                        player.hp = 0
            new_case = self._find_case(step, cases)
            if new_case is None or new_case.case_type == CaseType.WALL:
                path = self.BFS.find_path(
                    (case.x, case.y),
                    (player.pos_x, player.pos_y)
                )
                new_case = self._find_case(path, cases) \
                    if path is not None else None
            if new_case is None or new_case.case_type == CaseType.WALL:
                continue
            if type(new_case.entity) is Bolgrot:
                continue
            elif type(new_case.entity) is Flame or type(new_case.entity) is Player:
                player.hp = 0
            elif new_case.entity is None:
                new_case.entity = case.entity
                case.entity = None
