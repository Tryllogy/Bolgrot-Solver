from __future__ import annotations
from ..entity import Entity
from .. import constant
from ..case import Case, CaseType

SYMBOL_MAP: dict[str, CaseType] = {
    '.': CaseType.FREE,
    '#': CaseType.WALL,
    "|": CaseType.EMPTY,
}


class Map:
    """The diamond-shaped board: a ``(x, y) -> Case`` dict built from file."""

    def __init__(
        self,
        map_conf: str = constant.MAP_CONF,
    ) -> None:
        """Build the map by parsing the map file at ``map_conf``."""
        # Live set of flame positions, kept in sync by ``Case.entity``'s setter
        # so the RL/search hot path reads flames in O(#flames) instead of
        # scanning every cell. Shared by reference into each Case (below).
        self.flames: set[tuple[int, int]] = set()
        self.cases: dict[tuple[int, int], Case] = {}
        self.parse_map(map_conf)

    def parse_map(
        self,
        map_conf: str
    ) -> None:
        """Parse the isometric symbol grid into ``self.cases``.

        Each line is walked diagonally to assign grid coordinates; ``N``
        skips a slot (no cell), other symbols map via ``SYMBOL_MAP``.
        Sets ``grid_max_x``/``grid_max_y`` to the parsed bounds.
        """
        with open(map_conf, "r") as f:
            lines = [line.rstrip('\n').replace(" ", "")
                     for line in f if line.strip()]

        start_x: int = 0
        start_y: int = len(lines[0]) - 1
        for n_line, line in enumerate(lines):
            x: int = start_x
            y: int = start_y
            for symbol in line:
                if symbol == "N":
                    y -= 1
                    x += 1
                    continue
                case = Case(
                    x, y, case_type=SYMBOL_MAP.get(symbol, CaseType.FREE))
                case._flames = self.flames   # share the live flame-position set
                self.cases[(x, y)] = case
                x += 1
                y -= 1
            if n_line % 2 == 1:
                start_x += 1
            if n_line % 2 == 0:
                start_y += 1

        self.grid_max_x = max(case.x for case in self.cases.values())
        self.grid_max_y = max(case.y for case in self.cases.values())

    def place_entity(
        self,
        pattern: list[tuple[int, int]],
        entity: Entity
    ) -> None:
        """Place ``entity`` on the first valid cell in ``pattern``.

        Skips positions not on the map; a cell already holding the player is
        left untouched. Raises ``ValueError`` if ``entity`` is not an
        ``Entity`` or no pattern position matches a cell.
        """
        from ..entity import Player
        if not isinstance(entity, Entity):
            raise ValueError(
                "Entity must be an instance of Bolgrot or Player.")
        for x, y in pattern:
            case = self.cases.get((x, y))
            if case is None:
                continue
            if type(case.entity) is Player:
                return
            case.entity = entity
            return
        raise ValueError(
            f"Invalid position ({entity.pos_x}, "
            f"{entity.pos_y}) for entity {entity}. No matching case found.")
