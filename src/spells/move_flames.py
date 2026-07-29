from __future__ import annotations
from typing import TYPE_CHECKING

from ..BFS import BFS
from .spells import Spells, TypeSpell

if TYPE_CHECKING:
    from ..entity import Player
    from ..map import Map


class MoveFlames(Spells):
    """Attract all flames one tile toward a target; costs 5 HP, can't kill."""

    def __init__(
            self,
            name: str = "Inaction",
            description: str = "",
            cost: int = 1,
            max_use: int = 99,
            effects: list[str] | None = None,
            type_spell: list[tuple[TypeSpell, int]] | None = None,
            bfs: BFS | None = None,
            line_of_sight: bool = False,
            sprite: str = "inaction.png"
    ):
        """Configure the spell with DIAGONAL range 1 and no line-of-sight."""
        super().__init__(
            name, description, cost, max_use,
            effects, type_spell, bfs, line_of_sight, sprite)
        self.description: str = "Les ennemis sont attirés sur la case." \
            "Le lanceur perd 5 PV." \
            "Les ennemis ne peuvent pas tuer le lanceur avec ce sort."
        self.time_used: int = 0
        self.type_spell: list[tuple[TypeSpell, int]] = [
            (TypeSpell.DIAGONAL, 1)
        ]
        self.line_of_sight: bool = line_of_sight

    def play(
        self,
        map: Map,
        player: Player,
        tile_clicked: tuple[int, int],
    ) -> None:
        """Attract every flame one tile toward ``tile_clicked``.

        No-op unless affordable and uses remain. Flames cannot kill the
        caster here (``killable=False``); the caster loses 5 HP and a use.
        """
        if player.pa < self.cost or self.time_used >= self.max_use:
            return
        self.attract_flames(map.cases, tile_clicked=tile_clicked,
                            killable=False)
        player.hp -= 5
        player.pa -= self.cost
        self.time_used += 1

    def next_turn(self):
        """Reset the per-turn usage count so the spell can be cast again."""
        self.time_used = 0
