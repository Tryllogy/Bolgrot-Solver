from __future__ import annotations
from typing import TYPE_CHECKING
from .spells import Spells, TypeSpell
from ..BFS import BFS
from ..entity import Flame

if TYPE_CHECKING:
    from ..entity import Player
    from ..map import Map


class LongJump(Spells):
    """Teleport up to two tiles in a line; tracks per-turn usage count."""

    def __init__(
            self,
            name: str = "Double saut",
            description: str = "",
            cost: int = 2,
            max_use: int = 2,
            effects: list[str] | None = None,
            type_spell: list[tuple[TypeSpell, int]] | None = None,
            bfs: BFS | None = None,
            line_of_sight: bool = True,
            sprite: str = "long_jump.png"
    ):
        """Configure the spell with LINE range 2 and reset the usage count."""
        super().__init__(
            name, description, cost, max_use,
            effects, type_spell, bfs, line_of_sight, sprite)
        self.effects: list[str] = [
            "Se téléporte sur la case",
            "Attire les ennemis d'1 case",
            "-1 PV"
        ]
        self.description: str = "Se téléporte sur la case." \
            "Perd 1 PV." \
            "Peut cibler une case occupée pour tuer l'ennemi." \
            "Si un ennemi est tué, récupère 1 PV." \
            "Tous les ennemis sont attirés vers le lanceur" \
            " après la téléportation." \
            "Si un ennemi ne peut pas être déplacé, vous mourez."
        self.time_used: int = 0
        self.type_spell: list[tuple[TypeSpell, int]] = [
            (TypeSpell.LINE, 2)
        ]

    def play(
        self,
        map: Map,
        player: Player,
        tile_clicked: tuple[int, int],
    ) -> None:
        """Teleport (range 2) to ``tile_clicked`` and resolve its effects.

        No-op unless affordable, uses remain, and the target is valid. Same
        HP/flame resolution as ShortJump, and increments ``time_used``.
        """
        if player.pa < self.cost or self.time_used >= self.max_use:
            return
        if tile_clicked not in self.previsu(
                (player.pos_x, player.pos_y), map.cases):
            return
        src = self._find_case((player.pos_x, player.pos_y), map.cases)
        dst = self._find_case(tile_clicked, map.cases)
        killed_flame = False
        if src is None or dst is None:
            return
        if type(dst.entity) is Flame:
            killed_flame = True
            player.hp += 1
        src.entity = None
        player.pos_x, player.pos_y = tile_clicked
        player.hp -= 1
        player.pa -= self.cost
        dst.entity = player
        self.time_used += 1
        if killed_flame:
            self.push_flames(player, map.cases)
        self.attract_flames(map.cases, player=player)

    def next_turn(
        self
    ) -> None:
        """Reset the per-turn usage count so the spell can be cast again."""
        self.time_used = 0
