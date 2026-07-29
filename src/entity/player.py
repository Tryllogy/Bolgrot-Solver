from __future__ import annotations
from typing import TYPE_CHECKING
from .entity import Entity, TypeEntity
from .. import constant

if TYPE_CHECKING:
    from ..spells import Spells


class Player(Entity):
    """The controllable character: tracks position, HP, AP and spells."""

    def __init__(
            self,
            x: int,
            y: int,
    ) -> None:
        """Create the player at (x, y) with default HP/AP and no spells.

        The spell list is populated by ``Game`` once the map (and therefore
        the BFS pathfinder) exists.
        """
        super().__init__()
        self.type_entity: TypeEntity = TypeEntity.PLAYER
        self.blocks_sight: bool = False
        self.pos_x: int = x
        self.pos_y: int = y
        self.hp: int = constant.MAX_HP
        self.base_PA: int = 10
        self.pa: int = self.base_PA
        self.spells: list[Spells] = []
