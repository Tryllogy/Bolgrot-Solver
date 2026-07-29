from abc import ABC
from enum import Enum


class TypeEntity(Enum):
    """Discriminator used by the renderer to draw each entity kind."""
    NONE = 0
    FLAME = 1
    PLAYER = 2
    BOLGROT = 3


class Entity(ABC):
    """Abstract base for anything that can occupy a grid cell."""

    pos_x: int
    pos_y: int

    def __init__(self) -> None:
        """Initialise the shared entity flags (sight-blocking, killable)."""
        self.type_entity: TypeEntity | None = None
        self.blocks_sight: bool = True
        self.killable: bool = True
