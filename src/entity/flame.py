from .entity import Entity, TypeEntity


class Flame(Entity):
    """A flame tile that threatens the player and can be attracted/pushed."""

    def __init__(
            self,
            x: int,
            y: int
    ) -> None:
        """Create a flame at grid position (x, y)."""
        super().__init__()
        self.type_entity: TypeEntity = TypeEntity.FLAME
        self.pos_x: int = x
        self.pos_y: int = y
