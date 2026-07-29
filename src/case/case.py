from __future__ import annotations
from enum import Enum
import pygame
from ..entity import Entity, Flame
from .. import constant


class CaseType(Enum):
    """Walkability of a cell: free, a sight/movement-blocking wall, or void."""
    FREE = 0
    WALL = 1
    EMPTY = 2


class Case:
    """A single grid cell holding its coordinates, type and optional entity."""

    def __init__(
        self,
        x: int,
        y: int,
        entity: Entity | None = None,
        case_type: CaseType = CaseType.FREE,
    ) -> None:
        """Create a cell at grid (x, y) with an optional entity and type."""
        self.x = x
        self.y = y
        self.case_type: CaseType = case_type
        # ``_flames`` is a Map-shared set of flame positions, injected by Map
        # after construction; the ``entity`` setter keeps it in sync so callers
        # never have to scan every cell to find the flames (the search's hot
        # path). None until injected -> the setter is a no-op during __init__.
        self._flames: set[tuple[int, int]] | None = None
        self._entity: Entity | None = None
        self.entity = entity

    @property
    def entity(self) -> Entity | None:
        return self._entity

    @entity.setter
    def entity(self, value: Entity | None) -> None:
        flames = self._flames
        if flames is not None:
            was = type(self._entity) is Flame
            now = type(value) is Flame
            if was and not now:
                flames.discard((self.x, self.y))
            elif now and not was:
                flames.add((self.x, self.y))
        self._entity = value

    def contains(
        self,
        mouse_x: int,
        mouse_y: int,
        offset: tuple[int, int],
    ) -> bool:
        """Return True if the mouse (screen px) maps to this cell's grid pos.

        Inverts the isometric projection using ``offset`` (the screen pixel
        position of iso origin) to recover grid coordinates.
        """
        rx = mouse_x - offset[0]
        ry = mouse_y - offset[1]
        half_w = constant.CASE_WIDTH / 2
        half_h = constant.CASE_HEIGHT / 2
        gx = round((ry / half_h + rx / half_w) / 2)
        gy = round((ry / half_h - rx / half_w) / 2)
        return (gx, gy) == (self.x, self.y)

    def draw(
        self,
        screen: pygame.Surface,
        offset: tuple[int, int],
        color: tuple[int, int, int],
        font: pygame.font.Font | None = None,
        show_coords: bool = False,
    ) -> None:
        """Draw this cell as an isometric diamond, optionally labelled.

        ``offset`` is the screen pixel position of iso origin; ``color`` fills
        the diamond and a black outline is drawn on top. When ``show_coords``
        is set and a ``font`` is given, the grid coordinates are blitted.
        """
        iso_x = int((self.x - self.y) * (constant.CASE_WIDTH / 2))
        iso_y = int((self.x + self.y) * (constant.CASE_HEIGHT / 2))
        cx = iso_x + offset[0]
        cy = iso_y + offset[1]
        top = (cx, cy - constant.CASE_HEIGHT / 2)
        right = (cx + constant.CASE_WIDTH / 2, cy)
        bottom_pt = (cx, cy + constant.CASE_HEIGHT / 2)
        left = (cx - constant.CASE_WIDTH / 2, cy)
        pygame.draw.polygon(screen, color, [top, right, bottom_pt, left])
        pygame.draw.polygon(screen, (0, 0, 0), [
                            top, right, bottom_pt, left], 1)
        if show_coords and font:
            text = font.render(f"{self.x},{self.y}", True, (0, 0, 0))
            screen.blit(text, text.get_rect(center=(cx, cy)))
