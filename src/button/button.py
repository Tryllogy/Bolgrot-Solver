from __future__ import annotations
import pygame
from .. import constant


class Button:
    """A fixed-size labelled rectangle that hit-tests and draws itself."""

    def __init__(
        self,
        screen: pygame.Surface,
        font: pygame.font.Font,
        x: int,
        y: int,
        label: str,
    ):
        """Store the surface, font, top-left position and label text."""
        self.screen = screen
        self.font = font
        self.x = x
        self.y = y
        self.label = label

    def contains(self, mouse_x: int, mouse_y: int) -> bool:
        """Return True if the mouse position is inside the button's rect."""
        return (self.x <= mouse_x < self.x + constant.BUTTON_W
                and self.y <= mouse_y < self.y + constant.BUTTON_H)

    def draw(self, mouse_x: int, mouse_y: int) -> None:
        """Draw the button, highlighting it while the mouse hovers over it."""
        color = [73, 161, 108] if self.contains(
            mouse_x, mouse_y) else [123, 161, 58]
        pygame.draw.rect(
            self.screen, color,
            [self.x, self.y, constant.BUTTON_W, constant.BUTTON_H],
        )
        text = self.font.render(self.label, True, (0, 0, 0))
        self.screen.blit(text, (
            self.x + (constant.BUTTON_W - text.get_width()) // 2,
            self.y + (constant.BUTTON_H - text.get_height()) // 2,
        ))
