from __future__ import annotations
import pygame

from .button import Button
from .case import Case
from . import constant
from .entity import TypeEntity, Player
from .spells import Spells


class Renderer:
    """Holds rendering state and draws the map, entities, HUD and spells."""

    def __init__(
        self,
        screen: pygame.Surface,
        font_title: pygame.font.Font,
        font_txt: pygame.font.Font,
        cases: dict[tuple[int, int], Case],
    ):
        """Cache fonts/surface, compute the map offset and lay out the UI."""
        self.screen = screen
        self.font_title = font_title
        self.font_txt = font_txt
        sw, sh = screen.get_width(), screen.get_height()
        self.avail_w = sw - constant.RIGHT_PANEL_W
        self.offset = self._compute_map_offset(cases, sw, sh)
        self.spell_y = self._map_screen_bottom(cases) + 20
        bx = self.avail_w + (constant.RIGHT_PANEL_W - constant.BUTTON_W) // 2
        by = sh // 2
        self.end_turn_button = Button(screen, font_title, bx, by, "End turn")
        self.spell_renders: list[tuple[Spells, int, int]] = []

    @staticmethod
    def _compute_map_offset(
        cases: dict[tuple[int, int], Case],
        screen_w: int,
        screen_h: int,
        spell_bar_h: int = 120,
    ) -> tuple[int, int]:
        """Return the iso-origin screen offset that centres the map.

        Centres the map's isometric bounding box within the available area
        (screen minus the right panel and the bottom spell bar).
        """
        iso_xs = [(case.x - case.y) * constant.CASE_WIDTH //
                  2 for case in cases.values()]
        iso_ys = [(case.x + case.y) * constant.CASE_HEIGHT //
                  2 for case in cases.values()]
        iso_cx = (min(iso_xs) + max(iso_xs)) // 2
        iso_cy = (min(iso_ys) + max(iso_ys)) // 2
        avail_w = screen_w - constant.RIGHT_PANEL_W
        avail_h = screen_h - spell_bar_h
        return avail_w // 2 - iso_cx, avail_h // 2 - iso_cy

    def _map_screen_bottom(
        self, cases: dict[tuple[int, int], Case]
    ) -> int:
        """Return the screen y of the map's lowest point (for placing the bar).
        """
        iso_ys = [(case.x + case.y) * constant.CASE_HEIGHT //
                  2 for case in cases.values()]
        return self.offset[1] + max(iso_ys) + constant.CASE_HEIGHT // 2

    def draw_map(
        self,
        mouse_x: int,
        mouse_y: int,
        cases: dict[tuple[int, int], Case],
        previsualiation: list[tuple],
        spawn_pattern: list[tuple],
        show_coords: bool = False,
    ) -> None:
        """Draw every cell, colouring previsu/spawn tiles and the hovered one.
        """
        for case in cases.values():
            x, y = case.x, case.y
            if (x, y) in previsualiation:
                color = constant.PREVISU_COLOR
            elif (x, y) in spawn_pattern:
                color = constant.SPAWN_COLOR_1
            elif (y % 2 == 0 and x % 2 == 0) or (y % 2 == 1 and x % 2 == 1):
                color = constant.CASE_COLOR_1
            else:
                color = constant.CASE_COLOR_2
            if case.contains(mouse_x, mouse_y, self.offset):
                r, g, b = color
                color = (r, g + 50, b + 50)
            case.draw(self.screen, self.offset, color,
                      self.font_txt, show_coords)

    def draw_entities(
        self, cases: dict[tuple[int, int], Case]
    ) -> None:
        """Draw a coloured marker for each occupied cell's entity."""
        for case in cases.values():
            v = case.entity
            if v is None:
                continue
            cx = int((case.x - case.y) * (constant.CASE_WIDTH / 2)) + \
                self.offset[0]
            cy = int((case.x + case.y) *
                     (constant.CASE_HEIGHT / 2)) + self.offset[1]
            match v.type_entity:
                case TypeEntity.PLAYER:
                    pygame.draw.circle(self.screen, [0, 0, 255], (cx, cy), 15)
                case TypeEntity.BOLGROT:
                    pygame.draw.circle(self.screen, [0, 255, 0], (cx, cy), 15)
                case TypeEntity.FLAME:
                    pygame.draw.circle(self.screen, [255, 0, 0], (cx, cy), 15)

    def draw_timer(self, timer_text: pygame.Surface) -> None:
        """Blit the pre-rendered turn-timer text above the end-turn button."""
        btn = self.end_turn_button
        self.screen.blit(
            timer_text,
            (btn.x + (constant.BUTTON_W - timer_text.get_width()) // 2,
             btn.y - 60),
        )

    def draw_spells(
        self,
        mouse_x: int,
        mouse_y: int,
        spells: list[Spells],
    ) -> None:
        """Lay out and draw the spell bar; cache each icon's rect.

        Records ``(spell, x, y)`` in ``self.spell_renders`` so the event loop
        can hit-test clicks/hovers against the drawn icons.
        """
        total_w = (sum(s.image.get_width() for s in spells) +
                   constant.SPELL_GAP * max(0, len(spells) - 1))
        x = (self.avail_w - total_w) // 2
        self.spell_renders = []
        for s in spells:
            self.spell_renders.append((s, x, self.spell_y))
            s.draw(self.screen, x, self.spell_y, mouse_x, mouse_y,
                   self.font_title, self.font_txt)
            x += s.image.get_width() + constant.SPELL_GAP

    def draw_hp_player(
        self,
        player: Player,
    ) -> None:
        """Draw the player's current HP in the top-left corner."""
        hp_text = self.font_title.render(
            f"HP: {player.hp}", True, (255, 255, 255))
        self.screen.blit(hp_text, (10, 10))

    def draw_ap_player(
        self,
        player: Player,
    ) -> None:
        """Draw the player's current AP below the HP readout."""
        ap_text = self.font_title.render(
            f"AP: {player.pa}", True, (255, 255, 255))
        self.screen.blit(ap_text, (10, 50))
