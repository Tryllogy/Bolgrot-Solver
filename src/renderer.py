from __future__ import annotations
import pygame

from .button import Button
from .case import Case
from . import constant
from .entity import TypeEntity, Player
from .spells import Spells

# Display labels for the two hint engines (kept here so the renderer needs no
# import from the hint module, which pulls in the game).
_ENGINE_UI = {"az": "Rapide", "cnn": "Fort"}


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
        self.font_big = pygame.font.Font(None, 120)
        self._layout_hint_ui()
        self._layout_menus(sw, sh)

    # ---- home / game-over menus ------------------------------------------
    def _layout_menus(self, sw: int, sh: int) -> None:
        """Centred button rects for the home screen and game-over overlay."""
        cx, cy = sw // 2, sh // 2
        bw, bh = 460, 74
        self.home_play_rect = pygame.Rect(cx - bw // 2, cy - 20, bw, bh)
        self.home_custom_rect = pygame.Rect(
            cx - bw // 2, cy - 20 + bh + 22, bw, bh)
        gbw = 250
        self.over_replay_rect = pygame.Rect(cx - gbw - 12, cy + 40, gbw, bh)
        self.over_home_rect = pygame.Rect(cx + 12, cy + 40, gbw, bh)

    def _menu_button(self, rect: pygame.Rect, label: str,
                     mouse: tuple[int, int], enabled: bool = True) -> None:
        """Draw a large centred menu button (green, lit on hover)."""
        hover = enabled and rect.collidepoint(mouse)
        col = (90, 90, 70) if not enabled else (
            (73, 161, 108) if hover else (58, 118, 84))
        pygame.draw.rect(self.screen, col, rect, border_radius=10)
        self._center_label(rect, label)

    def draw_home(self, mouse_x: int, mouse_y: int) -> None:
        """Draw the welcome screen with the two play modes."""
        mouse = (mouse_x, mouse_y)
        self.screen.fill((18, 18, 22))
        cx = self.screen.get_width() // 2
        title = self.font_big.render("BOLGROT", True, (230, 220, 180))
        self.screen.blit(title, (cx - title.get_width() // 2,
                                 self.home_play_rect.y - 210))
        sub = self.font_txt.render(
            "Survivez aux vagues de flammes", True, (170, 170, 170))
        self.screen.blit(sub, (cx - sub.get_width() // 2,
                               self.home_play_rect.y - 90))
        self._menu_button(self.home_play_rect, "Jouer", mouse)
        self._menu_button(self.home_custom_rect, "Placer mes flammes", mouse)
        tip = self.font_txt.render("ÉCHAP : quitter", True, (120, 120, 120))
        self.screen.blit(tip, (cx - tip.get_width() // 2,
                               self.home_custom_rect.bottom + 30))

    def draw_game_over(self, won: bool, mouse_x: int, mouse_y: int) -> None:
        """Dim the board and show the result plus Rejouer / Accueil buttons."""
        mouse = (mouse_x, mouse_y)
        overlay = pygame.Surface(self.screen.get_size(), pygame.SRCALPHA)
        overlay.fill((0, 0, 0, 160))
        self.screen.blit(overlay, (0, 0))
        cx, cy = self.screen.get_width() // 2, self.screen.get_height() // 2
        msg = "Victoire !" if won else "Défaite"
        col = (120, 220, 120) if won else (220, 120, 120)
        t = self.font_big.render(msg, True, col)
        self.screen.blit(t, (cx - t.get_width() // 2, cy - 130))
        self._menu_button(self.over_replay_rect, "Rejouer", mouse)
        self._menu_button(self.over_home_rect, "Accueil", mouse)

    def draw_placement_hud(self, count: int, maxn: int, wave: int,
                           mouse_x: int, mouse_y: int) -> None:
        """Instruction banner + a Valider button while placing a wave.

        The placed tiles are the game's ``spawn_pattern`` and are already
        coloured by :meth:`draw_map`. Valider reuses the hint button's rect.
        """
        txt = (f"Vague {wave} — placez vos flammes : {count}/{maxn}   "
               "(clic : ajouter / retirer)")
        t = self.font_title.render(txt, True, (255, 255, 255))
        self.screen.blit(t, (self.avail_w // 2 - t.get_width() // 2, 20))
        r = self.hint_button_rect
        self._menu_button(r, "Valider", (mouse_x, mouse_y), enabled=count > 0)

    # ---- AI hint panel ---------------------------------------------------
    def _layout_hint_ui(self) -> None:
        """Compute the fixed rects of the hint panel (engine/budget/button)."""
        pad = 20
        ix = self.avail_w + pad
        iw = constant.RIGHT_PANEL_W - 2 * pad
        y = self.end_turn_button.y + constant.BUTTON_H + 45
        self._moteur_label_pos = (ix, y)
        y += 30
        bw = (iw - 10) // 2
        self.engine_rects = {
            "az": pygame.Rect(ix, y, bw, 42),
            "cnn": pygame.Rect(ix + bw + 10, y, iw - bw - 10, 42),
        }
        y += 42 + 18
        self._budget_label_pos = (ix, y)
        y += 30
        bw3 = (iw - 20) // 3
        self.budget_rects = {
            300: pygame.Rect(ix, y, bw3, 42),
            1000: pygame.Rect(ix + bw3 + 10, y, bw3, 42),
            8000: pygame.Rect(ix + 2 * (bw3 + 10), y, iw - 2 * (bw3 + 10), 42),
        }
        y += 42 + 18
        self.hint_button_rect = pygame.Rect(ix, y, iw, 50)
        y += 50 + 10
        self.step_button_rect = pygame.Rect(ix, y, iw, 50)
        y += 50 + 10
        self.autoplay_button_rect = pygame.Rect(ix, y, iw, 50)
        y += 50 + 14
        self._status_pos = (ix, y)
        self._status_w = iw

    def _toggle(self, rect: pygame.Rect, label: str, active: bool,
                mouse: tuple[int, int]) -> None:
        """Draw a small selectable button, outlined white when active."""
        hover = rect.collidepoint(mouse)
        color = (73, 161, 108) if active else (
            (95, 95, 80) if hover else (58, 58, 52))
        pygame.draw.rect(self.screen, color, rect, border_radius=6)
        if active:
            pygame.draw.rect(self.screen, (255, 255, 255), rect, 2,
                             border_radius=6)
        txt = self.font_txt.render(label, True, (255, 255, 255))
        self.screen.blit(txt, (rect.x + (rect.w - txt.get_width()) // 2,
                               rect.y + (rect.h - txt.get_height()) // 2))

    def draw_hint_panel(self, mouse_x: int, mouse_y: int, engine: str,
                        budget: int, status: str, busy: bool,
                        autoplay: bool = False) -> None:
        """Draw the engine/budget selectors, the action buttons and status."""
        mouse = (mouse_x, mouse_y)
        self.screen.blit(self.font_txt.render("Moteur", True, (220, 220, 220)),
                         self._moteur_label_pos)
        for key, rect in self.engine_rects.items():
            self._toggle(rect, _ENGINE_UI[key], key == engine, mouse)
        self.screen.blit(self.font_txt.render("Budget (simulations)", True,
                                              (220, 220, 220)),
                         self._budget_label_pos)
        for key, rect in self.budget_rects.items():
            self._toggle(rect, str(key), key == budget, mouse)
        # "Indice" — one-shot hint (dimmed while a search runs, and while
        # autoplay drives the game).
        r = self.hint_button_rect
        idle = not busy and not autoplay
        hover = r.collidepoint(mouse)
        col = (48, 108, 168) if not idle else (
            (60, 130, 200) if hover else (48, 108, 168))
        if not idle:
            col = (90, 90, 70)
        pygame.draw.rect(self.screen, col, r, border_radius=8)
        label = "Calcul…" if (busy and not autoplay) else "Indice"
        self._center_label(r, label)
        # "IA : 1 coup" — the AI plays exactly ONE move, then hands back.
        sr = self.step_button_rect
        shover = sr.collidepoint(mouse)
        scol = (90, 90, 70) if not idle else (
            (120, 100, 168) if shover else (96, 82, 140))
        pygame.draw.rect(self.screen, scol, sr, border_radius=8)
        self._center_label(sr, "IA : 1 coup")
        # "Autoplay" — toggle: the AI plays every move until stopped/done.
        ar = self.autoplay_button_rect
        ahover = ar.collidepoint(mouse)
        acol = (176, 66, 66) if autoplay else (
            (73, 161, 108) if ahover else (58, 118, 84))
        pygame.draw.rect(self.screen, acol, ar, border_radius=8)
        self._center_label(ar, "Arrêter l'auto" if autoplay else "Autoplay")
        if status:
            self._blit_wrapped(status, self._status_pos, self._status_w,
                               (255, 220, 120))

    def _center_label(self, rect: pygame.Rect, label: str) -> None:
        """Blit ``label`` (title font) centred in ``rect``."""
        t = self.font_title.render(label, True, (255, 255, 255))
        self.screen.blit(t, (rect.x + (rect.w - t.get_width()) // 2,
                             rect.y + (rect.h - t.get_height()) // 2))

    def _blit_wrapped(self, text: str, pos: tuple[int, int], width: int,
                      color: tuple[int, int, int]) -> None:
        """Word-wrap ``text`` to ``width`` px and blit it at ``pos``."""
        x, y = pos
        line = ""
        for word in text.split(" "):
            trial = f"{line} {word}".strip()
            if self.font_txt.size(trial)[0] > width and line:
                surf = self.font_txt.render(line, True, color)
                self.screen.blit(surf, (x, y))
                y += self.font_txt.get_height() + 2
                line = word
            else:
                line = trial
        if line:
            self.screen.blit(self.font_txt.render(line, True, color), (x, y))

    def draw_hint_end_turn(self) -> None:
        """Outline the end-turn button (the hint recommends passing the turn).

        Tile hints colour their target case green in :meth:`draw_map`; this is
        only for the ``end turn`` action, which has no board tile.
        """
        b = self.end_turn_button
        pygame.draw.rect(self.screen, constant.HINT_COLOR,
                         [b.x - 4, b.y - 4, constant.BUTTON_W + 8,
                          constant.BUTTON_H + 8], 3)

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
        hint_target: tuple[int, int] | None = None,
    ) -> None:
        """Draw every cell, colouring previsu/spawn/hint tiles and the hovered.
        """
        for case in cases.values():
            x, y = case.x, case.y
            if (x, y) == hint_target:
                color = constant.HINT_COLOR
            elif (x, y) in previsualiation:
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
