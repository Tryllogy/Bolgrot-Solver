import os
import pygame
from .entity import Player, Flame
from . import constant
from .game import Game
from .case import CaseType
from .actions import on_previsu_click
from .renderer import Renderer
from .hint import HintEngine
from .live.feed import LiveFeed

# App states.
HOME, PLAY, OVER = "home", "play", "over"
AUTOPLAY_DELAY_MS = 350           # pause between autoplay moves (watchable)
MAX_FLAMES = constant.CUSTOM_FLAMES_PER_WAVE


def main() -> None:
    """Run the app: a home screen, then the game (normal or custom flames).

    Serves as both the ``python -m src`` entry and the ``bolgrot`` console
    script. The window stays open on game over (a banner offers Rejouer /
    Accueil); ``ESC`` returns to the home screen, or quits from it.
    """
    pygame.init()
    screen: pygame.Surface = pygame.display.set_mode((0, 0), pygame.FULLSCREEN)
    clock: pygame.time.Clock = pygame.time.Clock()
    running: bool = True

    font_title: pygame.font.Font = pygame.font.Font(None, 50)
    font_txt: pygame.font.Font = pygame.font.Font(None, 25)

    timer_event: int = pygame.USEREVENT + 1
    timer_sec: int = constant.TIME_TURN
    timer_text: pygame.Surface = font_title.render(
        "02:00", True, (255, 255, 255))
    pygame.time.set_timer(timer_event, 1000)

    # The map is fixed, so one game builds the renderer's layout; real games
    # are (re)created when a mode is picked on the home screen.
    game: Game = Game(player=Player(*constant.BASE_PLAYER_POS))
    renderer = Renderer(screen, font_title, font_txt, game.map.cases)
    hint = HintEngine()
    live_feed = LiveFeed(MAX_FLAMES)

    state = HOME
    custom = False            # current game's mode
    live = False              # custom mode fed by the sniffed Dofus fight
    placing = False           # custom sub-phase: designing the next wave
    # "IA : tours" mode: the AI plays each turn's moves by itself, then waits
    # (autoplay_waiting) for the player to press End turn before the next.
    autoplay = False
    autoplay_waiting = False
    autoplay_next = 0
    ai_step = False           # one-shot: play the AI's next move, then stop
    # This turn's AI actions, in order, for the on-board trail and the panel
    # list: (spell index, spell name, origin tile, target tile).
    ai_trail: list[tuple[int, str, tuple[int, int], tuple[int, int]]] = []
    trail_hidden: set[int] = set()   # trail indices masked on the board

    def new_game(is_custom: bool, is_live: bool = False) -> None:
        nonlocal game, custom, live, placing, autoplay, autoplay_waiting
        nonlocal ai_step, timer_sec
        game = Game(player=Player(*constant.BASE_PLAYER_POS))
        custom = is_custom or is_live
        live = is_live
        if live:
            live_feed.start()       # (re)start listening, drop stale waves
        else:
            live_feed.stop()
        is_custom = custom
        placing = is_custom       # custom starts by placing wave 1
        if is_custom:
            game.spawn_pattern = []   # player designs it (drop the auto wave)
        autoplay = False
        autoplay_waiting = False
        ai_step = False
        ai_trail.clear()
        trail_hidden.clear()
        timer_sec = constant.TIME_TURN
        hint.clear()

    def apply_ai_move(action: int) -> None:
        """Play the AI's spell ``action`` and log it on this turn's trail."""
        nonlocal timer_sec
        spell_idx, (dx, dy) = Game.ACTIONS[action]
        origin = (game.player.pos_x, game.player.pos_y)
        game.step(action)
        if spell_idx is not None and not game.last_reward["illegal"]:
            ai_trail.append((spell_idx, game.player.spells[spell_idx].name,
                             origin, (origin[0] + dx, origin[1] + dy)))
        game.clear_previsu()
        timer_sec = constant.TIME_TURN

    def do_end_turn() -> None:
        nonlocal placing, timer_sec, autoplay_waiting
        game.end_turn()
        ai_trail.clear()
        trail_hidden.clear()
        autoplay_waiting = False          # AI mode: play the new turn
        hint.clear()
        game.clear_previsu()
        timer_sec = constant.TIME_TURN
        # Custom mode: design the next wave (if any) instead of the auto one.
        if custom and not game.done and game.waves_spawned < game.num_waves:
            game.spawn_pattern = []
            placing = True

    def confirm_placement() -> None:
        nonlocal placing, timer_sec
        if game.spawn_pattern:            # need at least one flame
            placing = False
            game.clear_previsu()
            timer_sec = constant.TIME_TURN

    while running:
        screen.fill((0, 0, 0))
        mouse_x, mouse_y = pygame.mouse.get_pos()

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

            if event.type == timer_event and state == PLAY and not placing:
                if timer_sec > 0:
                    timer_sec -= 1
                    time_str = ("01:%02d" % (timer_sec - 60)
                                if timer_sec >= 60 else "00:%02d" % timer_sec)
                    timer_text = font_title.render(
                        time_str, True, (255, 255, 255))
                elif not autoplay:
                    do_end_turn()
                else:
                    timer_sec = constant.TIME_TURN   # AI drives the turns

            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    if state == HOME:
                        running = False
                    else:
                        state = HOME
                        autoplay = False
                        autoplay_waiting = False
                        ai_step = False
                        live_feed.stop()
                        hint.clear()
                elif state == PLAY and placing:
                    if event.key == pygame.K_SPACE:
                        confirm_placement()
                elif state == PLAY:
                    if event.key == pygame.K_SPACE:
                        do_end_turn()
                    if event.key in (pygame.K_1, pygame.K_2, pygame.K_3):
                        game.select_spell(event.key - pygame.K_1)
                    if event.key == pygame.K_h:
                        hint.request(game)
                elif state == OVER:
                    if event.key in (pygame.K_RETURN, pygame.K_SPACE):
                        state = HOME
                    if event.key == pygame.K_r:
                        new_game(custom, live)
                        state = PLAY

            if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                pos = (mouse_x, mouse_y)
                if state == HOME:
                    if renderer.home_play_rect.collidepoint(pos):
                        new_game(False)
                        state = PLAY
                    elif renderer.home_custom_rect.collidepoint(pos):
                        new_game(True)
                        state = PLAY
                    elif renderer.home_live_rect.collidepoint(pos):
                        new_game(True, is_live=True)
                        state = PLAY
                elif state == OVER:
                    if renderer.over_replay_rect.collidepoint(pos):
                        new_game(custom, live)
                        state = PLAY
                    elif renderer.over_home_rect.collidepoint(pos):
                        state = HOME
                        live_feed.stop()
                elif state == PLAY and placing:
                    # Valider reuses the hint button rect; else toggle a tile.
                    if renderer.hint_button_rect.collidepoint(pos):
                        confirm_placement()
                    else:
                        cell = next(
                            (c for c in game.map.cases.values()
                             if c.contains(mouse_x, mouse_y, renderer.offset)),
                            None)
                        # Placeable = a free tile that is empty OR already
                        # holds a flame (the player may stack a new wave onto
                        # existing flames); never the player or Bolgrot.
                        if (cell is not None
                                and cell.case_type == CaseType.FREE
                                and (cell.entity is None
                                     or type(cell.entity) is Flame)):
                            p = (cell.x, cell.y)
                            if p in game.spawn_pattern:
                                game.spawn_pattern.remove(p)
                            elif len(game.spawn_pattern) < MAX_FLAMES:
                                game.spawn_pattern.append(p)
                elif state == PLAY:
                    trail_hit = next(
                        (i for i, r in enumerate(
                            renderer.trail_row_rects(len(ai_trail)))
                         if r.collidepoint(pos)), None)
                    engine_hit = next(
                        (k for k, r in renderer.engine_rects.items()
                         if r.collidepoint(pos)), None)
                    budget_hit = next(
                        (k for k, r in renderer.budget_rects.items()
                         if r.collidepoint(pos)), None)
                    if trail_hit is not None:
                        trail_hidden ^= {trail_hit}     # toggle its marks
                    elif engine_hit is not None:
                        hint.set_engine(engine_hit)
                    elif budget_hit is not None:
                        hint.set_budget(budget_hit)
                    elif renderer.autoplay_button_rect.collidepoint(pos):
                        autoplay = not autoplay
                        autoplay_waiting = False
                        ai_step = False
                        hint.clear()
                        autoplay_next = pygame.time.get_ticks()
                    elif renderer.step_button_rect.collidepoint(pos):
                        if not autoplay and not hint.busy:
                            ai_step = True
                            hint.clear()
                            hint.request(game)
                    elif renderer.hint_button_rect.collidepoint(pos):
                        ai_step = False
                        hint.request(game)
                    else:
                        is_on_previsu, previsu_index = on_previsu_click(
                            mouse_x, mouse_y, game.previsualiation,
                            renderer.offset)
                        spell_index = next(
                            (i for i, (s, sx, sy)
                             in enumerate(renderer.spell_renders)
                             if s.contains(mouse_x, mouse_y, sx, sy)),
                            None,
                        )
                        if is_on_previsu and previsu_index is not None:
                            game.play_selected_spell(
                                game.previsualiation[previsu_index])
                            hint.clear()
                        elif renderer.end_turn_button.contains(
                                mouse_x, mouse_y):
                            do_end_turn()
                        elif spell_index is not None:
                            game.select_spell(spell_index)
                        else:
                            game.clear_previsu()

        hint.poll()

        # Live Dofus: a sniffed wave fills the placement and validates it.
        if state == PLAY and placing and live:
            wave = live_feed.poll()
            if wave is not None:
                game.spawn_pattern = [
                    p for p in wave
                    if (c := game.map.cases.get(p)) is not None
                    and c.case_type == CaseType.FREE
                    and (c.entity is None or type(c.entity) is Flame)
                ][:MAX_FLAMES]
                confirm_placement()

        # "1 coup" : play the AI's move once it's computed, then hand back.
        if state == PLAY and not placing and ai_step and not game.done:
            if hint.result is not None and not hint.busy:
                if hint.result.target is None:   # AI ends the turn
                    do_end_turn()                # re-enters placement (custom)
                else:
                    apply_ai_move(hint.result.action)
                hint.clear()
                ai_step = False

        # AI turns: apply the AI's moves one by one; when it would end the
        # turn, stop there (End turn stays highlighted) until the player
        # presses End turn — do_end_turn() then lets it play the next turn.
        if (state == PLAY and not placing and autoplay
                and not autoplay_waiting and not game.done):
            now = pygame.time.get_ticks()
            if hint.result is not None and not hint.busy:
                if hint.result.target is None:   # AI's turn is over
                    # Keep the result so End turn stays highlighted.
                    autoplay_waiting = True
                    hint.status = ("Tour de l'IA terminé : appuyez sur "
                                   "« End turn »")
                else:
                    apply_ai_move(hint.result.action)
                    hint.clear()
                    autoplay_next = now + AUTOPLAY_DELAY_MS
            elif not hint.busy and now >= autoplay_next:
                hint.request(game)

        if state == PLAY and game.done:
            state = OVER
            autoplay = False

        # ---- draw ----------------------------------------------------------
        if state == HOME:
            renderer.draw_home(mouse_x, mouse_y)
        else:
            hint_target = (hint.result.target if (hint.result is not None
                           and not placing) else None)
            renderer.draw_map(mouse_x, mouse_y, game.map.cases,
                              game.previsualiation, game.spawn_pattern,
                              hint_target=hint_target,
                              trail=[t[3] for i, t in enumerate(ai_trail)
                                     if i not in trail_hidden])
            renderer.draw_entities(game.map.cases)
            renderer.draw_ai_trail(ai_trail, trail_hidden)
            renderer.draw_hp_player(game.player)
            renderer.draw_ap_player(game.player)
            if placing:
                renderer.draw_placement_hud(
                    len(game.spawn_pattern), MAX_FLAMES,
                    game.waves_spawned + 1, mouse_x, mouse_y,
                    live_status=live_feed.status if live else "")
            else:
                renderer.end_turn_button.draw(mouse_x, mouse_y)
                if hint.result is not None and hint.result.target is None:
                    renderer.draw_hint_end_turn()
                renderer.draw_timer(timer_text)
                renderer.draw_spells(mouse_x, mouse_y, game.player.spells)
                renderer.draw_hint_panel(mouse_x, mouse_y, hint.engine,
                                         hint.budget, hint.status, hint.busy,
                                         autoplay)
                renderer.draw_trail_list(ai_trail, trail_hidden,
                                         mouse_x, mouse_y)
            if state == OVER:
                renderer.draw_game_over(game.won, mouse_x, mouse_y)

        pygame.display.update()
        pygame.display.flip()
        clock.tick(60)

    pygame.quit()
    live_feed.stop()
    # If a hint search is still running (e.g. quit mid Fort@8000), a daemon
    # thread caught inside torch aborts the C++ teardown at exit — skip it.
    if hint.has_active():
        os._exit(0)


if __name__ == "__main__":
    main()
