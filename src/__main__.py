import pygame
from .entity import Player
from . import constant
from .game import Game
from .actions import on_previsu_click
from .renderer import Renderer


def main() -> None:
    """Run the game: init pygame, then loop over input, update and draw.

    Serves as both the ``python -m src`` entry and the ``bolgrot`` console
    script. Returns when the window is closed or the player dies.
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

    game: Game = Game(player=Player(*constant.BASE_PLAYER_POS))
    renderer = Renderer(screen, font_title, font_txt, game.map.cases)

    while running:
        screen.fill((0, 0, 0))
        mouse_x, mouse_y = pygame.mouse.get_pos()

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

            if event.type == timer_event:
                if timer_sec > 0:
                    timer_sec -= 1
                    time_str = ("01:%02d" % (timer_sec - 60)
                                if timer_sec >= 60 else "00:%02d" % timer_sec)
                    timer_text = font_title.render(
                        time_str, True, (255, 255, 255))
                else:
                    pygame.time.set_timer(timer_event, 1000)
                    timer_sec = constant.TIME_TURN
                    game.end_turn()

            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                if event.key == pygame.K_SPACE:
                    timer_sec = constant.TIME_TURN
                    game.end_turn()
                if event.key in (pygame.K_1, pygame.K_2, pygame.K_3):
                    game.select_spell(event.key - pygame.K_1)

            if event.type == pygame.MOUSEBUTTONDOWN:
                if event.button == 1:
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
                    elif renderer.end_turn_button.contains(mouse_x, mouse_y):
                        timer_sec = constant.TIME_TURN
                        game.end_turn()
                    elif spell_index is not None:
                        game.select_spell(spell_index)
                    else:
                        game.clear_previsu()

        renderer.draw_map(mouse_x, mouse_y, game.map.cases,
                          game.previsualiation, game.spawn_pattern)
        renderer.draw_entities(game.map.cases)
        renderer.end_turn_button.draw(mouse_x, mouse_y)
        renderer.draw_timer(timer_text)
        renderer.draw_spells(mouse_x, mouse_y, game.player.spells)
        renderer.draw_hp_player(game.player)
        renderer.draw_ap_player(game.player)

        if game.done:
            running = False
            print("Victory: all flames cleared!" if game.won
                  else "Game Over: Player is dead.")
        pygame.display.update()
        pygame.display.flip()
        clock.tick(60)

    pygame.quit()


if __name__ == "__main__":
    main()
