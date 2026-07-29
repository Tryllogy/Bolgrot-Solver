from pygame import Surface
from . import constant


def on_button_end_turn_click(
        mouse_x: int,
        mouse_y: int,
        bx: int,
        by: int
) -> bool:
    """Return True if the mouse falls within the end-turn button at (bx, by).
    """
    return (bx <= mouse_x < bx + constant.BUTTON_W
            and by <= mouse_y < by + constant.BUTTON_H)


def on_spell_hover(
        mouse_x: int,
        mouse_y: int,
        spell_renders: list[tuple[Surface, int, int]],
) -> tuple[bool, int | None]:
    """Find the hovered spell icon.

    Returns ``(True, index)`` for the first spell whose rendered rect
    contains the mouse, else ``(False, None)``.
    """
    for i, (img, sx, sy) in enumerate(spell_renders):
        if (sx <= mouse_x < sx + img.get_width()
                and sy <= mouse_y < sy + img.get_height()):
            return True, i
    return False, None


def on_previsu_click(
        mouse_x: int,
        mouse_y: int,
        previsualiation: list[tuple[int, int]],
        map_offset: tuple[int, int]
) -> tuple[bool, int | None]:
    """Test whether a click landed on a highlighted previsualisation tile.

    Inverts the isometric projection (using ``map_offset``) to map the mouse
    to grid coordinates. Returns ``(True, index)`` for the matching tile in
    ``previsualiation``, else ``(False, None)``.
    """
    for pos in previsualiation:
        x = mouse_x - map_offset[0]
        y = mouse_y - map_offset[1]
        gx = round((y / (constant.CASE_HEIGHT / 2) + x /
                    (constant.CASE_WIDTH / 2)) / 2)
        gy = round((y / (constant.CASE_HEIGHT / 2) - x /
                    (constant.CASE_WIDTH / 2)) / 2)
        tile: tuple[int, int] = (gx, gy)
        if tile == pos:
            return True, previsualiation.index(pos)
    return False, None
