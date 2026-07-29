import os
from importlib.resources import files

# 37
GRID_MAX_X = 37
# 34
GRID_MAX_Y = 34

CASE_HEIGHT = 32
CASE_WIDTH = 64

# (14,15)
BASE_PLAYER_POS = (16, 16)
# (21, 10)
BASE_BOLGROT_POS = (21, 11)

# HEXA COLOR
CASE_COLOR_1 = (205, 178, 111)
CASE_COLOR_2 = (189, 185, 132)
SPAWN_COLOR_1 = (151, 13, 158)
PREVISU_COLOR = (9, 91, 158)
BACKGROUND_POPUP = (79, 79, 61)

RIGHT_PANEL_W = 400
BUTTON_W = 200
BUTTON_H = 60
SPELL_GAP = 10

# IN SECONDS
TIME_TURN = 120

# Number of flame waves that spawn over a game (exactly this many, no repeats).
NB_WAVES = 6

# Player starting hit points (used for observation normalisation).
MAX_HP = 40

# Curriculum `spawn_formation="line"`: how far from the player the flame line
# starts. 2 keeps the nearest flame inside LongJump's range-2 reach.
SPAWN_LINE_OFFSET = 2

# Observation encoding: how many nearest flames / next-spawn tiles the AI sees.
# Each contributes 3 floats (present flag, dx, dy). Changing these changes the
# observation vector length and therefore the policy network's input size.
K_NEAREST_FLAMES = 8
K_NEAREST_SPAWN = 6

# Observation MODE: "flat" (the K-nearest vector above, fed to the MLP — this is
# what az.pt and the deployment use) or "grid" (a 2D multi-channel board for a
# CNN, full spatial observability). "grid" is an EXPERIMENT flag and is
# INCOMPATIBLE with flat-trained nets (az.pt) — the observation length and the
# network class both change. Keep "flat" committed; set "grid" only in a
# working-tree experiment and NEVER deploy it (prod runs the flat az.pt).
# Overridable by the BOLGROT_OBS_MODE env var so the deployment/hint server and
# a CNN training run can coexist from the SAME checkout (they share this module):
# the committed default is "flat" (az.pt / prod hint — a deploy is always safe),
# and a grid-CNN experiment sets BOLGROT_OBS_MODE=grid in its OWN launch env. The
# env var is inherited by spawn workers (unlike a runtime module global), so
# training workers see grid while a server started without the var sees flat.
# NEVER deploy with BOLGROT_OBS_MODE=grid — grid obs (CNN) breaks the flat az.pt.
OBS_MODE = os.environ.get("BOLGROT_OBS_MODE", "flat")
# Grid obs (OBS_MODE="grid"): an absolute board grid indexed [x, y], with a few
# binary channels + a small tail of scalar features. GRID_H/W must cover every
# cell coordinate (parsed grid_max is ~31/32; 34 leaves margin — asserted in
# Game._grid_observation so a larger map fails loud instead of truncating).
GRID_H = 34
GRID_W = 34
# Channels: 0=free-cell mask (board shape), 1=flames, 2=next-spawn tiles,
# 3=player, 4=bolgrot.
GRID_CHANNELS = 5
# Scalar tail (non-spatial): hp, pa, waves_spawned, flame_count, castable_0..2.
GRID_SCALARS = 7

# Reward weights for the AI (`Game.step`). The dominant term is REWARD_CLEAR:
# clearing flames is the only way to make real progress toward a win, and it
# must out-earn passively ending turns. Survival is kept deliberately tiny so
# "sit still and end turns" is not a winning strategy. Flame *spawns* are never
# penalised (the agent cannot prevent them) — only flame *reductions* pay out.
REWARD_CLEAR_FLAME = 10.0    # per flame removed from the board
REWARD_SURVIVE_TURN = 0.1   # per turn survived (small; anti-suicide only)
REWARD_HP_DELTA = 0.02      # per HP gained/lost (tiny nudge against waste)
REWARD_WIN = 100.0          # terminal: board cleared after all waves
REWARD_DEATH = -10.0        # terminal: player died (base cost)
# Extra death cost per wave still unspawned when the player dies. Dying early
# (many waves left) must hurt more than the handful of flames clearable before
# then, so "clear a few then die" can never beat surviving. See `Game.step`.
REWARD_DEATH_PER_WAVE_LEFT = -5.0
REWARD_ILLEGAL = -0.1       # picked an illegal action (guard; masked in play)
# Anti-stall: penalty applied on end-turn, scaled by the number of flames
# left ALIVE on the board at that moment (before the new wave spawns — spawns
# are never penalised). Ending a turn with a clean board still nets
# +REWARD_SURVIVE_TURN; ending it with N uncleared flames costs
# REWARD_END_TURN_PER_FLAME * N, so passively ending turns only pays once the
# board is cleared. This is what the agent *controls* (killing a flame before
# end-turn removes its penalty), unlike the fixed flames-still-to-spawn
# schedule. Keep its magnitude well under the death penalty (REWARD_DEATH +
# REWARD_DEATH_PER_WAVE_LEFT) so it never makes suicide preferable.
REWARD_END_TURN_PER_FLAME = -0.5

# Shaping on top of the outcome rewards above, to guide *how* the agent plays.
# Approach: reward the player for physically moving toward the nearest flame or
# next-spawn tile (only the two jump spells move the player).
REWARD_APPROACH = 1.0      # per tile closer to the nearest flame / spawn tile
# Per-spell cast bias: kept near zero on purpose. An *unconditional* per-cast
# reward is farmable (the agent banks it by casting uselessly until PA drains),
# so progress is paid through REWARD_CLEAR/REWARD_APPROACH instead. Only the
# costly no-progress "Inaction" flame-drag keeps a small negative nudge.
# Indexed by spell_index (0=ShortJump, 1=LongJump, 2=MoveFlames) in `step`.
REWARD_CAST_SHORTJUMP = 0.0     # "Astral leap"  — neutral (reward the outcome)
REWARD_CAST_LONGJUMP = 0.0      # "Double leap"  — neutral (reward the outcome)
REWARD_CAST_MOVEFLAMES = -0.2   # "Inaction"     — discouraged
REWARD_CAST = (
    REWARD_CAST_SHORTJUMP,
    REWARD_CAST_LONGJUMP,
    REWARD_CAST_MOVEFLAMES,
)

# Package-data paths resolved relative to the installed `src` package so they
# work regardless of the current working directory.
MAP_CONF = str(files("src") / "config" / "bolgrot.map")
SPRITES_DIR = str(files("src") / "sprites_png")
PATTERNS_CONF = str(files("src") / "patterns" / "patterns.json")
