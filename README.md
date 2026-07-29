# Bolgrot

A turn‑based, isometric grid survival game built with [pygame](https://www.pygame.org/).
You play a lone character on a diamond‑shaped map. Each turn a new wave of
**flames** spawns; the boss **Bolgrot** sits on the board as an indestructible
obstacle. Survive every wave and clear the board of flames to win — run out of
HP and you lose.

> 🎮 **Play it online:** a live web version is available at
> **[bolgrot.web.app](https://bolgrot.web.app/)** — playable directly in the
> browser, with the trained AI assistant suggesting the best move (or playing
> for you). No install required.

The project is also designed to be driven *headlessly* by an AI: all game rules
live in a pure‑Python `Game` class with **no pygame dependency**, so the logic
can be stepped programmatically (e.g. by a reinforcement‑learning agent) as well
as played by a human.

---

## Gameplay

- The map is a diamond of isometric tiles loaded from a text file
  (`src/config/bolgrot.map`).
- Every **120 seconds** (one turn), a flame **wave** spawns. Exactly
  `NB_WAVES` (6) distinct waves spawn over a game — no repeats.
- The player has **40 HP** and **10 AP** (action points, refreshed each turn)
  and three spells.
- Once all `NB_WAVES` waves have spawned, the player loses **1 HP per turn** —
  pressure to finish clearing the board quickly.
- **Win:** all waves have spawned *and* no flames remain on the map.
- **Lose:** the player's HP drops to 0 (a flame moving onto the player, or a
  flame that cannot be pushed, is lethal).

### Spells

| Key | Name          | Type        | Range | AP | Uses/turn | Effect |
|-----|---------------|-------------|-------|----|-----------|--------|
| `1` | Astral leap   | Line        | 1     | 1  | unlimited | Teleport 1 tile; −1 HP; landing on a flame kills it and restores 1 HP; then all flames are attracted one tile toward you. |
| `2` | Double leap   | Line        | 2     | 2  | 2         | Same as Astral leap but range 2. |
| `3` | Inaction      | Diagonal    | 1     | 1  | unlimited | Attract all flames one tile toward a target tile (no line‑of‑sight needed); −5 HP; flames can't kill you while casting. |

**Flame attraction** is the core mechanic shared by every spell: flames are
sorted nearest‑first (Manhattan distance) and each moves **one** tile toward the
player along its axis of greatest distance, staying inside its quadrant. If the
target tile is a wall or off‑map, the flame falls back to BFS pathfinding.

### Controls

| Input            | Action |
|------------------|--------|
| `1` / `2` / `3`  | Select the spell at that slot (shows its previsualisation tiles) |
| Left click       | Cast the selected spell on a highlighted tile, click a spell icon, or hit the end‑turn button |
| `SPACE`          | End the turn (or, while placing flames, validate the wave) |
| `H`              | Ask the AI for a move hint (same as the **Indice** button) |
| `ESC`            | Back to the home screen (or quit from the home screen) |

### Game modes

Launching opens a **home screen** with two modes:

- **Jouer** — the normal game: each wave spawns from the built‑in random
  patterns.
- **Placer mes flammes** — *you* design every wave. Before each turn you click
  tiles to place that wave's flames (up to **6 per wave, over 6 waves**), then
  **Valider** (or `SPACE`) and play the turn. Tiles may be empty **or already
  holding a flame** (you can stack a new wave onto existing flames); walls, the
  player and Bolgrot are off‑limits. Same idea as the online custom mode.

The window **stays open when a game ends**: a *Victoire !* / *Défaite* banner
offers **Rejouer** (`R`) or **Accueil**. `ESC` returns to the home screen at any
time.

### AI move hints

The right‑hand panel has an **Indice** button (or press `H`) that asks a trained
AI what to play in the current position and rings the recommended tile (or the
end‑turn button). You pick:

- **Moteur** — *Rapide* (the flat MLP `az.pt`) or *Fort* (the grid‑CNN
  `az_cnn_deep.pt`, stronger but slower).
- **Budget** — search simulations per move: **300 / 1000 / 8000** (more = stronger
  and slower; 8000 with *Fort* takes tens of seconds).

The **Autoplay** button lets the chosen agent play the whole game on its own —
it applies its own move each step (with a short pause so you can watch) until the
board is cleared, the player dies, or you press **Autoplay** again to stop and
take back control. The engine/budget selectors apply to autoplay too.

The search runs on a background thread so the window stays responsive, and it
reuses the exact agents from `src/ai/`. **Hints and autoplay require the `ai`
extra** (`uv sync --extra ai` or `pip install -e ".[ai]"`); without it the game
still plays and the buttons just report the missing dependency.

---

## Installation & running

**Requirements:** Python **3.9+**. Playing the game opens a window (pygame needs a
display); AI training and benchmarking run headlessly.

The project has two dependency levels, so you only install what you need:

| Level | Adds | Use it to |
|-------|------|-----------|
| **base** | `pygame` | **play** the game |
| **`ai` extra** | `torch`, `numpy` | run the **MCTS / AlphaZero** agents (train, benchmark, AI hints) |

### Option A — [uv](https://github.com/astral-sh/uv) (recommended)

`uv` reads `pyproject.toml` / `uv.lock` and creates a local `.venv` for you. Install
uv once (see the [install guide](https://docs.astral.sh/uv/getting-started/installation/)),
then:

```bash
uv sync                 # base: game only
uv sync --extra ai      # base + the AI solver (torch/numpy)

uv run python -m src    # play the game   (equivalently: uv run bolgrot)
```

### Option B — pip

```bash
python -m venv .venv
source .venv/bin/activate            # Windows: .venv\Scripts\activate

pip install -e .                     # base: game only
pip install -e ".[ai]"               # base + the AI solver

python -m src                        # play the game   (equivalently: bolgrot)
```

`pip install -e .` also installs the console script **`bolgrot`**, so once the
virtualenv is active you can launch the game by just typing `bolgrot`.

### Make shortcuts (thin wrappers over uv)

```bash
make install      # uv sync — install dependencies
make run          # install + launch the game
make play         # watch a trained policy play (pulls the ai extra)
make train        # train the PPO agent (make train ITERS=500)
make resume       # continue training the last checkpoint (make resume ITERS=300)
make lint         # flake8 + mypy
make clean        # remove __pycache__ / build caches
```

> **Command convention below:** the AI examples are written for uv as
> `uv run --extra ai python -m …`. If you installed with **pip** (`.[ai]`, venv
> active), just drop the `uv run --extra ai` prefix and run `python -m …` directly.

### Training an AI (PPO)

A [PPO](https://arxiv.org/abs/1707.06347) reinforcement‑learning agent can learn
to play the game headlessly. It lives in `src/ai/` behind the optional `ai`
dependency group (`torch`) and drives the pure‑Python `Game` through its
observation/action/step API — no game logic leaks into the agent.

```bash
uv run --extra ai python -m src.ai.train --iterations 300            # → src/ai/ppo.pt
uv run --extra ai python -m src.ai.train --iterations 300 --resume   # continue
uv run --extra ai python -m src.ai.play                              # watch it play
```

Training prints the rolling episode return plus a periodic evaluation win rate,
and saves the best‑scoring policy to `src/ai/ppo.pt`. It also writes a resumable
training checkpoint (weights + optimizer + progress) to `src/ai/ppo_ckpt.pt`;
`--resume` picks it up so you can train across several sessions.

### Stronger agents: MCTS & AlphaZero

Model‑free PPO never wins the full 6‑wave game (the winning line is ~60 moves
deep — too deep for random exploration). The agents that **do** win are search
based, driving the deterministic `Game` via `clone`/`get_state`/`set_state`:

- **`src/ai/mcts.py`** — Monte‑Carlo Tree Search with shaped‑return backups and a
  survival‑aware greedy rollout. Wins the full game (~44 % over held‑out seeds at
  300 sims/move).
- **`src/ai/alphazero.py`** (`az_train.py` CLI) — AlphaZero‑style **PUCT** self‑play:
  the network's policy head supplies the prior and its value head replaces the
  rollout, trained on the search's visit distributions and shaped return‑to‑go,
  bootstrapped from MCTS wins on a difficulty **curriculum**. Two network types:
  a flat‑obs MLP (`az.pt`) and a **grid CNN** (`ConvActorCritic`, `--net-type cnn`,
  requires `BOLGROT_OBS_MODE=grid`). The CNN's spatial structure makes it the
  strongest learned net — `az_cnn_deep.pt` wins **~77 % @300, ~94 % @1000** sims
  over held‑out seeds, and win rate keeps climbing with the play‑time search
  budget (a single net reaches ~100 % by budget escalation).

```bash
uv run --extra ai python -m src.ai.az_train --curriculum --iterations 100   # train AlphaZero
uv run --extra ai python -m src.ai.baselines --az src/ai/az.pt --az-sims 300 # benchmark agents
uv run --extra ai python -m src.ai.play --policy src/ai/az.pt                # watch it play
```

---

## Building a standalone executable

A [PyInstaller](https://pyinstaller.org/) spec (`bolgrot.spec`) freezes the game
**with the AI bundled** (torch + the trained nets), so hints and autoplay work
fully offline — no Python install needed to play.

```bash
# CPU-only torch (the default index pulls multi-GB CUDA on Linux):
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install numpy
pip install -e ".[build]"          # base game + PyInstaller
pyinstaller bolgrot.spec --noconfirm --clean
# -> dist/bolgrot/                 (a folder; run bolgrot(.exe) inside it)
```

Notes:

- **AI-bundled, onedir (~300-500 MB).** Because torch is large, this is a
  **folder** (`dist/bolgrot/`), not a single file — unpacking a few-hundred-MB
  torch on every launch (onefile) would make startup very slow. Ship the whole
  folder (the CI zips it). Verify a build with
  `dist/bolgrot/bolgrot --selftest` (loads a net + runs one search).
- **Per-OS.** PyInstaller does not cross-compile: build on Windows for a `.exe`,
  on Linux for an ELF binary, on macOS for a `.app`. The
  `.github/workflows/build.yml` workflow builds all three on GitHub's runners
  and attaches the zips to a Release on a `v*` tag.
- The map, sprites, spawn patterns **and nets** are bundled via the spec's
  `datas` (the game loads them through `importlib.resources`, so they must ship
  with their package paths). Set `console=True` in the spec to see tracebacks
  while debugging a build.

---

## How it's built

### Architecture overview

The codebase is split so that **all rules live in pygame‑free modules** and
pygame only appears in the rendering and event‑loop layers.

```
src/
├── __main__.py     # pygame event loop only — translates input into Game calls
├── game.py         # Game: owns all state + the turn/action API (no pygame)
├── renderer.py     # Renderer: all drawing; never mutates game state
├── actions.py      # stateless click/hover hit‑testing helpers
├── constant.py     # all magic numbers + resource paths
├── entity/         # Entity (ABC) → Player, Bolgrot, Flame
├── spells/         # Spells (ABC) → ShortJump, LongJump, MoveFlames
├── map/            # Map: dict[(x,y) → Case], parses bolgrot.map
├── case/           # Case (grid cell) + CaseType enum, iso draw + hit‑test
├── BFS/            # 4‑directional breadth‑first pathfinding
├── button/         # Button widget for the end‑turn button
├── patterns/       # Patterns: the flame spawn‑wave pool (+ patterns.json)
├── config/         # bolgrot.map — the map layout
└── sprites_png/    # spell icons (.png, with .xcf GIMP sources)
```

**Separation of concerns**
- `Game` exposes the action API (`select_spell`, `play_selected_spell`,
  `clear_previsu`, `end_turn`, `reset`) and holds no rendering code. It also
  carries stubbed `step()` / `get_observation()` hooks for a future AI runner.
- `Renderer` receives game state as arguments and only draws — it never mutates
  state.
- `__main__.py` is purely the input → `Game` glue and the draw loop.

### Map representation

`Map.cases` is a `dict[tuple[int, int], Case]` keyed by `(x, y)`, giving O(1)
lookups. The map is parsed from a symbol grid in `bolgrot.map`
(`.` = free, `#` = wall, `|` = empty, `N` = no cell). Each `Case` holds its
coordinates, an optional `entity`, and a `case_type`.

### Isometric coordinate system

Grid → screen conversion is done inline wherever needed:

```
iso_x = (x - y) * (CASE_WIDTH  / 2)
iso_y = (x + y) * (CASE_HEIGHT / 2)
```

All rendering is relative to an `offset` (the screen pixel position of iso
origin `(0, 0)`), computed once to centre the map. The inverse (mouse → grid)
lives in `Case.contains()` and `actions.on_previsu_click()`.

### Dependency layering (acyclic)

Imports form a clean DAG:

```
constant → entity → case/map → BFS → spells → game
```

`Player` is deliberately **spell‑agnostic** (it imports `Spells` only under
`TYPE_CHECKING`); `Game.__init__` constructs the concrete spells with one shared
`BFS(map)` and assigns them. Because `entity` doesn't depend on the higher
layers, every module imports cleanly regardless of entry point.

### Flame spawn patterns

`Patterns` loads a vocabulary of 3‑tile "atoms" from `patterns.json` and builds
the legal wave pool as the union of two atoms (excluding spatially‑overlapping
pairs). A seeded `random.Random` makes a run's wave sequence **reproducible**.

- `draw_opening()` picks the first wave from a fixed opening set.
- `draw(occupied)` picks each subsequent wave **without replacement**, and
  **prioritises patterns that land on unoccupied cells**: it scores every
  remaining pattern by how many of its tiles overlap occupied tiles (flames or
  the player), keeps only those with the fewest overlaps, and randomly selects
  among them. Any residual occupied tiles are then dropped from the returned
  wave, so a flame is never spawned on top of an existing flame or the player.

`Game.end_turn()` places the current wave, advances the turn, then draws the
next wave passing in `Game._occupied_tiles()` (the set of tiles currently
holding a flame or the player).

### Packaging & resources

`src` is a regular package. Data files (`config/*.map`, `sprites_png/*.png`,
`patterns/*.json`) are declared as `package-data` and resolved at runtime via
`importlib.resources.files("src") / ...`, so they work from any working
directory, not just the repo root. Sprite images are lazy‑loaded on first
access.

### Tooling

- **`uv`** for dependency management and running.
- **`mypy`** for static type checking (the codebase is fully type‑annotated;
  `from __future__ import annotations` is used throughout).
- **`flake8`** for linting.
