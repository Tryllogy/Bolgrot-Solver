# PyInstaller build spec — standalone game executable.
#
#   pyinstaller bolgrot.spec        # -> dist/bolgrot(.exe)
#
# This is the GAME-ONLY build: torch is excluded, so the executable is small
# (~40-70 MB) and needs no dependencies. The AI hint / autoplay buttons still
# appear but report that the optional `ai` extra is unavailable (a frozen app
# can't `pip install` it) — for the AI, run from source / pip (see README).
#
# Data files (the map, sprite PNGs and spawn patterns) are loaded at runtime via
# importlib.resources.files("src"), so they MUST be bundled with their package
# paths — that is what the `datas` entries below do.
#
# Build on the OS you target: Windows -> .exe, Linux -> ELF, macOS -> .app
# (PyInstaller does not cross-compile).

a = Analysis(
    ['run_game.py'],
    pathex=['.'],
    binaries=[],
    datas=[
        ('src/config', 'src/config'),
        ('src/patterns', 'src/patterns'),
        ('src/sprites_png', 'src/sprites_png'),
    ],
    # The hint code imports these lazily (inside functions), so PyInstaller's
    # static scan misses them; list them so a clean "install the ai extra"
    # message shows (they hit the excluded torch and fail gracefully).
    hiddenimports=['src.ai.policy', 'src.ai.alphazero'],
    hookspath=[],
    runtime_hooks=[],
    # Keep the game-only build lean: never bundle the heavy ML stack.
    excludes=['torch', 'numpy', 'torchvision', 'torchaudio'],
    noarchive=False,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.datas,
    [],
    name='bolgrot',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    runtime_tmpdir=None,
    console=False,          # windowed game; set True to see tracebacks/logs
    disable_windowed_traceback=False,
    argv_emulation=False,
)
