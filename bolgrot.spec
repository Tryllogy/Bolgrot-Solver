# PyInstaller build spec — standalone game executable WITH the AI.
#
#   pyinstaller bolgrot.spec --noconfirm --clean      # -> dist/bolgrot/
#
# This bundles torch + the trained nets, so the AI hint / autoplay work fully
# offline. It is therefore large (~300-500 MB) and built as a **onedir** app
# (a folder holding bolgrot(.exe) + libs), NOT a single file: unpacking a
# few-hundred-MB torch on every launch (onefile) would make startup painfully
# slow. Ship the whole `dist/bolgrot` folder (the CI zips it per OS).
#
# IMPORTANT: build in an env with **CPU-only torch**
# (pip install torch --index-url https://download.pytorch.org/whl/cpu),
# otherwise collect_all pulls the multi-GB CUDA libraries.
#
# Build on the OS you target: Windows -> .exe, Linux -> ELF, macOS -> .app
# (PyInstaller does not cross-compile).

from PyInstaller.utils.hooks import collect_all

# Pull torch's submodules, extension libraries and data files.
torch_datas, torch_binaries, torch_hidden = collect_all('torch')

a = Analysis(
    ['run_game.py'],
    pathex=['.'],
    binaries=torch_binaries,
    datas=[
        ('src/config', 'src/config'),
        ('src/patterns', 'src/patterns'),
        ('src/sprites_png', 'src/sprites_png'),
        # The two nets the hint engine loads (Rapide = az.pt, Fort = the CNN).
        ('src/ai/az.pt', 'src/ai'),
        ('src/ai/az_cnn_deep.pt', 'src/ai'),
    ] + torch_datas,
    hiddenimports=['src.ai.policy', 'src.ai.alphazero'] + torch_hidden,
    hookspath=[],
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,      # onedir: binaries go in the COLLECT folder
    name='bolgrot',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,                  # UPX + torch DLLs are a known bad mix
    console=False,              # windowed game; set True to see tracebacks
    disable_windowed_traceback=False,
    argv_emulation=False,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=False,
    name='bolgrot',
)
