# -*- mode: python ; coding: utf-8 -*-


a = Analysis(
    ['SolarFinish.py'],
    pathex=[],
    binaries=[],
    datas=[],
    hiddenimports=[],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=['pandas', 'scipy.stats', 'scipy.optimize', 'matplotlib', 'lxml', 'astropy.utils.iers', 'jedi', 'scipy.spatial', 'astropy.wcs', 'Pythonwin', 'scipy.interpolate', 'scipy.signal', 'scipy.integrate'],
    noarchive=False,
)

to_exclude = {'cv2\\opencv_videoio_ffmpeg480_64.dll','libopenblas64__v0.3.21-gcc_10_3_0.dll'}
a.binaries -= TOC([(os.path.normcase(x), None, None) for x in to_exclude])

pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.datas,
    [],
    name='SolarFinish',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=True,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
