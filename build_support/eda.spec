# -*- mode: python ; coding: utf-8 -*-
from pathlib import Path
import hashlib
import json
import os
from PyInstaller.utils.hooks import collect_data_files, copy_metadata

project = Path(SPECPATH).parent
release_version = os.environ.get('EDA_RELEASE_VERSION', '')
if release_version != '0.3.0':
    raise RuntimeError('EDA_RELEASE_VERSION must be 0.3.0; use scripts/build_windows_current.ps1')
build_info_path = project / 'src' / 'eda_toolkit' / 'resources' / 'build_info.json'
build_info = json.loads(build_info_path.read_text(encoding='utf-8'))
if build_info.get('version') != release_version or build_info.get('fingerprint_scope') != 'package_python_except_ui_v1':
    raise RuntimeError('build_info.json does not match this version and fingerprint scope; run the v0.2 inventory prepare step')
icon_names = ['app_icon.svg', 'app_icon.ico', 'app_icon_16.png', 'app_icon_24.png',
              'app_icon_32.png', 'app_icon_48.png', 'app_icon_64.png',
              'app_icon_128.png', 'app_icon_256.png']
root_icon_dir = project / 'assets'
package_icon_dir = project / 'src' / 'eda_toolkit' / 'resources' / 'icons'
for icon_name in icon_names:
    root_icon = root_icon_dir / icon_name
    package_icon = package_icon_dir / icon_name
    if not root_icon.is_file() or not package_icon.is_file():
        raise RuntimeError(f'missing source/package icon asset: {icon_name}')
    if hashlib.sha256(root_icon.read_bytes()).hexdigest() != hashlib.sha256(package_icon.read_bytes()).hexdigest():
        raise RuntimeError(f'root and package icon assets differ: {icon_name}')
datas = collect_data_files('eda_toolkit', include_py_files=False)
for name in ('eda-toolkit','numpy','scipy','pandas','PySide6','PySide6_Essentials','shiboken6',
             'matplotlib','statsmodels','openpyxl','reportlab','pillow','python-dateutil','packaging','patsy'):
    datas += copy_metadata(name)
hidden = ['scipy.signal','scipy.stats','scipy.interpolate','statsmodels.regression.mixed_linear_model',
          'statsmodels.stats.multitest','openpyxl','reportlab.pdfbase.ttfonts',
          'matplotlib.backends.backend_agg','matplotlib.backends.backend_qtagg']
common = dict(pathex=[str(project / 'src')], binaries=[], datas=datas, hiddenimports=hidden,
              hookspath=[], runtime_hooks=[], excludes=['tkinter','PyQt5','PyQt6','PySide2',
              'PySide6.QtWebEngineCore','PySide6.QtWebEngineWidgets','PySide6.QtQml',
              'pytest','_pytest','IPython','jupyter','notebook'],
              noarchive=False)
cli = Analysis([str(project / 'build_support' / 'cli_entry.py')], **common)
gui = Analysis([str(project / 'build_support' / 'gui_entry.py')], **common)
# Qt 6.12 wheels use the Windows system ICU API (unsuffixed exports). When the
# interpreter is Conda-based, bindepend may collect Conda ICU 78 under the same
# icuuc.dll basename; its suffixed exports break QtCore. Keep system ICU external.
for analysis in (cli, gui):
    analysis.binaries = [item for item in analysis.binaries
                         if Path(item[0]).name.lower() not in {'icuuc.dll', 'icudt78.dll'}]
    # This Widgets workbench does not use QML/Quick/VirtualKeyboard/WebEngine or
    # the PDF image-format plug-in. Do not collect unrelated Qt feature modules.
    analysis.binaries = [item for item in analysis.binaries
                         if not Path(item[0]).name.lower().startswith(('qt6qml','qt6quick','qt6virtualkeyboard','qt6webengine','qt6pdf'))
                         and '/platforminputcontexts/' not in item[0].replace('\\','/').lower()
                         and Path(item[0]).name.lower() not in {'qpdf.dll','qopensslbackend.dll'}]
pyz = PYZ(cli.pure, gui.pure)
icon = str(project / 'assets' / 'app_icon.ico')
cli_exe = EXE(pyz, cli.scripts, [], exclude_binaries=True, name='eda_cli', console=True,
              icon=icon,
              version=str(project / 'build_support' / 'version_info_cli.txt'),
              debug=False, strip=False, upx=False)
gui_exe = EXE(pyz, gui.scripts, [], exclude_binaries=True, name='eda_gui', console=False,
              icon=icon,
              version=str(project / 'build_support' / 'version_info_gui.txt'),
              debug=False, strip=False, upx=False)
bundle = COLLECT(cli_exe, gui_exe, cli.binaries, gui.binaries, cli.datas, gui.datas,
                 strip=False, upx=False, name='EDA_Toolkit-v0.3.0')
