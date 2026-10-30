"""记录实际安装/分发物、许可证副本、锁定与SHA256；不冒称法律审查。"""
import argparse
import hashlib
from importlib import metadata
import json
from pathlib import Path
import shutil
import sys
import tomllib

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / 'src'))
from eda_toolkit.core_export import atomic_json, sha256_file
from eda_toolkit.capabilities import SOURCE_FINGERPRINT_SCOPE, source_fingerprint
from eda_toolkit.config import PRESETS, get_preset, save_config
from eda_toolkit import __version__


def _project_version():
    with (PROJECT / 'pyproject.toml').open('rb') as stream:
        return tomllib.load(stream)['project']['version']


def _source_snapshot(scope):
    package_root = PROJECT / 'src' / 'eda_toolkit'
    sources = {}
    for path in sorted(package_root.rglob('*.py')):
        relative = path.relative_to(package_root)
        if relative.parts and relative.parts[0] == 'ui':
            continue
        sources[relative.as_posix()] = sha256_file(path)
    payload = {'fingerprint_scope': scope, 'files': sources}
    fingerprint = hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(',', ':')).encode('utf-8')).hexdigest()
    if fingerprint != source_fingerprint():
        raise RuntimeError('release source snapshot disagrees with runtime source_fingerprint()')
    return fingerprint, sources


def _validate_release_version(version):
    declared = {'pyproject.toml': _project_version(), 'eda_toolkit.__version__': __version__,
                'editable_distribution': metadata.version('eda-toolkit')}
    if any(value != version for value in declared.values()):
        raise RuntimeError(f'release version mismatch: requested={version}; observed={declared}')
    return declared


def _build_info(version, source_scope):
    if source_scope != SOURCE_FINGERPRINT_SCOPE:
        raise RuntimeError(f'unknown source fingerprint scope: {source_scope!r}')
    version_sources = _validate_release_version(version)
    fingerprint, files = _source_snapshot(source_scope)
    payload = {'version': version, 'fingerprint_scope': source_scope,
               'source_fingerprint_scope': source_scope,
               'source_fingerprint': fingerprint, 'source_files': files,
               'python': sys.version, 'version_sources': version_sources}
    atomic_json(PROJECT / 'src' / 'eda_toolkit' / 'resources' / 'build_info.json', payload)
    return payload


def _verify_prepared_build_info(version, source_scope):
    path = PROJECT / 'src' / 'eda_toolkit' / 'resources' / 'build_info.json'
    payload = json.loads(path.read_text(encoding='utf-8'))
    fingerprint, files = _source_snapshot(source_scope)
    if (payload.get('version') != version or
            payload.get('fingerprint_scope') != source_scope or
            payload.get('source_fingerprint_scope') != source_scope or
            payload.get('source_fingerprint') != fingerprint or
            payload.get('source_files') != files):
        raise RuntimeError('prepared build_info.json is stale or belongs to another release version')
    _validate_release_version(version)
    return payload


def _ignore_skill_artifacts(directory, names):
    ignored = {"__pycache__", ".pytest_cache", ".mypy_cache", ".ruff_cache", "validation-summary.json"}
    return [name for name in names if name in ignored or name.endswith((".pyc", ".pyo"))]


def _copy_release_metadata(bundle: Path, version: str, source_scope: str):
    if not bundle.is_dir():
        raise FileNotFoundError(f'PyInstaller bundle not found: {bundle}')
    if not bundle.is_relative_to((PROJECT / 'dist' / f'v{version}').resolve()):
        raise RuntimeError(f'v{version} bundle must be under its versioned dist directory: {bundle}')
    if bundle.name != f'EDA_Toolkit-v{version}':
        raise RuntimeError(f'bundle name must be EDA_Toolkit-v{version}; observed {bundle.name!r}')
    if (bundle / 'release_manifest.json').exists():
        raise FileExistsError('refusing to overwrite a finalized release manifest')
    for exe in ('eda_cli.exe', 'eda_gui.exe'):
        if not (bundle / exe).is_file():
            raise FileNotFoundError(f'PyInstaller output is missing {exe}')
    _verify_prepared_build_info(version, source_scope)
    destination = bundle / 'third_party'
    shutil.copytree(PROJECT / 'third_party', destination, dirs_exist_ok=True)
    for name in ('README.md', 'requirements.lock.txt', 'requirements.lock.sha256.txt'):
        source = PROJECT / name
        if source.is_file():
            shutil.copy2(source, bundle / name)
    shutil.copytree(PROJECT / 'docs', bundle / 'docs', dirs_exist_ok=True)
    shutil.copy2(PROJECT / 'src' / 'eda_toolkit' / 'resources' / 'help' / 'EDA使用指南.pdf',
                 bundle / 'docs' / 'EDA使用指南.pdf')
    shutil.copytree(PROJECT / 'assets', bundle / 'assets', dirs_exist_ok=True)
    shutil.copytree(PROJECT / 'skills', bundle / 'skills', dirs_exist_ok=True,
                    ignore=_ignore_skill_artifacts)
    preset_root = PROJECT / 'src' / 'eda_toolkit' / 'resources' / 'presets'
    shutil.copytree(preset_root, bundle / 'presets', dirs_exist_ok=True)
    return {'bundle': str(bundle), 'version': version,
            'source_fingerprint_scope': source_scope,
            'source_fingerprint': _verify_prepared_build_info(version, source_scope)['source_fingerprint']}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--version', default=__version__)
    parser.add_argument('--build-dir', type=Path, default=PROJECT / 'build')
    parser.add_argument('--source-scope', default=SOURCE_FINGERPRINT_SCOPE)
    parser.add_argument('--bundle', type=Path)
    args = parser.parse_args()
    _validate_release_version(args.version)
    if args.bundle is None:
        build_info = _build_info(args.version, args.source_scope)
    else:
        build_info = _verify_prepared_build_info(args.version, args.source_scope)
    preset_root = PROJECT / 'src' / 'eda_toolkit' / 'resources' / 'presets'
    preset_root.mkdir(parents=True, exist_ok=True)
    for name in PRESETS:
        save_config(get_preset(name), preset_root / (name + '.json'))
    destination = PROJECT / 'third_party'
    license_root = destination / 'licenses'
    license_root.mkdir(parents=True, exist_ok=True)
    runtime_license = Path(sys.base_prefix) / 'LICENSE_PYTHON.txt'
    if runtime_license.is_file():
        shutil.copy2(runtime_license, license_root / 'Python-LICENSE.txt')
    components, lines, seen = [], [], set()
    for distribution in sorted(metadata.distributions(), key=lambda item: item.metadata['Name'].lower()):
        name, version = distribution.metadata['Name'], distribution.version
        identity = (name.lower().replace('_', '-'), version)
        if identity in seen:
            continue
        seen.add(identity)
        if name.lower() != 'eda-toolkit':
            lines.append(f'{name}=={version}')
        license_files = []
        for relative in distribution.files or []:
            basename = Path(str(relative)).name.lower()
            if basename.startswith(('license', 'licence', 'copying', 'notice', 'copyright')):
                source = Path(distribution.locate_file(relative))
                if source.is_file():
                    target = license_root / name / str(relative).replace('..', '_').replace(':', '_')
                    target.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(source, target)
                    license_files.append({'path': target.relative_to(PROJECT).as_posix(), 'sha256': sha256_file(target)})
        license_id = distribution.metadata.get('License-Expression') or distribution.metadata.get('License')
        components.append({'name': name, 'version': version, 'license_declared': license_id,
                           'license_files': license_files, 'installer': distribution.read_text('INSTALLER'),
                           'classification': 'installed_development_environment',
                           'distribution_review': 'artifact_inventory_only_not_legal_clearance'})
    (PROJECT / 'requirements.lock.txt').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    wheels = args.build_dir.resolve() / 'wheelhouse'
    if wheels.is_dir():
        from packaging.utils import canonicalize_name, parse_wheel_filename
        hashes, wheel_records = {}, []
        for path in sorted(wheels.glob('*.whl')):
            name, version, wheel_build_tag, tags = parse_wheel_filename(path.name)
            digest = sha256_file(path)
            hashes[(canonicalize_name(name), str(version))] = digest
            wheel_records.append({'distribution': str(name), 'version': str(version), 'build_tag': str(wheel_build_tag), 'file': path.name,
                                  'sha256': digest, 'size': path.stat().st_size, 'tags': sorted(map(str,tags))})
        locked = []
        for line in lines:
            name, version = line.split('==')
            digest = hashes.get((canonicalize_name(name),version))
            if digest:
                locked.append(line + ' --hash=sha256:' + digest)
        if len(locked) == len(lines):
            (PROJECT / 'requirements.lock.sha256.txt').write_text('\n'.join(locked) + '\n', encoding='utf-8')
        atomic_json(destination / 'wheel_inventory.json', {'acquisition_command': 'python -m pip download --no-deps --only-binary=:all: -r requirements.lock.txt --dest build/wheelhouse',
                    'scope': 'exact build environment wheels for Windows x64/Python 3.14; includes test/build tools',
                    'locked_distributions': len(lines), 'hashed_lock_complete': len(locked)==len(lines), 'wheels':wheel_records})
    atomic_json(destination / 'sbom-installed.json', {'format': 'EDA-inventory-1.0', 'python': sys.version,
                'components': components, 'scope': 'actual isolated build environment; includes build/test dependencies'})
    notice = ['# 第三方告知', '', '本软件使用以下实际安装组件。许可证原文副本位于 licenses/。',
              '本清单是实际制品记录，不是所有分发义务已完成的法律结论。', '',
              'PySide6/Qt 使用动态库。不要删除第三方告知或限制用户依法替换/调试Qt库的权利。',
              '只在核对实际用到的Qt模块、对应源码提供方式和最终许可决策后对外分发。',
              'Qt来源与许可说明：https://doc.qt.io/qtforpython-6/licenses.html', '',
              'cvxEDA、Ledalab与GPU依赖没有包含在基础算法路径。', '']
    notice.extend(f"- {item['name']} {item['version']}：{str(item['license_declared'] or '见许可证原文').splitlines()[0]}" for item in components)
    (destination / 'THIRD_PARTY_NOTICES.md').write_text('\n'.join(notice), encoding='utf-8')
    bundled = _copy_release_metadata(args.bundle.resolve(), args.version, args.source_scope) if args.bundle else None
    print(json.dumps({'version': args.version, 'source_fingerprint_scope': args.source_scope,
                      'source_fingerprint': build_info['source_fingerprint'],
                      'components': len(components),
                      'license_files': sum(len(item['license_files']) for item in components),
                      'lock': str(PROJECT / 'requirements.lock.txt'), 'bundle': bundled}, ensure_ascii=False))


if __name__ == '__main__':
    main()
