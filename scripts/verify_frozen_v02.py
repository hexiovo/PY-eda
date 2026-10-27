"""Run version-isolated v0.2 frozen-process and hand-off acceptance checks.

This records development-host evidence with a restricted PATH. It is not a clean
Windows VM, real-device validation, or a taskbar visual inspection.
"""
from __future__ import annotations

import argparse
import ctypes
import hashlib
import json
import os
from pathlib import Path
import platform
import re
import struct
import subprocess
import sys
import time
import uuid

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / 'src'))
from eda_toolkit.capabilities import SOURCE_FINGERPRINT_SCOPE
from eda_toolkit.core_export import sha256_file, verify_manifest


VERSION = '0.2.0'
EXPECTED_APP_USER_MODEL_ID = 'EDA.AnalysisWorkbench.0_2'
EXPECTED_ICON_SIZES = {16, 24, 32, 48, 64, 128, 256}
EXTENDED_RECORD_TABLES = {
    'trial_table', 'trial_summaries', 'trial_fit_diagnostics', 'spectra',
    'complexity_diagnostics', 'ns_classification', 'scr_events_extended',
}


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _version_strings(path: Path) -> dict[str, str]:
    if os.name != 'nt':
        raise RuntimeError('PE version resources can only be checked on Windows')
    version = ctypes.WinDLL('version', use_last_error=True)
    size_fn = version.GetFileVersionInfoSizeW
    size_fn.argtypes = [ctypes.c_wchar_p, ctypes.POINTER(ctypes.c_uint32)]
    size_fn.restype = ctypes.c_uint32
    get_fn = version.GetFileVersionInfoW
    get_fn.argtypes = [ctypes.c_wchar_p, ctypes.c_uint32, ctypes.c_uint32, ctypes.c_void_p]
    get_fn.restype = ctypes.c_int
    query_fn = version.VerQueryValueW
    query_fn.argtypes = [ctypes.c_void_p, ctypes.c_wchar_p,
                         ctypes.POINTER(ctypes.c_void_p), ctypes.POINTER(ctypes.c_uint32)]
    query_fn.restype = ctypes.c_int
    ignored = ctypes.c_uint32(0)
    size = size_fn(str(path), ctypes.byref(ignored))
    if not size:
        raise ctypes.WinError(ctypes.get_last_error())
    data = ctypes.create_string_buffer(size)
    if not get_fn(str(path), 0, size, data):
        raise ctypes.WinError(ctypes.get_last_error())
    values = {}
    for key in ('ProductName', 'ProductVersion', 'FileVersion', 'FileDescription',
                'OriginalFilename', 'InternalName'):
        pointer = ctypes.c_void_p()
        length = ctypes.c_uint32()
        query = f'\\StringFileInfo\\040904B0\\{key}'
        if query_fn(data, query, ctypes.byref(pointer), ctypes.byref(length)) and pointer.value:
            values[key] = ctypes.wstring_at(pointer.value, max(0, length.value - 1))
    return values


def _icon_group_sizes(path: Path) -> tuple[set[int], int]:
    import pefile

    pe = pefile.PE(str(path), fast_load=False)
    if not hasattr(pe, 'DIRECTORY_ENTRY_RESOURCE'):
        raise RuntimeError(f'{path.name} has no PE resource directory')
    groups = []
    icon_resource_count = 0
    for resource_type in pe.DIRECTORY_ENTRY_RESOURCE.entries:
        resource_id = getattr(resource_type, 'id', None)
        if resource_id == 3 and getattr(resource_type, 'directory', None):
            icon_resource_count += len(resource_type.directory.entries)
        if resource_id != 14 or not getattr(resource_type, 'directory', None):
            continue
        for named_group in resource_type.directory.entries:
            directory = getattr(named_group, 'directory', None)
            if directory is None:
                continue
            for language in directory.entries:
                data_entry = getattr(language, 'data', None)
                if data_entry is None:
                    continue
                structure = data_entry.struct
                blob = pe.get_data(structure.OffsetToData, structure.Size)
                if len(blob) < 6:
                    continue
                reserved, icon_type, count = struct.unpack_from('<HHH', blob, 0)
                if reserved != 0 or icon_type != 1 or len(blob) < 6 + 14 * count:
                    continue
                sizes = set()
                for index in range(count):
                    width, height, _colors, _reserved, _planes, _bits, _bytes, _icon_id = struct.unpack_from(
                        '<BBBBHHIH', blob, 6 + 14 * index)
                    sizes.add(width or 256)
                    sizes.add(height or 256)
                groups.append(sizes)
    pe.close()
    if len(groups) != 1:
        raise RuntimeError(f'{path.name} must contain exactly one RT_GROUP_ICON; got {len(groups)}')
    return groups[0], icon_resource_count


def _shell_extract_icon_sizes(path: Path) -> set[int]:
    """Ask Windows to extract every expected size from the actual PE image."""
    if os.name != 'nt':
        raise RuntimeError('Windows shell icon extraction is only available on Windows')

    class IconInfo(ctypes.Structure):
        _fields_ = [('fIcon', ctypes.c_int), ('xHotspot', ctypes.c_uint32),
                    ('yHotspot', ctypes.c_uint32), ('hbmMask', ctypes.c_void_p),
                    ('hbmColor', ctypes.c_void_p)]

    class Bitmap(ctypes.Structure):
        _fields_ = [('bmType', ctypes.c_long), ('bmWidth', ctypes.c_long),
                    ('bmHeight', ctypes.c_long), ('bmWidthBytes', ctypes.c_long),
                    ('bmPlanes', ctypes.c_ushort), ('bmBitsPixel', ctypes.c_ushort),
                    ('bmBits', ctypes.c_void_p)]

    user32 = ctypes.WinDLL('user32', use_last_error=True)
    extract = user32.PrivateExtractIconsW
    extract.argtypes = [ctypes.c_wchar_p, ctypes.c_int, ctypes.c_int, ctypes.c_int,
                        ctypes.POINTER(ctypes.c_void_p), ctypes.POINTER(ctypes.c_uint32),
                        ctypes.c_uint32, ctypes.c_uint32]
    extract.restype = ctypes.c_uint32
    get_info = user32.GetIconInfo
    get_info.argtypes = [ctypes.c_void_p, ctypes.POINTER(IconInfo)]
    get_info.restype = ctypes.c_int
    destroy_icon = user32.DestroyIcon
    destroy_icon.argtypes = [ctypes.c_void_p]
    destroy_icon.restype = ctypes.c_int
    gdi32 = ctypes.WinDLL('gdi32', use_last_error=True)
    get_object = gdi32.GetObjectW
    get_object.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.POINTER(Bitmap)]
    get_object.restype = ctypes.c_int
    delete_object = gdi32.DeleteObject
    delete_object.argtypes = [ctypes.c_void_p]
    delete_object.restype = ctypes.c_int

    extracted = set()
    for requested in EXPECTED_ICON_SIZES:
        handles = (ctypes.c_void_p * 1)()
        resource_ids = (ctypes.c_uint32 * 1)()
        count = extract(str(path), 0, requested, requested, handles, resource_ids, 1, 0)
        if count != 1 or not handles[0]:
            raise ctypes.WinError(ctypes.get_last_error())
        info = IconInfo()
        bitmaps = []
        try:
            if not get_info(handles[0], ctypes.byref(info)):
                raise ctypes.WinError(ctypes.get_last_error())
            for handle in (info.hbmColor, info.hbmMask):
                if handle:
                    bitmaps.append(handle)
            image_handle = info.hbmColor or info.hbmMask
            bitmap = Bitmap()
            if not image_handle or not get_object(image_handle, ctypes.sizeof(bitmap), ctypes.byref(bitmap)):
                raise ctypes.WinError(ctypes.get_last_error())
            height = bitmap.bmHeight if info.hbmColor else bitmap.bmHeight // 2
            if bitmap.bmWidth != requested or height != requested:
                raise RuntimeError(f'Windows extracted {requested}px request as {bitmap.bmWidth}x{height}')
            extracted.add(requested)
        finally:
            destroy_icon(handles[0])
            for handle in bitmaps:
                delete_object(handle)
    return extracted


def _check_executable_resources(bundle: Path) -> dict:
    expected = {
        'eda_cli.exe': 'EDA Toolkit Command-Line Analyzer',
        'eda_gui.exe': 'EDA Toolkit Desktop Workbench',
    }
    results = {}
    for name, description in expected.items():
        path = bundle / name
        if not path.is_file():
            raise FileNotFoundError(path)
        version = _version_strings(path)
        required = {'ProductName': 'EDA Toolkit', 'ProductVersion': VERSION,
                    'FileVersion': '0.2.0.0', 'FileDescription': description,
                    'OriginalFilename': name, 'InternalName': Path(name).stem}
        if version != required:
            raise RuntimeError(f'{name} version strings do not match: {version!r}')
        icon_sizes, icon_count = _icon_group_sizes(path)
        if icon_sizes != EXPECTED_ICON_SIZES or icon_count < len(EXPECTED_ICON_SIZES):
            raise RuntimeError(f'{name} embedded icon sizes differ: sizes={sorted(icon_sizes)}, icon_resource_count={icon_count}')
        shell_sizes = _shell_extract_icon_sizes(path)
        if shell_sizes != EXPECTED_ICON_SIZES:
            raise RuntimeError(f'{name} Windows shell extraction sizes differ: {sorted(shell_sizes)}')
        results[name] = {'version_info': version, 'icon_sizes': sorted(icon_sizes),
                         'windows_shell_extracted_icon_sizes': sorted(shell_sizes),
                         'icon_resource_count': icon_count, 'sha256': sha256_file(path),
                         'size': path.stat().st_size}
    return results


def _restricted_environment() -> dict[str, str]:
    environment = {key: value for key, value in os.environ.items()
                   if not key.upper().startswith(('PYTHON', 'CONDA', 'VIRTUAL_ENV', 'EDA_RELEASE'))}
    system_root = Path(os.environ.get('SystemRoot', r'C:\Windows'))
    environment['PATH'] = str(system_root / 'System32')
    environment['QT_QPA_PLATFORM'] = 'offscreen'
    environment['PYTHONUTF8'] = '1'
    return environment


def _prepare_source_queue(config_path: Path, work_root: Path) -> tuple[Path, str, dict]:
    from eda_toolkit.config import resolve_config
    from eda_toolkit.project_store import open_project
    from eda_toolkit.runtime_preflight import preflight_run
    from eda_toolkit.task_store import enqueue_task

    project_path = work_root / 'persistent queue project'
    project = open_project(project_path)
    config = resolve_config(config_path)
    config['workspace']['project_directory'] = str(project_path.resolve())
    config['output']['directory'] = str((work_root / 'queued-task-results').resolve())
    config['output']['formats'] = ['json']
    input_manifest = config['input'].get('manifest')
    report = preflight_run(config, input_manifest, project)
    if report.get('eligible_to_enqueue') is not True:
        raise RuntimeError(f'queue fixture preflight blocked: {report.get("errors")}')
    frozen = {'config': config, 'input_manifest': input_manifest,
              'project_revision': project['project_revision'], 'draft_revision': project['draft_revision'],
              'marker_revision': project['marker_revision'], 'qc_revision': project['qc_revision']}
    task_id = enqueue_task(project, frozen, report)
    return project_path, task_id, report


def _write_short_failure_fixture(work_root: Path) -> Path:
    signal_path = work_root / 'short-record.csv'
    rows = [f'{index / 4:.2f},{2 + 0.01 * (index % 3):.6f}' for index in range(20)]
    signal_path.write_text('time_s,eda_uS\n' + '\n'.join(rows) + '\n', encoding='utf-8')
    config = {
        'input': {'signals': [str(signal_path)]},
        'mapping': {'columns': {'time_s': 'time_s', 'eda_uS': 'eda_uS'},
                    'sampling_rate_hz': 4.0, 'eda_unit': 'uS', 'time_unit': 's'},
        'output': {'directory': str((work_root / 'short-results').resolve()), 'formats': ['json']},
    }
    path = work_root / 'short-record-config.json'
    path.write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding='utf-8')
    return path


def _source_identity_probe() -> dict:
    """Exercise the shared Windows AppUserModelID API in a real Qt process.

    The separate frozen GUI smoke still owns frozen-window startup; this probe
    does not claim a taskbar or desktop screenshot check.
    """
    if os.name != 'nt':
        raise RuntimeError('identity probe requires Windows')
    os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
    from eda_toolkit.runtime_identity import APP_USER_MODEL_ID, configure_app_identity, icon_path
    configured = configure_app_identity()
    actual_icon_path = Path(icon_path()).resolve()
    expected_source_icon = (PROJECT / 'src' / 'eda_toolkit' / 'resources' / 'icons' / 'app_icon.ico').resolve()
    if actual_icon_path != expected_source_icon or not actual_icon_path.is_file():
        raise RuntimeError(f'source Qt icon did not resolve to its packaged asset: {actual_icon_path}')
    shell = ctypes.WinDLL('shell32', use_last_error=True)
    getter = shell.GetCurrentProcessExplicitAppUserModelID
    getter.argtypes = [ctypes.POINTER(ctypes.c_void_p)]
    getter.restype = ctypes.c_long
    pointer = ctypes.c_void_p()
    code = getter(ctypes.byref(pointer))
    if code != 0 or not pointer.value:
        raise ctypes.WinError(code & 0xFFFFFFFF)
    try:
        observed = ctypes.wstring_at(pointer.value)
    finally:
        ctypes.WinDLL('ole32').CoTaskMemFree(pointer)
    if not configured.get('configured') or observed != EXPECTED_APP_USER_MODEL_ID or APP_USER_MODEL_ID != observed:
        raise RuntimeError(f'AppUserModelID mismatch: api={configured}, observed={observed!r}')

    from eda_toolkit.ui.app import create_window
    window, state = create_window()
    try:
        state['app'].processEvents()
        app_icon_ok = not state['app'].windowIcon().isNull()
        window_icon_ok = not window.windowIcon().isNull()
        font_metadata = state.get('font_metadata', {})
        font_path = Path(font_metadata.get('path', '')).resolve()
        expected_source_font = (PROJECT / 'src' / 'eda_toolkit' / 'resources' / 'help' /
                                'fonts' / 'NotoSansSC-Regular.ttf').resolve()
        page_names = sorted(state['widgets']['pages'])
        required_widgets = ('navigation', 'project_autosave_status', 'marker_table',
                            'log_results', 'log_search', 'log_export', 'start_queue', 'preflight')
        widget_presence = {name: name in state['widgets'] for name in required_widgets}
        if (not app_icon_ok or not window_icon_ok or not font_path.is_file() or
                font_metadata.get('source') != 'bundled' or font_path != expected_source_font or
                not all(widget_presence.values())):
            raise RuntimeError(f'Qt identity/font/widget probe failed: app_icon={app_icon_ok}, '
                               f'window_icon={window_icon_ok}, font={font_metadata}, widgets={widget_presence}')
        return {'api': configured, 'app_user_model_id': observed,
                'app_icon_set': app_icon_ok, 'window_icon_set': window_icon_ok,
                'font_metadata': font_metadata,
                'visible_page_count': len(page_names), 'page_names': page_names,
                'marker_log_autosave_queue_widgets_present': widget_presence,
                'scope': 'source_Qt_process_shared_identity_api_not_frozen_taskbar_visual_review',
                'icon_path': str(actual_icon_path)}
    finally:
        window.close()
        state['shutdown']()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument('--bundle', type=Path,
                        default=PROJECT / 'dist' / 'v0.2' / 'EDA_Toolkit-v0.2.0')
    parser.add_argument('--report', type=Path,
                        default=PROJECT / 'artifacts' / 'verification' / 'v0.2' / 'frozen_application.json')
    parser.add_argument('--timeout', type=int, default=300)
    args = parser.parse_args()
    bundle, report_path = args.bundle.resolve(), args.report.resolve()
    dist_root = (PROJECT / 'dist' / 'v0.2').resolve()
    evidence_root = (PROJECT / 'artifacts' / 'verification' / 'v0.2').resolve()
    if not bundle.is_relative_to(dist_root) or bundle.name != f'EDA_Toolkit-v{VERSION}':
        raise RuntimeError('frozen verifier accepts only the isolated v0.2.0 bundle')
    if not report_path.is_relative_to(evidence_root) or report_path.exists():
        raise RuntimeError('verification report must be a fresh v0.2 evidence path; existing reports are immutable')
    if not bundle.is_dir():
        raise FileNotFoundError(bundle)
    cli = bundle / 'eda_cli.exe'
    gui = bundle / 'eda_gui.exe'
    executable_resources = _check_executable_resources(bundle)
    build_info_candidates = list(bundle.rglob('build_info.json'))
    if len(build_info_candidates) != 1:
        raise RuntimeError(f'expected exactly one bundled build_info.json, found {len(build_info_candidates)}')
    build_info = json.loads(build_info_candidates[0].read_text(encoding='utf-8'))
    if build_info.get('version') != VERSION or build_info.get('fingerprint_scope') != SOURCE_FINGERPRINT_SCOPE:
        raise RuntimeError('bundled build_info version/scope mismatch')
    if not Path(build_info_candidates[0]).resolve().is_relative_to(bundle):
        raise RuntimeError('build_info.json resolved outside the new v0.2 bundle')

    work_root = evidence_root / ('native-smoke-' + uuid.uuid4().hex[:10])
    work_root.mkdir(parents=True, exist_ok=False)
    environment = _restricted_environment()
    case_dir = work_root / 'process-logs'
    case_dir.mkdir()
    records = []
    status, error = 'passed', None

    def run(executable: Path, arguments: list[str], label: str, *, expected_code=0, timeout=None,
            parse_json=True):
        safe = re.sub(r'[^A-Za-z0-9_.-]+', '_', label).strip('_')[:90] or 'case'
        stdout_path, stderr_path = case_dir / f'{safe}.stdout.txt', case_dir / f'{safe}.stderr.txt'
        started = time.monotonic()
        result = subprocess.run([str(executable), *map(str, arguments)], cwd=work_root, env=environment,
                                capture_output=True, text=True, encoding='utf-8', errors='replace',
                                timeout=timeout or args.timeout,
                                creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0) if os.name == 'nt' else 0)
        elapsed = time.monotonic() - started
        stdout_path.write_text(result.stdout, encoding='utf-8')
        stderr_path.write_text(result.stderr, encoding='utf-8')
        item = {'label': label, 'executable': executable.name, 'arguments': list(map(str, arguments)),
                'returncode': result.returncode, 'expected_returncode': expected_code,
                'elapsed_s': elapsed,
                'stdout': {'path': stdout_path.relative_to(work_root).as_posix(), 'sha256': sha256_file(stdout_path),
                           'size': stdout_path.stat().st_size},
                'stderr': {'path': stderr_path.relative_to(work_root).as_posix(), 'sha256': sha256_file(stderr_path),
                           'size': stderr_path.stat().st_size}}
        records.append(item)
        if result.returncode != expected_code:
            raise RuntimeError(f'{label} exit {result.returncode}, expected {expected_code}: {result.stderr[-1200:]} {result.stdout[-1200:]}')
        if parse_json and result.stdout.strip().startswith('{'):
            return json.loads(result.stdout)
        return result.stdout

    try:
        version = run(cli, ['version', '--json'], 'version')
        if version.get('version') != VERSION or version.get('schema_version') != '1.1':
            raise RuntimeError(f'CLI version/schema mismatch: {version}')
        bundled_resources = run(cli, ['resources', '--json'], 'resolved_frozen_resource_paths')
        runtime_root = Path(bundled_resources.get('runtime_root', '')).resolve()
        if bundled_resources.get('frozen') is not True or not runtime_root.is_relative_to(bundle):
            raise RuntimeError(f'frozen runtime root is outside the new bundle: {bundled_resources}')
        expected_resources = {
            'icon': PROJECT / 'src' / 'eda_toolkit' / 'resources' / 'icons' / 'app_icon.ico',
            'font': PROJECT / 'src' / 'eda_toolkit' / 'resources' / 'help' / 'fonts' / 'NotoSansSC-Regular.ttf',
            'help': PROJECT / 'src' / 'eda_toolkit' / 'resources' / 'help' / 'EDA使用指南.pdf',
        }
        resolved_resource_paths = {}
        for resource_name in ('icon', 'font', 'help'):
            item = bundled_resources.get('resources', {}).get(resource_name, {})
            resource_path = Path(item.get('path', '')).resolve()
            if not resource_path.is_file() or not resource_path.is_relative_to(runtime_root):
                raise RuntimeError(f'{resource_name} did not resolve from the frozen bundle: {item}')
            expected_path = expected_resources[resource_name]
            if not expected_path.is_file() or item.get('sha256') != sha256_file(expected_path):
                raise RuntimeError(f'frozen {resource_name} hash differs from the packaged source resource')
            resolved_resource_paths[resource_name] = {
                'path': str(resource_path), 'sha256': item.get('sha256'), 'size': item.get('size'),
                'source_sha256': sha256_file(expected_path)}
        capabilities = run(cli, ['capabilities', '--json'], 'capabilities')
        if capabilities.get('source_fingerprint') != build_info.get('source_fingerprint'):
            raise RuntimeError('frozen CLI capabilities fingerprint differs from bundled build_info')
        if capabilities.get('methods', {}).get('cvxeda', {}).get('enabled') is not False:
            raise RuntimeError('disabled cvxEDA capability changed unexpectedly')

        current_plan = run(cli, ['plan', '--json'], 'current_applicable_feature_definitions')
        current_ids = current_plan.get('feature_definitions', {})
        if len(current_ids) != 124 or 'clean_sample_entropy' not in current_ids:
            raise RuntimeError(f'single preset feature definitions not current: count={len(current_ids)}')
        all_config = run(cli, ['presets', '--name', 'group_sync', '--json'], 'get_group_sync_preset')
        all_config['features']['extended']['enabled'] = True
        all_config['features']['ids'] = []
        all_config['synchrony']['enabled'] = True
        all_config_path = work_root / 'all-capability-config.json'
        all_config_path.write_text(json.dumps(all_config, ensure_ascii=False, indent=2), encoding='utf-8')
        all_plan = run(cli, ['plan', '--config', all_config_path, '--json'], 'all_capability_feature_definitions')
        all_ids = all_plan.get('feature_definitions', {})
        if len(all_ids) != 135:
            raise RuntimeError(f'expected 135 current all-capability definitions, observed {len(all_ids)}')

        demo_dir = work_root / 'SYNTHETIC_ONLY_demo'
        demo = run(cli, ['demo', '--directory', demo_dir, '--json'], 'create_synthetic_demo')
        preflight = run(cli, ['preflight', '--config', demo['single_config'], '--json'], 'preflight_single_demo')
        if preflight.get('eligible_to_enqueue') is not True:
            raise RuntimeError(f'synthetic demo preflight did not pass: {preflight.get("errors")}')

        project_path, task_id, queue_report = _prepare_source_queue(Path(demo['single_config']), work_root)
        queued = run(cli, ['queue-list', '--project', project_path, '--json'], 'queue_list_before_start')
        if queued.get('task_count') != 1 or queued['tasks'][0].get('task_id') != task_id or queued['tasks'][0].get('state') != 'queued':
            raise RuntimeError(f'persistent queue fixture did not remain queued: {queued}')
        queue_start = run(cli, ['queue-start', '--project', project_path, '--json'], 'queue_start_frozen_cli', timeout=480)
        if queue_start.get('status') != 'succeeded' or not queue_start.get('queue_exhausted'):
            raise RuntimeError(f'frozen queue CLI failed its single task: {queue_start}')
        queued_after = run(cli, ['queue-list', '--project', project_path, '--json'], 'queue_list_after_start')
        if queued_after.get('tasks', [{}])[0].get('state') != 'succeeded':
            raise RuntimeError(f'persistent task was not committed succeeded: {queued_after}')
        task_outcome = next((item for item in queue_start.get('tasks', []) if item.get('task_id') == task_id), None)
        if not task_outcome or task_outcome.get('status') != 'succeeded' or not task_outcome.get('directory'):
            raise RuntimeError(f'queue outcome lacks the completed run directory: {task_outcome}')
        queued_run = Path(task_outcome['directory'])
        queue_resume = run(cli, ['resume', queued_run, '--json'], 'verify_queued_run_manifest')
        if queue_resume.get('resume_action') != 'already_complete_verified' or not verify_manifest(queued_run)['valid']:
            raise RuntimeError('queued frozen run did not pass its stored manifest check')
        queue_cached_records = []
        for record_dir in sorted((queued_run / 'records').iterdir()):
            if not record_dir.is_dir():
                continue
            table_names = {path.stem for path in record_dir.glob('*.json')}
            missing = sorted(EXTENDED_RECORD_TABLES - table_names)
            if missing:
                raise RuntimeError(f'queued record cache is missing extended tables: {record_dir.name}: {missing}')
            if not verify_manifest(record_dir)['valid']:
                raise RuntimeError(f'queued record cache manifest invalid: {record_dir}')
            queue_cached_records.append({'path': str(record_dir), 'missing_extended_tables': missing})

        # Create a source-generated partial run, then make the bundled executable resume
        # it. This proves the release process can load every committed extension cache.
        from eda_toolkit.config import resolve_config
        from eda_toolkit.pipeline import run_pipeline
        cache_config = resolve_config(Path(demo['single_config']))
        cache_config['features']['ids'] = ['mean', 'clean_sample_entropy']
        cache_config['windows'].update({'whole_record': False, 'fixed_length_s': 60.0,
                                        'fixed_step_s': 60.0, 'tail_policy': 'drop'})
        cache_config['output']['directory'] = str((work_root / 'cache-resume-results').resolve())
        cache_config['output']['formats'] = ['json']
        stop = {'value': False}

        def cancel_after_first_record(update):
            if update.get('stage') == 'record_done' and int(update.get('completed', 0)) >= 1:
                stop['value'] = True

        partial = run_pipeline(cache_config, progress_callback=cancel_after_first_record,
                               cancel_callback=lambda: stop['value'])
        if partial.get('status') != 'cancelled' or partial.get('completed_records') != 1:
            raise RuntimeError(f'cache fixture did not cancel after one committed record: {partial}')
        resumed = run(cli, ['resume', partial['directory'], '--json'], 'frozen_cli_resume_committed_record', timeout=480)
        if resumed.get('status') != 'succeeded' or resumed.get('completed_records') != 4:
            raise RuntimeError(f'frozen CLI could not resume remaining committed work: {resumed}')
        resumed_dir = Path(resumed['directory'])
        if not verify_manifest(resumed_dir)['valid']:
            raise RuntimeError('frozen resume output manifest is invalid')
        resumed_caches = []
        for record_dir in sorted((resumed_dir / 'records').iterdir()):
            if not record_dir.is_dir():
                continue
            cache_manifest = verify_manifest(record_dir)
            table_names = {path.stem for path in record_dir.glob('*.json')}
            missing = sorted(EXTENDED_RECORD_TABLES - table_names)
            record_summary = json.loads((record_dir / 'record_summary.json').read_text(encoding='utf-8'))
            if not cache_manifest['valid'] or missing:
                raise RuntimeError(f'frozen resume cache invalid for {record_dir}: {cache_manifest}, missing={missing}')
            resumed_caches.append({'path': str(record_dir), 'reused_from': record_summary.get('reused_from'),
                                   'extended_tables_present': sorted(EXTENDED_RECORD_TABLES)})
        if not any(item.get('reused_from') == partial['directory'] for item in resumed_caches):
            raise RuntimeError('frozen resume did not reuse the pre-cancel committed record cache')

        # One selected single-advanced run proves real, non-empty advanced result export.
        advanced_config = resolve_config(demo['single_config'])
        signal_file = Path(demo['directory']) / 'synthetic_001.csv'
        advanced_config['input'].update({'manifest': None, 'signals': [str(signal_file.resolve())],
                                         'participants': None, 'events': None, 'conditions': None,
                                         'group_membership': None, 'sync_anchors': None,
                                         'manual_qc': None, 'statistical_design': None,
                                         'base_directory': str(signal_file.parent.resolve())})
        advanced_config['features']['extended']['enabled'] = True
        advanced_config['features']['extended']['spectral'].update({
            'segment_s': 32.0, 'overlap': 0.5, 'min_segments': 2, 'min_cycles': 2.0,
            'bands': [{'id': 'grid', 'low_hz': 0.125, 'high_hz': 0.375}],
            'normalization_band_hz': [0.125, 0.5],
        })
        advanced_config['features']['ids'] = ['mean', 'clean_sample_entropy',
                                               'clean_bandpower_grid_uS2',
                                               'clean_bandpower_grid_normalized']
        advanced_config['windows'].update({'whole_record': False, 'fixed_length_s': 60.0,
                                           'fixed_step_s': 60.0, 'tail_policy': 'drop'})
        advanced_config['output']['directory'] = str((work_root / 'single-advanced-results').resolve())
        advanced_config['output']['formats'] = ['json']
        advanced_config_path = work_root / 'single-advanced-config.json'
        advanced_config_path.write_text(json.dumps(advanced_config, ensure_ascii=False, indent=2), encoding='utf-8')
        advanced_preflight = run(cli, ['preflight', '--config', advanced_config_path, '--json'],
                                 'preflight_single_advanced')
        if advanced_preflight.get('eligible_to_enqueue') is not True:
            raise RuntimeError(f'advanced synthetic input preflight blocked: {advanced_preflight.get("errors")}')
        advanced = run(cli, ['run', '--config', advanced_config_path, '--json'],
                       'run_single_advanced_nonempty_output', timeout=240)
        if advanced.get('status') != 'succeeded':
            raise RuntimeError(f'single advanced run did not succeed: {advanced}')
        advanced_dir = Path(advanced['directory'])
        if not verify_manifest(advanced_dir)['valid']:
            raise RuntimeError('single advanced manifest is invalid')
        spectra_path = next((advanced_dir / 'tables').glob('spectra_*.json'))
        diagnostics_path = next((advanced_dir / 'tables').glob('complexity_diagnostics_*.json'))
        spectra_rows = json.loads(spectra_path.read_text(encoding='utf-8'))
        complexity_rows = json.loads(diagnostics_path.read_text(encoding='utf-8'))
        if not isinstance(spectra_rows, list) or not spectra_rows or not isinstance(complexity_rows, list) or not complexity_rows:
            raise RuntimeError('single advanced run lacks non-empty spectral or complexity tables')
        spectra_valid = sum(row.get('status') == 'valid' for row in spectra_rows)
        complexity_valid = sum(row.get('status') == 'valid' for row in complexity_rows)
        if spectra_valid < 1 or complexity_valid < 1:
            raise RuntimeError('single advanced tables have no valid PSD/complexity diagnostics rows')

        screenshot = work_root / 'frozen-gui-startup.png'
        run(gui, ['--smoke', '--config', demo['single_config'], '--screenshot', screenshot],
            'frozen_gui_window_smoke', timeout=120, parse_json=False)
        if not screenshot.is_file() or screenshot.stat().st_size < 10_000:
            raise RuntimeError('frozen GUI smoke did not save a non-empty window screenshot')

        invalid_config = work_root / 'invalid-config.json'
        invalid_config.write_text('{"compute":{"workers":true}}', encoding='utf-8')
        run(cli, ['plan', '--config', invalid_config, '--json'], 'invalid_configuration_exit_2', expected_code=2)
        disabled_config = work_root / 'disabled-method.json'
        disabled_config.write_text('{"decomposition":{"method":"cvxeda"}}', encoding='utf-8')
        run(cli, ['plan', '--config', disabled_config, '--json'], 'disabled_method_exit_3', expected_code=3)
        short_config = _write_short_failure_fixture(work_root)
        run(cli, ['run', '--config', short_config, '--json'], 'short_record_partial_exit_5',
            expected_code=5, timeout=90)
        budget_config = json.loads(Path(demo['single_config']).read_text(encoding='utf-8'))
        budget_config['compute']['memory_limit_mb'] = 1
        budget_config['output']['directory'] = str((work_root / 'resource-budget-results').resolve())
        budget_config['output']['formats'] = ['json']
        budget_path = work_root / 'resource-budget-config.json'
        budget_path.write_text(json.dumps(budget_config, ensure_ascii=False, indent=2), encoding='utf-8')
        run(cli, ['run', '--config', budget_path, '--json'], 'resource_budget_failure_exit_4',
            expected_code=4, timeout=120)

        identity_probe = _source_identity_probe()
        status = 'passed'
    except Exception as caught:
        status, error = 'failed', f'{type(caught).__name__}: {caught}'
        identity_probe = None

    executable_report = {
        name: {'sha256': sha256_file(bundle / name), 'size': (bundle / name).stat().st_size}
        for name in ('eda_cli.exe', 'eda_gui.exe')
    }
    report = {
        'status': status, 'error': error, 'version': VERSION,
        'directory': str(work_root), 'bundle': str(bundle),
        'source_fingerprint_scope': SOURCE_FINGERPRINT_SCOPE,
        'source_fingerprint': build_info.get('source_fingerprint'),
        'frozen_resolved_resources': resolved_resource_paths if 'resolved_resource_paths' in locals() else None,
        'executables': executable_report, 'native_resources': executable_resources,
        'identity': {'app_user_model_id': EXPECTED_APP_USER_MODEL_ID,
                     'source_Qt_process_probe': identity_probe,
                     'frozen_gui_taskbar_visual_review': 'not_claimed'},
        'counts': {'current_applicable_feature_ids': len(current_ids) if 'current_ids' in locals() else None,
                   'all_capability_feature_ids': len(all_ids) if 'all_ids' in locals() else None,
                   'queue_source_preflight_eligible': bool(queue_report.get('eligible_to_enqueue')) if 'queue_report' in locals() else None,
                   'queue_cached_record_count': len(queue_cached_records) if 'queue_cached_records' in locals() else None,
                   'resume_cached_record_count': len(resumed_caches) if 'resumed_caches' in locals() else None,
                   'spectra_rows': len(spectra_rows) if 'spectra_rows' in locals() else None,
                   'spectra_valid_rows': spectra_valid if 'spectra_valid' in locals() else None,
                   'complexity_diagnostics_rows': len(complexity_rows) if 'complexity_rows' in locals() else None,
                   'complexity_valid_rows': complexity_valid if 'complexity_valid' in locals() else None},
        'queue_cached_records': queue_cached_records if 'queue_cached_records' in locals() else [],
        'resumed_caches': resumed_caches if 'resumed_caches' in locals() else [],
        'advanced_tables': ({
            'spectra': {'path': str(spectra_path), 'sha256': sha256_file(spectra_path), 'rows': len(spectra_rows)},
            'complexity_diagnostics': {'path': str(diagnostics_path), 'sha256': sha256_file(diagnostics_path), 'rows': len(complexity_rows)},
            'manifest': {'path': str(advanced_dir / 'manifest.json'), 'sha256': sha256_file(advanced_dir / 'manifest.json'),
                         'valid': verify_manifest(advanced_dir)['valid']},
        } if 'spectra_path' in locals() and 'diagnostics_path' in locals() else None),
        'cases': records,
        'scope': 'Windows development host with restricted external PATH; synthetic engineering fixtures only',
        'qualification': {'clean_windows': 'not_tested', 'real_device_science': 'pending_representative_real_data',
                          'license_clearance': 'pending', 'desktop_taskbar_visual_review': 'not_tested'},
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    print(json.dumps({'status': status, 'report': str(report_path), 'error': error,
                      'source_fingerprint': report['source_fingerprint']}, ensure_ascii=False))
    return 0 if status == 'passed' else 1


if __name__ == '__main__':
    raise SystemExit(main())
