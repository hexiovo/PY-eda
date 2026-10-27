"""Finalize only the version-isolated v0.2.0 bundle after real process checks."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import sys
import tomllib
import xml.etree.ElementTree as ET
import zipfile

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / 'src'))
from eda_toolkit import __version__
from eda_toolkit.capabilities import SOURCE_FINGERPRINT_SCOPE, source_fingerprint
from eda_toolkit.core_export import atomic_json, sha256_file, verify_manifest


RELEASE_VERSION = '0.2.0'
VERSION_DIST_ROOT = (PROJECT / 'dist' / 'v0.2').resolve()
EVIDENCE_ROOT = (PROJECT / 'artifacts' / 'verification' / 'v0.2').resolve()


def _read_json(path: Path):
    return json.loads(path.read_text(encoding='utf-8-sig'))


def _is_within(path: Path, root: Path) -> bool:
    return path.resolve().is_relative_to(root.resolve())


def _verify_test_xml(path: Path) -> dict:
    root = ET.parse(path).getroot()
    if root.tag == 'testsuites':
        suites = list(root.iter('testsuite'))
        cases = list(root.iter('testcase'))
        failures = sum(int(suite.attrib.get('failures', '0')) for suite in suites)
        errors = sum(int(suite.attrib.get('errors', '0')) for suite in suites)
        skipped = sum(int(suite.attrib.get('skipped', '0')) for suite in suites)
    elif root.tag == 'testsuite':
        cases = list(root.iter('testcase'))
        failures = int(root.attrib.get('failures', '0'))
        errors = int(root.attrib.get('errors', '0'))
        skipped = int(root.attrib.get('skipped', '0'))
    else:
        raise ValueError(f'unrecognized pytest XML root: {root.tag!r}')
    tests = len(cases)
    if tests < 1 or failures or errors or skipped:
        raise RuntimeError(f'pytest XML is empty or not fully green: tests={tests}, failures={failures}, errors={errors}, skipped={skipped}')
    return {'tests': tests, 'failures': failures, 'errors': errors, 'skipped': skipped}


def _hash_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _zip_member_name(prefix: str, relative: str) -> str:
    """ZIP uses forward slashes even when release generation runs on Windows."""
    return f'{prefix}/{Path(relative).as_posix()}'


def _copy_current_docs(bundle: Path, pdf_evidence: dict) -> dict:
    pdf_source = (PROJECT / 'src' / 'eda_toolkit' / 'resources' / 'help' / 'EDA使用指南.pdf').resolve()
    reported_pdf = Path(pdf_evidence.get('pdf', '')).resolve()
    if reported_pdf != pdf_source or not pdf_source.is_file():
        raise RuntimeError('final help PDF evidence must refer to the current packaged resource PDF')
    if pdf_evidence.get('status') != 'passed':
        raise RuntimeError('final help PDF visual QA is not marked passed')
    page_count = pdf_evidence.get('page_count')
    if isinstance(page_count, bool) or not isinstance(page_count, int) or page_count < 1:
        raise RuntimeError('final help PDF evidence has no valid page count')
    expected_pages = list(range(1, page_count + 1))
    if pdf_evidence.get('rendered_pages') != expected_pages or pdf_evidence.get('reviewed_pages') != expected_pages:
        raise RuntimeError('final help PDF evidence does not show render and review of every page in order')
    visual = pdf_evidence.get('visual_qa', {})
    if (visual.get('all_pages_contact_sheet_reviewed') is not True or
            visual.get('visible_clipping_or_overlap') is not False or
            visual.get('visible_missing_glyphs') is not False):
        raise RuntimeError('final help PDF visual QA flags are incomplete or report visible defects')
    renders = pdf_evidence.get('page_renders', [])
    if len(renders) != page_count or [item.get('page') for item in renders] != expected_pages:
        raise RuntimeError('final help PDF evidence is missing one or more rendered pages')
    for item in renders:
        render_path = Path(item.get('path', '')).resolve()
        if (not _is_within(render_path, EVIDENCE_ROOT) or not render_path.is_file() or
                sha256_file(render_path) != item.get('sha256')):
            raise RuntimeError(f'help PDF page render evidence is missing or changed: {item.get("page")}')
    contact_sheet = pdf_evidence.get('contact_sheet', {})
    contact_path = Path(contact_sheet.get('path', '')).resolve()
    if (not _is_within(contact_path, EVIDENCE_ROOT) or not contact_path.is_file() or
            sha256_file(contact_path) != contact_sheet.get('sha256')):
        raise RuntimeError('final help PDF contact sheet evidence is missing or changed')
    spot_checks = {item.get('page') for item in pdf_evidence.get('spot_checks', [])}
    if not {1, 4, 5, 13, 14} <= spot_checks:
        raise RuntimeError('final help PDF evidence lacks the required cover, method, live screenshot, and help-page checks')
    expected_hash = pdf_evidence.get('sha256')
    if not isinstance(expected_hash, str) or sha256_file(pdf_source) != expected_hash:
        raise RuntimeError('current help PDF does not match the final visual-QA evidence hash')
    if pdf_evidence.get('size') != pdf_source.stat().st_size:
        raise RuntimeError('current help PDF size does not match final visual-QA evidence')
    if not (PROJECT / 'README.md').is_file() or not (PROJECT / 'docs').is_dir():
        raise FileNotFoundError('release README/docs are missing')
    shutil.copy2(PROJECT / 'README.md', bundle / 'README.md')
    shutil.copytree(PROJECT / 'docs', bundle / 'docs', dirs_exist_ok=True)
    shutil.copy2(pdf_source, bundle / 'docs' / 'EDA使用指南.pdf')
    return {'path': 'docs/EDA使用指南.pdf', 'sha256': sha256_file(bundle / 'docs' / 'EDA使用指南.pdf'),
            'size': (bundle / 'docs' / 'EDA使用指南.pdf').stat().st_size,
            'visual_qa_status': pdf_evidence['status'], 'page_count': page_count,
            'reviewed_pages': expected_pages, 'contact_sheet_sha256': contact_sheet['sha256'],
            'visual_qa_evidence_sha256': _hash_bytes(json.dumps(pdf_evidence, ensure_ascii=False,
                                                                 sort_keys=True).encode('utf-8'))}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument('--bundle', type=Path,
                        default=PROJECT / 'dist' / 'v0.2' / 'EDA_Toolkit-v0.2.0')
    parser.add_argument('--archive', type=Path,
                        default=PROJECT / 'dist' / 'EDA_Toolkit-0.2.0-Windows-x64.zip')
    parser.add_argument('--verification-report', type=Path,
                        default=EVIDENCE_ROOT / 'frozen_application.json')
    parser.add_argument('--test-xml', type=Path, default=EVIDENCE_ROOT / 'pytest-results-final.xml')
    parser.add_argument('--test-log', type=Path, default=EVIDENCE_ROOT / 'pytest-output-final.log')
    parser.add_argument('--test-exit-code', type=Path, default=EVIDENCE_ROOT / 'pytest-exit-code-final.txt')
    parser.add_argument('--help-pdf-evidence', type=Path, default=EVIDENCE_ROOT / 'help-pdf-final.json')
    parser.add_argument('--summary', type=Path, default=EVIDENCE_ROOT / 'release_summary.json')
    args = parser.parse_args()

    bundle = args.bundle.resolve()
    archive = args.archive.resolve()
    report_path = args.verification_report.resolve()
    test_xml = args.test_xml.resolve()
    test_log = args.test_log.resolve()
    test_exit_code = args.test_exit_code.resolve()
    help_evidence_path = args.help_pdf_evidence.resolve()
    summary_path = args.summary.resolve()
    if not _is_within(bundle, VERSION_DIST_ROOT) or bundle.name != f'EDA_Toolkit-v{RELEASE_VERSION}':
        raise RuntimeError('bundle is not the isolated v0.2.0 directory')
    if not _is_within(archive, (PROJECT / 'dist').resolve()) or archive.name != 'EDA_Toolkit-0.2.0-Windows-x64.zip':
        raise RuntimeError('archive must use the versioned v0.2.0 distribution path')
    if (not _is_within(report_path, EVIDENCE_ROOT) or not _is_within(test_xml, EVIDENCE_ROOT) or
            not _is_within(test_log, EVIDENCE_ROOT) or not _is_within(test_exit_code, EVIDENCE_ROOT) or
            not _is_within(help_evidence_path, EVIDENCE_ROOT) or not _is_within(summary_path, EVIDENCE_ROOT)):
        raise RuntimeError('final evidence inputs must be stored under artifacts/verification/v0.2')
    for path in (report_path, test_xml, test_log, test_exit_code, help_evidence_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    if test_exit_code.read_text(encoding='ascii').strip() != '0':
        raise RuntimeError('the recorded full pytest process exit code is not zero')
    if (bundle / 'release_manifest.json').exists() or archive.exists() or summary_path.exists():
        raise FileExistsError('refusing to overwrite an existing v0.2 manifest, ZIP, or summary')

    report = _read_json(report_path)
    if report.get('status') != 'passed' or report.get('version') != RELEASE_VERSION:
        raise RuntimeError('frozen native process report is not a passed v0.2.0 report')
    if not report.get('cases') or any(case.get('returncode') != case.get('expected_returncode', 0)
                                       for case in report['cases']):
        raise RuntimeError('one or more frozen application cases returned an unexpected exit code')
    if not bundle.is_dir():
        raise FileNotFoundError(bundle)
    if sys.platform != 'win32':
        raise RuntimeError('the v0.2 native finalizer must run on Windows')
    if __version__ != RELEASE_VERSION:
        raise RuntimeError(f'installed source version mismatch: {__version__}')
    with (PROJECT / 'pyproject.toml').open('rb') as stream:
        pyproject_version = tomllib.load(stream)['project']['version']
    if pyproject_version != RELEASE_VERSION:
        raise RuntimeError(f'pyproject version mismatch: {pyproject_version}')

    build_info_candidates = list(bundle.rglob('build_info.json'))
    if len(build_info_candidates) != 1:
        raise RuntimeError(f'expected exactly one frozen build_info.json, found {len(build_info_candidates)}')
    build_info = _read_json(build_info_candidates[0])
    source_fp = source_fingerprint()
    if (build_info.get('version') != RELEASE_VERSION or
            build_info.get('fingerprint_scope') != SOURCE_FINGERPRINT_SCOPE or
            build_info.get('source_fingerprint') != source_fp):
        raise RuntimeError('frozen source fingerprint/version no longer matches the final source tree')
    if report.get('source_fingerprint') != source_fp or report.get('source_fingerprint_scope') != SOURCE_FINGERPRINT_SCOPE:
        raise RuntimeError('native process report was not produced from the final source fingerprint')
    executable_hashes = report.get('executables', {})
    for name in ('eda_cli.exe', 'eda_gui.exe'):
        target = bundle / name
        if not target.is_file() or executable_hashes.get(name, {}).get('sha256') != sha256_file(target):
            raise RuntimeError(f'{name} differs from its actual native-process verification hash')

    test_counts = _verify_test_xml(test_xml)
    help_evidence = _read_json(help_evidence_path)
    help_artifact = _copy_current_docs(bundle, help_evidence)
    evidence_dir = bundle / 'validation'
    evidence_dir.mkdir(parents=True, exist_ok=True)
    evidence_files = {}
    for source in (report_path, test_xml, test_log, test_exit_code, help_evidence_path):
        target = evidence_dir / source.name
        shutil.copy2(source, target)
        evidence_files[target.relative_to(bundle).as_posix()] = {
            'sha256': sha256_file(target), 'size': target.stat().st_size}
    qa_render_dir = evidence_dir / 'help-pdf-rendered'
    qa_render_dir.mkdir()
    copied_page_renders = []
    for item in help_evidence['page_renders']:
        source = Path(item['path']).resolve()
        target = qa_render_dir / f'page-{item["page"]:02d}.png'
        shutil.copy2(source, target)
        digest = sha256_file(target)
        if digest != item['sha256']:
            raise RuntimeError(f'copied help PDF page render differs from QA evidence: {item["page"]}')
        copied_page_renders.append({'page': item['page'], 'path': target.relative_to(bundle).as_posix(),
                                    'sha256': digest, 'size': target.stat().st_size})
    contact_source = Path(help_evidence['contact_sheet']['path']).resolve()
    contact_target = qa_render_dir / 'contact-sheet.png'
    shutil.copy2(contact_source, contact_target)
    if sha256_file(contact_target) != help_evidence['contact_sheet']['sha256']:
        raise RuntimeError('copied PDF contact sheet differs from QA evidence')
    for target in sorted(qa_render_dir.glob('*.png')):
        evidence_files[target.relative_to(bundle).as_posix()] = {
            'sha256': sha256_file(target), 'size': target.stat().st_size}
    render_index = evidence_dir / 'help-pdf-render-index.json'
    atomic_json(render_index, {
        'status': 'passed', 'source_evidence_sha256': sha256_file(help_evidence_path),
        'page_count': help_evidence['page_count'], 'pages': copied_page_renders,
        'contact_sheet': {'path': contact_target.relative_to(bundle).as_posix(),
                          'sha256': sha256_file(contact_target), 'size': contact_target.stat().st_size}})
    evidence_files[render_index.relative_to(bundle).as_posix()] = {
        'sha256': sha256_file(render_index), 'size': render_index.stat().st_size}
    native_workdir = Path(report.get('directory', '')).resolve()
    if not _is_within(native_workdir, EVIDENCE_ROOT):
        raise RuntimeError('native verification working directory is outside v0.2 evidence root')
    native_logs = native_workdir / 'process-logs'
    if not native_logs.is_dir():
        raise FileNotFoundError(f'native process logs are missing: {native_logs}')
    logs_target = evidence_dir / 'process-logs'
    shutil.copytree(native_logs, logs_target)
    for target in sorted(logs_target.rglob('*')):
        if target.is_file():
            evidence_files[target.relative_to(bundle).as_posix()] = {
                'sha256': sha256_file(target), 'size': target.stat().st_size}
    screenshot = native_workdir / 'frozen-gui-startup.png'
    if screenshot.is_file():
        screenshot_target = evidence_dir / screenshot.name
        shutil.copy2(screenshot, screenshot_target)
        evidence_files[screenshot_target.relative_to(bundle).as_posix()] = {
            'sha256': sha256_file(screenshot_target), 'size': screenshot_target.stat().st_size}
    independent_review = EVIDENCE_ROOT / 'independent_review.json'
    if independent_review.is_file():
        target = evidence_dir / independent_review.name
        shutil.copy2(independent_review, target)
        evidence_files[target.relative_to(bundle).as_posix()] = {
            'sha256': sha256_file(target), 'size': target.stat().st_size}

    manifest_files = [
        {'path': path.relative_to(bundle).as_posix(), 'size': path.stat().st_size,
         'sha256': sha256_file(path)}
        for path in sorted(bundle.rglob('*'))
        if path.is_file() and path.name != 'release_manifest.json'
    ]
    manifest = {
        'version': RELEASE_VERSION,
        'classification': 'Windows_x64_portable_engineering_candidate',
        'source_fingerprint_scope': SOURCE_FINGERPRINT_SCOPE,
        'source_fingerprint': source_fp,
        'source_files': build_info.get('source_files', {}),
        'executables': {name: {'sha256': sha256_file(bundle / name), 'size': (bundle / name).stat().st_size}
                        for name in ('eda_cli.exe', 'eda_gui.exe')},
        'identity': report.get('identity', {}),
        'help_pdf': help_artifact,
        'tests': {'xml_path': 'validation/' + test_xml.name,
                  'log_path': 'validation/' + test_log.name,
                  'exit_code_path': 'validation/' + test_exit_code.name,
                  **test_counts},
        'verification_evidence': evidence_files,
        'qualification': {
            'development_host_native_process_checks': 'passed_restricted_PATH_recorded_cases',
            'clean_windows': 'not_tested',
            'real_device_science': 'pending_representative_real_data',
            'public_distribution_license_clearance': 'pending',
        },
        'files': manifest_files,
    }
    manifest_path = bundle / 'release_manifest.json'
    atomic_json(manifest_path, manifest)
    if not verify_manifest(bundle, {'artifacts': manifest_files, 'status': 'succeeded'})['valid']:
        raise RuntimeError('bundle files changed while creating the final manifest')

    archive.parent.mkdir(parents=True, exist_ok=True)
    prefix = bundle.name
    with zipfile.ZipFile(archive, 'x', compression=zipfile.ZIP_DEFLATED, compresslevel=6) as target:
        for path in sorted(bundle.rglob('*')):
            if path.is_file():
                target.write(path, _zip_member_name(prefix, path.relative_to(bundle).as_posix()))
    with zipfile.ZipFile(archive, 'r') as source:
        bad = source.testzip()
        if bad is not None:
            raise RuntimeError(f'ZIP CRC check failed: {bad}')
        expected_members = {_zip_member_name(prefix, item['path']) for item in manifest_files}
        expected_members.add(_zip_member_name(prefix, 'release_manifest.json'))
        if set(source.namelist()) != expected_members:
            raise RuntimeError('ZIP members do not exactly match the finalized bundle')
        for item in manifest_files:
            data = source.read(_zip_member_name(prefix, item['path']))
            if len(data) != item['size'] or _hash_bytes(data) != item['sha256']:
                raise RuntimeError(f'ZIP member hash/size mismatch: {item["path"]}')
        if _hash_bytes(source.read(_zip_member_name(prefix, 'release_manifest.json'))) != sha256_file(manifest_path):
            raise RuntimeError('ZIP release manifest differs from the bundle manifest')

    summary = {
        'version': RELEASE_VERSION,
        'bundle': str(bundle),
        'bundle_file_count': len(manifest_files),
        'release_manifest': str(manifest_path),
        'release_manifest_sha256': sha256_file(manifest_path),
        'source_fingerprint_scope': SOURCE_FINGERPRINT_SCOPE,
        'source_fingerprint': source_fp,
        'archive': str(archive),
        'archive_sha256': sha256_file(archive),
        'archive_size': archive.stat().st_size,
        'zip_crc_valid': True,
        'zip_members_verified': len(expected_members),
        'test_counts': test_counts,
        'clean_windows': 'not_tested',
        'science_qualification': 'pending_representative_real_data',
        'distribution_license_clearance': 'pending',
    }
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    atomic_json(summary_path, summary)
    print(json.dumps(summary, ensure_ascii=False))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
