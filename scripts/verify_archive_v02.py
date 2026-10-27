"""Independently verify the frozen v0.2 bundle and ZIP against their manifest."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
import xml.etree.ElementTree as ET
import zipfile

PROJECT = Path(__file__).resolve().parents[1]
EVIDENCE_ROOT = (PROJECT / 'artifacts' / 'verification' / 'v0.2').resolve()
BUNDLE = (PROJECT / 'dist' / 'v0.2' / 'EDA_Toolkit-v0.2.0').resolve()
ARCHIVE = (PROJECT / 'dist' / 'EDA_Toolkit-0.2.0-Windows-x64.zip').resolve()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def _sha256_zip_member(archive: zipfile.ZipFile, name: str) -> tuple[int, str]:
    digest = hashlib.sha256()
    size = 0
    with archive.open(name, 'r') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
            size += len(block)
    return size, digest.hexdigest()


def _test_counts(path: Path) -> dict:
    root = ET.parse(path).getroot()
    cases = list(root.iter('testcase'))
    if root.tag == 'testsuites':
        failures = sum(int(row.attrib.get('failures', '0')) for row in root.iter('testsuite'))
        errors = sum(int(row.attrib.get('errors', '0')) for row in root.iter('testsuite'))
        skipped = sum(int(row.attrib.get('skipped', '0')) for row in root.iter('testsuite'))
    elif root.tag == 'testsuite':
        failures = int(root.attrib.get('failures', '0'))
        errors = int(root.attrib.get('errors', '0'))
        skipped = int(root.attrib.get('skipped', '0'))
    else:
        raise ValueError(f'unrecognized pytest XML root: {root.tag!r}')
    return {'tests': len(cases), 'failures': failures, 'errors': errors, 'skipped': skipped}


def verify(report_path: Path) -> dict:
    if not report_path.resolve().is_relative_to(EVIDENCE_ROOT) or report_path.exists():
        raise RuntimeError('archive integrity report must be a fresh file under v0.2 verification evidence')
    summary_path = EVIDENCE_ROOT / 'release_summary.json'
    frozen_path = EVIDENCE_ROOT / 'frozen-application-r2.json'
    manifest_path = BUNDLE / 'release_manifest.json'
    summary = json.loads(summary_path.read_text(encoding='utf-8'))
    frozen = json.loads(frozen_path.read_text(encoding='utf-8'))
    manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    archive_sha = _sha256_file(ARCHIVE)
    actual_archive_size = ARCHIVE.stat().st_size
    if archive_sha != summary['archive_sha256'] or actual_archive_size != summary['archive_size']:
        raise RuntimeError('archive file size or hash differs from the finalizer summary')

    entries = manifest['files']
    if len(entries) != summary['bundle_file_count']:
        raise RuntimeError('release manifest entry count differs from the release summary')
    expected_disk = {row['path'] for row in entries}
    actual_disk = {path.relative_to(BUNDLE).as_posix() for path in BUNDLE.rglob('*') if path.is_file()
                   and path.name != 'release_manifest.json'}
    if expected_disk != actual_disk:
        raise RuntimeError('bundle disk files do not exactly match release_manifest.files')

    prefix = BUNDLE.name
    expected_members = {f'{prefix}/{row["path"]}' for row in entries}
    expected_members.add(f'{prefix}/release_manifest.json')
    verified_members = []
    with zipfile.ZipFile(ARCHIVE, 'r') as archive:
        bad_member = archive.testzip()
        if bad_member is not None:
            raise RuntimeError(f'ZIP CRC check failed at {bad_member}')
        names = set(archive.namelist())
        if names != expected_members:
            raise RuntimeError(f'ZIP member inventory mismatch: missing={len(expected_members - names)}, extra={len(names - expected_members)}')
        for row in entries:
            disk_path = BUNDLE / Path(row['path'])
            disk_size, disk_hash = disk_path.stat().st_size, _sha256_file(disk_path)
            if disk_size != row['size'] or disk_hash != row['sha256']:
                raise RuntimeError(f'bundle file differs from release manifest: {row["path"]}')
            member_name = f'{prefix}/{row["path"]}'
            member_size, member_hash = _sha256_zip_member(archive, member_name)
            if member_size != row['size'] or member_hash != row['sha256']:
                raise RuntimeError(f'ZIP member differs from release manifest: {row["path"]}')
            verified_members.append(row['path'])
        manifest_size, manifest_hash = _sha256_zip_member(archive, f'{prefix}/release_manifest.json')
        if manifest_size != manifest_path.stat().st_size or manifest_hash != _sha256_file(manifest_path):
            raise RuntimeError('ZIP release_manifest.json differs from the bundle manifest')

    executable_checks = {}
    for name in ('eda_cli.exe', 'eda_gui.exe'):
        path = BUNDLE / name
        digest = _sha256_file(path)
        if manifest['executables'][name]['sha256'] != digest or frozen['executables'][name]['sha256'] != digest:
            raise RuntimeError(f'{name} is not bound consistently across disk, manifest, and frozen report')
        executable_checks[name] = {'sha256': digest, 'size': path.stat().st_size}

    source_fingerprint = manifest['source_fingerprint']
    if (summary['source_fingerprint'] != source_fingerprint or
            frozen['source_fingerprint'] != source_fingerprint):
        raise RuntimeError('manifest, release summary, and frozen report source fingerprints differ')
    test_xml = EVIDENCE_ROOT / 'pytest-results-final.xml'
    test_counts = _test_counts(test_xml)
    if test_counts != summary['test_counts'] or test_counts != {'tests': 166, 'failures': 0, 'errors': 0, 'skipped': 0}:
        raise RuntimeError(f'final full-suite evidence is not fully green or differs from summary: {test_counts}')
    help_evidence = json.loads((EVIDENCE_ROOT / 'help-pdf-final.json').read_text(encoding='utf-8'))
    bundled_pdf = BUNDLE / 'docs' / 'EDA使用指南.pdf'
    if (_sha256_file(bundled_pdf) != help_evidence['sha256'] or
            manifest['help_pdf']['sha256'] != help_evidence['sha256'] or
            manifest['help_pdf']['page_count'] != help_evidence['page_count'] or
            help_evidence['status'] != 'passed'):
        raise RuntimeError('bundled help PDF is not bound to the passed final visual QA evidence')

    result = {
        'status': 'passed', 'version': manifest['version'], 'classification': manifest['classification'],
        'bundle': str(BUNDLE), 'bundle_file_count': len(entries),
        'release_manifest': str(manifest_path), 'release_manifest_sha256': _sha256_file(manifest_path),
        'archive': str(ARCHIVE), 'archive_sha256': archive_sha, 'archive_size': actual_archive_size,
        'zip_crc_valid': True, 'zip_members_verified': len(expected_members),
        'bundle_files_verified': len(verified_members), 'executables': executable_checks,
        'source_fingerprint_scope': manifest['source_fingerprint_scope'],
        'source_fingerprint': source_fingerprint, 'tests': test_counts,
        'help_pdf_sha256': help_evidence['sha256'], 'help_pdf_pages_reviewed': help_evidence['reviewed_pages'],
        'qualification': manifest['qualification'],
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument('--report', type=Path, default=EVIDENCE_ROOT / 'archive_integrity.json')
    args = parser.parse_args()
    result = verify(args.report.resolve())
    print(json.dumps(result, ensure_ascii=False))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
