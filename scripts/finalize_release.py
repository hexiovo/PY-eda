"""Legacy v0.1 finalizer kept disabled to protect the released artifact."""
raise SystemExit('The v0.1 release is frozen and this legacy finalizer is disabled. Use scripts/finalize_release_v02.py for isolated v0.2 outputs.')

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import sys
import zipfile

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(PROJECT/'src'))
from eda_toolkit.core_export import atomic_json, sha256_file


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--archive',action='store_true')
    args = parser.parse_args()
    bundle = PROJECT/'dist'/'EDA_Toolkit'
    evidence = PROJECT/'artifacts'/'verification'
    report_path = evidence/'frozen_application.json'
    report = json.loads(report_path.read_text(encoding='utf-8'))
    if report['status']!='passed' or any(case['returncode']!=case.get('expected_returncode',0) for case in report['cases']):
        raise RuntimeError('当前进程验证没有全部按预期完成，不能标记候选通过。')
    executables = {name:{'sha256':sha256_file(bundle/name),'size':(bundle/name).stat().st_size}
                   for name in ('eda_cli.exe','eda_gui.exe')}
    if report.get('executables') and report['executables']!=executables:
        raise RuntimeError('双EXE与已绑定的验证快照发生变化，必须重新实测。')
    if not report.get('executables'):
        shutil.copy2(report_path,evidence/'frozen_application_process_results.json')
    report['executables'] = executables
    report['binding_note'] = 'Executable hashes observed immediately after successful process tests; application binaries unchanged between build and verification.'
    atomic_json(report_path,report)
    validation = bundle/'validation'
    validation.mkdir(exist_ok=True)
    for name in ('frozen_application.json','tests-release.xml','help-pdf-final.json','independent_review.json','freeze_runtime_diagnosis.json'):
        if (evidence/name).is_file():
            shutil.copy2(evidence/name,validation/name)
    if (evidence/'surrogate_diagnostic'/'diagnostic.json').is_file():
        shutil.copy2(evidence/'surrogate_diagnostic'/'diagnostic.json',validation/'exploratory_surrogate_diagnostic.json')
    shutil.copy2(PROJECT/'README.md',bundle/'README.md')
    shutil.copytree(PROJECT/'docs',bundle/'docs',dirs_exist_ok=True)
    shutil.copy2(PROJECT/'src'/'eda_toolkit'/'resources'/'help'/'EDA使用指南.pdf',bundle/'docs'/'EDA使用指南.pdf')
    sources = {path.relative_to(PROJECT/'src').as_posix():sha256_file(path) for path in sorted((PROJECT/'src'/'eda_toolkit').rglob('*.py'))}
    source_tree_sha = hashlib.sha256(json.dumps(sources,sort_keys=True).encode()).hexdigest()
    manifest = {'version':'0.1.0','classification':'Windows_x64_portable_engineering_candidate',
                'qualification':{'development_machine_frozen':'passed_restricted_PATH_14_process_cases',
                                 'clean_windows':'not_tested','real_device_science':'not_tested',
                                 'public_distribution_license_clearance':'pending'},
                'source_tree_sha256':source_tree_sha,'source_files':sources,'executables':executables,
                'verification_report_sha256':sha256_file(report_path),
                'files':[{'path':p.relative_to(bundle).as_posix(),'size':p.stat().st_size,'sha256':sha256_file(p)}
                         for p in sorted(bundle.rglob('*')) if p.is_file() and p.name!='release_manifest.json']}
    atomic_json(bundle/'release_manifest.json',manifest)
    output = {'bundle':str(bundle),'file_count':len(manifest['files']),'executables':executables,
              'manifest_sha256':sha256_file(bundle/'release_manifest.json'),'verification_sha256':sha256_file(report_path)}
    if args.archive:
        archive = PROJECT/'dist'/'EDA_Toolkit-0.1.0-Windows-x64.zip'
        with zipfile.ZipFile(archive,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as target:
            for path in sorted(bundle.rglob('*')):
                if path.is_file():
                    target.write(path,Path('EDA_Toolkit')/path.relative_to(bundle))
        output.update({'archive':str(archive),'archive_sha256':sha256_file(archive),'archive_size':archive.stat().st_size})
    atomic_json(evidence/'release_summary.json',output)
    print(json.dumps(output,ensure_ascii=False))


if __name__=='__main__':
    main()
