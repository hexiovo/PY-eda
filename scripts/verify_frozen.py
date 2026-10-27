"""Legacy v0.1 verifier kept disabled to protect its recorded evidence."""
raise SystemExit('The v0.1 verification report is immutable. Use scripts/verify_frozen_v02.py for new v0.2 evidence.')

import argparse
from datetime import datetime
import json
import os
from pathlib import Path
import subprocess
import sys
import uuid

PROJECT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--probe', type=Path)
    parser.add_argument('--bundle', type=Path)
    args = parser.parse_args()
    root = PROJECT / 'artifacts' / 'verification' / ('冻结 中文路径 ' + uuid.uuid4().hex[:8])
    root.mkdir(parents=True)
    environment = {key: value for key, value in os.environ.items()
                   if not key.upper().startswith(('PYTHON', 'CONDA', 'VIRTUAL_ENV'))}
    environment['PATH'] = str(Path(os.environ.get('SystemRoot', 'C:/Windows')) / 'System32')
    environment['QT_QPA_PLATFORM'] = 'offscreen'
    records = []
    def run(executable, arguments, label, timeout=120, expected_code=0):
        result = subprocess.run([str(executable), *map(str, arguments)], cwd=root, env=environment,
                                capture_output=True, text=True, encoding='utf-8', errors='replace', timeout=timeout,
                                creationflags=0x08000000 if os.name == 'nt' else 0)
        record = {'label': label, 'returncode': result.returncode, 'expected_returncode': expected_code, 'stdout': result.stdout, 'stderr': result.stderr}
        records.append(record)
        if result.returncode != expected_code:
            raise RuntimeError(label + ' failed: ' + result.stderr[-2000:] + result.stdout[-2000:])
        return json.loads(result.stdout) if result.stdout.strip().startswith('{') else result.stdout
    status, error = 'passed', None
    try:
        if args.probe:
            run(args.probe.resolve(), [], 'native_runtime_probe')
        else:
            bundle = args.bundle.resolve()
            cli, gui = bundle / 'eda_cli.exe', bundle / 'eda_gui.exe'
            run(cli, ['version', '--json'], 'version')
            run(cli, ['capabilities', '--json'], 'capabilities')
            run(cli, ['templates', '--json'], 'bundled_templates')
            demo = run(cli, ['demo', '--directory', root / '合成 输入', '--json'], 'create_synthetic_demo')
            config = demo['single_config']
            run(cli, ['validate', '--config', config, '--json'], 'validate_synthetic')
            summary = run(cli, ['run', '--config', config, '--output', root / '分析 输出', '--json'], 'run_single_pipeline', timeout=240)
            if summary['status'] != 'succeeded' or summary['completed_records'] != 4:
                raise RuntimeError('冻结分析未完整成功。')
            run(cli, ['resume', summary['directory'], '--json'], 'verify_complete_resume')
            group_path = Path(demo['group_config'])
            group_config = json.loads(group_path.read_text(encoding='utf-8'))
            group_config['synchrony'].update(null_replicates=5, max_attempts=100)
            group_config['output']['formats'] = ['csv','json','html','png']
            group_path.write_text(json.dumps(group_config,ensure_ascii=False,indent=2),encoding='utf-8')
            run(cli, ['run', '--config', group_path, '--output', root / '同步 输出', '--json'], 'run_group_pipeline', timeout=240)
            if demo.get('statistics_config'):
                run(cli, ['run', '--config', demo['statistics_config'], '--output', root / '统计 输出', '--json'], 'run_actual_mixed_model', timeout=180)
            invalid = root / '非法配置.json'
            invalid.write_text('{"compute":{"workers":true}}',encoding='utf-8')
            run(cli,['plan','--config',invalid,'--json'],'invalid_config_exit_2',expected_code=2)
            disabled = root / '禁用方法.json'
            disabled.write_text('{"decomposition":{"method":"cvxeda"}}',encoding='utf-8')
            run(cli,['plan','--config',disabled,'--json'],'disabled_capability_exit_3',expected_code=3)
            short = root / '短记录.csv'
            short.write_text('time_s,eda_uS\n'+'\n'.join(f'{i/4},{2+.01*(i%3)}' for i in range(20))+'\n',encoding='utf-8')
            short_config = root / '短记录配置.json'
            short_config.write_text(json.dumps({'input':{'signals':[str(short)]},'output':{'directory':str(root/'部分结果'),'formats':['json']}}),encoding='utf-8')
            run(cli,['run','--config',short_config,'--json'],'no_detector_support_partial_exit_5',expected_code=5)
            budget_config = json.loads(Path(config).read_text(encoding='utf-8'))
            budget_config['compute']['memory_limit_mb'] = 1
            budget_config['output'].update(directory=str(root/'资源拒绝'),formats=['json'])
            budget = root/'资源上限.json'
            budget.write_text(json.dumps(budget_config,ensure_ascii=False),encoding='utf-8')
            run(cli,['run','--config',budget,'--json'],'resource_limit_runtime_exit_4',expected_code=4)
            screenshot = root / '真实 冻结界面.png'
            run(gui, ['--smoke', '--config', config, '--screenshot', screenshot], 'gui_smoke', timeout=90)
            if not screenshot.is_file():
                raise RuntimeError('冻结GUI未生成实际窗口截图。')
    except Exception as caught:
        status, error = 'failed', str(caught)
    report = {'status': status, 'error': error, 'cases': records, 'directory': str(root),
              'scope': 'development_host_restricted_PATH_no_external_Python_paths_not_clean_machine',
              'clean_windows_qualification': 'not_tested'}
    output = PROJECT / 'artifacts' / 'verification' / ('frozen_runtime_probe.json' if args.probe else 'frozen_application.json')
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps({'status': status, 'report': str(output), 'error': error}, ensure_ascii=False))
    return 0 if status == 'passed' else 1


if __name__ == '__main__':
    raise SystemExit(main())
