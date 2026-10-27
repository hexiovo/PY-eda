"""按实际CSV生成当前证据索引；历史需求报告和基线快照保持原样。"""
import csv
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

root = Path(__file__).resolve().parents[1]
baseline = root / 'docs' / 'baseline'
inventory = {}
for name in ('requirements.csv', 'method_evidence.csv', 'license_inventory.csv'):
    path = baseline / name
    with path.open(encoding='utf-8-sig', newline='') as stream:
        rows = list(csv.DictReader(stream))
    record = {'rows':len(rows), 'sha256':hashlib.sha256(path.read_bytes()).hexdigest()}
    if name == 'requirements.csv':
        record['priority_by_implementation_status'] = dict(Counter(f"{row['priority']} / {row['implementation_status']}" for row in rows))
        record['mandatory_without_evidence_path'] = [row['requirement_id'] for row in rows if row['priority']=='mandatory' and not row['evidence_path'].strip()]
    inventory[name] = record
output = {'updated_at_utc':datetime.now(timezone.utc).isoformat(), 'scope':'actual current living registers; separate from initial P00 historical snapshot',
          'files':inventory}
(baseline/'inventory_snapshot.json').write_text(json.dumps(output,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps({name:item['rows'] for name,item in inventory.items()}))
